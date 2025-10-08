import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
import logging
import os
from tqdm import tqdm
import math
import json
from subsapce_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, MultiTaskGaLoreAdamW
from adaptive_gating import GradientBalanceController, AdaptiveTemperatureScheduler
from null_space_utils import NullSpaceProjector, get_target_layers

def evaluate(model, valid_loader, collator, device, args):
    """验证函数"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating", disable=args.rank != 0):
            # 处理batch数据
            batch = collator(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 移除task_type参数，因为模型不接受这个参数
            model_inputs = {k: v for k, v in batch.items() if k != "task_type"}
            
            # 前向传播
            outputs = model(**model_inputs)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if args.distributed:
        # 在分布式环境中同步损失
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        torch.distributed.all_reduce(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item() / args.world_size
    
    return avg_loss


def compute_gradient_balanced_weights(rec_grads, src_grads):
    """
    基于梯度范数计算平衡权重
    """
    rec_grad_norm = 0.0
    src_grad_norm = 0.0
    
    # 计算推荐任务梯度范数
    for param in rec_grads:
        if rec_grads[param] is not None:
            rec_grad_norm += rec_grads[param].norm().item() ** 2
    rec_grad_norm = rec_grad_norm ** 0.5
    
    # 计算搜索任务梯度范数
    for param in src_grads:
        if src_grads[param] is not None:
            src_grad_norm += src_grads[param].norm().item() ** 2
    src_grad_norm = src_grad_norm ** 0.5
    
    # 让梯度范数大的任务获得更多权重
    total_norm = rec_grad_norm + src_grad_norm
    if total_norm == 0:
        device = next(iter(rec_grads.values())).device if rec_grads else next(iter(src_grads.values())).device
        return torch.tensor([0.5, 0.5], device=device)

    rec_weight = rec_grad_norm / total_norm
    src_weight = src_grad_norm / total_norm
    
    device = next(iter(rec_grads.values())).device if rec_grads else next(iter(src_grads.values())).device
    return torch.tensor([rec_weight, src_weight], device=device)


def compute_balanced_weights(rec_grads, src_grads, task_types=None, adjustment_factor=0.3):
    """
    Compute balanced weights based on task ratios and gradient norms.
    First, assign base weights based on task ratios in the batch, then adjust using gradient norms.

    Args:
        rec_grads: Gradients for recommendation task
        src_grads: Gradients for search task
        task_types: Task types in the batch
        adjustment_factor: Adjustment factor for gradient norms (0-1, larger values mean greater adjustment)
    """
    if rec_grads is None and src_grads is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        return torch.tensor([0.5, 0.5], device=device)
    
    if rec_grads is None:
        device = next(iter(src_grads.values())).device
        return torch.tensor([0.0, 1.0], device=device)
    
    if src_grads is None:
        device = next(iter(rec_grads.values())).device
        return torch.tensor([1.0, 0.0], device=device)
    
    # 获取设备信息
    device = next(iter(rec_grads.values())).device
    
    # 1. 先获取基于任务比例的基础权重
    if task_types is not None:
        base_weights = get_task_ratio_weights(task_types, device)
    else:
        base_weights = torch.tensor([0.5, 0.5], device=device)
    
    # 2. 复用compute_gradient_balanced_weights获取梯度权重
    grad_weights = compute_gradient_balanced_weights(rec_grads, src_grads)
    
    # 3. 基于任务比例进行梯度调整
    # 基础权重 + 调整因子 * (梯度权重 - 0.5)
    adjusted_weights = base_weights + adjustment_factor * (grad_weights - 0.5)
    # 确保权重为正且和为1
    adjusted_weights = torch.clamp(adjusted_weights, min=0.01)  # 避免权重为0
    adjusted_weights = adjusted_weights / adjusted_weights.sum()
    return adjusted_weights

# 新增：根据batch内search/rec样本比例分配权重
def get_task_ratio_weights(task_types, device):
    """
    根据batch内search/rec样本比例分配权重
    task_types: tensor/list，0=rec, 1=search
    """
    if isinstance(task_types, torch.Tensor):
        rec_count = (task_types == 0).sum().item()
        src_count = (task_types == 1).sum().item()
    else:
        rec_count = sum([1 for t in task_types if t == 0])
        src_count = sum([1 for t in task_types if t == 1])
    total = rec_count + src_count
    if total == 0:
        return torch.tensor([0.5, 0.5], device=device)
    return torch.tensor([rec_count / total, src_count / total], device=device)


def train_epoch_multi_task(model, train_loader, optimizer, scheduler, collator, device, args, epoch, global_step, save_checkpoint_func, valid_loader=None, tokenizer=None, best_loss=float('inf'), patience_counter=0, null_space_projections=None):
    """使用共享-特定子空间优化策略训练一个epoch - 按token数分别backward再加权"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    steps_per_epoch = len(train_loader)
    save_interval = steps_per_epoch // args.save_steps_per_epoch if args.save_steps_per_epoch > 0 else steps_per_epoch
    
    # 初始化累积梯度、loss和样本计数的字典 - 三种梯度
    accumulated_whole_grads = {}
    accumulated_rec_grads = {}
    accumulated_src_grads = {}
    accumulated_sample_counts = {'rec': 0, 'src': 0}
    accumulated_rec_loss = 0.0
    accumulated_src_loss = 0.0
    
    # 初始化自适应门控控制器
    if not hasattr(train_epoch_multi_task, 'gradient_controller'):
        use_learnable_gating = getattr(args, 'use_learnable_gating', True)
        gating_learning_rate = getattr(args, 'gating_learning_rate', 1e-4)
        initial_temperature = getattr(args, 'initial_temperature', 1.0)
        hidden_dim = getattr(args, 'gating_hidden_dim', 128)
        gating_dropout = getattr(args, 'gating_dropout', 0.1)
        loss_weight_alpha = getattr(args, 'loss_weight_alpha', 0.4)
        grad_weight_beta = getattr(args, 'grad_weight_beta', 0.4)
        sample_weight_gamma = getattr(args, 'sample_weight_gamma', 0.2)
        
        train_epoch_multi_task.gradient_controller = GradientBalanceController(
            use_learnable_gating=use_learnable_gating,
            gating_learning_rate=gating_learning_rate,
            hidden_dim=hidden_dim,
            gating_dropout=gating_dropout,
            initial_temperature=initial_temperature,
            loss_weight_alpha=loss_weight_alpha,
            grad_weight_beta=grad_weight_beta,
            sample_weight_gamma=sample_weight_gamma
        )
        
        min_temp = getattr(args, 'min_temperature', 0.1)
        temp_decay_factor = getattr(args, 'temp_decay_factor', 0.99)
        train_epoch_multi_task.temp_scheduler = AdaptiveTemperatureScheduler(
            initial_temp=initial_temperature,
            min_temp=min_temp,
            decay_factor=temp_decay_factor
        )
    
    gradient_controller = train_epoch_multi_task.gradient_controller
    temp_scheduler = train_epoch_multi_task.temp_scheduler
    
    if args.rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        batch = collator(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        task_types = batch["task_type"]
        rec_indices = [i for i, t in enumerate(task_types) if t == 0]
        src_indices = [i for i, t in enumerate(task_types) if t == 1]

        rec_loss, src_loss = None, None
        rec_tokens, src_tokens = 0, 0
        
        # 先计算整个batch的loss和各个任务的loss（不进行backward）
        whole_batch_loss = None
        
        # 整个batch前向传播
        model_inputs = {k: v for k, v in batch.items() if k != "task_type"}
        whole_outputs = model(**model_inputs)
        whole_batch_loss = whole_outputs.loss
        
        # 处理推荐任务
        if len(rec_indices) > 0:
            rec_in = {k: v[rec_indices] for k, v in batch.items() if k != "task_type"}
            
            rec_outputs = model(**rec_in)
            rec_loss = rec_outputs.loss
            accumulated_rec_loss += rec_loss.item()

        # 处理搜索任务  
        if len(src_indices) > 0:
            src_in = {k: v[src_indices] for k, v in batch.items() if k != "task_type"}
                
            src_outputs = model(**src_in)
            src_loss = src_outputs.loss
            accumulated_src_loss += src_loss.item()
        
        # 计算三种梯度：whole batch、rec 和 search
        model_params = list(model.parameters())
        
        # 1. 计算whole batch梯度
        if whole_batch_loss is not None:
            whole_batch_loss.backward(retain_graph=True)
            whole_batch_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                               for param in model_params]
            # 清零梯度，准备计算下一个
            for param in model_params:
                if param.grad is not None:
                    param.grad.zero_()
        else:
            whole_batch_grads = [torch.zeros_like(param) for param in model_params]
        
        # 2. 计算rec任务梯度
        if rec_loss is not None:
            rec_loss.backward(retain_graph=True)
            rec_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                        for param in model_params]
            # 清零梯度，准备计算下一个
            for param in model_params:
                if param.grad is not None:
                    param.grad.zero_()
        else:
            rec_grads = [torch.zeros_like(param) for param in model_params]
        
        # 3. 计算search任务梯度
        if src_loss is not None:
            src_loss.backward(retain_graph=False)  # 最后一个不需要retain_graph
            src_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                        for param in model_params]
        else:
            src_grads = [torch.zeros_like(param) for param in model_params]
        
        # 累积三种梯度
        if not accumulated_whole_grads:  # 第一次初始化
            accumulated_whole_grads = {p: torch.zeros_like(p) for p in model_params}
            accumulated_rec_grads = {p: torch.zeros_like(p) for p in model_params}
            accumulated_src_grads = {p: torch.zeros_like(p) for p in model_params}
        
        # 累积三种梯度
        for param, whole_grad, rec_grad, src_grad in zip(model_params, whole_batch_grads, rec_grads, src_grads):
            accumulated_whole_grads[param] += whole_grad
            accumulated_rec_grads[param] += rec_grad
            accumulated_src_grads[param] += src_grad
        
        # 累积计数
        accumulated_sample_counts['rec'] += len(rec_indices)
        accumulated_sample_counts['src'] += len(src_indices)

        # 梯度累积边界
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # 准备三种梯度数据给GaLore优化器
            model_params = list(model.parameters())
            
            # 1. 整个batch的平均梯度
            whole_task_grads = {}
            for param in model_params:
                whole_task_grads[param] = accumulated_whole_grads[param] / args.gradient_accumulation_steps
            
            # 2. rec任务的平均梯度
            rec_task_grads = {}
            for param in model_params:
                rec_task_grads[param] = accumulated_rec_grads[param] / args.gradient_accumulation_steps
            
            # 3. search任务的平均梯度
            src_task_grads = {}
            for param in model_params:
                src_task_grads[param] = accumulated_src_grads[param] / args.gradient_accumulation_steps
            
            # 设置梯度到参数上（使用whole batch梯度作为主梯度）
            for param in model_params:
                param.grad = whole_task_grads[param]
            
            # 准备task_gradients参数（传递三种不同的梯度）
            task_gradients = {
                'merged': whole_task_grads,  # 使用whole batch的梯度作为merged
                'rec': rec_task_grads,       # 使用rec任务的梯度
                'src': src_task_grads        # 使用search任务的梯度
            }
            
            # 计算梯度范数用于自适应门控
            rec_grad_norm = 0.0
            src_grad_norm = 0.0
            for param in model_params:
                if param in rec_task_grads:
                    rec_grad_norm += rec_task_grads[param].norm().item() ** 2
                if param in src_task_grads:
                    src_grad_norm += src_task_grads[param].norm().item() ** 2
            rec_grad_norm = rec_grad_norm ** 0.5
            src_grad_norm = src_grad_norm ** 0.5
            
            # 使用自适应门控计算权重
            task_losses = [accumulated_rec_loss, accumulated_src_loss]
            task_grad_norms = [rec_grad_norm, src_grad_norm]
            task_sample_counts_list = [accumulated_sample_counts['rec'], accumulated_sample_counts['src']]
            
            # 计算元损失用于门控网络更新
            meta_loss = (accumulated_rec_loss + accumulated_src_loss) / 2
            
            # 获取自适应权重
            adaptive_weights = gradient_controller.compute_gates(
                task_losses, task_grad_norms, task_sample_counts_list, 
                meta_loss if args.rank == 0 else None  # 只在主进程更新门控网络
            )
            
            gates = {
                'rec': adaptive_weights[0],
                'src': adaptive_weights[1]
            }
            
            # 更新温度调度器
            if args.rank == 0:
                current_temp = temp_scheduler.step()
                if hasattr(gradient_controller, 'gating_net'):
                    gradient_controller.gating_net.temperature.data.fill_(current_temp)

            # 调用GaLore优化器的step方法
            optimizer.step(task_gradients=task_gradients, gates=gates, task_sample_counts=accumulated_sample_counts, null_space_projections=null_space_projections)
            
            # 调试：检查参数更新
            if args.rank == 0 and global_step % (args.print_steps * 10) == 0:
                param_norm = 0.0
                for param in model.parameters():
                    param_norm += param.data.norm().item() ** 2
                param_norm = param_norm ** 0.5
                logging.info(f"Step {global_step}: Parameter norm = {param_norm:.6f}")
            
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()

            # 清空累积值 - 清空三种梯度
            accumulated_whole_grads.clear()
            accumulated_rec_grads.clear()
            accumulated_src_grads.clear()
            accumulated_sample_counts = {'rec': 0, 'src': 0}
            accumulated_rec_loss = 0.0
            accumulated_src_loss = 0.0
            
            global_step += 1
            
            if args.rank == 0 and global_step % args.print_steps == 0:
                rec_loss_val = rec_loss.item() if rec_loss is not None else 0.0
                src_loss_val = src_loss.item() if src_loss is not None else 0.0
                
                # 调试：比较三种梯度的范数
                whole_grad_norm = 0.0
                rec_grad_norm = 0.0
                src_grad_norm = 0.0
                
                for param in model_params:
                    if param in whole_task_grads:
                        whole_grad_norm += whole_task_grads[param].norm().item() ** 2
                    if param in rec_task_grads:
                        rec_grad_norm += rec_task_grads[param].norm().item() ** 2
                    if param in src_task_grads:
                        src_grad_norm += src_task_grads[param].norm().item() ** 2
                
                whole_grad_norm = whole_grad_norm ** 0.5
                rec_grad_norm = rec_grad_norm ** 0.5
                src_grad_norm = src_grad_norm ** 0.5
                
                logging.info(f"Epoch {epoch}, Step {global_step}: Rec Loss = {rec_loss_val:.4f}, Src Loss = {src_loss_val:.4f}")
                logging.info(f"Gradient Norms - Whole: {whole_grad_norm:.6f}, Rec: {rec_grad_norm:.6f}, Src: {src_grad_norm:.6f}")
                logging.info(f"Adaptive Gates - Rec: {gates['rec']:.4f}, Src: {gates['src']:.4f}")
                
                # 记录温度信息
                if hasattr(gradient_controller, 'gating_net'):
                    current_temp = gradient_controller.gating_net.temperature.item()
                    logging.info(f"Gating Temperature: {current_temp:.4f}")
                
                # 检查是否有 NaN 或 Inf
                for grad_name, grad_norm in [("Whole", whole_grad_norm), ("Rec", rec_grad_norm), ("Src", src_grad_norm)]:
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        logging.warning(f"{grad_name} gradient norm is {grad_norm}, may cause training instability!")

        # 计算用于日志记录的meta loss（简单平均）
        meta_loss = 0.0
        if rec_loss is not None and src_loss is not None:
            # 简单平均两个任务的loss
            meta_loss = (rec_loss.item() + src_loss.item()) / 2
        elif rec_loss is not None:
            meta_loss = rec_loss.item()
        elif src_loss is not None:
            meta_loss = src_loss.item()

        total_loss += meta_loss
        num_batches += 1
        
        if (batch_idx + 1) % save_interval == 0 and args.rank == 0:
            step_loss = total_loss / num_batches
            if valid_loader is not None:
                valid_loss = evaluate(model, valid_loader, collator, args.device, args)
                logging.info(f"Epoch {epoch} Step {batch_idx}: Train Loss: {step_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                
                should_stop, new_best_loss, new_patience_counter = save_checkpoint_func(
                    model, optimizer, scheduler, epoch, valid_loss, args, 
                    tokenizer, f"epoch_{epoch}_step_{batch_idx}", step_loss, 
                    False, best_loss, patience_counter
                )
                best_loss = new_best_loss
                patience_counter = new_patience_counter
                
                if should_stop:
                    logging.info("Early stopping triggered, ending training")
                    return total_loss / num_batches, global_step, best_loss, patience_counter
            else:
                logging.info(f"Epoch {epoch} Step {batch_idx}: Train Loss: {step_loss:.4f}")
        
        if args.rank == 0:
            pbar.set_postfix({'loss': f'{meta_loss:.4f}'})
    
    avg_loss = total_loss / num_batches
    if args.distributed:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        torch.distributed.all_reduce(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item() / args.world_size
    
    return avg_loss, global_step, best_loss, patience_counter


def train_epoch(model, train_loader, optimizer, scheduler, collator, device, args, epoch, global_step, save_checkpoint_func, valid_loader=None, tokenizer=None, best_loss=float('inf'), patience_counter=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 计算每个epoch需要保存/验证几次
    steps_per_epoch = len(train_loader)
    save_interval = steps_per_epoch // args.save_steps_per_epoch if args.save_steps_per_epoch > 0 else steps_per_epoch
    
    # 增加早停检查频率 - 每N个batch检查一次
    early_stop_check_interval = max(1, steps_per_epoch // getattr(args, 'early_stop_check_frequency', 10))  # 每个epoch检查指定次数
    
    # 创建进度条
    if args.rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        # 处理batch数据
        batch = collator(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 移除task_type参数，因为模型不接受这个参数
        model_inputs = {k: v for k, v in batch.items() if k != "task_type"}
        
        # 前向传播
        outputs = model(**model_inputs)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪
            if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 更新全局步数
            global_step += 1
            
            # 按step打印loss
            if args.rank == 0 and global_step % args.print_steps == 0:
                logging.info(f"Epoch {epoch}, Step {global_step}: Loss = {loss.item():.4f}")
        
        total_loss += loss.item()
        num_batches += 1
        
        # 按step验证和保存checkpoint
        if (batch_idx + 1) % save_interval == 0 and args.rank == 0:
            step_loss = total_loss / num_batches
            
            # 验证
            if valid_loader is not None:
                valid_loss = evaluate(model, valid_loader, collator, args.device, args)
                logging.info(f"Epoch {epoch} Step {batch_idx}: Train Loss: {step_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                
                # 调用保存函数，传递验证损失
                should_stop, new_best_loss, new_patience_counter = save_checkpoint_func(
                    model, optimizer, scheduler, epoch, valid_loss, args, 
                    tokenizer, None, step_loss, 
                    False, best_loss, patience_counter
                )
                best_loss = new_best_loss
                patience_counter = new_patience_counter
                
                if should_stop:
                    logging.info("Early stopping triggered, ending training")
                    return total_loss / num_batches, global_step, best_loss, patience_counter
            else:
                logging.info(f"Epoch {epoch} Step {batch_idx}: Train Loss: {step_loss:.4f}")
          
        # 更新进度条
        if args.rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    
    if args.distributed:
        # 在分布式环境中同步损失
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        torch.distributed.all_reduce(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item() / args.world_size
    
    return avg_loss, global_step, best_loss, patience_counter


def save_checkpoint(model, optimizer, scheduler, epoch, loss, args, tokenizer, checkpoint_name, train_loss=None, is_best=False, best_loss=float('inf'), patience_counter=0):
    """保存检查点 - 只保存最佳模型"""
    if args.rank != 0:
        return False, best_loss, patience_counter
    
    print(f"Saving checkpoint at epoch {epoch}, loss: {loss:.4f}, best_loss: {best_loss:.4f}, patience_counter: {patience_counter}")
    # 检查是否是最佳模型
    is_best_model = loss < best_loss
    if is_best_model:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    # 只保存最佳模型
    if is_best_model:
        best_dir = os.path.join(args.output_dir, 'checkpoint-best')
        try:
            if os.path.exists(best_dir):
                import shutil
                shutil.rmtree(best_dir)
            os.makedirs(best_dir, exist_ok=True)
            
            # 保存模型（使用transformers的save_pretrained方法）
            model_to_save = model.module if args.distributed else model
            model_to_save.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            
            # 保存训练状态信息
            checkpoint_info = {
                'epoch': epoch,
                'best_loss': best_loss,
                'patience_counter': patience_counter,
                'train_loss': train_loss
            }
            
            import json
            with open(os.path.join(best_dir, 'training_state.json'), 'w') as f:
                json.dump(checkpoint_info, f)
            
            logging.info(f"Saved best model with loss: {loss:.4f} at {best_dir}")
        except Exception as e:
            logging.warning(f"Failed to save best checkpoint: {e}")
    else:
        logging.info(f"Validation loss {loss:.4f} not better than best loss {best_loss:.4f}, skipping checkpoint save")
    
    # 早停检查
    if patience_counter >= args.early_stopping_patience:  # 使用参数设置早停耐心值
        logging.info(f"Early stopping triggered after {patience_counter} validation steps without improvement")
        return True, best_loss, patience_counter  # 返回True表示需要早停
    
    return False, best_loss, patience_counter  # 返回False表示继续训练


def load_checkpoint(model, optimizer, scheduler, args, tokenizer):
    """加载检查点 - 从transformers标准格式加载"""
    # 尝试从最佳checkpoint加载
    best_path = os.path.join(args.output_dir, 'checkpoint-best')
    if not os.path.exists(best_path):
        return 0, float('inf'), 0  # 返回 start_epoch, best_loss, patience_counter
    
    checkpoint_dir = best_path
    
    # 使用transformers的from_pretrained方法加载模型
    model_to_load = model.module if args.distributed else model
    loaded_model = type(model_to_load).from_pretrained(checkpoint_dir)
    
    # 更新模型权重
    if args.distributed:
        model.module.load_state_dict(loaded_model.state_dict())
    else:
        model.load_state_dict(loaded_model.state_dict())
    
    # 加载训练状态信息
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        try:
            import json
            with open(training_state_path, 'r') as f:
                checkpoint_info = json.load(f)
            
            start_epoch = checkpoint_info.get('epoch', 0) + 1  # 从下一个epoch开始
            best_loss = checkpoint_info.get('best_loss', float('inf'))
            patience_counter = checkpoint_info.get('patience_counter', 0)
            
            logging.info(f"Resumed from checkpoint-best: epoch {start_epoch-1}, best_loss: {best_loss:.4f}, patience_counter: {patience_counter}")
            return start_epoch, best_loss, patience_counter
        except Exception as e:
            logging.warning(f"Failed to load training state: {e}, using default values")
    
    # 如果没有training_state.json或加载失败，使用默认值
    logging.info("Loaded model from checkpoint-best but no training state found, using default values")
    return 0, float('inf'), 0


def create_data_loaders(train_data, valid_data, args):
    """创建数据加载器"""
    train_sampler = None
    valid_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=False)
    
    train_loader = DataLoader(
        train_data,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.per_device_batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, valid_loader, train_sampler, valid_sampler


def create_optimizer_and_scheduler(model, train_loader, args, null_space_projections=None):
    """创建优化器和学习率调度器"""
    # 计算总训练步数
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    # 使用 GaLore 优化器
    # 只对 attention 和 MLP 层的权重使用 GaLore
    galore_params = []
    # T5 模型中的关键层名称
    target_modules_list = [
        "SelfAttention.q", "SelfAttention.k", "SelfAttention.v", "SelfAttention.o",  # Self Attention
        "EncDecAttention.q", "EncDecAttention.k", "EncDecAttention.v", "EncDecAttention.o",  # Cross Attention
        "DenseReluDense.wi_0", "DenseReluDense.wi_1", "DenseReluDense.wo"  # MLP
    ]
    
    print("Searching for GaLore parameters...")
    param_to_layer_name = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # 打印所有找到的 Linear 层，帮助调试
        # print(f"Found Linear layer: {module_name}")
        if any(target_key in module_name for target_key in target_modules_list):
            # print(f'Enable GaLore for weights in module: {module_name}')
            galore_params.append(module.weight)
            param_to_layer_name[id(module.weight)] = module_name

    id_galore_params = [id(p) for p in galore_params]
    # 其他参数使用普通优化
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

    print(f"Number of GaLore parameters: {len(galore_params)}")
    print(f"Number of regular parameters: {len(regular_params)}")

    param_groups = [{'params': regular_params}]
    for param in galore_params:
        layer_name = param_to_layer_name[id(param)]
        projection_matrix = null_space_projections.get(layer_name)
        param_group = {
            'params': [param],
            'rank': getattr(args, 'galore_rank', 1024),
            'update_proj_gap': getattr(args, 'galore_update_proj_gap', 200),
            'scale': getattr(args, 'galore_scale', 0.25),
            'proj_type': 'std',
            'multi_task': True,  # 标记这是多任务参数组
            'null_space_projector': NullSpaceProjector(),
            'projection_matrix': projection_matrix
        }
        print(f'added null space projection for layer: {layer_name}')
        param_groups.append(param_group)


    # param_groups = [
    #     {'params': regular_params},
    #     {
    #         'params': galore_params,
    #         'rank': getattr(args, 'galore_rank', 1024),
    #         'update_proj_gap': getattr(args, 'galore_update_proj_gap', 200),
    #         'scale': getattr(args, 'galore_scale', 0.25),
    #         'proj_type': 'std',
    #         'multi_task': True  # 标记这是多任务参数组
    #     }
    # ]
    
    gradient_merge_strategy = getattr(args, 'gradient_merge_strategy', 'weighted')
    optimizer = MultiTaskGaLoreAdamW(
        param_groups, 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        gradient_merge_strategy=gradient_merge_strategy
        )


    # 定义学习率调度器
    if args.lr_scheduler_type == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif args.lr_scheduler_type == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_model(model, train_loader, valid_loader, optimizer, scheduler, collator, args, tokenizer):
    """训练模型的主函数"""
    # 加载检查点（如果存在）
    start_epoch, best_loss, patience_counter = load_checkpoint(model, optimizer, scheduler, args, tokenizer)

    # 训练循环
    model.config.use_cache = False
    global_step = 0

    # 选择训练函数
    train_func = train_epoch_multi_task if getattr(args, 'use_dual_space', False) else train_epoch

    for epoch in range(start_epoch, args.epochs):
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch，传入验证数据加载器
        train_loss, global_step, best_loss, patience_counter = train_func(
            model, train_loader, optimizer, scheduler,
            collator, args.device, args, epoch, global_step,
            lambda *args, **kwargs: save_checkpoint(*args, **kwargs),
            valid_loader, tokenizer,
            best_loss,  # 传入当前的best_loss
            patience_counter
        )
        
        # 记录epoch结束的日志
        if args.rank == 0:
            logging.info(f"Epoch {epoch} completed: Train Loss: {train_loss:.4f}")
        
        # 检查是否早停
        if patience_counter >= args.early_stopping_patience:
            if args.rank == 0:
                logging.info("Training stopped due to early stopping")
            break

    return model