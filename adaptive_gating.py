import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGatingNetwork(nn.Module):
    """
    Adaptive gating network for learning gradient fusion weights for search and recommendation tasks.

    Args:
        hidden_dim: Hidden dimension size for the network.
        num_tasks: Number of tasks (default: 2).
        dropout: Dropout rate for regularization.
        initial_temperature: Initial temperature for softmax sharpness.
    """
    def __init__(self, hidden_dim=128, num_tasks=2, dropout=0.1, initial_temperature=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_tasks * 3, hidden_dim),  # Input: loss, grad_norm, sample_count for each task
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating weight generation
        self.gate_generator = nn.Linear(hidden_dim // 2, num_tasks)
        
        # Temperature parameter to control the sharpness of softmax
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        
    def forward(self, task_losses, task_grad_norms, task_sample_counts):
        """
        Args:
            task_losses: [rec_loss, src_loss]
            task_grad_norms: [rec_grad_norm, src_grad_norm] 
            task_sample_counts: [rec_count, src_count]
        Returns:
            gates: [rec_weight, src_weight] (sum to 1)
        """
        # Feature normalization
        losses = torch.tensor(task_losses, dtype=torch.float32)
        grad_norms = torch.tensor(task_grad_norms, dtype=torch.float32)
        sample_counts = torch.tensor(task_sample_counts, dtype=torch.float32)
        
        # Avoid division by zero
        total_loss = losses.sum()
        total_samples = sample_counts.sum()
        
        if total_loss > 1e-8:
            loss_ratios = losses / total_loss
        else:
            loss_ratios = torch.ones_like(losses) / len(losses)
            
        if total_samples > 0:
            sample_ratios = sample_counts / total_samples
        else:
            sample_ratios = torch.ones_like(sample_counts) / len(sample_counts)
            
        grad_norm_total = grad_norms.sum()
        if grad_norm_total > 1e-8:
            grad_ratios = grad_norms / grad_norm_total
        else:
            grad_ratios = torch.ones_like(grad_norms) / len(grad_norms)
        
        # Concatenate features
        features = torch.cat([loss_ratios, grad_ratios, sample_ratios])
        
        # Extract features
        hidden = self.feature_extractor(features)
        
        # Generate gating weights
        logits = self.gate_generator(hidden)
        
        # Temperature-controlled softmax
        gates = F.softmax(logits / self.temperature, dim=0)
        
        return gates


class GradientBalanceController:
    """
    梯度平衡控制器，结合多种策略
    """
    def __init__(self, use_learnable_gating=True, momentum=0.9, 
                 gating_learning_rate=1e-4, hidden_dim=128, gating_dropout=0.1, 
                 initial_temperature=1.0, loss_weight_alpha=0.4, grad_weight_beta=0.4, 
                 sample_weight_gamma=0.2):
        self.use_learnable_gating = use_learnable_gating
        self.momentum = momentum
        self.ema_loss_ratios = None
        self.ema_grad_ratios = None
        
        # 启发式权重参数
        self.loss_weight_alpha = loss_weight_alpha
        self.grad_weight_beta = grad_weight_beta
        self.sample_weight_gamma = sample_weight_gamma
        
        if use_learnable_gating:
            self.gating_net = AdaptiveGatingNetwork(
                hidden_dim=hidden_dim, 
                dropout=gating_dropout,
                initial_temperature=initial_temperature
            )
            self.gating_optimizer = torch.optim.Adam(
                self.gating_net.parameters(), 
                lr=gating_learning_rate
            )
    
    def compute_gates(self, task_losses, task_grad_norms, task_sample_counts, 
                     meta_loss=None, update_gating=True):
        """
        计算门控权重
        """
        if self.use_learnable_gating:
            gates = self.gating_net(task_losses, task_grad_norms, task_sample_counts)
            
            # 如果提供了meta_loss，更新门控网络
            if meta_loss is not None and update_gating:
                self._update_gating_network(meta_loss)
                
            return gates.detach().cpu().numpy()
        else:
            # 使用启发式方法
            return self._heuristic_gates(task_losses, task_grad_norms, task_sample_counts)
    
    def _heuristic_gates(self, task_losses, task_grad_norms, task_sample_counts):
        """
        启发式门控策略：结合loss、梯度范数和样本数
        """
        # 基于loss的权重（loss大的任务权重大）
        total_loss = sum(task_losses)
        if total_loss > 1e-8:
            loss_weights = [l / total_loss for l in task_losses]
        else:
            loss_weights = [0.5, 0.5]
        
        # 基于梯度范数的权重（梯度大的任务权重大）
        total_grad = sum(task_grad_norms)
        if total_grad > 1e-8:
            grad_weights = [g / total_grad for g in task_grad_norms]
        else:
            grad_weights = [0.5, 0.5]
        
        # 基于样本数的权重
        total_samples = sum(task_sample_counts)
        if total_samples > 0:
            sample_weights = [s / total_samples for s in task_sample_counts]
        else:
            sample_weights = [0.5, 0.5]
        
        # 加权组合 (使用可配置的系数)
        alpha, beta, gamma = self.loss_weight_alpha, self.grad_weight_beta, self.sample_weight_gamma
        final_weights = [
            alpha * loss_weights[i] + beta * grad_weights[i] + gamma * sample_weights[i]
            for i in range(len(task_losses))
        ]
        
        # 归一化
        total = sum(final_weights)
        return [w / total for w in final_weights]
    
    def _update_gating_network(self, meta_loss):
        """
        更新门控网络，目标是最小化元损失
        """
        if hasattr(meta_loss, 'backward'):
            self.gating_optimizer.zero_grad()
            meta_loss.backward(retain_graph=True)
            self.gating_optimizer.step()


class AdaptiveTemperatureScheduler:
    """
    自适应温度调度器，用于控制门控的锐度
    """
    def __init__(self, initial_temp=1.0, min_temp=0.1, decay_factor=0.99):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.decay_factor = decay_factor
        self.current_temp = initial_temp
        
    def step(self, performance_improvement=None):
        """
        根据性能改善情况调整温度
        """
        if performance_improvement is not None:
            # 如果性能在改善，保持当前温度；否则降低温度
            if performance_improvement > 0:
                self.current_temp = min(self.current_temp * 1.01, self.initial_temp)
            else:
                self.current_temp = max(self.current_temp * self.decay_factor, self.min_temp)
        else:
            # 默认衰减
            self.current_temp = max(self.current_temp * self.decay_factor, self.min_temp)
            
        return self.current_temp
