import torch
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import os
from pathlib import Path


class SecondMoment:
    """计算二阶矩统计量（协方差矩阵）"""
    
    def __init__(self):
        self.count = 0
        self.mom2 = None

    def add(self, a):
        """添加batch数据到统计量中"""
        if len(a.shape) > 2:
            a = a.view(-1, a.shape[-1])  # flatten to [batch*seq_len, hidden_dim]
        
        if len(a) == 0:
            return
            
        # 初始化
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        
        batch_count = a.shape[0]
        self.count += batch_count
        # 累积 a^T @ a
        self.mom2 += a.t().mm(a)

    def moment(self):
        """返回协方差矩阵 E[x x^T]"""
        return self.mom2 / self.count if self.count > 0 else self.mom2

    def to(self, device):
        if self.mom2 is not None:
            self.mom2 = self.mom2.to(device)
        return self


class NullSpaceProjector:
    """Null space projection工具类"""
    
    def __init__(self, nullspace_threshold: float = 2e-2):
        self.nullspace_threshold = nullspace_threshold
        self.projection_matrices = {}
        
    def compute_covariance_from_data(self, model, tokenizer, layer_name: str, 
                                   dataset_name: str = "wikipedia", 
                                   n_samples: int = 10000,
                                   batch_size: int = 1,
                                   max_seq_length: int = 512):
        """
        Compute the covariance matrix for a specific layer from data.

        Args:
            model: The model containing the layer.
            tokenizer: Tokenizer for processing input data.
            layer_name: Name of the layer to compute covariance for.
            dataset_name: Dataset to use for computation (default: Wikipedia).
            n_samples: Number of samples to process.
            batch_size: Batch size for processing.
            max_seq_length: Maximum sequence length for tokenization.
        """
        
        print(f"Computing covariance matrix for layer: {layer_name}")
        
        # 首先获取目标层的权重形状来确定需要计算哪个维度的协方差
        target_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None or not hasattr(target_module, 'weight'):
            raise ValueError(f"Layer {layer_name} not found or has no weight")
        
        weight_shape = target_module.weight.shape  # [out_features, in_features]
        out_features, in_features = weight_shape
        print(f"Target layer weight shape: {weight_shape}")
        
        raw_ds = load_dataset("/nas/user/zhaoxin/chiyin/modelscope/swift___wikipedia/20220301.en/2.0.0",split="train")
        # 初始化统计量
        stat = SecondMoment()
        
        model.eval()
        device = next(model.parameters()).device
        
        # 准备hook来捕获激活和输出
        activations = {}
        outputs = {}
        
        def get_activation_and_output(name):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    activations[name] = input[0].detach()
                else:
                    activations[name] = input.detach()
                outputs[name] = output.detach()
            return hook
        
        handle = target_module.register_forward_hook(get_activation_and_output(layer_name))
        
        try:
            processed_samples = 0
            with torch.no_grad():
                for item in tqdm(raw_ds, desc="Processing samples", total=n_samples):
                    if processed_samples >= n_samples:
                        break
                    
                    # 处理文本
                    if dataset_name == "wikipedia":
                        text = item.get("text", "")
                    else:
                        text = item.get("text", "")
                    
                    if len(text.strip()) < 50:  # 跳过太短的文本
                        continue
                    
                    # Tokenize
                    inputs = tokenizer(
                        text,
                        max_length=max_seq_length,
                        truncation=True,
                        padding=False,
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    inputs["decoder_input_ids"] = inputs["input_ids"].clone()
                    
                    # 前向传播
                    _ = model(**inputs)
                    
                    # 提取输出特征来构建输出特征空间的协方差矩阵
                    if layer_name in outputs:
                        output_feats = outputs[layer_name]  # 层的输出
                        
                        # 检查输出形状并进行适当的处理
                        original_shape = output_feats.shape
                        print(f"Debug: Layer {layer_name} output shape: {original_shape}, expected out_features: {out_features}")
                        
                        # 处理不同维度的输出
                        if output_feats.dim() == 4:  # [batch, heads, seq_len, seq_len] for attention bias
                            # 对于attention bias等特殊层，重塑为 [batch*heads*seq_len, seq_len]
                            output_feats = output_feats.view(-1, original_shape[-1])
                        elif output_feats.dim() == 3:  # [batch, seq_len, out_features]
                            # 使用attention_mask来过滤padding
                            if "attention_mask" in inputs:
                                mask = inputs["attention_mask"].bool().unsqueeze(-1)
                                if mask.shape[-1] == 1 and output_feats.shape[-1] != 1:
                                    mask = mask.expand(-1, -1, output_feats.shape[-1])
                                try:
                                    output_feats = output_feats[mask.expand_as(output_feats)]
                                    # 计算实际的特征数量
                                    valid_tokens = mask.sum().item()
                                    if valid_tokens > 0:
                                        output_feats = output_feats.view(valid_tokens, -1)
                                except:
                                    # 如果mask操作失败，直接reshape
                                    output_feats = output_feats.view(-1, output_feats.shape[-1])
                            else:
                                output_feats = output_feats.view(-1, output_feats.shape[-1])
                        elif output_feats.dim() == 2:
                            # 已经是 [batch*seq_len, out_features] 形状
                            pass
                        else:
                            # 处理其他维度，展平为2D
                            output_feats = output_feats.view(-1, output_feats.shape[-1])
                        
                        # 检查最终形状是否合理
                        if output_feats.numel() > 0 and output_feats.shape[-1] > 0:
                            # 如果最后一个维度与预期的out_features不匹配，调整处理方式
                            if output_feats.shape[-1] != out_features:
                                print(f"Warning: Output features last dim {output_feats.shape[-1]} != expected {out_features}")
                                print(f"Reshaping output to match expected dimensions...")
                                # 尝试重塑以匹配out_features
                                total_elements = output_feats.numel()
                                if total_elements % out_features == 0:
                                    output_feats = output_feats.view(-1, out_features)
                                else:
                                    print(f"Cannot reshape {total_elements} elements to [-1, {out_features}], skipping this sample")
                                    continue
                            
                            stat.add(output_feats.float())
                            processed_samples += 1
                        else:
                            print(f"Warning: Empty output features for layer {layer_name}, skipping")
                    
                    # 清理激活缓存
                    activations.clear()
                    outputs.clear()
                    
        except Exception as e:
            print(f"Error during covariance computation: {e}")
            raise
        finally:
            handle.remove()
        
        print(f"Processed {processed_samples} samples")
        cov_matrix = stat.moment().cpu()
        print(f"Computed covariance matrix shape: {cov_matrix.shape}")
        return cov_matrix
    
    def compute_projection_matrix(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        """从协方差矩阵计算null space投影矩阵"""
        
        print(f"Computing null space projection matrix...")
        print(f"Covariance matrix shape: {cov_matrix.shape}")
        
        # 确保协方差矩阵是float类型进行SVD计算（SVD需要高精度）
        original_dtype = cov_matrix.dtype
        if cov_matrix.dtype != torch.float32:
            cov_matrix = cov_matrix.float()
        
        # SVD分解
        U, S, _ = torch.linalg.svd(cov_matrix, full_matrices=False)
        
        print(f"Singular values range: {S.min():.6f} - {S.max():.6f}")
        
        # 找到小奇异值对应的索引（null space）
        small_singular_indices = (S < self.nullspace_threshold).nonzero(as_tuple=True)[0]
        
        print(f"Number of null space dimensions: {len(small_singular_indices)} / {len(S)}")
        print(f"Null space threshold: {self.nullspace_threshold}")
        
        if len(small_singular_indices) == 0:
            print("Warning: No null space found with current threshold!")
            # 使用最小的几个奇异值
            small_singular_indices = torch.argsort(S)[:max(1, len(S) // 20)]
            print(f"Using smallest {len(small_singular_indices)} dimensions instead")
        
        # 构建投影矩阵 P = U_null @ U_null^T
        U_null = U[:, small_singular_indices]
        P = U_null @ U_null.T
        
        print(f"Projection matrix shape: {P.shape}")
        return P
    
    def get_or_compute_projection_matrix(self, model, tokenizer, layer_name: str,
                                       cache_dir: str = "/nas/user/zhaoxin/chiyin/rag/qilin_rerank/GSRS_v1_galore_moe_null/null_space_cache",
                                       force_recompute: bool = False,
                                       dataset_name: str = "wikipedia",
                                       n_samples: int = 10000) -> torch.Tensor:
        """获取或计算投影矩阵（带缓存功能）"""
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件名
        model_name = model.config._name_or_path.replace("/", "_")
        cache_file = Path(cache_dir) / f"{model_name}_{layer_name.replace('.', '_')}_nullspace_projection_th{self.nullspace_threshold}.pt"
        
        # 尝试加载缓存
        if cache_file.exists() and not force_recompute:
            print(f"Loading cached projection matrix from {cache_file}")
            try:
                P = torch.load(cache_file, map_location='cpu')
                return P
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        
        # 计算协方差矩阵
        print(f"Computing covariance matrix for {layer_name}...")
        cov_matrix = self.compute_covariance_from_data(
            model, tokenizer, layer_name, 
            dataset_name=dataset_name, 
            n_samples=n_samples
        )
        
        # 计算投影矩阵
        P = self.compute_projection_matrix(cov_matrix)
        
        # 保存到缓存
        try:
            torch.save(P, cache_file)
            print(f"Cached projection matrix to {cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
        return P
    
    def project_to_null_space(self, tensor: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """将张量投影到null space"""
        original_shape = tensor.shape
        
        # 确保投影矩阵和张量有相同的数据类型和设备
        projection_matrix = projection_matrix.to(tensor.device, dtype=tensor.dtype)
        
        # 检查形状兼容性
        if len(original_shape) == 2:
            # 权重矩阵: [out_features, in_features]
            out_features, in_features = original_shape
            
            # 投影矩阵应该是 [out_features, out_features]
            if projection_matrix.shape != (out_features, out_features):
                print(f"Shape mismatch: projection_matrix {projection_matrix.shape} vs tensor {original_shape}")
                print(f"Expected projection_matrix shape: ({out_features}, {out_features})")
                # 如果形状不匹配，跳过投影
                return tensor
            
            # 执行投影: P @ W，其中 P 是投影矩阵，W 是权重矩阵
            projected = projection_matrix @ tensor
        else:
            # 处理其他形状的张量
            tensor_2d = tensor.view(tensor.shape[0], -1)
            if projection_matrix.shape[0] != tensor_2d.shape[0]:
                print(f"Shape mismatch: projection_matrix {projection_matrix.shape} vs reshaped tensor {tensor_2d.shape}")
                return tensor
            projected_2d = projection_matrix @ tensor_2d
            projected = projected_2d.view(original_shape)
        
        return projected


def get_target_layers(model, target_modules_list: List[str] = ["attn", "mlp"]) -> List[str]:
    """获取目标层的名称列表，支持T5和其他模型"""
    target_layers = []
    
    # 检查是否是T5模型
    is_t5_model = hasattr(model.config, 'model_type') and 't5' in model.config.model_type.lower()
    
    if is_t5_model:
        print("T5 model detected. Using T5-specific layer targeting.")
        # T5模型的特殊处理
        for module_name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
                # 排除一些特殊层
                if any(exclude_key in module_name for exclude_key in ['relative_attention_bias', 'embed_tokens', 'lm_head']):
                    continue
                
                # T5的注意力层
                if 'attn' in target_modules_list:
                    if any(attn_key in module_name for attn_key in ['SelfAttention', 'EncDecAttention']):
                        if any(weight_key in module_name for weight_key in ['.q.', '.k.', '.v.', '.o.']):
                            target_layers.append(module_name)
                
                # T5的MLP层
                if 'mlp' in target_modules_list:
                    if 'DenseReluDense' in module_name:
                        if any(weight_key in module_name for weight_key in ['.wi.', '.wo.']):
                            target_layers.append(module_name)
    else:
        # 其他模型的通用处理
        for module_name, module in model.named_modules():
            if hasattr(module, 'weight') and any(target_key in module_name for target_key in target_modules_list):
                target_layers.append(module_name)
    
    return target_layers


def compute_null_space_projections(model, tokenizer, target_modules_list: List[str] = ["attn", "mlp"],
                                 cache_dir: str = "./null_space_cache",
                                 nullspace_threshold: float = 2e-2,
                                 force_recompute: bool = False) -> Dict[str, torch.Tensor]:
    """为所有目标层计算null space投影矩阵"""
    
    projector = NullSpaceProjector(nullspace_threshold=nullspace_threshold)
    projection_matrices = {}
    
    # 获取目标层
    target_layers = get_target_layers(model, target_modules_list)
    print(f"Found {len(target_layers)} target layers")
    
    for layer_name in tqdm(target_layers, desc="Computing null space projections"):
        try:
            P = projector.get_or_compute_projection_matrix(
                model, tokenizer, layer_name, cache_dir, force_recompute
            )
            projection_matrices[layer_name] = P
            print(f"✓ Computed projection matrix for {layer_name}")
        except Exception as e:
            print(f"✗ Failed to compute projection matrix for {layer_name}: {e}")
    
    return projection_matrices