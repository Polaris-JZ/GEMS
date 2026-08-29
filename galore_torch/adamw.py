# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable this warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                    
                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group['dim'] <=2:
                            state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class MultiTaskGaLoreAdamW(AdamW):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-6,
        correct_bias: bool = True,
        gradient_merge_strategy: str = "weighted"
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.gradient_merge_strategy = gradient_merge_strategy

    @torch.no_grad()
    def step(self, closure: Callable = None, task_gradients=None, gates=None, task_sample_counts=None, null_space_projections=None):
        """
        Perform optimization step, supporting shared-specific subspace decomposition and gating mechanism.
        task_gradients: Dictionary containing gradients for each task {'rec': grad_rec, 'src': grad_src}
        gates: Dictionary containing gating coefficients for each task {'rec': alpha_r, 'src': alpha_s}
        task_sample_counts: Dictionary containing sample counts for each task {'rec': N_rec, 'src': N_src}
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None and (task_gradients is None or (task_gradients['rec'].get(p, None) is None and task_gradients['src'].get(p, None) is None)):
                    continue

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2

                # 共享-特定 GaLore 投影
                if "rank" in group and group.get("multi_task", False):
                    null_space_projector = group.get("null_space_projector", None)
                    projection_matrix = group.get("projection_matrix", None)
                    # 计算子平面维度：rec和src特定子平面的维度是共享子平面的一半
                    shared_rank = group["rank"]
                    specific_rank = max(1, shared_rank // 2)  # 确保至少为1
                    # 1. 初始化三个投影器
                    if "projector_shared" not in state:
                        state["projector_shared"] = GaLoreProjector(shared_rank, update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"], null_space_projector=null_space_projector, projection_matrix=projection_matrix)
                    if "projector_rec_specific" not in state:
                        state["projector_rec_specific"] = GaLoreProjector(specific_rank, update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"], null_space_projector=null_space_projector, projection_matrix=projection_matrix)
                    if "projector_src_specific" not in state:
                        state["projector_src_specific"] = GaLoreProjector(specific_rank, update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"], null_space_projector=null_space_projector, projection_matrix=projection_matrix)

                    state["step"] += 1
                    
                    if task_gradients is not None:
                        rec_grad = task_gradients['rec'].get(p, None)
                        src_grad = task_gradients['src'].get(p, None)
                        total_grad = task_gradients['merged'].get(p, None)  # 总的token加权合并梯度（等价于统一backward）


                        # 3. Perform optimization in respective subspaces
                        def optimize_in_subspace(grad, projector_name, state_prefix):
                            if grad is None:
                                return None, None
                            
                            projector = state[projector_name]
                            low_rank_grad = projector.project(grad, state["step"])
                            
                            if f"exp_avg_{state_prefix}" not in state:
                                state[f"exp_avg_{state_prefix}"] = torch.zeros_like(low_rank_grad)
                                state[f"exp_avg_sq_{state_prefix}"] = torch.zeros_like(low_rank_grad)
                            
                            exp_avg, exp_avg_sq = state[f"exp_avg_{state_prefix}"], state[f"exp_avg_sq_{state_prefix}"]
                            beta1, beta2 = group["betas"]
                            
                            exp_avg.mul_(beta1).add_(low_rank_grad, alpha=(1.0 - beta1))
                            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)
                            denom = exp_avg_sq.sqrt().add_(group["eps"])
                            
                            step_size = group["lr"]
                            if group["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            
                            norm_grad_subspace = exp_avg / denom
                            return projector.project_back(norm_grad_subspace), step_size

                        # 计算三个更新量
                        shared_update, step_size = optimize_in_subspace(total_grad, "projector_shared", "shared")
                        rec_specific_update, _ = optimize_in_subspace(rec_grad, "projector_rec_specific", "rec_specific")
                        src_specific_update, _ = optimize_in_subspace(src_grad, "projector_src_specific", "src_specific")

                        if shared_update is None:
                            continue

                        # 4. 门控聚合更新
                        final_update = shared_update
                        if gates is not None:
                            if rec_specific_update is not None and 'rec' in gates:
                                final_update = final_update + gates['rec'] * rec_specific_update
                            if src_specific_update is not None and 'src' in gates:
                                final_update = final_update + gates['src'] * src_specific_update
                        
                        # 更新参数
                        p.add_(final_update, alpha=-step_size)
                        
                        # 权重衰减
                        if group["weight_decay"] > 0.0:
                            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                        
                        continue

                # 对于非多任务参数，使用标准 Adam 更新
                grad = p.grad
                if grad is None:
                    continue
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                norm_grad = exp_avg / denom
                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        
        return loss
