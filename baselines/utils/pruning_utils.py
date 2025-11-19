#!/usr/bin/env python3
"""
通用剪枝工具函数

提供各种 baseline 方法共享的工具函数。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable


def get_layer_groups(model) -> List[nn.Module]:
    """
    获取模型的 Transformer 层列表

    Args:
        model: Transformer 模型

    Returns:
        层列表
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("无法识别模型结构，请手动指定层列表")


def aggregate_to_gqa_groups(
    head_importance: torch.Tensor,
    num_kv_heads: int,
    gqa_ratio: int = 4
) -> torch.Tensor:
    """
    将 Q 头重要性聚合到 GQA group

    Args:
        head_importance: Q 头重要性 [num_q_heads]
        num_kv_heads: KV 头数量
        gqa_ratio: Q:KV 比例

    Returns:
        GQA group 重要性 [num_kv_heads]
    """
    num_q_heads = len(head_importance)
    assert num_q_heads == num_kv_heads * gqa_ratio, \
        f"Q头数量({num_q_heads}) != KV头数量({num_kv_heads}) * 比例({gqa_ratio})"

    group_importance = torch.zeros(num_kv_heads, device=head_importance.device)

    for kv_idx in range(num_kv_heads):
        # 获取属于这个 KV 头的 Q 头
        q_start = kv_idx * gqa_ratio
        q_end = q_start + gqa_ratio
        # 聚合（可以用 sum、mean、max 等）
        group_importance[kv_idx] = head_importance[q_start:q_end].sum()

    return group_importance


def compute_channel_norms(
    weight: torch.Tensor,
    dim: int = 0,
    norm_type: str = 'l2'
) -> torch.Tensor:
    """
    计算权重通道的范数

    Args:
        weight: 权重张量 [out_features, in_features]
        dim: 计算维度 (0=输出通道, 1=输入通道)
        norm_type: 范数类型 ('l1', 'l2', 'linf')

    Returns:
        通道范数
    """
    if norm_type == 'l1':
        return weight.abs().sum(dim=1-dim)
    elif norm_type == 'l2':
        return weight.norm(p=2, dim=1-dim)
    elif norm_type == 'linf':
        return weight.abs().max(dim=1-dim).values
    else:
        raise ValueError(f"未知范数类型: {norm_type}")


class ActivationHook:
    """
    激活值捕获钩子

    用于收集层的输入/输出激活值。
    """

    def __init__(self, capture_input: bool = True, capture_output: bool = False):
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.inputs = []
        self.outputs = []

    def __call__(self, module, input, output):
        if self.capture_input and len(input) > 0:
            self.inputs.append(input[0].detach())
        if self.capture_output:
            if isinstance(output, tuple):
                self.outputs.append(output[0].detach())
            else:
                self.outputs.append(output.detach())

    def clear(self):
        self.inputs = []
        self.outputs = []

    def get_mean_activation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取平均激活值"""
        mean_input = None
        mean_output = None

        if self.inputs:
            stacked = torch.cat(self.inputs, dim=0)
            mean_input = stacked.mean(dim=0)

        if self.outputs:
            stacked = torch.cat(self.outputs, dim=0)
            mean_output = stacked.mean(dim=0)

        return mean_input, mean_output


def register_activation_hooks(
    model,
    target_modules: List[str] = None
) -> Dict[str, ActivationHook]:
    """
    为模型注册激活值钩子

    Args:
        model: 模型
        target_modules: 目标模块名称列表 (默认所有 Linear)

    Returns:
        {module_name: hook} 字典
    """
    hooks = {}
    handles = []

    for name, module in model.named_modules():
        if target_modules is None:
            # 默认只为 Linear 层注册
            if isinstance(module, nn.Linear):
                hook = ActivationHook()
                handle = module.register_forward_hook(hook)
                hooks[name] = hook
                handles.append(handle)
        else:
            if any(t in name for t in target_modules):
                hook = ActivationHook()
                handle = module.register_forward_hook(hook)
                hooks[name] = hook
                handles.append(handle)

    return hooks, handles


def remove_hooks(handles: List) -> None:
    """移除所有钩子"""
    for handle in handles:
        handle.remove()


def select_indices_to_prune(
    importance: torch.Tensor,
    pruning_ratio: float,
    min_keep: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据重要性选择要剪枝的索引

    Args:
        importance: 重要性分数
        pruning_ratio: 剪枝率
        min_keep: 最少保留数量

    Returns:
        (keep_indices, prune_indices)
    """
    num_total = len(importance)
    num_prune = int(num_total * pruning_ratio)
    num_keep = max(num_total - num_prune, min_keep)

    # 按重要性排序
    sorted_indices = torch.argsort(importance, descending=True)

    keep_indices = sorted_indices[:num_keep]
    prune_indices = sorted_indices[num_keep:]

    # 排序以保持原始顺序
    keep_indices = torch.sort(keep_indices).values
    prune_indices = torch.sort(prune_indices).values

    return keep_indices, prune_indices


def compute_reconstruction_error(
    original_output: torch.Tensor,
    pruned_output: torch.Tensor
) -> float:
    """
    计算剪枝前后的重建误差

    Args:
        original_output: 原始输出
        pruned_output: 剪枝后输出

    Returns:
        MSE 误差
    """
    return torch.mean((original_output - pruned_output) ** 2).item()
