"""
剪枝方法模块
包含各种神经网络剪枝算法的实现
"""

from .gqa_aware import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

__all__ = [
    'compute_gqa_group_importance',
    'select_gqa_groups_to_prune',
    'prune_attention_by_gqa_groups',
]
