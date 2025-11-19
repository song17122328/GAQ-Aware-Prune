#!/usr/bin/env python3
"""
Baseline 剪枝方法模块

包含多种经典的LLM结构化剪枝方法，用于与 GAQ-Aware-Prune 进行对比实验。

方法优先级:
- 第一阶段（必须实现）: LLM-Pruner, Wanda-Structured, Magnitude
- 第二阶段（后续实现）: ShortGPT
- 第三阶段（视难度）: SlimGPT, SparseGPT
- 第四阶段（可选）: FLAP, Random
"""

from .methods import (
    BasePruner,
    AVAILABLE_METHODS,
    get_pruner
)

__all__ = [
    'BasePruner',
    'AVAILABLE_METHODS',
    'get_pruner'
]

__version__ = '0.1.0'
