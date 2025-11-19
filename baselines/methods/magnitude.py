#!/usr/bin/env python3
"""
Magnitude Pruning 实现

核心思想:
- 最简单的剪枝方法
- 按权重绝对值大小排序
- 移除最小的权重/通道

实现优先级: 第一阶段（必须实现）
"""

import torch
from typing import Dict, Any
from .base_pruner import BasePruner


class MagnitudePruner(BasePruner):
    """
    Magnitude 剪枝器

    基于权重绝对值的简单剪枝方法。

    特点:
    - 实现简单
    - 不需要校准数据
    - 作为基线参考
    """

    def __init__(self, model, tokenizer, device='cuda', logger=None):
        super().__init__(model, tokenizer, device, logger)
        self.method_name = 'Magnitude'

    def compute_importance(
        self,
        calibration_data: torch.Tensor = None,
        **kwargs
    ) -> Dict[int, torch.Tensor]:
        """
        计算 Magnitude 重要性

        重要性 = ||W||_2 (L2 范数)

        Args:
            calibration_data: 不需要校准数据
            **kwargs: 额外参数

        Returns:
            {layer_idx: importance_tensor}
        """
        # TODO: 实现 Magnitude 重要性计算
        # 1. 遍历每层
        # 2. 计算每个通道/头的 L2 范数
        # 3. 返回重要性字典

        raise NotImplementedError("Magnitude 重要性计算尚未实现")

    def prune(
        self,
        pruning_ratio: float,
        calibration_data: torch.Tensor = None,
        **kwargs
    ) -> None:
        """
        执行 Magnitude 剪枝

        Args:
            pruning_ratio: 目标剪枝率
            calibration_data: 不需要
            **kwargs: 额外参数
                - prune_mlp: 是否剪枝 MLP
                - gqa_aware: 是否保持 GQA 结构

        流程:
        1. 计算每个通道的 L2 范数
        2. 按大小排序
        3. 移除最小的通道
        4. 更新统计信息
        """
        # TODO: 实现完整的剪枝流程

        raise NotImplementedError("Magnitude 剪枝尚未实现")


# 参考实现说明
"""
实现步骤:

1. 核心公式:
   对于线性层 W ∈ R^{out x in}:
   importance[i] = ||W[i, :]||_2  (输出通道重要性)

   对于注意力头:
   importance[head] = ||W_q[head]||_2 + ||W_k[head]||_2 + ||W_v[head]||_2

2. GQA 兼容性:
   - 将 Q 头重要性聚合到 KV 组
   - group_importance = sum(Q_heads) + K_head + V_head
   - 保持 4:1 比例

3. 实现优势:
   - 不需要前向传播
   - 不需要校准数据
   - 计算速度最快

4. 局限性:
   - 不考虑激活值分布
   - 不考虑参数间依赖
   - 通常效果不如其他方法

5. 测试用例:
   python baselines/run_baseline.py \\
       --method magnitude \\
       --base_model /path/to/model \\
       --pruning_ratio 0.25

6. 代码框架:
   def compute_head_importance(self, layer):
       q_proj = layer.self_attn.q_proj.weight
       k_proj = layer.self_attn.k_proj.weight
       v_proj = layer.self_attn.v_proj.weight

       # 计算每个头的范数
       head_dim = q_proj.shape[0] // num_heads
       q_norms = [q_proj[i*head_dim:(i+1)*head_dim].norm() for i in range(num_heads)]
       ...
"""
