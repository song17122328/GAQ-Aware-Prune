#!/usr/bin/env python3
"""
剪枝方法基类

定义所有剪枝方法的统一接口，确保一致性和可比较性。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer


class BasePruner(ABC):
    """
    剪枝方法基类

    所有 baseline 方法都需要继承此类并实现必要的抽象方法。
    这确保了所有方法具有统一的接口，便于对比实验。

    Attributes:
        model: 待剪枝的模型
        tokenizer: 分词器
        device: 计算设备
        config: 剪枝配置
        logger: 日志记录器
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda',
        logger: Any = None
    ):
        """
        初始化剪枝器

        Args:
            model: 预训练模型
            tokenizer: 分词器
            device: 设备 ('cuda' or 'cpu')
            logger: 可选的日志记录器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger

        # 将模型移动到指定设备
        self.model.to(device)

        # 剪枝统计信息
        self.stats = {
            'original_params': self._count_parameters(),
            'pruned_params': None,
            'pruning_ratio': None,
            'layer_stats': {}
        }

    def log(self, message: str) -> None:
        """
        记录日志

        Args:
            message: 日志消息
        """
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def _count_parameters(self) -> int:
        """
        统计模型参数数量

        Returns:
            参数总数
        """
        return sum(p.numel() for p in self.model.parameters())

    @abstractmethod
    def compute_importance(
        self,
        calibration_data: torch.Tensor,
        **kwargs
    ) -> Dict[int, torch.Tensor]:
        """
        计算各层/通道的重要性分数

        Args:
            calibration_data: 校准数据 [batch_size, seq_len]
            **kwargs: 方法特定参数

        Returns:
            {layer_idx: importance_scores} 字典
        """
        pass

    @abstractmethod
    def prune(
        self,
        pruning_ratio: float,
        calibration_data: torch.Tensor = None,
        **kwargs
    ) -> None:
        """
        执行剪枝

        Args:
            pruning_ratio: 目标剪枝率 (0-1)
            calibration_data: 校准数据
            **kwargs: 方法特定参数

        注意:
            此方法应该修改 self.model 并更新 self.stats
        """
        pass

    def get_pruned_model(self) -> PreTrainedModel:
        """
        获取剪枝后的模型

        Returns:
            剪枝后的模型
        """
        return self.model

    def get_stats(self) -> Dict[str, Any]:
        """
        获取剪枝统计信息

        Returns:
            包含剪枝统计的字典
        """
        # 更新剪枝后参数数量
        self.stats['pruned_params'] = self._count_parameters()
        if self.stats['original_params'] > 0:
            actual_ratio = 1 - (self.stats['pruned_params'] / self.stats['original_params'])
            self.stats['pruning_ratio'] = actual_ratio

        return self.stats

    def save_checkpoint(
        self,
        save_path: str,
        additional_info: Dict = None
    ) -> None:
        """
        保存剪枝后的模型检查点

        Args:
            save_path: 保存路径 (.bin 文件)
            additional_info: 额外的元信息
        """
        save_dict = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'stats': self.get_stats(),
            'method': self.__class__.__name__
        }

        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, save_path)
        self.log(f"模型已保存到: {save_path}")

    def validate_gqa_ratio(self, expected_ratio: int = 4) -> Tuple[bool, Dict[int, Tuple[int, int]]]:
        """
        验证 GQA 比例是否保持一致

        Args:
            expected_ratio: 预期的 Q:KV 比例 (默认 4:1)

        Returns:
            (is_valid, {layer_idx: (num_q_heads, num_kv_heads)})
        """
        layer_ratios = {}
        is_valid = True

        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                num_q = getattr(attn, 'num_heads', None)
                num_kv = getattr(attn, 'num_key_value_heads', None)

                if num_q is not None and num_kv is not None:
                    layer_ratios[idx] = (num_q, num_kv)
                    if num_kv > 0 and num_q // num_kv != expected_ratio:
                        is_valid = False

        return is_valid, layer_ratios

    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        执行前向传播

        Args:
            input_ids: 输入 token ids
            attention_mask: 注意力掩码

        Returns:
            模型输出的 logits
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
        return outputs.logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算损失（用于重要性计算）

        Args:
            input_ids: 输入 token ids
            labels: 标签 (默认使用 input_ids)

        Returns:
            损失值
        """
        if labels is None:
            labels = input_ids.clone()

        self.model.train()
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            labels=labels.to(self.device)
        )
        return outputs.loss

    @staticmethod
    def prune_linear_layer(
        layer: nn.Linear,
        keep_indices: torch.Tensor,
        dim: int = 0
    ) -> nn.Linear:
        """
        剪枝线性层

        Args:
            layer: 原始线性层
            keep_indices: 保留的索引
            dim: 剪枝维度 (0=output, 1=input)

        Returns:
            剪枝后的新线性层
        """
        keep_indices = keep_indices.to(layer.weight.device)

        if dim == 0:
            # 剪枝输出维度
            new_weight = layer.weight.index_select(0, keep_indices)
            new_bias = layer.bias.index_select(0, keep_indices) if layer.bias is not None else None
            new_layer = nn.Linear(layer.in_features, len(keep_indices), bias=layer.bias is not None)
        else:
            # 剪枝输入维度
            new_weight = layer.weight.index_select(1, keep_indices)
            new_bias = layer.bias  # bias 不变
            new_layer = nn.Linear(len(keep_indices), layer.out_features, bias=layer.bias is not None)

        new_layer.weight.data = new_weight
        if new_bias is not None:
            new_layer.bias.data = new_bias

        return new_layer.to(layer.weight.device)

    def print_summary(self) -> None:
        """
        打印剪枝摘要
        """
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print(f"剪枝方法: {self.__class__.__name__}")
        print("=" * 50)
        print(f"原始参数量: {stats['original_params']:,}")
        print(f"剪枝后参数量: {stats['pruned_params']:,}")
        print(f"实际剪枝率: {stats['pruning_ratio']*100:.2f}%")

        # 验证 GQA 比例
        is_valid, ratios = self.validate_gqa_ratio()
        if is_valid:
            print("GQA 比例验证: ✅ 所有层保持 4:1")
        else:
            print("GQA 比例验证: ❌ 部分层比例异常")
            for idx, (q, kv) in ratios.items():
                if kv > 0 and q // kv != 4:
                    print(f"  Layer {idx}: {q}Q:{kv}KV")

        print("=" * 50 + "\n")
