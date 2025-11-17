#!/usr/bin/env python3
"""
模型微调模块
用于剪枝后模型的性能恢复
"""

import torch
from typing import Optional, Dict, Any
from ..datasets.example_samples import get_examples


class FineTuner:
    """剪枝后模型的微调器"""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        logger = None
    ):
        """
        初始化微调器

        Args:
            model: 要微调的模型
            tokenizer: tokenizer
            device: 设备
            logger: 日志记录器（可选）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger

    def log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def finetune(
        self,
        dataset_name: str = 'wikitext',
        num_samples: int = 500,
        seq_len: int = 512,
        lr: float = 1e-5,
        epochs: int = 1,
        batch_size: int = 1,
        split: str = 'train'
    ) -> Dict[str, Any]:
        """
        微调模型

        Args:
            dataset_name: 数据集名称
            num_samples: 样本数量
            seq_len: 序列长度
            lr: 学习率
            epochs: 训练轮数
            batch_size: batch大小
            split: 数据集划分

        Returns:
            微调统计信息字典
        """
        self.log(f"从 {dataset_name} {split} 集加载 {num_samples} 个样本...")

        # 加载训练数据
        finetune_data = get_examples(
            dataset_name,
            self.tokenizer,
            num_samples=num_samples,
            seq_len=seq_len,
            split=split
        )
        self.log(f"✅ 微调数据加载完成，shape: {finetune_data.shape}")

        # 准备优化器
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # 打印配置
        self.log(f"\n微调配置:")
        self.log(f"  学习率: {lr}")
        self.log(f"  轮数: {epochs}")
        self.log(f"  样本数: {num_samples}")
        self.log(f"  Batch size: {batch_size}")
        self.log(f"  序列长度: {seq_len}")

        # 训练统计
        epoch_losses = []

        # 微调循环
        for epoch in range(epochs):
            self.log(f"\n开始第 {epoch + 1}/{epochs} 轮微调...")

            total_loss = 0
            num_batches = (len(finetune_data) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                # 获取batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(finetune_data))
                batch = finetune_data[start_idx:end_idx].to(self.device)

                # 前向传播
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 每10%进度打印一次
                if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    progress = (batch_idx + 1) / num_batches * 100
                    self.log(f"  进度: {progress:.0f}% | 平均Loss: {avg_loss:.4f}")

            avg_epoch_loss = total_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            self.log(f"✅ 第 {epoch + 1} 轮完成，平均Loss: {avg_epoch_loss:.4f}")

        self.log("\n✅ 微调完成！")

        # 返回统计信息
        return {
            'lr': lr,
            'epochs': epochs,
            'samples': num_samples,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'epoch_losses': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else None
        }

    def save_finetuned_model(
        self,
        save_path: str,
        layer_pruning_rates: Dict[int, float],
        layer_importance: Dict[int, float],
        finetune_stats: Dict[str, Any],
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        保存微调后的模型

        Args:
            save_path: 保存路径
            layer_pruning_rates: 层剪枝率
            layer_importance: 层重要性
            finetune_stats: 微调统计信息
            extra_info: 额外信息（可选）
        """
        self.log(f"保存微调后的模型到: {save_path}")

        self.model.half()

        save_dict = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'layer_pruning_rates': layer_pruning_rates,
            'layer_importance': layer_importance,
            'pruning_method': 'gqa_aware_taylor',
            'finetuned': True,
            'finetune_config': finetune_stats,
        }

        if extra_info:
            save_dict['config'] = extra_info

        torch.save(save_dict, save_path)
        self.log(f"✅ 微调后模型已保存")
