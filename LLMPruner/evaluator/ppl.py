#!/usr/bin/env python3
"""
困惑度(Perplexity)评估工具
用于评估语言模型在各种数据集上的困惑度
"""

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Optional


class PPLMetric:
    """
    困惑度评估类

    用法:
        ppl_metric = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'])
        results = ppl_metric  # 自动计算并返回结果字典
    """

    def __init__(
        self,
        model,
        tokenizer,
        datasets: List[str],
        seq_len: int = 128,
        device: str = 'cuda',
        stride: int = None,
        batch_size: int = 1
    ):
        """
        初始化PPL评估器

        Args:
            model: 语言模型
            tokenizer: tokenizer实例
            datasets: 要评估的数据集列表，如 ['wikitext2', 'ptb', 'c4']
            seq_len: 序列长度
            device: 计算设备
            stride: 滑动窗口步长（None则等于seq_len，即不重叠）
            batch_size: 批处理大小
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_names = datasets
        self.seq_len = seq_len
        self.device = device
        self.stride = stride if stride is not None else seq_len
        self.batch_size = batch_size

        # 确保模型在正确的设备上
        if hasattr(model, 'to'):
            self.model.to(device)

        self.model.eval()

        # 自动计算PPL
        self.results = self._evaluate_all()

    def _evaluate_all(self) -> Dict[str, float]:
        """评估所有数据集"""
        results = {}

        for dataset_name in self.dataset_names:
            try:
                ppl = self._evaluate_dataset(dataset_name)
                # 使用与数据集加载一致的键名格式
                if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
                    key = 'wikitext2 (wikitext-2-raw-v1)'
                elif dataset_name.lower() in ['wikitext103', 'wikitext-103']:
                    key = 'wikitext103 (wikitext-103-raw-v1)'
                elif dataset_name.lower() in ['ptb', 'penn-treebank']:
                    key = 'ptb'
                elif dataset_name.lower() == 'c4':
                    key = 'c4'
                else:
                    key = dataset_name

                results[key] = ppl
                print(f"✓ {key}: PPL = {ppl:.2f}")

            except Exception as e:
                print(f"✗ 评估 {dataset_name} 时出错: {e}")
                results[dataset_name] = float('inf')

        return results

    def _evaluate_dataset(self, dataset_name: str) -> float:
        """
        评估单个数据集的困惑度

        Args:
            dataset_name: 数据集名称

        Returns:
            float: 困惑度值
        """
        # 加载数据集
        encodings = self._load_dataset(dataset_name)

        # 计算PPL
        ppl = self._calculate_perplexity(encodings)

        return ppl

    def _load_dataset(self, dataset_name: str) -> torch.Tensor:
        """
        加载数据集并tokenize

        Args:
            dataset_name: 数据集名称

        Returns:
            torch.Tensor: tokenized数据
        """
        # 根据数据集名称加载
        if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            text_field = 'text'
        elif dataset_name.lower() in ['wikitext103', 'wikitext-103']:
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
            text_field = 'text'
        elif dataset_name.lower() in ['ptb', 'penn-treebank']:
            dataset = load_dataset('ptb_text_only', split='test')
            text_field = 'sentence'
        elif dataset_name.lower() == 'c4':
            # C4数据集较大，只取部分
            dataset = load_dataset('c4', 'en', split='validation', streaming=True)
            dataset = list(dataset.take(1000))
            text_field = 'text'
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        # 合并所有文本
        if isinstance(dataset, list):
            texts = [item[text_field] for item in dataset]
        else:
            texts = [item[text_field] for item in dataset]

        # 过滤空文本并合并
        text = '\n\n'.join([t for t in texts if t.strip()])

        # Tokenize整个文本
        encodings = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=False
        )

        return encodings['input_ids'].squeeze(0)

    def _calculate_perplexity(self, encodings: torch.Tensor) -> float:
        """
        使用滑动窗口计算困惑度

        Args:
            encodings: tokenized input_ids

        Returns:
            float: 困惑度
        """
        seq_len = self.seq_len
        stride = self.stride

        nlls = []  # negative log-likelihoods
        prev_end_loc = 0

        # 滑动窗口遍历
        for begin_loc in tqdm(range(0, encodings.size(0), stride), desc="计算PPL"):
            end_loc = min(begin_loc + seq_len, encodings.size(0))
            trg_len = end_loc - prev_end_loc  # 可能小于stride的最后一个序列

            input_ids = encodings[begin_loc:end_loc].unsqueeze(0).to(self.device)

            # 如果序列太短，跳过
            if input_ids.size(1) < 2:
                break

            target_ids = input_ids.clone()

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # outputs.loss 已经是平均loss
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == encodings.size(0):
                break

        # 计算困惑度
        ppl = torch.exp(torch.stack(nlls).mean())

        return ppl.item()

    def get(self, key: str, default=None):
        """字典式访问接口"""
        return self.results.get(key, default)

    def __getitem__(self, key: str):
        """支持 ppl['wikitext2'] 访问"""
        return self.results[key]

    def __repr__(self):
        """打印结果"""
        return str(self.results)

    def __str__(self):
        """转为字符串"""
        lines = []
        for dataset, ppl_value in self.results.items():
            lines.append(f"  {dataset}: {ppl_value:.2f}")
        return "\n".join(lines)


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = 'wikitext2',
    seq_len: int = 128,
    device: str = 'cuda'
) -> float:
    """
    快捷函数：评估单个数据集的困惑度

    Args:
        model: 语言模型
        tokenizer: tokenizer
        dataset_name: 数据集名称
        seq_len: 序列长度
        device: 设备

    Returns:
        float: 困惑度
    """
    metric = PPLMetric(model, tokenizer, [dataset_name], seq_len, device)
    return list(metric.results.values())[0]


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("测试 PPLMetric 类...")

    # 使用一个小模型进行测试
    print("\n加载测试模型 (gpt2)...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("\n计算 wikitext2 PPL...")
    try:
        ppl_metric = PPLMetric(
            model,
            tokenizer,
            datasets=['wikitext2'],
            seq_len=512,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print(f"\n结果:")
        print(ppl_metric)

        # 测试字典访问
        print(f"\n字典访问测试:")
        for key in ppl_metric.results:
            print(f"  ppl['{key}'] = {ppl_metric[key]:.2f}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成！")
