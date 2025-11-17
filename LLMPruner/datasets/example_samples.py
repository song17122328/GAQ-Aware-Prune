#!/usr/bin/env python3
"""
样本数据加载工具
用于从各种数据集中加载示例样本，供模型剪枝时计算梯度使用
"""

import torch
from datasets import load_dataset
from typing import Optional


def get_examples(
    dataset_name: str,
    tokenizer,
    num_samples: int = 10,
    seq_len: int = 128,
    split: str = 'train'
) -> torch.Tensor:
    """
    从指定数据集加载样本数据并进行tokenization

    Args:
        dataset_name: 数据集名称，支持 'wikitext', 'c4', 'ptb' 等
        tokenizer: HuggingFace tokenizer实例
        num_samples: 需要的样本数量
        seq_len: 序列长度
        split: 数据集划分 ('train', 'test', 'validation')

    Returns:
        torch.Tensor: tokenized input_ids, shape [num_samples, seq_len]
    """

    # 根据数据集名称加载对应数据
    if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        text_field = 'text'
    elif dataset_name.lower() in ['wikitext103', 'wikitext-103']:
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_field = 'text'
    elif dataset_name.lower() == 'c4':
        dataset = load_dataset('c4', 'en', split=split, streaming=True)
        text_field = 'text'
    elif dataset_name.lower() in ['ptb', 'penn-treebank']:
        dataset = load_dataset('ptb_text_only', split=split)
        text_field = 'sentence'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported: wikitext, wikitext103, c4, ptb")

    # 收集文本样本
    texts = []
    for item in dataset:
        text = item[text_field].strip()
        # 过滤掉太短的文本
        if len(text) > 50:
            texts.append(text)

        if len(texts) >= num_samples:
            break

    # 如果样本不够，重复使用
    while len(texts) < num_samples:
        texts.extend(texts[:num_samples - len(texts)])

    texts = texts[:num_samples]

    # Tokenization
    encodings = tokenizer(
        texts,
        return_tensors='pt',
        max_length=seq_len,
        truncation=True,
        padding='max_length'
    )

    return encodings['input_ids']


def get_examples_from_text(
    texts: list,
    tokenizer,
    seq_len: int = 128
) -> torch.Tensor:
    """
    从给定的文本列表创建样本

    Args:
        texts: 文本列表
        tokenizer: HuggingFace tokenizer实例
        seq_len: 序列长度

    Returns:
        torch.Tensor: tokenized input_ids
    """
    encodings = tokenizer(
        texts,
        return_tensors='pt',
        max_length=seq_len,
        truncation=True,
        padding='max_length'
    )

    return encodings['input_ids']


def get_calibration_data(
    dataset_name: str = 'wikitext',
    tokenizer = None,
    num_samples: int = 128,
    seq_len: int = 2048
) -> torch.Tensor:
    """
    获取校准数据（用于量化等场景）

    Args:
        dataset_name: 数据集名称
        tokenizer: tokenizer实例
        num_samples: 样本数量
        seq_len: 序列长度

    Returns:
        torch.Tensor: calibration data
    """
    return get_examples(dataset_name, tokenizer, num_samples, seq_len)


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer

    print("测试 get_examples 函数...")

    # 使用一个小模型的tokenizer进行测试
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 测试加载wikitext
    print("\n1. 测试加载 wikitext 数据集:")
    try:
        examples = get_examples('wikitext', tokenizer, num_samples=5, seq_len=64)
        print(f"   ✓ 成功加载，shape: {examples.shape}")
        print(f"   样本示例: {tokenizer.decode(examples[0][:20])}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")

    print("\n测试完成！")
