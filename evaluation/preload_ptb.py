#!/usr/bin/env python3
"""
PTB (Penn Treebank) 数据集预加载脚本

从 GitHub 下载 PTB 数据集并保存到本地缓存
解决 HuggingFace 上数据集被删除的问题

用法:
    python evaluation/preload_ptb.py
"""

import os
import json
import requests
from datasets import Dataset, DatasetDict

# GitHub 原始数据 URL (from wojzaremba/lstm repo)
PTB_URLS = {
    "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "validation": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
    "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
}


def download_file(url, local_path):
    """
    下载文件到本地（支持代理）

    Args:
        url: 远程文件 URL
        local_path: 本地保存路径
    """
    print(f"  下载: {url.split('/')[-1]}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        f.write(response.content)

    return local_path


def preload_ptb():
    """
    从 GitHub 下载 PTB 数据集并保存到本地
    """
    print("=" * 60)
    print("PTB 数据集预加载 (从 GitHub)")
    print("=" * 60)

    try:
        # 创建本地下载目录
        local_data_dir = os.path.expanduser("~/.cache/huggingface/datasets/ptb_text_only_local")
        os.makedirs(local_data_dir, exist_ok=True)

        # 1. 先用 requests 下载文件到本地（支持代理）
        print("\n[1/3] 从 GitHub 下载数据...")

        local_train = os.path.join(local_data_dir, "ptb.train.txt")
        local_valid = os.path.join(local_data_dir, "ptb.valid.txt")
        local_test = os.path.join(local_data_dir, "ptb.test.txt")

        download_file(PTB_URLS['train'], local_train)
        download_file(PTB_URLS['validation'], local_valid)
        download_file(PTB_URLS['test'], local_test)

        print("✓ 文件下载完成")

        # 2. 转换为 HuggingFace 数据集格式
        print("\n[2/3] 转换为数据集格式...")

        def load_ptb_file(file_path):
            """加载 PTB 文件，每行作为一个句子"""
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            return {'sentence': sentences}

        train_data = load_ptb_file(local_train)
        valid_data = load_ptb_file(local_valid)
        test_data = load_ptb_file(local_test)

        dataset = DatasetDict({
            'train': Dataset.from_dict(train_data),
            'validation': Dataset.from_dict(valid_data),
            'test': Dataset.from_dict(test_data)
        })

        print(f"✓ 数据集创建成功")
        print(f"  训练集: {len(dataset['train'])} 句子")
        print(f"  验证集: {len(dataset['validation'])} 句子")
        print(f"  测试集: {len(dataset['test'])} 句子")

        # 3. 保存数据集
        print("\n[3/3] 保存数据集...")

        # 保存为 HF 格式
        save_dir = os.path.expanduser("~/.cache/huggingface/datasets/ptb_text_only")
        dataset.save_to_disk(save_dir)
        print(f"✓ HF 格式已保存: {save_dir}")

        # 验证样本
        print(f"\n样本示例:")
        sample = dataset['validation'][0]
        print(f"  sentence: {sample['sentence'][:80]}...")

        print("\n" + "=" * 60)
        print("✓ PTB 预加载完成!")
        print("=" * 60)
        print("\nPTB 数据集现在可用于 PPL 评估")

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ 网络下载失败: {e}")
        print("请检查网络连接或使用代理")
        return False

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = preload_ptb()
    exit(0 if success else 1)
