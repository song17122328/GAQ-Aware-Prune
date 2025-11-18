#!/usr/bin/env python3
"""
预加载 PIQA 数据集

将通过 `hf download ybisk/piqa --repo-type=dataset` 下载的数据
转换为 datasets 库可用的缓存格式。

用法:
    python evaluation/preload_piqa.py
"""

import os
from datasets import load_dataset

def preload_piqa():
    """
    预加载 PIQA 数据集到 datasets 缓存

    hf download 下载的数据在 ~/.cache/huggingface/hub/datasets--ybisk--piqa/
    load_dataset 会自动从这里读取并创建 datasets 缓存
    """
    print("=" * 60)
    print("预加载 PIQA 数据集")
    print("=" * 60)

    # 设置离线模式，强制从本地缓存加载
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    try:
        print("\n从本地缓存加载 PIQA...")

        # 加载训练集和验证集
        # datasets 库会自动从 hub 缓存读取并创建 datasets 缓存
        train_ds = load_dataset('ybisk/piqa', split='train')
        valid_ds = load_dataset('ybisk/piqa', split='validation')

        print(f"\n✓ PIQA 数据集加载成功!")
        print(f"  训练集样本数: {len(train_ds)}")
        print(f"  验证集样本数: {len(valid_ds)}")

        # 打印样本示例
        print(f"\n样本示例:")
        sample = valid_ds[0]
        print(f"  goal: {sample.get('goal', 'N/A')[:50]}...")
        print(f"  sol1: {sample.get('sol1', 'N/A')[:50]}...")
        print(f"  sol2: {sample.get('sol2', 'N/A')[:50]}...")

        print("\n" + "=" * 60)
        print("✓ PIQA 预加载完成!")
        print("=" * 60)
        print("\n现在可以运行评估了:")
        print("python evaluation/run_evaluation.py --metrics zeroshot ...")

        return True

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        print("\n请确保已经运行过:")
        print("  HF_ENDPOINT='https://hf-mirror.com' hf download ybisk/piqa --repo-type=dataset")

        # 尝试检查缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--ybisk--piqa")
        if os.path.exists(cache_dir):
            print(f"\n缓存目录存在: {cache_dir}")
            print("目录内容:")
            for item in os.listdir(cache_dir):
                print(f"  {item}")
        else:
            print(f"\n缓存目录不存在: {cache_dir}")

        return False

if __name__ == '__main__':
    success = preload_piqa()
    exit(0 if success else 1)
