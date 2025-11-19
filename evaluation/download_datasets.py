#!/usr/bin/env python3
"""
统一数据集下载脚本

将所有评估需要的数据集下载到本地 data/ 目录，
这样评估时可以直接从本地加载，不需要网络连接。

使用方法:
    python evaluation/download_datasets.py

数据集将保存到:
    data/
    ├── wikitext2/          # WikiText-2 数据集
    ├── ptb/                # Penn TreeBank 数据集
    └── c4/                 # C4 数据集 (可选，较大)
"""

import os
import sys
import argparse

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_wikitext2(save_dir: str):
    """下载 WikiText-2 数据集"""
    from datasets import load_dataset

    print("\n" + "="*50)
    print("下载 WikiText-2 数据集")
    print("="*50)

    save_path = os.path.join(save_dir, "wikitext2")

    if os.path.exists(save_path):
        print(f"  数据集已存在: {save_path}")
        return

    try:
        print("  正在从 HuggingFace 下载...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

        print(f"  保存到: {save_path}")
        dataset.save_to_disk(save_path)

        # 显示数据集信息
        print(f"  ✓ 下载完成")
        print(f"    - train: {len(dataset['train'])} 样本")
        print(f"    - validation: {len(dataset['validation'])} 样本")
        print(f"    - test: {len(dataset['test'])} 样本")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        raise


def download_ptb(save_dir: str):
    """下载 Penn TreeBank 数据集"""
    import requests
    from datasets import Dataset, DatasetDict

    print("\n" + "="*50)
    print("下载 Penn TreeBank 数据集")
    print("="*50)

    save_path = os.path.join(save_dir, "ptb")

    if os.path.exists(save_path):
        print(f"  数据集已存在: {save_path}")
        return

    # 从 GitHub 下载
    PTB_URLS = {
        "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "validation": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
    }

    try:
        datasets = {}

        for split, url in PTB_URLS.items():
            print(f"  下载 {split} 集...")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 按行分割
            lines = response.text.strip().split('\n')
            datasets[split] = Dataset.from_dict({'sentence': lines})
            print(f"    - {len(lines)} 行")

        # 创建 DatasetDict 并保存
        dataset_dict = DatasetDict(datasets)

        print(f"  保存到: {save_path}")
        dataset_dict.save_to_disk(save_path)
        print(f"  ✓ 下载完成")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        raise


def download_c4(save_dir: str, num_samples: int = 10000):
    """下载 C4 数据集 (部分)"""
    from datasets import load_dataset

    print("\n" + "="*50)
    print(f"下载 C4 数据集 (前 {num_samples} 样本)")
    print("="*50)

    save_path = os.path.join(save_dir, "c4")

    if os.path.exists(save_path):
        print(f"  数据集已存在: {save_path}")
        return

    try:
        print("  正在从 HuggingFace 下载 (这可能需要一些时间)...")

        # 只下载 validation 集的前 N 个样本
        try:
            dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=False, trust_remote_code=True)
        except:
            dataset = load_dataset('c4', 'en', split='validation', streaming=False, trust_remote_code=True)

        # 只保留前 N 个样本
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"  保存到: {save_path}")
        dataset.save_to_disk(save_path)

        print(f"  ✓ 下载完成")
        print(f"    - validation: {len(dataset)} 样本")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  提示: C4 数据集较大，如果下载失败，可以跳过它，使用 WikiText2 评估 PPL")
        raise


def download_zeroshot_datasets(save_dir: str):
    """下载 Zero-shot 评估所需的 7 个数据集到本地缓存"""
    from datasets import load_dataset

    print("\n" + "="*50)
    print("下载 Zero-shot 评估数据集 (7个)")
    print("="*50)

    # 设置缓存目录为 data/zeroshot
    zeroshot_dir = os.path.join(save_dir, "zeroshot")
    os.makedirs(zeroshot_dir, exist_ok=True)

    # 设置环境变量，让数据集下载到指定目录
    os.environ['HF_DATASETS_CACHE'] = zeroshot_dir

    # Zero-shot 任务对应的数据集
    ZEROSHOT_DATASETS = {
        'boolq': ('google/boolq', None),
        'piqa': ('piqa', None),
        'hellaswag': ('Rowan/hellaswag', None),
        'winogrande': ('winogrande', 'winogrande_xl'),
        'arc_easy': ('allenai/ai2_arc', 'ARC-Easy'),
        'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge'),
        'openbookqa': ('allenai/openbookqa', 'main'),
    }

    for task_name, (dataset_name, config) in ZEROSHOT_DATASETS.items():
        try:
            print(f"  下载 {task_name} ({dataset_name})...")
            if config:
                dataset = load_dataset(dataset_name, config, cache_dir=zeroshot_dir)
            else:
                dataset = load_dataset(dataset_name, cache_dir=zeroshot_dir)

            # 显示数据集大小
            if hasattr(dataset, 'num_rows'):
                print(f"    ✓ 完成 ({dataset.num_rows} 样本)")
            else:
                total = sum(len(split) for split in dataset.values())
                print(f"    ✓ 完成 ({total} 样本)")

        except Exception as e:
            print(f"    ✗ 失败: {e}")

    print(f"\n  数据集已缓存到: {zeroshot_dir}")
    print(f"  评估时会自动从此目录加载")


def main():
    parser = argparse.ArgumentParser(description='下载评估数据集到本地')
    parser.add_argument('--save_dir', type=str, default='data',
                       help='保存目录 (默认: data)')
    parser.add_argument('--datasets', type=str, default='wikitext2,ptb,zeroshot',
                       help='要下载的数据集，逗号分隔 (默认: wikitext2,ptb,zeroshot)')
    parser.add_argument('--c4_samples', type=int, default=10000,
                       help='C4 数据集样本数 (默认: 10000)')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*50)
    print("数据集下载工具")
    print("="*50)
    print(f"保存目录: {os.path.abspath(args.save_dir)}")

    # 解析数据集列表
    datasets = [d.strip().lower() for d in args.datasets.split(',')]

    # 下载各数据集
    if 'wikitext2' in datasets or 'wikitext' in datasets:
        download_wikitext2(args.save_dir)

    if 'ptb' in datasets:
        download_ptb(args.save_dir)

    if 'c4' in datasets:
        download_c4(args.save_dir, args.c4_samples)

    if 'zeroshot' in datasets:
        download_zeroshot_datasets(args.save_dir)

    print("\n" + "="*50)
    print("下载完成！")
    print("="*50)
    print(f"\n数据集已保存到: {os.path.abspath(args.save_dir)}")
    print("\n现在可以运行评估，数据将从本地加载:")
    print("  python evaluation/run_evaluation.py --model_path <model> --metrics ppl")
    print("  python evaluation/run_evaluation.py --model_path <model> --metrics zero_shot")


if __name__ == '__main__':
    main()
