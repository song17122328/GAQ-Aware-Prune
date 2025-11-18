#!/usr/bin/env python3
"""
数据集缓存清理工具

用于清理损坏的HuggingFace datasets缓存，解决以下问题：
- NonMatchingSplitsSizesError
- 数据集下载失败
- 缓存损坏

用法：
    # 清理Zero-shot相关数据集缓存
    python evaluation/clean_dataset_cache.py --zeroshot

    # 清理PPL相关数据集缓存
    python evaluation/clean_dataset_cache.py --ppl

    # 清理特定数据集
    python evaluation/clean_dataset_cache.py --dataset hellaswag

    # 清理所有缓存（危险！）
    python evaluation/clean_dataset_cache.py --all
"""

import argparse
import os
import shutil
from pathlib import Path

# 数据集分组
ZEROSHOT_DATASETS = [
    'hellaswag',
    'Rowan___hellaswag',  # 可能的别名
    'piqa',
    'winogrande',
    'ai2_arc',
    'ARC-Easy',
    'google___boolq',
    'boolq'
]

PPL_DATASETS = [
    'wikitext',
    'ptb_text_only',
    'ptb-text-only',
    'c4',
    'allenai___c4'
]


def get_cache_dir():
    """获取HuggingFace缓存目录"""
    # 从环境变量获取，或使用默认位置
    cache_dir = os.environ.get('HF_DATASETS_CACHE')
    if cache_dir:
        return Path(cache_dir)

    # 默认位置
    home = Path.home()
    return home / '.cache' / 'huggingface' / 'datasets'


def list_cached_datasets(cache_dir: Path):
    """列出所有缓存的数据集"""
    if not cache_dir.exists():
        print(f"缓存目录不存在: {cache_dir}")
        return []

    datasets = []
    for item in cache_dir.iterdir():
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            datasets.append((item.name, size_mb))

    return datasets


def clean_dataset(cache_dir: Path, dataset_name: str):
    """清理特定数据集的缓存"""
    dataset_path = cache_dir / dataset_name

    if not dataset_path.exists():
        print(f"  ⚠️  数据集缓存不存在: {dataset_name}")
        return False

    # 计算大小
    size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
    size_mb = size / (1024 * 1024)

    # 删除
    try:
        shutil.rmtree(dataset_path)
        print(f"  ✓ 已删除: {dataset_name} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ 删除失败: {dataset_name} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='清理HuggingFace数据集缓存')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--zeroshot', action='store_true',
                      help='清理Zero-shot评估相关数据集')
    group.add_argument('--ppl', action='store_true',
                      help='清理PPL评估相关数据集')
    group.add_argument('--dataset', type=str,
                      help='清理特定数据集（数据集名称）')
    group.add_argument('--all', action='store_true',
                      help='清理所有缓存（危险！会删除所有数据集）')
    group.add_argument('--list', action='store_true',
                      help='列出所有缓存的数据集')

    parser.add_argument('--cache_dir', type=str,
                       help='自定义缓存目录（默认：~/.cache/huggingface/datasets/）')

    args = parser.parse_args()

    # 获取缓存目录
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = get_cache_dir()

    print(f"缓存目录: {cache_dir}")

    if not cache_dir.exists():
        print("✗ 缓存目录不存在！")
        return

    # 列出缓存
    if args.list:
        print("\n已缓存的数据集:")
        print("=" * 60)
        datasets = list_cached_datasets(cache_dir)
        if not datasets:
            print("  (无缓存)")
        else:
            total_size = 0
            for name, size in sorted(datasets, key=lambda x: x[1], reverse=True):
                print(f"  {name:<40} {size:>10.2f} MB")
                total_size += size
            print("=" * 60)
            print(f"  总计: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        return

    # 清理操作
    print("\n开始清理...")
    print("=" * 60)

    cleaned = 0

    if args.zeroshot:
        print("清理 Zero-shot 相关数据集:")
        for dataset in ZEROSHOT_DATASETS:
            if clean_dataset(cache_dir, dataset):
                cleaned += 1

    elif args.ppl:
        print("清理 PPL 相关数据集:")
        for dataset in PPL_DATASETS:
            if clean_dataset(cache_dir, dataset):
                cleaned += 1

    elif args.dataset:
        print(f"清理数据集: {args.dataset}")
        if clean_dataset(cache_dir, args.dataset):
            cleaned += 1

    elif args.all:
        print("⚠️  警告：将删除所有数据集缓存！")
        confirm = input("确认删除？输入 'yes' 继续: ")
        if confirm.lower() == 'yes':
            datasets = list_cached_datasets(cache_dir)
            for name, _ in datasets:
                if clean_dataset(cache_dir, name):
                    cleaned += 1
        else:
            print("已取消")
            return

    print("=" * 60)
    print(f"✓ 完成！共清理 {cleaned} 个数据集")
    print("\n提示：下次运行评估时，这些数据集会自动重新下载")


if __name__ == '__main__':
    main()
