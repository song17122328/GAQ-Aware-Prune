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

def find_piqa_data_dir():
    """查找 PIQA 数据目录"""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--ybisk--piqa")

    if not os.path.exists(cache_base):
        return None

    snapshots_dir = os.path.join(cache_base, "snapshots")
    if not os.path.exists(snapshots_dir):
        return None

    snapshots = os.listdir(snapshots_dir)
    if not snapshots:
        return None

    return os.path.join(snapshots_dir, snapshots[0])

def preload_piqa():
    """
    预加载 PIQA 数据集到 datasets 缓存

    由于新版 datasets 不支持旧的脚本格式，需要直接从 jsonl 文件加载
    """
    print("=" * 60)
    print("预加载 PIQA 数据集")
    print("=" * 60)

    # 查找数据目录
    data_dir = find_piqa_data_dir()
    if data_dir is None:
        print("\n✗ 未找到 PIQA 缓存目录")
        print("请先运行: hf download ybisk/piqa --repo-type=dataset")
        return False

    print(f"\n找到数据目录: {data_dir}")

    # 检查文件
    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")
    train_labels = os.path.join(data_dir, "train-labels.lst")
    valid_labels = os.path.join(data_dir, "valid-labels.lst")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"\n✗ 未找到数据文件")
        print(f"目录内容: {os.listdir(data_dir)}")
        return False

    try:
        print("\n从本地 jsonl 文件加载 PIQA...")

        # 直接从 jsonl 文件加载
        dataset = load_dataset(
            'json',
            data_files={
                'train': train_file,
                'validation': valid_file
            }
        )

        # 加载标签文件
        if os.path.exists(train_labels) and os.path.exists(valid_labels):
            print("加载标签文件...")
            with open(train_labels) as f:
                train_label_list = [int(x.strip()) for x in f.readlines()]
            with open(valid_labels) as f:
                valid_label_list = [int(x.strip()) for x in f.readlines()]

            # 添加标签列
            dataset['train'] = dataset['train'].add_column('label', train_label_list)
            dataset['validation'] = dataset['validation'].add_column('label', valid_label_list)

        print(f"\n✓ PIQA 数据集加载成功!")
        print(f"  训练集样本数: {len(dataset['train'])}")
        print(f"  验证集样本数: {len(dataset['validation'])}")

        # 打印样本示例
        print(f"\n样本示例:")
        sample = dataset['validation'][0]
        print(f"  goal: {sample.get('goal', 'N/A')[:50]}...")
        print(f"  sol1: {sample.get('sol1', 'N/A')[:50]}...")
        print(f"  sol2: {sample.get('sol2', 'N/A')[:50]}...")
        if 'label' in sample:
            print(f"  label: {sample['label']}")

        # 保存为标准格式供 lm-eval 使用
        save_dir = os.path.expanduser("~/.cache/huggingface/datasets/piqa_local")
        dataset.save_to_disk(save_dir)
        print(f"\n✓ 数据集已保存到: {save_dir}")

        print("\n" + "=" * 60)
        print("✓ PIQA 预加载完成!")
        print("=" * 60)
        print("\n注意: lm-eval 使用 'piqa' 任务时会自动下载数据集。")
        print("如果遇到网络问题，可以暂时跳过 piqa 任务:")
        print("--zeroshot_tasks boolq,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa")

        return True

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = preload_piqa()
    exit(0 if success else 1)
