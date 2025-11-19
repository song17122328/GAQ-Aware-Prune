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


def download_piqa_from_official(save_dir: str):
    """从官方 GitHub 下载 PIQA 数据集"""
    import requests
    import json
    from datasets import Dataset, DatasetDict

    piqa_dir = os.path.join(save_dir, "piqa")
    os.makedirs(piqa_dir, exist_ok=True)

    # PIQA 官方数据 URL
    BASE_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data"

    files = {
        'train': (f"{BASE_URL}/train.jsonl", f"{BASE_URL}/train-labels.lst"),
        'validation': (f"{BASE_URL}/valid.jsonl", f"{BASE_URL}/valid-labels.lst"),
    }

    datasets = {}

    for split, (data_url, labels_url) in files.items():
        print(f"    下载 {split} 集...")

        # 下载数据
        data_response = requests.get(data_url, timeout=60)
        data_response.raise_for_status()

        # 下载标签
        labels_response = requests.get(labels_url, timeout=60)
        labels_response.raise_for_status()

        # 解析数据
        data_lines = data_response.text.strip().split('\n')
        labels = labels_response.text.strip().split('\n')

        goals = []
        sol1s = []
        sol2s = []
        label_list = []

        for line, label in zip(data_lines, labels):
            item = json.loads(line)
            goals.append(item['goal'])
            sol1s.append(item['sol1'])
            sol2s.append(item['sol2'])
            label_list.append(int(label))

        datasets[split] = Dataset.from_dict({
            'goal': goals,
            'sol1': sol1s,
            'sol2': sol2s,
            'label': label_list
        })

        print(f"      - {len(goals)} 样本")

    # 保存数据集 (HuggingFace 格式)
    dataset_dict = DatasetDict(datasets)
    dataset_dict.save_to_disk(piqa_dir)

    # 同时保存 jsonl 格式供 lm-eval 使用
    import json

    for split in ['train', 'validation']:
        jsonl_path = os.path.join(piqa_dir, f"{split}.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            ds = datasets[split]
            for i in range(len(ds)):
                item = {
                    'goal': ds['goal'][i],
                    'sol1': ds['sol1'][i],
                    'sol2': ds['sol2'][i],
                    'label': ds['label'][i]
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"      - 保存 {jsonl_path}")

    return piqa_dir


def download_zeroshot_datasets(save_dir: str):
    """下载 Zero-shot 评估所需的 7 个数据集到本地，保存为 jsonl 格式"""
    from datasets import load_dataset
    import json

    print("\n" + "="*50)
    print("下载 Zero-shot 评估数据集 (7个)")
    print("="*50)

    # 设置缓存目录为 data/zeroshot
    zeroshot_dir = os.path.join(save_dir, "zeroshot")
    os.makedirs(zeroshot_dir, exist_ok=True)

    # 先处理 piqa (从官方源下载)
    try:
        print(f"  下载 piqa (从官方 GitHub)...")
        piqa_path = download_piqa_from_official(zeroshot_dir)
        print(f"    ✓ 完成，保存到: {piqa_path}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")

    # Zero-shot 任务对应的数据集配置
    # 格式: task_name -> (hf_dataset, config, validation_split, fields_to_save)
    ZEROSHOT_DATASETS = {
        'boolq': ('google/boolq', None, 'validation', ['question', 'passage', 'answer']),
        'hellaswag': ('Rowan/hellaswag', None, 'validation', ['ctx', 'ctx_a', 'ctx_b', 'endings', 'label']),
        'winogrande': ('winogrande', 'winogrande_xl', 'validation', ['sentence', 'option1', 'option2', 'answer']),
        'arc_easy': ('allenai/ai2_arc', 'ARC-Easy', 'validation', ['question', 'choices', 'answerKey']),
        'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge', 'validation', ['question', 'choices', 'answerKey']),
        'openbookqa': ('allenai/openbookqa', 'main', 'validation', ['question_stem', 'choices', 'answerKey']),
    }

    # 下载并保存为 jsonl
    for task_name, (dataset_name, config, val_split, fields) in ZEROSHOT_DATASETS.items():
        task_dir = os.path.join(zeroshot_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        jsonl_path = os.path.join(task_dir, "validation.jsonl")

        # 如果已存在则跳过
        if os.path.exists(jsonl_path):
            print(f"  {task_name}: 已存在，跳过")
            continue

        try:
            print(f"  下载 {task_name} ({dataset_name})...")

            if config:
                dataset = load_dataset(dataset_name, config, split=val_split)
            else:
                dataset = load_dataset(dataset_name, split=val_split)

            # 保存为 jsonl
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    # 只保留需要的字段
                    row = {k: item[k] for k in fields if k in item}
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

            print(f"    ✓ 完成 ({len(dataset)} 样本) -> {jsonl_path}")

        except Exception as e:
            print(f"    ✗ 失败: {e}")

    print(f"\n  数据集已保存到: {zeroshot_dir}")
    print(f"  每个任务都保存为 jsonl 格式，评估时从本地加载")


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
