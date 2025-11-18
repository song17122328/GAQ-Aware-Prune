#!/usr/bin/env python3
"""
PIQA 数据集预加载脚本

从 GitHub 下载 PIQA 数据集并保存到本地缓存
解决 HuggingFace 上数据集被删除的问题

用法:
    python evaluation/preload_piqa.py
"""

import os
import json
import requests
from datasets import load_dataset

# GitHub 原始数据 URL
PIQA_URLS = {
    "train_data": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train.jsonl",
    "train_labels": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train-labels.lst",
    "validation_data": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl",
    "validation_labels": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"
}


def generate_lm_eval_task(valid_file):
    """
    生成 lm-eval 的自定义 PIQA 任务 YAML 文件

    Args:
        valid_file: 验证集 jsonl 文件路径（包含标签）
    """
    # 获取任务目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_dir = os.path.join(script_dir, 'tasks')
    os.makedirs(tasks_dir, exist_ok=True)

    yaml_path = os.path.join(tasks_dir, 'piqa_local.yaml')

    # 生成 YAML 内容（使用绝对路径）
    yaml_content = f"""task: piqa_local
dataset_path: json
dataset_kwargs:
  data_files:
    validation: {valid_file}
output_type: multiple_choice
validation_split: validation
doc_to_text: "Question: {{{{goal}}}}\\nAnswer:"
doc_to_target: label
doc_to_choice:
  - "{{{{sol1}}}}"
  - "{{{{sol2}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✓ 已生成 lm-eval 任务文件: {yaml_path}")


def add_labels_from_url(dataset_split, url):
    """
    下载标签文件并将其作为 'label' 列添加到数据集中

    Args:
        dataset_split: 数据集分割
        url: 标签文件 URL

    Returns:
        添加了标签列的数据集
    """
    print(f"  下载标签: {url.split('/')[-1]}")
    response = requests.get(url)
    response.raise_for_status()

    labels = [int(line.strip()) for line in response.text.strip().split('\n')]

    assert len(dataset_split) == len(labels), \
        f"数据和标签数量不匹配！数据: {len(dataset_split)}, 标签: {len(labels)}"

    return dataset_split.add_column("label", labels)


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


def preload_piqa():
    """
    从 GitHub 下载 PIQA 数据集并保存到本地
    """
    print("=" * 60)
    print("PIQA 数据集预加载 (从 GitHub)")
    print("=" * 60)

    try:
        # 创建本地下载目录
        local_data_dir = os.path.expanduser("~/.cache/huggingface/datasets/piqa_local_jsonl")
        os.makedirs(local_data_dir, exist_ok=True)

        # 1. 先用 requests 下载文件到本地（支持代理）
        print("\n[1/4] 从 GitHub 下载数据...")

        local_train = os.path.join(local_data_dir, "train.jsonl")
        local_valid = os.path.join(local_data_dir, "valid.jsonl")
        local_train_labels = os.path.join(local_data_dir, "train-labels.lst")
        local_valid_labels = os.path.join(local_data_dir, "valid-labels.lst")

        download_file(PIQA_URLS['train_data'], local_train)
        download_file(PIQA_URLS['validation_data'], local_valid)
        download_file(PIQA_URLS['train_labels'], local_train_labels)
        download_file(PIQA_URLS['validation_labels'], local_valid_labels)

        print("✓ 文件下载完成")

        # 2. 从本地文件加载数据集
        print("\n[2/4] 从本地文件加载数据集...")

        dataset = load_dataset('json', data_files={
            'train': local_train,
            'validation': local_valid
        })

        print(f"✓ JSONL 数据加载成功")
        print(f"  训练集: {len(dataset['train'])} 样本")
        print(f"  验证集: {len(dataset['validation'])} 样本")

        # 3. 加载标签并合并
        print("\n[3/4] 合并标签...")

        with open(local_train_labels) as f:
            train_labels = [int(line.strip()) for line in f.readlines()]
        with open(local_valid_labels) as f:
            valid_labels = [int(line.strip()) for line in f.readlines()]

        dataset['train'] = dataset['train'].add_column("label", train_labels)
        dataset['validation'] = dataset['validation'].add_column("label", valid_labels)
        print("✓ 标签合并完成")

        # 4. 验证数据并保存
        print("\n[4/4] 验证数据并保存...")
        sample = dataset['validation'][0]
        print(f"  样本字段: {list(sample.keys())}")
        print(f"  goal: {sample['goal'][:50]}...")
        print(f"  sol1: {sample['sol1'][:50]}...")
        print(f"  sol2: {sample['sol2'][:50]}...")
        print(f"  label: {sample['label']}")
        print("✓ 数据验证通过")

        # 保存为 HF 格式
        save_dir = os.path.expanduser("~/.cache/huggingface/datasets/piqa_local")
        dataset.save_to_disk(save_dir)
        print(f"✓ HF 格式已保存: {save_dir}")

        # 创建包含标签的 jsonl 文件供 lm-eval 使用
        valid_jsonl_path = os.path.join(local_data_dir, "valid_with_labels.jsonl")
        with open(valid_jsonl_path, 'w') as f:
            for item in dataset['validation']:
                f.write(json.dumps(dict(item)) + '\n')
        print(f"✓ JSONL 格式已保存: {valid_jsonl_path}")

        # 生成 lm-eval 任务文件
        generate_lm_eval_task(valid_jsonl_path)

        print("\n" + "=" * 60)
        print("✓ PIQA 预加载完成!")
        print("=" * 60)
        print("\n现在可以使用完整的7个数据集进行评估:")
        print("python evaluation/run_evaluation.py \\")
        print("    --model_path prune_log/experiment/pytorch_model.bin \\")
        print("    --metrics zeroshot \\")
        print("    --output results/evaluation.json")

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
    success = preload_piqa()
    exit(0 if success else 1)
