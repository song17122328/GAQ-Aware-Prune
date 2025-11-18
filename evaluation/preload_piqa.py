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
    yaml_content = f"""group: piqa_local
task: piqa_local
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


def preload_piqa():
    """
    从 GitHub 下载 PIQA 数据集并保存到本地
    """
    print("=" * 60)
    print("PIQA 数据集预加载 (从 GitHub)")
    print("=" * 60)

    try:
        # 1. 从 GitHub 加载 JSONL 数据
        print("\n[1/4] 从 GitHub 下载数据...")
        print(f"  训练集: {PIQA_URLS['train_data'].split('/')[-1]}")
        print(f"  验证集: {PIQA_URLS['validation_data'].split('/')[-1]}")

        dataset = load_dataset('json', data_files={
            'train': PIQA_URLS['train_data'],
            'validation': PIQA_URLS['validation_data']
        })

        print(f"✓ JSONL 数据加载成功")
        print(f"  训练集: {len(dataset['train'])} 样本")
        print(f"  验证集: {len(dataset['validation'])} 样本")

        # 2. 下载并合并标签
        print("\n[2/4] 下载并合并标签...")
        dataset['train'] = add_labels_from_url(
            dataset['train'],
            PIQA_URLS['train_labels']
        )
        dataset['validation'] = add_labels_from_url(
            dataset['validation'],
            PIQA_URLS['validation_labels']
        )
        print("✓ 标签合并完成")

        # 3. 验证数据
        print("\n[3/4] 验证数据...")
        sample = dataset['validation'][0]
        print(f"  样本字段: {list(sample.keys())}")
        print(f"  goal: {sample['goal'][:50]}...")
        print(f"  sol1: {sample['sol1'][:50]}...")
        print(f"  sol2: {sample['sol2'][:50]}...")
        print(f"  label: {sample['label']}")
        print("✓ 数据验证通过")

        # 4. 保存到本地
        print("\n[4/4] 保存数据到本地...")

        # 保存为 HF 格式
        save_dir = os.path.expanduser("~/.cache/huggingface/datasets/piqa_local")
        dataset.save_to_disk(save_dir)
        print(f"✓ HF 格式已保存: {save_dir}")

        # 创建包含标签的 jsonl 文件供 lm-eval 使用
        local_data_dir = os.path.expanduser("~/.cache/huggingface/datasets/piqa_local_jsonl")
        os.makedirs(local_data_dir, exist_ok=True)

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
