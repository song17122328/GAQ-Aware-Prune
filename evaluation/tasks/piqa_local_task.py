#!/usr/bin/env python3
"""
自定义 PIQA 任务 - 从本地 jsonl 文件加载

用于 lm-eval 评估，解决 PIQA 数据集下载问题
"""

import os
import glob
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.registry import register_task


def find_piqa_snapshot_dir():
    """查找 PIQA 数据快照目录"""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--ybisk--piqa")
    snapshots_dir = os.path.join(cache_base, "snapshots")

    if not os.path.exists(snapshots_dir):
        return None

    snapshots = os.listdir(snapshots_dir)
    if not snapshots:
        return None

    return os.path.join(snapshots_dir, snapshots[0])


def get_piqa_task_config():
    """获取 PIQA 任务配置"""
    data_dir = find_piqa_snapshot_dir()

    if data_dir is None:
        raise FileNotFoundError(
            "未找到 PIQA 本地数据。请先运行:\n"
            "  hf download ybisk/piqa --repo-type=dataset\n"
            "然后运行:\n"
            "  python evaluation/preload_piqa.py"
        )

    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")
    train_labels = os.path.join(data_dir, "train-labels.lst")
    valid_labels = os.path.join(data_dir, "valid-labels.lst")

    # 检查文件是否存在
    for f in [train_file, valid_file, train_labels, valid_labels]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"未找到文件: {f}")

    return {
        "task": "piqa_local",
        "dataset_path": "json",
        "dataset_kwargs": {
            "data_files": {
                "train": train_file,
                "validation": valid_file
            }
        },
        "output_type": "multiple_choice",
        "training_split": "train",
        "validation_split": "validation",
        "doc_to_text": "Question: {{goal}}\nAnswer:",
        "doc_to_target": "label",
        "doc_to_choice": ["{{sol1}}", "{{sol2}}"],
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
        ]
    }


class PIQALocalTask(ConfigurableTask):
    """从本地文件加载的 PIQA 任务"""

    VERSION = 1.0

    def __init__(self, **kwargs):
        data_dir = find_piqa_snapshot_dir()
        if data_dir is None:
            raise FileNotFoundError("未找到 PIQA 本地数据")

        train_file = os.path.join(data_dir, "train.jsonl")
        valid_file = os.path.join(data_dir, "valid.jsonl")
        train_labels = os.path.join(data_dir, "train-labels.lst")
        valid_labels = os.path.join(data_dir, "valid-labels.lst")

        # 加载标签
        with open(train_labels) as f:
            self._train_labels = [int(x.strip()) for x in f.readlines()]
        with open(valid_labels) as f:
            self._valid_labels = [int(x.strip()) for x in f.readlines()]

        config = {
            "task": "piqa_local",
            "dataset_path": "json",
            "dataset_kwargs": {
                "data_files": {
                    "train": train_file,
                    "validation": valid_file
                }
            },
            "output_type": "multiple_choice",
            "training_split": "train",
            "validation_split": "validation",
            "doc_to_text": self._doc_to_text,
            "doc_to_target": self._doc_to_target,
            "doc_to_choice": self._doc_to_choice,
        }

        super().__init__(config=config, **kwargs)

    def _doc_to_text(self, doc):
        return f"Question: {doc['goal']}\nAnswer:"

    def _doc_to_target(self, doc):
        # 从标签文件获取标签
        if hasattr(doc, '_index'):
            return self._valid_labels[doc._index]
        return 0

    def _doc_to_choice(self, doc):
        return [doc['sol1'], doc['sol2']]


def register_piqa_local():
    """注册本地 PIQA 任务到 lm-eval"""
    try:
        # 检查数据是否存在
        data_dir = find_piqa_snapshot_dir()
        if data_dir is None:
            print("警告: 未找到 PIQA 本地数据，跳过注册 piqa_local 任务")
            return False

        # 注册任务
        register_task("piqa_local")(PIQALocalTask)
        print("✓ 已注册本地 PIQA 任务: piqa_local")
        return True

    except Exception as e:
        print(f"注册 piqa_local 任务失败: {e}")
        return False


if __name__ == "__main__":
    # 测试任务注册
    success = register_piqa_local()
    if success:
        print("PIQA 本地任务注册成功")
    else:
        print("PIQA 本地任务注册失败")
