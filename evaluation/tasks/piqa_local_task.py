#!/usr/bin/env python3
"""
自定义 PIQA 任务 - 从本地 jsonl 文件加载

用于 lm-eval 评估，解决 PIQA 数据集下载问题。
数据从 data/zeroshot/piqa/ 目录加载。

使用方法:
    1. 先运行 python evaluation/download_datasets.py 下载数据
    2. 在 evaluate_zeroshot 中使用 'piqa_local' 替代 'piqa'
"""

import os
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.registry import register_task


def find_piqa_data_dir():
    """查找 PIQA 数据目录

    查找顺序:
    1. 项目根目录下的 data/zeroshot/piqa/
    2. 用户 home 目录下的 .cache/huggingface/hub/datasets--ybisk--piqa/
    """
    # 方法1: 项目根目录
    # 从 evaluation/tasks/ 向上两级到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_piqa_dir = os.path.join(project_root, "data", "zeroshot", "piqa")

    if os.path.exists(local_piqa_dir):
        validation_file = os.path.join(local_piqa_dir, "validation.jsonl")
        if os.path.exists(validation_file):
            return local_piqa_dir

    # 方法2: HuggingFace 缓存目录 (旧的方式)
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--ybisk--piqa")
    snapshots_dir = os.path.join(cache_base, "snapshots")

    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[0])

    return None


def get_piqa_task_config():
    """获取 PIQA 任务配置"""
    data_dir = find_piqa_data_dir()

    if data_dir is None:
        raise FileNotFoundError(
            "未找到 PIQA 本地数据。请先运行:\n"
            "  python evaluation/download_datasets.py\n"
            "这将从官方 GitHub 下载 PIQA 数据到 data/zeroshot/piqa/"
        )

    validation_file = os.path.join(data_dir, "validation.jsonl")

    # 检查文件是否存在
    if not os.path.exists(validation_file):
        raise FileNotFoundError(f"未找到文件: {validation_file}")

    return {
        "task": "piqa_local",
        "dataset_path": "json",
        "dataset_kwargs": {
            "data_files": {
                "validation": validation_file
            }
        },
        "output_type": "multiple_choice",
        "validation_split": "validation",
        "doc_to_text": "Question: {{goal}}\nAnswer:",
        "doc_to_target": "{{label}}",
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
        data_dir = find_piqa_data_dir()
        if data_dir is None:
            raise FileNotFoundError(
                "未找到 PIQA 本地数据。请先运行: python evaluation/download_datasets.py"
            )

        validation_file = os.path.join(data_dir, "validation.jsonl")

        if not os.path.exists(validation_file):
            raise FileNotFoundError(f"未找到文件: {validation_file}")

        config = {
            "task": "piqa_local",
            "dataset_path": "json",
            "dataset_kwargs": {
                "data_files": {
                    "validation": validation_file
                }
            },
            "output_type": "multiple_choice",
            "validation_split": "validation",
            "doc_to_text": "Question: {{goal}}\nAnswer:",
            "doc_to_target": "{{label}}",
            "doc_to_choice": ["{{sol1}}", "{{sol2}}"],
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
            ]
        }

        super().__init__(config=config, **kwargs)


def register_piqa_local():
    """注册本地 PIQA 任务到 lm-eval"""
    try:
        # 检查数据是否存在
        data_dir = find_piqa_data_dir()
        if data_dir is None:
            print("警告: 未找到 PIQA 本地数据，跳过注册 piqa_local 任务")
            print("请先运行: python evaluation/download_datasets.py")
            return False

        # 注册任务
        register_task("piqa_local")(PIQALocalTask)
        print(f"✓ 已注册本地 PIQA 任务: piqa_local (数据目录: {data_dir})")
        return True

    except Exception as e:
        print(f"注册 piqa_local 任务失败: {e}")
        return False


if __name__ == "__main__":
    # 测试任务配置
    data_dir = find_piqa_data_dir()
    if data_dir:
        print(f"找到 PIQA 数据目录: {data_dir}")
        print("\n文件列表:")
        for f in os.listdir(data_dir):
            fpath = os.path.join(data_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / 1024
                print(f"  - {f} ({size:.1f} KB)")

        # 测试任务注册
        print("\n测试任务注册:")
        success = register_piqa_local()
        if success:
            print("PIQA 本地任务注册成功")
    else:
        print("未找到 PIQA 本地数据")
        print("请先运行: python evaluation/download_datasets.py")
