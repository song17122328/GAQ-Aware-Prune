#!/usr/bin/env python3
"""
本地 Zero-shot 任务配置

为所有 7 个 zero-shot 评估任务提供本地版本，
从 data/zeroshot/{task_name}/validation.jsonl 加载数据，
避免网络请求。

使用方法:
    1. 运行 python evaluation/download_datasets.py 下载数据
    2. 在评估时使用 *_local 任务名称
"""

import os
import yaml


def get_project_root():
    """获取项目根目录"""
    # 从 evaluation/tasks/ 向上两级
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_local_data_path(task_name: str) -> str:
    """获取本地数据文件路径"""
    project_root = get_project_root()
    return os.path.join(project_root, "data", "zeroshot", task_name, "validation.jsonl")


def check_local_data_exists(task_name: str) -> bool:
    """检查本地数据是否存在"""
    return os.path.exists(get_local_data_path(task_name))


# 所有本地任务的配置模板
LOCAL_TASK_CONFIGS = {
    'boolq_local': {
        'task': 'boolq_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': '{{passage}}\nQuestion: {{question}}?\nAnswer:',
        'doc_to_target': '{% if answer %}1{% else %}0{% endif %}',
        'doc_to_choice': ['no', 'yes'],
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
    'piqa_local': {
        'task': 'piqa_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': 'Question: {{goal}}\nAnswer:',
        'doc_to_target': '{{label}}',
        'doc_to_choice': ['{{sol1}}', '{{sol2}}'],
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
            {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
    'hellaswag_local': {
        'task': 'hellaswag_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': '{{ctx}}',
        'doc_to_target': '{{label}}',
        'doc_to_choice': '{{endings}}',
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
            {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
    'winogrande_local': {
        'task': 'winogrande_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': '{{sentence}}',
        'doc_to_target': '{% if answer == "1" %}0{% else %}1{% endif %}',
        'doc_to_choice': ['{{option1}}', '{{option2}}'],
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
    'arc_easy_local': {
        'task': 'arc_easy_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': 'Question: {{question}}\nAnswer:',
        'doc_to_target': "{{choices.label.index(answerKey)}}",
        'doc_to_choice': '{{choices.text}}',
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
            {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
    'arc_challenge_local': {
        'task': 'arc_challenge_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': 'Question: {{question}}\nAnswer:',
        'doc_to_target': "{{choices.label.index(answerKey)}}",
        'doc_to_choice': '{{choices.text}}',
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
            {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
    'openbookqa_local': {
        'task': 'openbookqa_local',
        'dataset_path': 'json',
        'output_type': 'multiple_choice',
        'validation_split': 'validation',
        'doc_to_text': '{{question_stem}}',
        'doc_to_target': "{{choices.label.index(answerKey)}}",
        'doc_to_choice': '{{choices.text}}',
        'metric_list': [
            {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
            {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}
        ]
    },
}


def get_task_config(task_name: str) -> dict:
    """获取指定任务的完整配置（包含数据路径）"""
    if task_name not in LOCAL_TASK_CONFIGS:
        raise ValueError(f"未知任务: {task_name}")

    # 获取基础任务名（去掉 _local 后缀）
    base_name = task_name.replace('_local', '')
    data_path = get_local_data_path(base_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到 {task_name} 的本地数据: {data_path}\n"
            f"请先运行: python evaluation/download_datasets.py"
        )

    # 复制配置并添加数据路径
    config = LOCAL_TASK_CONFIGS[task_name].copy()
    config['dataset_kwargs'] = {
        'data_files': {
            'validation': data_path
        }
    }

    return config


def generate_yaml_configs(output_dir: str = None):
    """生成所有本地任务的 YAML 配置文件"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    for task_name in LOCAL_TASK_CONFIGS:
        base_name = task_name.replace('_local', '')

        # 检查数据是否存在
        if not check_local_data_exists(base_name):
            print(f"跳过 {task_name}: 数据不存在")
            continue

        config = get_task_config(task_name)
        yaml_path = os.path.join(output_dir, f"{task_name}.yaml")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ 生成 {yaml_path}")

    print(f"\n配置文件已生成到: {output_dir}")


def get_available_local_tasks() -> list:
    """获取所有可用的本地任务（数据已下载）"""
    available = []
    for task_name in LOCAL_TASK_CONFIGS:
        base_name = task_name.replace('_local', '')
        if check_local_data_exists(base_name):
            available.append(task_name)
    return available


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='管理本地 Zero-shot 任务')
    parser.add_argument('--generate', action='store_true', help='生成 YAML 配置文件')
    parser.add_argument('--check', action='store_true', help='检查本地数据状态')
    args = parser.parse_args()

    if args.generate:
        generate_yaml_configs()
    elif args.check:
        print("本地任务数据状态:")
        print("="*50)
        for task_name in LOCAL_TASK_CONFIGS:
            base_name = task_name.replace('_local', '')
            data_path = get_local_data_path(base_name)
            exists = os.path.exists(data_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {task_name}: {data_path}")

        available = get_available_local_tasks()
        print(f"\n可用任务 ({len(available)}/{len(LOCAL_TASK_CONFIGS)}):")
        print(f"  {', '.join(available) if available else '无'}")
    else:
        parser.print_help()
