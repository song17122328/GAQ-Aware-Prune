#!/usr/bin/env python3
"""
性能指标评估

包括:
1. PPL (Perplexity) - 多数据集
2. Zero-shot准确率 - 常识推理、阅读理解等
3. Few-shot准确率 - MMLU等（可选）
"""

import sys
import os
import torch
from typing import Dict, List, Optional, Union

# 添加LLMPruner到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def evaluate_ppl(
    model,
    tokenizer,
    datasets: List[str] = ['wikitext2', 'ptb', 'c4'],
    seq_len: int = 128,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    评估多个数据集上的PPL

    Args:
        model: 模型
        tokenizer: tokenizer
        datasets: 数据集列表，支持 'wikitext2', 'ptb', 'c4'
        seq_len: 序列长度
        device: 设备

    Returns:
        {dataset_name: ppl_value}
    """
    from LLMPruner.evaluator.ppl import PPLMetric

    print(f"\n{'='*60}")
    print(f"评估 PPL (seq_len={seq_len})")
    print(f"{'='*60}")

    results = {}

    for dataset in datasets:
        print(f"\n测试数据集: {dataset}")
        try:
            ppl_metric = PPLMetric(
                model=model,
                tokenizer=tokenizer,
                datasets=[dataset],
                seq_len=seq_len,
                device=device
            )

            # PPLMetric返回的是类似dict的对象
            for key, value in ppl_metric.items():
                results[key] = value
                print(f"  ✓ {key}: {value:.2f}")

        except Exception as e:
            print(f"  ✗ 评估失败: {e}")
            results[dataset] = None

    return results


def evaluate_zeroshot(
    model_path: str,
    tasks: List[str] = None,
    batch_size: int = 8,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    使用lm-evaluation-harness评估Zero-shot任务

    Args:
        model_path: 模型路径（HF格式或checkpoint路径）
        tasks: 任务列表，默认为常用的5个任务
        batch_size: 批次大小
        device: 设备

    Returns:
        评估结果字典

    注意：
        需要安装 lm-eval: pip install lm-eval
        checkpoint需要先转换为HF格式，或者临时保存
    """
    if tasks is None:
        tasks = [
            'hellaswag',      # 常识推理
            'piqa',           # 物理常识
            'winogrande',     # 代词消歧
            'arc_easy',       # 科学问答（简单）
            'boolq'           # 是非问答
        ]

    print(f"\n{'='*60}")
    print(f"评估 Zero-shot 任务")
    print(f"{'='*60}")
    print(f"任务: {', '.join(tasks)}")

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # 检查是否是checkpoint文件
        if model_path.endswith('.bin'):
            print("⚠️  检测到checkpoint文件，需要先转换为HF格式")
            print("请使用以下方式之一:")
            print("  1. 在剪枝时使用 --save_model 保存完整模型目录")
            print("  2. 手动转换checkpoint为HF格式")
            return None

        # 使用lm-eval评估
        print(f"\n开始评估...")

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=float16,device={device}",
            tasks=tasks,
            batch_size=batch_size,
            log_samples=False
        )

        # 提取关键结果
        summary = {}
        for task in tasks:
            if task in results['results']:
                task_results = results['results'][task]
                # 提取准确率（不同任务的metric名称可能不同）
                if 'acc_norm' in task_results:
                    acc = task_results['acc_norm']
                elif 'acc' in task_results:
                    acc = task_results['acc']
                else:
                    acc = None

                summary[task] = {
                    'accuracy': acc,
                    'full_results': task_results
                }

                print(f"  ✓ {task}: {acc*100:.2f}%" if acc is not None else f"  ✓ {task}: N/A")

        return summary

    except ImportError:
        print("✗ lm-eval未安装")
        print("请安装: pip install lm-eval")
        return None

    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_fewshot(
    model_path: str,
    tasks: List[str] = None,
    num_fewshot: int = 5,
    batch_size: int = 8,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    评估Few-shot任务（可选）

    Args:
        model_path: 模型路径
        tasks: 任务列表，默认为MMLU
        num_fewshot: few-shot样本数
        batch_size: 批次大小
        device: 设备

    Returns:
        评估结果字典
    """
    if tasks is None:
        tasks = ['mmlu']  # MMLU是最常用的few-shot任务

    print(f"\n{'='*60}")
    print(f"评估 {num_fewshot}-shot 任务")
    print(f"{'='*60}")
    print(f"任务: {', '.join(tasks)}")
    print("⚠️  Few-shot评估较慢，建议只在最终对比时使用")

    try:
        import lm_eval

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=float16,device={device}",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            log_samples=False
        )

        summary = {}
        for task in tasks:
            if task in results['results']:
                task_results = results['results'][task]
                if 'acc' in task_results:
                    acc = task_results['acc']
                else:
                    acc = None

                summary[task] = {
                    'accuracy': acc,
                    'full_results': task_results
                }

                print(f"  ✓ {task}: {acc*100:.2f}%" if acc is not None else f"  ✓ {task}: N/A")

        return summary

    except ImportError:
        print("✗ lm-eval未安装")
        return None

    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return None


def compute_average_accuracy(zeroshot_results: Dict[str, Dict]) -> float:
    """
    计算多个任务的平均准确率

    Args:
        zeroshot_results: evaluate_zeroshot返回的结果

    Returns:
        平均准确率 (0-1)
    """
    if not zeroshot_results:
        return 0.0

    accuracies = []
    for task, results in zeroshot_results.items():
        if results and 'accuracy' in results and results['accuracy'] is not None:
            accuracies.append(results['accuracy'])

    if not accuracies:
        return 0.0

    avg_acc = sum(accuracies) / len(accuracies)
    return avg_acc


if __name__ == '__main__':
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--metrics', type=str, default='ppl,zeroshot',
                       help='逗号分隔的指标: ppl, zeroshot, fewshot')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    metrics = args.metrics.split(',')

    # 加载模型
    from evaluation.utils.model_loader import load_model_and_tokenizer

    if 'ppl' in metrics:
        model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
        ppl_results = evaluate_ppl(model, tokenizer, device=args.device)
        print(f"\nPPL结果: {ppl_results}")

    if 'zeroshot' in metrics:
        zeroshot_results = evaluate_zeroshot(args.model_path, device=args.device)
        if zeroshot_results:
            avg_acc = compute_average_accuracy(zeroshot_results)
            print(f"\n平均准确率: {avg_acc*100:.2f}%")

    if 'fewshot' in metrics:
        fewshot_results = evaluate_fewshot(args.model_path, device=args.device)
