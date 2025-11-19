#!/usr/bin/env python3
"""
自定义 Zero-shot 评估器

不依赖 lm-eval，直接从本地 jsonl 文件加载数据并评估。
通过计算每个选项的 log-likelihood 来选择答案。

支持的任务:
- piqa: 物理常识推理
- boolq: 是非问答
- hellaswag: 常识推理
- winogrande: 代词消歧
- arc_easy/arc_challenge: 科学问答
- openbookqa: 科学推理
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(file_path: str) -> List[dict]:
    """加载 jsonl 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_loglikelihood(
    model,
    tokenizer,
    context: str,
    continuation: str,
    device: str
) -> float:
    """
    计算给定上下文后续文本的 log-likelihood

    Args:
        model: 语言模型
        tokenizer: tokenizer
        context: 上下文文本
        continuation: 续写文本
        device: 设备

    Returns:
        log-likelihood 值
    """
    # 编码上下文和完整文本
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_text = context + continuation
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # 获取续写部分的 token ids
    continuation_ids = full_ids[len(context_ids):]

    if len(continuation_ids) == 0:
        return float('-inf')

    # 准备输入
    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # 计算续写部分的 log-likelihood
    # logits shape: [1, seq_len, vocab_size]
    # 我们需要从 context 结束位置开始计算
    start_pos = len(context_ids) - 1  # -1 因为我们用前一个 token 预测当前 token

    log_likelihood = 0.0
    for i, token_id in enumerate(continuation_ids):
        pos = start_pos + i
        if pos >= logits.shape[1]:
            break

        # 获取该位置的 logits 并计算 log_softmax
        token_logits = logits[0, pos, :]
        log_probs = F.log_softmax(token_logits, dim=-1)
        log_likelihood += log_probs[token_id].item()

    return log_likelihood


def evaluate_multiple_choice(
    model,
    tokenizer,
    questions: List[dict],
    format_fn,
    device: str,
    task_name: str = ""
) -> Tuple[float, int, int]:
    """
    评估多选题任务

    Args:
        model: 语言模型
        tokenizer: tokenizer
        questions: 问题列表
        format_fn: 格式化函数，返回 (context, choices, label)
        device: 设备
        task_name: 任务名称（用于显示）

    Returns:
        (accuracy, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0

    desc = f"评估 {task_name}" if task_name else "评估中"

    for item in tqdm(questions, desc=desc):
        context, choices, label = format_fn(item)

        # 计算每个选项的 log-likelihood
        scores = []
        for choice in choices:
            ll = compute_loglikelihood(model, tokenizer, context, choice, device)
            scores.append(ll)

        # 选择得分最高的选项
        pred = scores.index(max(scores))

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


# ============== 任务格式化函数 ==============

def format_piqa(item: dict) -> Tuple[str, List[str], int]:
    """PIQA 格式化"""
    context = f"Question: {item['goal']}\nAnswer:"
    choices = [item['sol1'], item['sol2']]
    label = item['label']
    return context, choices, label


def format_boolq(item: dict) -> Tuple[str, List[str], int]:
    """BoolQ 格式化"""
    context = f"{item['passage']}\nQuestion: {item['question']}?\nAnswer:"
    choices = ["no", "yes"]
    label = 1 if item['answer'] else 0
    return context, choices, label


def format_hellaswag(item: dict) -> Tuple[str, List[str], int]:
    """HellaSwag 格式化"""
    context = item['ctx']
    choices = item['endings']
    label = int(item['label'])
    return context, choices, label


def format_winogrande(item: dict) -> Tuple[str, List[str], int]:
    """Winogrande 格式化"""
    context = item['sentence']
    choices = [item['option1'], item['option2']]
    # answer 是 "1" 或 "2"，转换为 0 或 1
    label = int(item['answer']) - 1
    return context, choices, label


def format_arc(item: dict) -> Tuple[str, List[str], int]:
    """ARC (Easy/Challenge) 格式化"""
    context = f"Question: {item['question']}\nAnswer:"
    choices = item['choices']['text']
    labels = item['choices']['label']
    answer_key = item['answerKey']
    label = labels.index(answer_key)
    return context, choices, label


def format_openbookqa(item: dict) -> Tuple[str, List[str], int]:
    """OpenBookQA 格式化"""
    context = item['question_stem']
    choices = item['choices']['text']
    labels = item['choices']['label']
    answer_key = item['answerKey']
    label = labels.index(answer_key)
    return context, choices, label


# ============== 主评估函数 ==============

TASK_CONFIGS = {
    'piqa': {
        'file': 'piqa/validation.jsonl',
        'format_fn': format_piqa
    },
    'boolq': {
        'file': 'boolq/validation.jsonl',
        'format_fn': format_boolq
    },
    'hellaswag': {
        'file': 'hellaswag/validation.jsonl',
        'format_fn': format_hellaswag
    },
    'winogrande': {
        'file': 'winogrande/validation.jsonl',
        'format_fn': format_winogrande
    },
    'arc_easy': {
        'file': 'arc_easy/validation.jsonl',
        'format_fn': format_arc
    },
    'arc_challenge': {
        'file': 'arc_challenge/validation.jsonl',
        'format_fn': format_arc
    },
    'openbookqa': {
        'file': 'openbookqa/validation.jsonl',
        'format_fn': format_openbookqa
    }
}


def evaluate_zeroshot_custom(
    model,
    tokenizer,
    tasks: List[str] = None,
    device: str = 'cuda',
    data_dir: str = None
) -> Dict[str, Dict]:
    """
    自定义 Zero-shot 评估（不使用 lm-eval）

    Args:
        model: 语言模型
        tokenizer: tokenizer
        tasks: 任务列表，默认所有任务
        device: 设备
        data_dir: 数据目录，默认 data/zeroshot

    Returns:
        {task_name: {'accuracy': float, 'correct': int, 'total': int}}
    """
    if tasks is None:
        tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande',
                 'arc_easy', 'arc_challenge', 'openbookqa']

    if data_dir is None:
        data_dir = os.path.join(get_project_root(), 'data', 'zeroshot')

    print(f"\n{'='*60}")
    print(f"自定义 Zero-shot 评估")
    print(f"{'='*60}")
    print(f"任务: {', '.join(tasks)}")
    print(f"数据目录: {data_dir}\n")

    results = {}

    for task in tasks:
        if task not in TASK_CONFIGS:
            print(f"⚠ 未知任务: {task}，跳过")
            continue

        config = TASK_CONFIGS[task]
        file_path = os.path.join(data_dir, config['file'])

        if not os.path.exists(file_path):
            print(f"⚠ 数据文件不存在: {file_path}，跳过")
            continue

        # 加载数据
        questions = load_jsonl(file_path)
        print(f"\n{task}: 加载 {len(questions)} 个样本")

        # 评估
        accuracy, correct, total = evaluate_multiple_choice(
            model, tokenizer, questions,
            config['format_fn'],
            device,
            task_name=task
        )

        results[task] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

        print(f"✓ {task}: {accuracy*100:.2f}% ({correct}/{total})")

    # 计算平均准确率
    if results:
        avg_acc = sum(r['accuracy'] for r in results.values()) / len(results)
        print(f"\n平均准确率: {avg_acc*100:.2f}%")

    return results


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from evaluation.utils.model_loader import load_model_and_tokenizer

    parser = argparse.ArgumentParser(description='自定义 Zero-shot 评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--tasks', type=str, default=None,
                       help='任务列表，逗号分隔')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录，默认 data/zeroshot')
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        force_single_device=True
    )

    # 解析任务
    tasks = args.tasks.split(',') if args.tasks else None

    # 评估
    results = evaluate_zeroshot_custom(
        model, tokenizer,
        tasks=tasks,
        device=args.device,
        data_dir=args.data_dir
    )

    print(f"\n完整结果: {json.dumps(results, indent=2)}")
