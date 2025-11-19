#!/usr/bin/env python3
"""
Baseline 模型批量评估脚本

评估所有 baseline 剪枝模型的性能指标：
- PPL (Perplexity)
- Zero-shot 准确率
- 推理速度 (tokens/sec)
- 显存占用

使用方法:
    # 评估所有 baseline 模型
    python baselines/evaluate_baselines.py \\
        --baselines_dir prune_log/baselines \\
        --metrics ppl,zero_shot,efficiency \\
        --output_dir evaluation_results/baselines

    # 只评估特定方法
    python baselines/evaluate_baselines.py \\
        --methods llm_pruner,wanda,magnitude \\
        --metrics ppl,zero_shot \\
        --output_dir evaluation_results/baselines
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='批量评估 Baseline 剪枝模型')

    # 输入参数
    parser.add_argument('--baselines_dir', type=str, default='prune_log/baselines',
                       help='baseline 模型目录')
    parser.add_argument('--methods', type=str, default=None,
                       help='要评估的方法，逗号分隔。默认评估所有可用模型')
    parser.add_argument('--pruning_ratio', type=int, default=25,
                       help='剪枝率百分比 (用于匹配目录名，如 25 表示 25pct)')

    # 评估参数
    parser.add_argument('--metrics', type=str, default='ppl,zero_shot',
                       help='评估指标，逗号分隔: ppl, zero_shot, efficiency')
    parser.add_argument('--ppl_datasets', type=str, default='wikitext2',
                       help='PPL 评估数据集')
    parser.add_argument('--zero_shot_tasks', type=str, default=None,
                       help='Zero-shot 任务，默认使用标准 7 个任务')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='评估批次大小')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='evaluation_results/baselines',
                       help='结果输出目录')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda')

    # 其他
    parser.add_argument('--include_original', action='store_true',
                       help='包含原始模型作为对照')
    parser.add_argument('--original_model', type=str, default=None,
                       help='原始模型路径（用于对照）')

    return parser.parse_args()


def find_baseline_models(baselines_dir: str, methods: List[str] = None,
                         pruning_ratio: int = 25) -> Dict[str, str]:
    """
    查找所有可用的 baseline 模型

    Args:
        baselines_dir: baseline 目录
        methods: 指定的方法列表
        pruning_ratio: 剪枝率百分比

    Returns:
        {method_name: checkpoint_path}
    """
    models = {}

    if not os.path.exists(baselines_dir):
        print(f"警告: 目录不存在 {baselines_dir}")
        return models

    # 遍历目录
    for item in os.listdir(baselines_dir):
        item_path = os.path.join(baselines_dir, item)
        if not os.path.isdir(item_path):
            continue

        # 检查是否匹配剪枝率
        if f"_{pruning_ratio}pct" not in item:
            continue

        # 提取方法名
        method_name = item.replace(f"_{pruning_ratio}pct", "")

        # 过滤指定方法
        if methods and method_name not in methods:
            continue

        # 检查模型文件是否存在
        checkpoint_path = os.path.join(item_path, 'pytorch_model.bin')
        if os.path.exists(checkpoint_path):
            models[method_name] = checkpoint_path
        else:
            print(f"警告: {method_name} 模型文件不存在: {checkpoint_path}")

    return models


def evaluate_ppl(checkpoint_path: str, datasets: List[str],
                 seq_len: int, device: str) -> Dict[str, float]:
    """评估 PPL"""
    from evaluation.utils.model_loader import load_model_and_tokenizer
    from LLMPruner.evaluator.ppl import PPLMetric

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device=device)

    ppl_metric = PPLMetric(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        seq_len=seq_len,
        device=device
    )

    # 清理显存
    del model
    torch.cuda.empty_cache()

    return ppl_metric.results


def evaluate_zero_shot(checkpoint_path: str, tasks: List[str],
                       batch_size: int, device: str) -> Dict[str, Dict]:
    """评估 Zero-shot"""
    from evaluation.metrics.performance import evaluate_zeroshot

    results = evaluate_zeroshot(
        model_path=checkpoint_path,
        tasks=tasks,
        batch_size=batch_size,
        device=device
    )

    torch.cuda.empty_cache()
    return results


def evaluate_efficiency(checkpoint_path: str, device: str) -> Dict[str, Any]:
    """评估效率指标"""
    from evaluation.utils.model_loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device=device)

    # 参数量
    num_params = sum(p.numel() for p in model.parameters())

    # 显存占用
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

        # 做一次前向传播
        dummy_input = torch.randint(0, 1000, (1, 128)).to(device)
        with torch.no_grad():
            _ = model(dummy_input)

        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0

    # 推理速度
    import time
    num_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randint(0, 1000, (1, 128)).to(device)
            _ = model(dummy_input)
            num_tokens += 128

    elapsed = time.time() - start_time
    tokens_per_sec = num_tokens / elapsed

    del model
    torch.cuda.empty_cache()

    return {
        'num_params': num_params,
        'memory_mb': memory_mb,
        'tokens_per_sec': tokens_per_sec
    }


def main():
    args = parse_args()

    # 解析参数
    metrics = [m.strip() for m in args.metrics.split(',')]
    methods = [m.strip() for m in args.methods.split(',')] if args.methods else None
    ppl_datasets = [d.strip() for d in args.ppl_datasets.split(',')]

    # 查找模型
    print("=" * 60)
    print("Baseline 模型批量评估")
    print("=" * 60)

    models = find_baseline_models(args.baselines_dir, methods, args.pruning_ratio)

    if not models:
        print("\n未找到任何 baseline 模型！")
        print(f"请确保模型已保存到 {args.baselines_dir}/{{method}}_{args.pruning_ratio}pct/pytorch_model.bin")
        return

    print(f"\n找到 {len(models)} 个模型:")
    for method, path in models.items():
        print(f"  - {method}: {path}")

    print(f"\n评估指标: {', '.join(metrics)}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 存储结果
    all_results = {}

    # 评估原始模型（如果指定）
    if args.include_original and args.original_model:
        print(f"\n评估原始模型: {args.original_model}")
        all_results['original'] = evaluate_model(
            'original', args.original_model, metrics,
            ppl_datasets, args.zero_shot_tasks,
            args.batch_size, args.seq_len, args.device
        )

    # 评估每个 baseline 模型
    for method, checkpoint_path in models.items():
        print(f"\n{'='*60}")
        print(f"评估: {method}")
        print(f"{'='*60}")

        all_results[method] = evaluate_model(
            method, checkpoint_path, metrics,
            ppl_datasets, args.zero_shot_tasks,
            args.batch_size, args.seq_len, args.device
        )

    # 生成报告
    generate_report(all_results, args.output_dir, args)


def evaluate_model(method: str, checkpoint_path: str, metrics: List[str],
                   ppl_datasets: List[str], zero_shot_tasks: List[str],
                   batch_size: int, seq_len: int, device: str) -> Dict:
    """评估单个模型"""
    result = {'status': 'success'}

    try:
        # PPL 评估
        if 'ppl' in metrics:
            print("\n评估 PPL...")
            result['ppl'] = evaluate_ppl(checkpoint_path, ppl_datasets, seq_len, device)
            for key, value in result['ppl'].items():
                print(f"  {key}: {value:.2f}")

        # Zero-shot 评估
        if 'zero_shot' in metrics:
            print("\n评估 Zero-shot...")
            result['zero_shot'] = evaluate_zero_shot(
                checkpoint_path, zero_shot_tasks, batch_size, device
            )
            if result['zero_shot']:
                for task, task_result in result['zero_shot'].items():
                    if task_result and 'accuracy' in task_result:
                        acc = task_result['accuracy']
                        print(f"  {task}: {acc*100:.2f}%")

        # 效率评估
        if 'efficiency' in metrics:
            print("\n评估效率...")
            result['efficiency'] = evaluate_efficiency(checkpoint_path, device)
            eff = result['efficiency']
            print(f"  参数量: {eff['num_params']:,}")
            print(f"  显存: {eff['memory_mb']:.1f} MB")
            print(f"  速度: {eff['tokens_per_sec']:.1f} tokens/sec")

    except Exception as e:
        import traceback
        result['status'] = 'error'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"❌ 错误: {e}")

    return result


def generate_report(results: Dict, output_dir: str, args) -> None:
    """生成评估报告"""
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    # 表格输出
    methods = list(results.keys())

    # PPL 表格
    if any('ppl' in r for r in results.values() if isinstance(r, dict)):
        print("\n### PPL 结果")
        print(f"{'方法':<20} {'WikiText-2':<15}")
        print("-" * 35)

        for method in methods:
            r = results[method]
            if r.get('status') == 'success' and 'ppl' in r:
                ppl_val = list(r['ppl'].values())[0] if r['ppl'] else '-'
                if isinstance(ppl_val, (int, float)):
                    print(f"{method:<20} {ppl_val:<15.2f}")
                else:
                    print(f"{method:<20} {ppl_val:<15}")
            else:
                print(f"{method:<20} {'错误':<15}")

    # Zero-shot 表格
    if any('zero_shot' in r for r in results.values() if isinstance(r, dict)):
        print("\n### Zero-shot 准确率 (%)")

        # 获取任务列表
        tasks = []
        for r in results.values():
            if isinstance(r, dict) and 'zero_shot' in r and r['zero_shot']:
                tasks = list(r['zero_shot'].keys())
                break

        if tasks:
            header = f"{'方法':<15}" + "".join(f"{t:<12}" for t in tasks) + f"{'平均':<10}"
            print(header)
            print("-" * len(header))

            for method in methods:
                r = results[method]
                if r.get('status') == 'success' and 'zero_shot' in r and r['zero_shot']:
                    row = f"{method:<15}"
                    accs = []
                    for task in tasks:
                        if task in r['zero_shot'] and r['zero_shot'][task]:
                            acc = r['zero_shot'][task].get('accuracy', 0)
                            if acc:
                                row += f"{acc*100:<12.2f}"
                                accs.append(acc)
                            else:
                                row += f"{'-':<12}"
                        else:
                            row += f"{'-':<12}"

                    avg = sum(accs) / len(accs) * 100 if accs else 0
                    row += f"{avg:<10.2f}"
                    print(row)
                else:
                    print(f"{method:<15}" + "错误")

    # 效率表格
    if any('efficiency' in r for r in results.values() if isinstance(r, dict)):
        print("\n### 效率指标")
        print(f"{'方法':<20} {'参数量':<15} {'显存(MB)':<12} {'速度(tok/s)':<12}")
        print("-" * 59)

        for method in methods:
            r = results[method]
            if r.get('status') == 'success' and 'efficiency' in r:
                eff = r['efficiency']
                params = f"{eff['num_params']/1e9:.2f}B"
                mem = f"{eff['memory_mb']:.1f}"
                speed = f"{eff['tokens_per_sec']:.1f}"
                print(f"{method:<20} {params:<15} {mem:<12} {speed:<12}")
            else:
                print(f"{method:<20} {'错误':<15}")

    # 保存 JSON 报告
    report = {
        'config': {
            'baselines_dir': args.baselines_dir,
            'metrics': args.metrics,
            'pruning_ratio': args.pruning_ratio,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }

    report_path = os.path.join(output_dir, f'baseline_evaluation_{args.pruning_ratio}pct.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n详细报告已保存到: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
