#!/usr/bin/env python3
"""
Baseline 方法对比脚本

对比多个剪枝方法的性能，生成对比报告。

使用方法:
    python baselines/compare_methods.py \\
        --methods llm_pruner,wanda,magnitude \\
        --base_model /path/to/model \\
        --pruning_ratio 0.25 \\
        --output_dir comparison_results/
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.methods import get_pruner, AVAILABLE_METHODS
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.evaluator.ppl import PPLMetric


def parse_args():
    parser = argparse.ArgumentParser(description='对比多个 Baseline 剪枝方法')

    parser.add_argument('--methods', type=str, required=True,
                       help='要对比的方法，逗号分隔 (如: llm_pruner,wanda,magnitude)')
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='结果输出目录')

    # 校准参数
    parser.add_argument('--calibration_samples', type=int, default=128,
                       help='校准样本数量')

    # 评估参数
    parser.add_argument('--eval_datasets', type=str, default='wikitext2',
                       help='评估数据集，逗号分隔')
    parser.add_argument('--eval_seq_len', type=int, default=128,
                       help='评估序列长度')

    # 设备
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析方法列表
    methods = [m.strip() for m in args.methods.split(',')]

    # 验证方法
    for method in methods:
        if method not in AVAILABLE_METHODS:
            print(f"错误: 未知方法 '{method}'")
            print(f"可用方法: {', '.join(AVAILABLE_METHODS.keys())}")
            return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("Baseline 方法对比")
    print("=" * 60)
    print(f"方法: {', '.join(methods)}")
    print(f"剪枝率: {args.pruning_ratio*100:.1f}%")
    print(f"模型: {args.base_model}")
    print("=" * 60)

    # 加载 tokenizer（共享）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 准备校准数据
    print("\n加载校准数据...")
    calibration_data = get_examples(
        dataset_name='bookcorpus',
        tokenizer=tokenizer,
        num_samples=args.calibration_samples,
        seq_len=128
    )

    # 存储结果
    results = {}

    # 对每个方法运行剪枝
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print(f"{'='*60}")

        method_info = AVAILABLE_METHODS[method]

        if method_info['status'] == 'pending':
            print(f"⏳ 方法 '{method}' 尚未实现，跳过...")
            results[method] = {
                'status': 'not_implemented',
                'error': '方法尚未实现'
            }
            continue

        try:
            # 每次重新加载模型
            from transformers import AutoModelForCausalLM
            print(f"加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            original_params = sum(p.numel() for p in model.parameters())

            # 初始化剪枝器
            pruner = get_pruner(
                method,
                model=model,
                tokenizer=tokenizer,
                device=args.device
            )

            # 执行剪枝
            print(f"执行剪枝...")
            pruner.prune(
                pruning_ratio=args.pruning_ratio,
                calibration_data=calibration_data
            )

            # 获取统计
            stats = pruner.get_stats()
            is_valid, _ = pruner.validate_gqa_ratio()

            # 评估 PPL
            print(f"评估 PPL...")
            eval_datasets = [d.strip() for d in args.eval_datasets.split(',')]
            ppl_metric = PPLMetric(
                model=pruner.get_pruned_model(),
                tokenizer=tokenizer,
                datasets=eval_datasets,
                seq_len=args.eval_seq_len,
                device=args.device
            )

            # 记录结果
            results[method] = {
                'status': 'success',
                'original_params': original_params,
                'pruned_params': stats['pruned_params'],
                'actual_pruning_ratio': stats['pruning_ratio'],
                'gqa_valid': is_valid,
                'ppl': ppl_metric.results
            }

            print(f"✅ 完成: PPL = {list(ppl_metric.results.values())[0]:.2f}")

            # 清理显存
            del model, pruner
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            results[method] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"❌ 错误: {e}")

    # 生成报告
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    # 表格输出
    print(f"\n{'方法':<20} {'剪枝率':<10} {'PPL':<10} {'GQA':<8} {'状态':<10}")
    print("-" * 58)

    for method, result in results.items():
        if result['status'] == 'success':
            ratio = f"{result['actual_pruning_ratio']*100:.1f}%"
            ppl_val = list(result['ppl'].values())[0]
            ppl = f"{ppl_val:.2f}"
            gqa = "✅" if result['gqa_valid'] else "❌"
            status = "成功"
        elif result['status'] == 'not_implemented':
            ratio = "-"
            ppl = "-"
            gqa = "-"
            status = "未实现"
        else:
            ratio = "-"
            ppl = "-"
            gqa = "-"
            status = "错误"

        print(f"{method:<20} {ratio:<10} {ppl:<10} {gqa:<8} {status:<10}")

    # 保存 JSON 报告
    report = {
        'config': {
            'methods': methods,
            'base_model': args.base_model,
            'pruning_ratio': args.pruning_ratio,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }

    report_path = os.path.join(args.output_dir, 'comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存到: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
