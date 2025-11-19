#!/usr/bin/env python3
"""
Baseline 方法统一运行入口

使用方法:
    python baselines/run_baseline.py \\
        --method llm_pruner \\
        --base_model /path/to/model \\
        --pruning_ratio 0.25 \\
        --save_model \\
        --test_after_prune
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.methods import get_pruner, list_methods, AVAILABLE_METHODS
from core.utils.logger import LoggerWithDepth
from core.datasets.example_samples import get_examples


def parse_args():
    parser = argparse.ArgumentParser(description='运行 Baseline 剪枝方法')

    # 基本参数
    parser.add_argument('--method', type=str, required=True,
                       choices=list(AVAILABLE_METHODS.keys()),
                       help='剪枝方法名称')
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--save_ckpt_log_name', type=str, default=None,
                       help='实验名称（用于日志和检查点）')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率 (0-1)')
    parser.add_argument('--prune_mlp', action='store_true',
                       help='是否剪枝 MLP 层')
    parser.add_argument('--gqa_aware', action='store_true', default=True,
                       help='是否保持 GQA 结构')

    # 校准数据参数
    parser.add_argument('--calibration_dataset', type=str, default='bookcorpus',
                       help='校准数据集')
    parser.add_argument('--calibration_samples', type=int, default=128,
                       help='校准样本数量')
    parser.add_argument('--calibration_seq_len', type=int, default=128,
                       help='校准序列长度')

    # 评估参数
    parser.add_argument('--test_after_prune', action='store_true',
                       help='剪枝后评估 PPL')
    parser.add_argument('--eval_seq_len', type=int, default=128,
                       help='评估序列长度')

    # 保存参数
    parser.add_argument('--save_model', action='store_true',
                       help='保存剪枝后的模型')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--list_methods', action='store_true',
                       help='列出所有可用方法并退出')

    return parser.parse_args()


def main():
    args = parse_args()

    # 列出方法
    if args.list_methods:
        list_methods()
        return

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置实验名称
    if args.save_ckpt_log_name is None:
        args.save_ckpt_log_name = f"{args.method}_{int(args.pruning_ratio*100)}pct"

    # 初始化日志 (保存到 prune_log/baselines/ 目录)
    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir='prune_log/baselines',
        setup_sublogger=True
    )

    logger.log("=" * 60)
    logger.log(f"Baseline 剪枝方法: {args.method}")
    logger.log("=" * 60)

    # 加载模型
    logger.log(f"\n加载模型: {args.base_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    logger.log(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 获取校准数据
    logger.log(f"\n加载校准数据: {args.calibration_dataset}")
    calibration_data = get_examples(
        dataset_name=args.calibration_dataset,
        tokenizer=tokenizer,
        num_samples=args.calibration_samples,
        seq_len=args.calibration_seq_len
    )
    logger.log(f"校准数据形状: {calibration_data.shape}")

    # 初始化剪枝器
    logger.log(f"\n初始化 {args.method} 剪枝器...")
    try:
        pruner = get_pruner(
            args.method,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            logger=logger
        )
    except NotImplementedError as e:
        logger.log(f"\n❌ 错误: {e}")
        logger.log("\n该方法尚未实现。请先完成实现后再运行。")
        logger.log("参考骨架文件中的实现说明。")
        return

    # 执行剪枝
    logger.log(f"\n开始剪枝 (目标剪枝率: {args.pruning_ratio*100:.1f}%)...")
    pruner.prune(
        pruning_ratio=args.pruning_ratio,
        calibration_data=calibration_data,
        prune_mlp=args.prune_mlp,
        gqa_aware=args.gqa_aware
    )

    # 打印摘要
    pruner.print_summary()

    # 评估 PPL
    if args.test_after_prune:
        logger.log("\n评估剪枝后 PPL...")
        from evaluation.metrics.ppl import PPLMetric

        pruned_model = pruner.get_pruned_model()
        ppl_metric = PPLMetric(
            model=pruned_model,
            tokenizer=tokenizer,
            datasets=['wikitext2'],
            seq_len=args.eval_seq_len,
            device=args.device
        )

        for key, value in ppl_metric.results.items():
            logger.log(f"  {key}: {value:.2f}")

    # 保存模型
    if args.save_model:
        save_path = os.path.join(
            'prune_log/baselines',
            args.save_ckpt_log_name,
            'pytorch_model.bin'
        )
        logger.log(f"\n保存模型到: {save_path}")
        pruner.save_checkpoint(
            save_path,
            additional_info={
                'args': args.__dict__,
                'timestamp': datetime.now().isoformat()
            }
        )

    logger.log("\n" + "=" * 60)
    logger.log("完成!")
    logger.log("=" * 60)


if __name__ == '__main__':
    main()
