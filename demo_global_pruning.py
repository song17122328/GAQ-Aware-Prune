#!/usr/bin/env python3
"""
全局剪枝分析表生成演示脚本

展示如何使用新的全局剪枝策略：
1. 计算每个 group 的 Taylor importance
2. 计算每个 group 的参数成本
3. 计算 Score = Importance / Cost
4. 生成全局分析表，按 Score 排序
5. 根据目标剪枝率选择要剪枝的 groups
"""

import torch
import argparse
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.methods.global_pruning import (
    build_global_group_table,
    select_groups_to_prune,
    save_group_table
)
from core.datasets.example_samples import get_examples
from core.utils.logger import LoggerWithDepth


def main():
    parser = argparse.ArgumentParser(description='全局剪枝分析表生成演示')
    parser.add_argument('--base_model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--save_table_path', type=str, default='prune_log/global_group_table.json',
                       help='分析表保存路径')
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率（相对于模型总参数）')
    parser.add_argument('--importance_method', type=str, default='taylor',
                       choices=['taylor', 'wanda'],
                       help='重要性计算方法')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='用于计算梯度的样本数')
    parser.add_argument('--layer_start', type=int, default=0,
                       help='起始层（debug用）')
    parser.add_argument('--layer_end', type=int, default=None,
                       help='结束层（debug用）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')

    args = parser.parse_args()

    # 设置 logger
    logger = LoggerWithDepth(
        env_name='global_pruning_demo',
        config=args.__dict__,
        root_dir='prune_log'
    )

    logger.log("="*60)
    logger.log("全局剪枝分析表生成演示")
    logger.log("="*60)
    logger.log(f"模型: {args.base_model}")
    logger.log(f"重要性方法: {args.importance_method}")
    logger.log(f"目标剪枝率: {args.pruning_ratio:.1%}")

    # ========== Step 1: 加载模型 ==========
    logger.log("\n[Step 1] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 获取实际使用的设备
    if hasattr(model, 'hf_device_map'):
        logger.log(f"  模型分布: {model.hf_device_map}")
        args.device = 'cuda'
    else:
        args.device = next(model.parameters()).device

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"✓ 模型加载完成")
    logger.log(f"  总参数量: {total_params:,}")

    # ========== Step 2: 计算梯度（Taylor方法需要）==========
    if args.importance_method == 'taylor':
        logger.log("\n[Step 2] 计算梯度（Taylor importance）...")
        logger.log(f"  加载 {args.num_samples} 个样本...")

        # 分批计算梯度以节省内存
        batch_size = 4
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        logger.log(f"  批次大小: {batch_size}, 总批次数: {num_batches}")

        model.zero_grad()
        total_loss = 0.0
        start_time = time.time()

        # 使用 tqdm 显示进度条
        pbar = tqdm(range(num_batches), desc="计算梯度", ncols=100)

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, args.num_samples)
            current_batch_size = end_idx - start_idx

            batch_start_time = time.time()

            # 加载当前批次
            input_ids = get_examples('wikitext', tokenizer, num_samples=current_batch_size, seq_len=128)
            input_ids = input_ids.to(args.device)

            # 前向+反向传播
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss / num_batches
            loss.backward()

            batch_time = time.time() - batch_start_time
            total_loss += loss.item() * num_batches

            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item() * num_batches:.4f}',
                'batch_time': f'{batch_time:.2f}s'
            })

            # 清理内存
            del input_ids, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        total_time = time.time() - start_time
        logger.log(f"✓ 梯度计算完成")
        logger.log(f"  平均 loss: {total_loss:.4f}")
        logger.log(f"  总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.log(f"  平均每批次: {total_time/num_batches:.2f}s")

        activations = None

    elif args.importance_method == 'wanda':
        logger.log("\n[Step 2] 收集激活值（Wanda importance）...")
        logger.log("⚠️ Wanda 方法需要实现 activation hooks，当前演示脚本暂不支持")
        logger.log("   请使用 --importance_method taylor")
        return

    # ========== Step 3: 构建全局分析表 ==========
    logger.log("\n[Step 3] 构建全局 Group 分析表...")

    df = build_global_group_table(
        model=model,
        importance_method=args.importance_method,
        activations=activations,
        layer_start=args.layer_start,
        layer_end=args.layer_end if args.layer_end else len(model.model.layers),
        head_dim=128,
        gqa_ratio=4,
        device=args.device
    )

    logger.log(f"\n✓ 分析表构建完成")
    logger.log(f"  总 Groups: {len(df)}")
    logger.log(f"  Attention Groups: {len(df[df['group_type']=='attention'])}")
    logger.log(f"  MLP Groups: {len(df[df['group_type']=='mlp'])}")

    # 展示 Score 最低的前10个 groups
    logger.log("\nScore 最低的前 10 个 groups (最优先剪枝):")
    logger.log("-" * 80)
    logger.log(f"{'Rank':<6} {'Layer':<8} {'Type':<12} {'Group':<8} {'Importance':<15} {'Cost':<10} {'Score':<12}")
    logger.log("-" * 80)

    for idx, row in df.head(10).iterrows():
        logger.log(
            f"{idx+1:<6} {row['layer_idx']:<8} {row['group_type']:<12} "
            f"{row['group_idx']:<8} {row['importance']:<15.6e} {row['cost']:<10} {row['score']:<12.6e}"
        )

    # 展示 Score 最高的前10个 groups
    logger.log("\nScore 最高的前 10 个 groups (最不应该剪枝):")
    logger.log("-" * 80)
    logger.log(f"{'Rank':<6} {'Layer':<8} {'Type':<12} {'Group':<8} {'Importance':<15} {'Cost':<10} {'Score':<12}")
    logger.log("-" * 80)

    for idx, row in df.tail(10).iterrows():
        logger.log(
            f"{idx+1:<6} {row['layer_idx']:<8} {row['group_type']:<12} "
            f"{row['group_idx']:<8} {row['importance']:<15.6e} {row['cost']:<10} {row['score']:<12.6e}"
        )

    # ========== Step 4: 选择要剪枝的 groups ==========
    logger.log(f"\n[Step 4] 根据剪枝率 {args.pruning_ratio:.1%} 选择要剪枝的 groups...")

    groups_to_prune = select_groups_to_prune(
        df=df,
        pruning_ratio=args.pruning_ratio,
        total_params=total_params
    )

    logger.log(f"\n✓ 选中 {len(groups_to_prune)} 个 groups 进行剪枝")

    # 统计各层的剪枝情况
    logger.log("\n各层剪枝统计:")
    logger.log("-" * 60)
    logger.log(f"{'Layer':<8} {'Attention剪枝':<15} {'MLP剪枝':<15}")
    logger.log("-" * 60)

    layer_start = args.layer_start
    layer_end = args.layer_end if args.layer_end else len(model.model.layers)

    for layer_idx in range(layer_start, layer_end):
        layer_attn = groups_to_prune[
            (groups_to_prune['layer_idx'] == layer_idx) &
            (groups_to_prune['group_type'] == 'attention')
        ]
        layer_mlp = groups_to_prune[
            (groups_to_prune['layer_idx'] == layer_idx) &
            (groups_to_prune['group_type'] == 'mlp')
        ]

        if len(layer_attn) > 0 or len(layer_mlp) > 0:
            logger.log(f"{layer_idx:<8} {len(layer_attn):<15} {len(layer_mlp):<15}")

    # ========== Step 5: 保存分析表 ==========
    logger.log(f"\n[Step 5] 保存分析表...")

    # 保存完整表
    save_group_table(df, args.save_table_path)

    # 保存要剪枝的 groups
    prune_table_path = args.save_table_path.replace('.json', '_to_prune.json')
    save_group_table(groups_to_prune, prune_table_path)

    logger.log(f"\n✓ 全部完成！")
    logger.log(f"\n生成的文件:")
    logger.log(f"  1. 完整分析表: {args.save_table_path}")
    logger.log(f"  2. 要剪枝的groups: {prune_table_path}")
    logger.log(f"\n后续步骤:")
    logger.log(f"  - 可以使用这些表格指导实际剪枝")
    logger.log(f"  - 按照 groups_to_prune 中的信息剪枝相应的 Attention 和 MLP groups")


if __name__ == '__main__':
    main()
