#!/usr/bin/env python3
"""
基于全局性价比排序的混合结构化剪枝

核心思想：
- 将剪枝问题建模为分数背包问题
- Score = Importance / Cost
- 全局排序，优先剪除"性价比"最低的 groups
- 自动实现深度剪枝（层移除）+ 宽度剪枝（神经元剪除）的混合策略
"""

import os
import torch
import argparse
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.methods.global_pruning import (
    build_global_group_table,
    select_groups_to_prune
)
from core.methods.gqa_aware import prune_attention_by_gqa_groups
from core.datasets.example_samples import get_examples
from evaluation.metrics.ppl import PPLMetric
from core.trainer.finetuner import FineTuner
from core.utils.logger import LoggerWithDepth


def apply_global_pruning(model, groups_to_prune_df, head_dim=128, gqa_ratio=4, logger=None):
    """
    根据全局分析表执行实际剪枝

    Args:
        model: 模型
        groups_to_prune_df: 要剪枝的 groups DataFrame
        head_dim: attention head 维度
        gqa_ratio: Q:KV 比例
        logger: 日志记录器

    Returns:
        pruned_layers: 被完全剪空的层列表
        pruning_stats: 剪枝统计信息
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    log("\n" + "="*60)
    log("执行全局剪枝")
    log("="*60)

    num_layers = len(model.model.layers)
    pruning_stats = {
        'attention': {},  # {layer_idx: (old_kv, new_kv)}
        'mlp': {},        # {layer_idx: (old_channels, new_channels)}
        'empty_layers': []
    }

    # 按层组织要剪枝的 groups
    layer_prune_info = {}
    for layer_idx in range(num_layers):
        layer_data = groups_to_prune_df[groups_to_prune_df['layer_idx'] == layer_idx]

        attn_groups = layer_data[layer_data['group_type'] == 'attention']['group_idx'].tolist()
        mlp_groups = layer_data[layer_data['group_type'] == 'mlp']['group_idx'].tolist()

        layer_prune_info[layer_idx] = {
            'attention': attn_groups,
            'mlp': mlp_groups
        }

    # 执行剪枝
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        prune_info = layer_prune_info[layer_idx]

        log(f"\n处理 Layer {layer_idx}:")

        # ========== Attention 剪枝 ==========
        attn_prune_indices = prune_info['attention']

        if len(attn_prune_indices) > 0:
            # 获取当前 KV heads 数量
            num_kv_heads = layer.self_attn.num_key_value_heads

            # 计算保留的 indices
            all_kv_indices = set(range(num_kv_heads))
            keep_kv_indices = sorted(list(all_kv_indices - set(attn_prune_indices)))

            old_q = layer.self_attn.num_heads
            old_kv = layer.self_attn.num_key_value_heads

            if len(keep_kv_indices) > 0:
                # 执行剪枝
                new_q, new_kv = prune_attention_by_gqa_groups(
                    layer,
                    keep_kv_indices,
                    head_dim=head_dim,
                    gqa_ratio=gqa_ratio
                )
                log(f"  Attention: {old_q}Q:{old_kv}KV → {new_q}Q:{new_kv}KV")
                pruning_stats['attention'][layer_idx] = (old_kv, new_kv)
            else:
                # 该层 Attention 被完全剪空
                log(f"  ⚠️ Attention 被完全剪空（{old_kv} → 0 KV heads）")
                pruning_stats['attention'][layer_idx] = (old_kv, 0)

        # ========== MLP 剪枝 ==========
        mlp_prune_indices = prune_info['mlp']

        if len(mlp_prune_indices) > 0:
            intermediate_size = layer.mlp.gate_proj.out_features

            # 计算保留的 indices
            all_mlp_indices = set(range(intermediate_size))
            keep_mlp_indices = sorted(list(all_mlp_indices - set(mlp_prune_indices)))

            if len(keep_mlp_indices) > 0:
                # 执行 MLP 剪枝
                keep_mlp_indices_tensor = torch.tensor(keep_mlp_indices, device=layer.mlp.gate_proj.weight.device)

                # 剪枝 gate_proj 和 up_proj（保留对应的行）
                layer.mlp.gate_proj.weight = torch.nn.Parameter(
                    layer.mlp.gate_proj.weight[keep_mlp_indices_tensor, :]
                )
                layer.mlp.up_proj.weight = torch.nn.Parameter(
                    layer.mlp.up_proj.weight[keep_mlp_indices_tensor, :]
                )

                # 剪枝 down_proj（保留对应的列）
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    layer.mlp.down_proj.weight[:, keep_mlp_indices_tensor]
                )

                # 更新 intermediate_size
                new_intermediate_size = len(keep_mlp_indices)
                layer.mlp.gate_proj.out_features = new_intermediate_size
                layer.mlp.up_proj.out_features = new_intermediate_size
                layer.mlp.down_proj.in_features = new_intermediate_size

                log(f"  MLP: {intermediate_size} → {new_intermediate_size} channels")
                pruning_stats['mlp'][layer_idx] = (intermediate_size, new_intermediate_size)
            else:
                # 该层 MLP 被完全剪空
                log(f"  ⚠️ MLP 被完全剪空（{intermediate_size} → 0 channels）")
                pruning_stats['mlp'][layer_idx] = (intermediate_size, 0)

        # 检查是否整层被剪空
        attn_empty = (layer_idx in pruning_stats['attention'] and
                     pruning_stats['attention'][layer_idx][1] == 0)
        mlp_empty = (layer_idx in pruning_stats['mlp'] and
                    pruning_stats['mlp'][layer_idx][1] == 0)

        if attn_empty and mlp_empty:
            log(f"  🔴 Layer {layer_idx} 被完全剪空（自动深度剪枝）")
            pruning_stats['empty_layers'].append(layer_idx)

    return pruning_stats


def remove_empty_layers(model, empty_layers, logger=None):
    """
    移除被完全剪空的层

    Args:
        model: 模型
        empty_layers: 要移除的层索引列表
        logger: 日志记录器
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    if len(empty_layers) == 0:
        log("\n✓ 没有层被完全剪空，跳过层移除")
        return

    log(f"\n{'='*60}")
    log(f"移除完全剪空的层")
    log(f"{'='*60}")
    log(f"要移除的层: {empty_layers}")

    # 创建保留的层列表
    num_layers = len(model.model.layers)
    keep_layers = [i for i in range(num_layers) if i not in empty_layers]

    # 重建 layers 列表
    new_layers = torch.nn.ModuleList([model.model.layers[i] for i in keep_layers])
    model.model.layers = new_layers

    # 更新配置
    model.config.num_hidden_layers = len(keep_layers)

    log(f"✓ 层数: {num_layers} → {len(keep_layers)}")


def main():
    parser = argparse.ArgumentParser(description='基于全局性价比的混合结构化剪枝')

    # 模型参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--save_ckpt_log_name', type=str, default='llama_global_prune',
                       help='实验名称')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率（相对于模型总参数）')
    parser.add_argument('--importance_method', type=str, default='taylor',
                       choices=['taylor', 'wanda'],
                       help='重要性计算方法')
    parser.add_argument('--num_samples', type=int, default=128,
                       help='用于计算重要性的样本数')
    parser.add_argument('--gradient_batch_size', type=int, default=4,
                       help='梯度计算时的批次大小（用于节省内存）')
    parser.add_argument('--remove_empty_layers', action='store_true',
                       help='是否移除被完全剪空的层（自动深度剪枝）')

    # GQA 配置
    parser.add_argument('--head_dim', type=int, default=128,
                       help='Attention head 维度')
    parser.add_argument('--gqa_ratio', type=int, default=4,
                       help='Q:KV 比例')

    # 评估参数
    parser.add_argument('--test_before_prune', action='store_true',
                       help='剪枝前评估基线 PPL')
    parser.add_argument('--test_after_prune', action='store_true',
                       help='剪枝后评估 PPL')

    # 微调参数
    parser.add_argument('--finetune', action='store_true',
                       help='剪枝后进行微调')
    parser.add_argument('--finetune_method', type=str, default='lora',
                       choices=['full', 'lora'],
                       help='微调方法')
    parser.add_argument('--finetune_samples', type=int, default=500,
                       help='微调样本数')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                       help='微调学习率')
    parser.add_argument('--finetune_epochs', type=int, default=1,
                       help='微调轮数')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')

    # 保存参数
    parser.add_argument('--save_model', action='store_true',
                       help='保存剪枝后的模型')

    # 其他
    from core.utils.get_best_gpu import get_best_gpu
    # bestDevice= "cuda:"+str(get_best_gpu())
    bestDevice = "cpu"
    parser.add_argument('--device', type=str, default=bestDevice,
                       help='设备')
    parser.add_argument('--layer_start', type=int, default=0,
                       help='起始层（debug用）')
    parser.add_argument('--layer_end', type=int, default=None,
                       help='结束层（debug用）')

    args = parser.parse_args()

    # 设置 logger
    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir='prune_log'
    )

    logger.log("="*60)
    logger.log("基于全局性价比的混合结构化剪枝")
    logger.log("="*60)
    logger.log(f"模型: {args.base_model}")
    logger.log(f"剪枝率: {args.pruning_ratio:.1%}")
    logger.log(f"重要性方法: {args.importance_method}")

    # ========== Step 1: 加载模型 ==========
    logger.log("\n[Step 1] 加载模型...")

    # 使用 device_map='auto' 让 transformers 自动管理内存
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map = args.device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 获取实际使用的设备
    if hasattr(model, 'hf_device_map'):
        logger.log(f"  模型分布: {model.hf_device_map}")
        # 获取第一个模块的设备（输入数据应该发送到这里）
        first_device = next(iter(model.hf_device_map.values()))
        args.device = f'cuda:{first_device}' if isinstance(first_device, int) else first_device
        logger.log(f"  输入设备: {args.device}")
    else:
        args.device = next(model.parameters()).device

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"✓ 模型加载完成")
    logger.log(f"  总参数量: {total_params:,}")

    # ========== Step 2: 评估基线 ==========
    if args.test_before_prune:
        logger.log("\n[Step 2] 评估基线 PPL...")
        baseline_ppl = PPLMetric(model, tokenizer, datasets=['wikitext2'], device=args.device)
        logger.log(f"✓ 基线 PPL: {baseline_ppl}")

    # ========== Step 3: 计算梯度 ==========
    if args.importance_method == 'taylor':
        logger.log(f"\n[Step 3] 计算梯度（Taylor importance）...")
        logger.log(f"  加载 {args.num_samples} 个样本...")

        # 分批计算梯度以节省内存
        batch_size = args.gradient_batch_size
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
            loss = outputs.loss / num_batches  # 归一化
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
    else:
        logger.log("\n[Step 3] Wanda 方法暂不支持，请使用 --importance_method taylor")
        return

    # ========== Step 4: 构建全局分析表 ==========
    logger.log("\n[Step 4] 构建全局 Group 分析表...")

    layer_end = args.layer_end if args.layer_end else len(model.model.layers)

    df = build_global_group_table(
        model=model,
        importance_method=args.importance_method,
        activations=activations,
        layer_start=args.layer_start,
        layer_end=layer_end,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
        device=args.device
    )

    logger.log(f"✓ 分析表构建完成")

    # ========== Step 5: 选择要剪枝的 groups ==========
    logger.log(f"\n[Step 5] 根据剪枝率选择要剪枝的 groups...")

    groups_to_prune = select_groups_to_prune(
        df=df,
        pruning_ratio=args.pruning_ratio,
        total_params=total_params
    )

    logger.log(f"✓ 选中 {len(groups_to_prune)} 个 groups 进行剪枝")

    # 保存分析表
    table_path = os.path.join(logger.env_dir, 'global_group_table.csv')
    df.to_csv(table_path, index=False)
    logger.log(f"✓ 分析表已保存: {table_path}")

    prune_table_path = os.path.join(logger.env_dir, 'groups_to_prune.csv')
    groups_to_prune.to_csv(prune_table_path, index=False)
    logger.log(f"✓ 剪枝列表已保存: {prune_table_path}")

    # ========== Step 6: 执行全局剪枝 ==========
    logger.log(f"\n[Step 6] 执行全局剪枝...")

    pruning_stats = apply_global_pruning(
        model=model,
        groups_to_prune_df=groups_to_prune,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
        logger=logger
    )

    logger.log("\n✓ 全局剪枝完成")

    # ========== Step 7: 移除空层（可选）==========
    if args.remove_empty_layers and len(pruning_stats['empty_layers']) > 0:
        logger.log(f"\n[Step 7] 移除空层...")
        remove_empty_layers(model, pruning_stats['empty_layers'], logger)

    # ========== Step 8: 统计剪枝结果 ==========
    logger.log(f"\n{'='*60}")
    logger.log(f"剪枝统计")
    logger.log(f"{'='*60}")

    after_params = sum(p.numel() for p in model.parameters())
    actual_ratio = (total_params - after_params) / total_params

    logger.log(f"参数统计:")
    logger.log(f"  剪枝前: {total_params:,}")
    logger.log(f"  剪枝后: {after_params:,}")
    logger.log(f"  实际剪枝率: {actual_ratio:.2%}")

    if len(pruning_stats['empty_layers']) > 0:
        logger.log(f"\n自动深度剪枝:")
        logger.log(f"  移除的层: {pruning_stats['empty_layers']}")
        logger.log(f"  剩余层数: {len(model.model.layers)}")

    # ========== Step 9: 评估剪枝后 PPL ==========
    if args.test_after_prune:
        logger.log(f"\n[Step 9] 评估剪枝后 PPL...")
        pruned_ppl = PPLMetric(model, tokenizer, datasets=['wikitext2'], device=args.device)
        logger.log(f"✓ 剪枝后 PPL: {pruned_ppl}")

        if args.test_before_prune:
            degradation = (pruned_ppl['wikitext2 (wikitext-2-raw-v1)'] /
                          baseline_ppl['wikitext2 (wikitext-2-raw-v1)'] - 1) * 100
            logger.log(f"  PPL 退化: {degradation:.2f}%")

    # ========== Step 10: 微调恢复（可选）==========
    if args.finetune:
        logger.log(f"\n[Step 10] 微调恢复...")

        finetuner = FineTuner(model, tokenizer, device=args.device, logger=logger)

        finetuner.finetune(
            dataset_name='wikitext',
            num_samples=args.finetune_samples,
            lr=args.finetune_lr,
            epochs=args.finetune_epochs,
            method=args.finetune_method,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )

        logger.log(f"✓ 微调完成")

        # 评估微调后 PPL
        if args.test_after_prune:
            logger.log(f"\n评估微调后 PPL...")
            finetuned_ppl = PPLMetric(model, tokenizer, datasets=['wikitext2'], device=args.device)
            logger.log(f"✓ 微调后 PPL: {finetuned_ppl}")

            if args.test_before_prune:
                final_degradation = (finetuned_ppl['wikitext2 (wikitext-2-raw-v1)'] /
                                    baseline_ppl['wikitext2 (wikitext-2-raw-v1)'] - 1) * 100
                logger.log(f"  最终 PPL 退化: {final_degradation:.2f}%")

    # ========== Step 11: 保存模型 ==========
    if args.save_model:
        logger.log(f"\n[Step 11] 保存模型...")

        save_path = os.path.join(logger.env_dir, 'pytorch_model.bin')

        save_dict = {
            'model': model,
            'tokenizer': tokenizer,
            'pruning_stats': pruning_stats,
            'pruning_ratio': args.pruning_ratio,
            'actual_ratio': actual_ratio,
            'method': 'global_pruning',
            'config': args.__dict__
        }

        torch.save(save_dict, save_path)
        logger.log(f"✓ 模型已保存: {save_path}")

    logger.log(f"\n{'='*60}")
    logger.log(f"✓ 全部完成！")
    logger.log(f"{'='*60}")


if __name__ == '__main__':
    main()
