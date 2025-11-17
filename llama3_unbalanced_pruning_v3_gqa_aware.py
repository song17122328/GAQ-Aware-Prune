#!/usr/bin/env python3
"""
Llama-3 非均衡结构化剪枝脚本 (v3 - GQA-Aware版本)

核心改进：
1. 保留层重要性评估和per-layer剪枝率计算
2. Attention使用GQA-aware Taylor importance剪枝
3. MLP也使用Taylor importance剪枝（综合gate/up/down三个投影）
4. 不依赖torch_pruning，完全手动控制剪枝过程
5. 确保4:1 GQA比例自然保持，基于importance选择GQA组

与v2的主要区别：
- v2: torch_pruning + 后处理简单截断 → PPL 71万
- v3: GQA-aware组级剪枝 + MLP Taylor importance → PPL显著改善
"""

import os
import gc
import sys
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM

from LLMPruner import (
    # 剪枝方法
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups,
    # 重要性分析
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator,
    # 评估和数据
    PPLMetric,
    get_examples,
    get_examples_from_text,
    # 训练
    FineTuner,
    # 工具
    LoggerWithDepth,
)


def main():
    parser = argparse.ArgumentParser(description='Llama-3 GQA-Aware非均衡结构化剪枝')

    # 模型参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='原始模型路径')
    parser.add_argument('--save_ckpt_log_name', type=str, default='llama_gqa_aware_prune',
                       help='日志和模型保存目录名称')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率（整体平均）')

    # 层重要度评估
    parser.add_argument('--importance_method', type=str, default='removal',
                       choices=['removal', 'activation'],
                       help='层重要度评估方法：removal(移除层) 或 activation(激活值)')
    parser.add_argument('--importance_samples', type=int, default=50,
                       help='用于评估层重要度的样本数量')
    parser.add_argument('--skip_importance_analysis', action='store_true',
                       help='跳过层重要度分析，使用已保存的配置')
    parser.add_argument('--importance_config', type=str, default='layer_importance_config.json',
                       help='层重要度配置文件路径')

    # 非均衡剪枝策略
    parser.add_argument('--pruning_strategy', type=str, default='inverse',
                       choices=['inverse', 'proportional', 'uniform'],
                       help='剪枝策略：inverse(重要层剪少), proportional(重要层剪多), uniform(均匀)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='重要性权重系数，越大差异越明显')
    parser.add_argument('--min_pruning_rate', type=float, default=0.15,
                       help='最小剪枝率（至少剪1个GQA组）')
    parser.add_argument('--max_pruning_rate', type=float, default=0.5,
                       help='最大剪枝率')

    # 剪枝范围
    parser.add_argument('--layer_start', type=int, default=0,
                       help='剪枝起始层')
    parser.add_argument('--layer_end', type=int, default=32,
                       help='剪枝结束层')

    # 其他参数
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Taylor重要性评估的样本数')
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--test_after_prune', action='store_true',
                       help='剪枝后是否评估PPL')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='PPL评估最大序列长度')

    # GQA配置
    parser.add_argument('--head_dim', type=int, default=128,
                       help='每个attention head的维度')
    parser.add_argument('--gqa_ratio', type=int, default=4,
                       help='Q:KV比例（Llama-3默认4:1）')

    # MLP剪枝
    parser.add_argument('--prune_mlp', action='store_true',
                       help='是否也剪枝MLP（默认只剪Attention）')

    # 微调参数
    parser.add_argument('--finetune', action='store_true',
                       help='剪枝后是否进行微调')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                       help='微调学习率')
    parser.add_argument('--finetune_epochs', type=int, default=1,
                       help='微调轮数')
    parser.add_argument('--finetune_samples', type=int, default=500,
                       help='微调使用的样本数量')
    parser.add_argument('--finetune_batch_size', type=int, default=1,
                       help='微调batch size')
    parser.add_argument('--finetune_seq_len', type=int, default=512,
                       help='微调序列长度')

    args = parser.parse_args()

    # 自动选择最优 GPU
    try:
        from LLMPruner.utils.get_best_gpu import get_best_gpu
        device = f"cuda:{get_best_gpu()}"
    except:
        device = "cuda:0"

    print(f"自动选择设备: {device}")
    args.device = device

    # 创建日志
    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # ==================== 步骤1: 加载模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤1: 加载模型")
    logger.log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 确保 tokenizer 有 pad_token (Llama 等模型默认没有)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.log("设置 pad_token = eos_token")

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    model.half()

    # 启用梯度
    for param in model.parameters():
        param.requires_grad_(True)

    num_layers = len(model.model.layers)
    logger.log(f"模型总层数: {num_layers}")

    # 统计剪枝前参数量（所有参数）
    before_pruning_parameters = sum(p.numel() for p in model.parameters())
    logger.log(f"剪枝前参数量: {before_pruning_parameters:,}")

    # ==================== 统一数据加载 ====================
    logger.log("\n" + "=" * 60)
    logger.log("加载评估数据")
    logger.log("=" * 60)

    # 计算所需的最大样本数
    max_samples = max(args.importance_samples, args.num_examples) if not args.skip_importance_analysis else args.num_examples

    # 一次性加载所有需要的样本（从 wikitext2）
    logger.log(f"从 wikitext2 加载 {max_samples} 个样本...")
    all_samples = get_examples('wikitext', tokenizer, num_samples=max_samples, seq_len=512, split='test')
    logger.log(f"✅ 加载完成，shape: {all_samples.shape}")

    # ==================== 步骤2: 评估层重要性 ====================
    if not args.skip_importance_analysis:
        logger.log("\n" + "=" * 60)
        logger.log("步骤2: 评估层重要性")
        logger.log("=" * 60)

        # 将 tokenized 样本转换回文本（用于层重要性评估）
        eval_samples = all_samples[:args.importance_samples]
        eval_texts = [tokenizer.decode(sample, skip_special_tokens=True) for sample in eval_samples]
        logger.log(f"准备了 {len(eval_texts)} 个样本用于层重要性评估")

        analyzer = LayerImportanceAnalyzer(model, tokenizer, device=args.device)

        if args.importance_method == 'removal':
            logger.log("使用层移除法评估重要性...")
            layer_importance = analyzer.measure_layer_importance_by_removal(
                eval_texts, num_layers=num_layers
            )
        else:
            logger.log("使用激活值法评估重要性...")
            layer_importance = analyzer.measure_layer_importance_by_activation(eval_texts)

        # 只打印统计信息和极值层
        importance_values = list(layer_importance.values())
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)

        logger.log(f"\n层重要性统计:")
        logger.log(f"  平均: {np.mean(importance_values):.6f}")
        logger.log(f"  标准差: {np.std(importance_values):.6f}")
        logger.log(f"  最大: {max(importance_values):.6f}")
        logger.log(f"  最小: {min(importance_values):.6f}")

        logger.log(f"\n最重要的5层:")
        for layer_idx, importance in sorted_layers[:5]:
            logger.log(f"  Layer {layer_idx}: {importance:.6f}")

        logger.log(f"最不重要的5层:")
        for layer_idx, importance in sorted_layers[-5:]:
            logger.log(f"  Layer {layer_idx}: {importance:.6f}")

    else:
        logger.log("跳过层重要度分析，加载已保存的配置...")
        calculator = UnbalancedStructuredPruningCalculator({}, num_layers)
        layer_pruning_rates = calculator.load_pruning_rates(args.importance_config)
        layer_importance = {i: 1.0 for i in range(num_layers)}

    # ==================== 步骤3: 计算各层剪枝率 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤3: 计算各层剪枝率")
    logger.log("=" * 60)

    calculator = UnbalancedStructuredPruningCalculator(layer_importance, num_layers)

    layer_pruning_rates = calculator.compute_layer_pruning_rates(
        target_overall_rate=args.pruning_ratio,
        strategy=args.pruning_strategy,
        alpha=args.alpha,
        min_rate=args.min_pruning_rate,
        max_rate=args.max_pruning_rate,
        use_log_transform=True
    )

    stats = calculator.verify_average_pruning_rate(layer_pruning_rates)
    logger.log(f"\n剪枝率统计:")
    logger.log(f"  平均剪枝率: {stats['average_pruning_rate']:.4f}")
    logger.log(f"  标准差: {stats['std_pruning_rate']:.4f}")
    logger.log(f"  最小剪枝率: {stats['min_pruning_rate']:.4f}")
    logger.log(f"  最大剪枝率: {stats['max_pruning_rate']:.4f}")

    # 不再打印所有32层的详细剪枝率，仅保存到JSON配置文件中

    # 保存配置
    config_path = os.path.join(logger.log_dir, args.importance_config)
    calculator.save_pruning_rates(layer_pruning_rates, config_path)

    # 可视化
    viz_path = os.path.join(logger.log_dir, 'pruning_strategy.png')
    calculator.visualize_pruning_strategy(layer_pruning_rates, save_path=viz_path)

    # ==================== 步骤4: GQA-Aware剪枝 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤4: GQA-Aware结构化剪枝")
    logger.log("=" * 60)
    logger.log("使用GQA组级Taylor Importance，保持4:1 Q:KV比例\n")

    # 准备样本数据用于计算梯度（复用已加载的数据，截断到64 token）
    example_prompts = all_samples[:args.num_examples, :64].to(args.device)
    logger.log(f"准备了 {args.num_examples} 个样本用于Taylor importance计算")

    # 确定要剪枝的层
    pruning_layers = [i for i in range(args.layer_start, min(args.layer_end, num_layers))
                     if layer_pruning_rates.get(i, 0.0) >= args.min_pruning_rate]

    logger.log(f"\n实际参与剪枝的层: {pruning_layers}")
    logger.log(f"跳过的层（剪枝率<{args.min_pruning_rate}）: {[i for i in range(num_layers) if i not in pruning_layers]}\n")

    # 记录已剪枝的层（用于禁用梯度计算）
    pruned_layer_indices = []

    # 逐层剪枝
    for layer_idx in pruning_layers:
        rate = layer_pruning_rates[layer_idx]
        logger.log(f"\n处理 Layer {layer_idx} (剪枝率: {rate:.2%})")

        layer = model.model.layers[layer_idx]

        # 禁用已剪枝层的梯度计算（避免形状不匹配）
        for pruned_idx in pruned_layer_indices:
            for param in model.model.layers[pruned_idx].parameters():
                param.requires_grad = False

        # 计算梯度
        model.zero_grad()
        loss = model(example_prompts, labels=example_prompts).loss
        loss.backward()

        # 计算importance
        group_imp = compute_gqa_group_importance(layer, args.head_dim, args.gqa_ratio)

        if args.prune_mlp:
            gate_salience = (layer.mlp.gate_proj.weight * layer.mlp.gate_proj.weight.grad).abs().sum(1)
            up_salience = (layer.mlp.up_proj.weight * layer.mlp.up_proj.weight.grad).abs().sum(1)
            down_salience = (layer.mlp.down_proj.weight * layer.mlp.down_proj.weight.grad).abs().sum(0)
            mlp_importance = gate_salience + up_salience + down_salience

        # Attention剪枝
        num_kv_heads = len(group_imp)
        num_groups_to_prune = int(num_kv_heads * rate)
        target_num_kv_heads = max(1, num_kv_heads - num_groups_to_prune)

        keep_indices, _ = select_gqa_groups_to_prune(group_imp, target_num_kv_heads)
        num_q, num_kv = prune_attention_by_gqa_groups(layer, keep_indices, args.head_dim, args.gqa_ratio)

        # MLP剪枝
        if args.prune_mlp:
            num_channels = mlp_importance.shape[0]
            num_channels_to_prune = int(num_channels * rate)
            num_channels_to_prune = (num_channels_to_prune // args.head_dim) * args.head_dim
            target_channels = max(args.head_dim, num_channels - num_channels_to_prune)

            _, sorted_indices = torch.sort(mlp_importance, descending=True)
            keep_indices_mlp = sorted(sorted_indices[:target_channels].tolist())

            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[keep_indices_mlp, :]
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[keep_indices_mlp, :]

            if layer.mlp.gate_proj.bias is not None:
                layer.mlp.gate_proj.bias.data = layer.mlp.gate_proj.bias.data[keep_indices_mlp]
            if layer.mlp.up_proj.bias is not None:
                layer.mlp.up_proj.bias.data = layer.mlp.up_proj.bias.data[keep_indices_mlp]

            layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, keep_indices_mlp]

            layer.mlp.gate_proj.out_features = target_channels
            layer.mlp.up_proj.out_features = target_channels
            layer.mlp.down_proj.in_features = target_channels

            # 输出完整日志（Attention + MLP）
            logger.log(f"  Attention: {32}Q:{8}KV → {num_q}Q:{num_kv}KV, MLP: {num_channels}→{target_channels}")
        else:
            # 仅输出 Attention
            logger.log(f"  Attention: {32}Q:{8}KV → {num_q}Q:{num_kv}KV")

        # 清理
        del loss
        model.zero_grad()
        for param in layer.parameters():
            if param.grad is not None:
                param.grad = None
        torch.cuda.empty_cache()

        pruned_layer_indices.append(layer_idx)

    # ==================== 步骤5: 保存模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤5: 保存剪枝后的模型")
    logger.log("=" * 60)

    if args.save_model:
        model.half()
        save_dict = {
            'model': model,
            'tokenizer': tokenizer,
            'layer_pruning_rates': layer_pruning_rates,
            'layer_importance': layer_importance,
            'pruning_method': 'gqa_aware_taylor',
            'config': args.__dict__
        }

        torch.save(save_dict, logger.best_checkpoint_path)
        logger.log(f"✅ 模型已保存到: {logger.best_checkpoint_path}")
    else:
        logger.log("⚠️ 未启用 --save_model，跳过模型保存")

    # ==================== 步骤6: 重新加载模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤6: 重新加载模型")
    logger.log("=" * 60)

    if args.save_model:
        # 删除原模型，释放内存
        logger.log("删除原模型副本，释放显存...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 重新加载保存的模型
        logger.log(f"从检查点重新加载模型: {logger.best_checkpoint_path}")
        checkpoint = torch.load(logger.best_checkpoint_path, weights_only=False)
        model = checkpoint['model']
        tokenizer = checkpoint['tokenizer']
        logger.log("✅ 模型重新加载成功")
    else:
        logger.log("⚠️ 未保存模型，使用内存中的模型继续")

    # ==================== 步骤7: 最终统计 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤7: 最终统计")
    logger.log("=" * 60)

    # 统计剪枝后参数量（所有参数，不管 requires_grad 状态）
    final_parameters = sum(p.numel() for p in model.parameters())
    logger.log(f"\n参数统计:")
    logger.log(f"  剪枝前: {before_pruning_parameters:,}")
    logger.log(f"  剪枝后: {final_parameters:,}")
    logger.log(f"  减少量: {before_pruning_parameters - final_parameters:,}")
    logger.log(f"  实际剪枝率: {(1 - final_parameters/before_pruning_parameters)*100:.2f}%")

    # 计算物理大小（假设 float16，每个参数 2 bytes）
    before_size_gb = before_pruning_parameters * 2 / (1024**3)
    final_size_gb = final_parameters * 2 / (1024**3)
    logger.log(f"\n模型大小（FP16）:")
    logger.log(f"  剪枝前: {before_size_gb:.2f} GB")
    logger.log(f"  剪枝后: {final_size_gb:.2f} GB")
    logger.log(f"  减少: {before_size_gb - final_size_gb:.2f} GB")

    # 验证所有层保持4:1 GQA比例
    gqa_ratios = [layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                  for layer in model.model.layers]
    all_4_to_1 = all(ratio == 4 for ratio in gqa_ratios)
    logger.log(f"\nGQA比例验证: {'✅ 所有层保持4:1' if all_4_to_1 else '❌ 存在不一致'}")

    # ==================== 步骤8: 评估剪枝后PPL ====================
    ppl_before_finetune = None
    if args.test_after_prune:
        logger.log("\n" + "=" * 60)
        logger.log("步骤8: 评估剪枝后困惑度")
        logger.log("=" * 60)

        model.to(args.device)
        model.eval()

        ppl_before_finetune = PPLMetric(model, tokenizer, ['wikitext2'],
                       seq_len=args.max_seq_len, device=args.device)
        logger.log(f"\n剪枝后 PPL: {ppl_before_finetune}")
    else:
        logger.log("\n⚠️ 未启用 --test_after_prune，跳过PPL评估")

    # ==================== 步骤9: 微调剪枝后的模型 ====================
    finetune_stats = None
    if args.finetune:
        logger.log("\n" + "=" * 60)
        logger.log("步骤9: 微调剪枝后的模型")
        logger.log("=" * 60)

        # 创建微调器
        finetuner = FineTuner(model, tokenizer, device=args.device, logger=logger)

        # 执行微调
        finetune_stats = finetuner.finetune(
            dataset_name='wikitext',
            num_samples=args.finetune_samples,
            seq_len=args.finetune_seq_len,
            lr=args.finetune_lr,
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            split='train'
        )

    else:
        logger.log("\n⚠️ 未启用 --finetune，跳过微调")

    # ==================== 步骤10: 保存微调后的模型 ====================
    if args.finetune and args.save_model:
        logger.log("\n" + "=" * 60)
        logger.log("步骤10: 保存微调后的模型")
        logger.log("=" * 60)

        finetuned_path = logger.best_checkpoint_path.replace('.bin', '_finetuned.bin')

        # 使用 FineTuner 的保存方法
        finetuner.save_finetuned_model(
            save_path=finetuned_path,
            layer_pruning_rates=layer_pruning_rates,
            layer_importance=layer_importance,
            finetune_stats=finetune_stats,
            extra_info=args.__dict__
        )

    # ==================== 步骤11: 评估微调后PPL ====================
    if args.finetune and args.test_after_prune:
        logger.log("\n" + "=" * 60)
        logger.log("步骤11: 评估微调后困惑度")
        logger.log("=" * 60)

        model.to(args.device)
        model.eval()

        ppl_after_finetune = PPLMetric(model, tokenizer, ['wikitext2'],
                                       seq_len=args.max_seq_len, device=args.device)
        logger.log(f"\n微调后 PPL: {ppl_after_finetune}")

        # 对比剪枝前后和微调前后的变化
        logger.log("\n" + "=" * 60)
        logger.log("性能对比总结")
        logger.log("=" * 60)
        if ppl_before_finetune:
            logger.log(f"剪枝后（微调前）: {ppl_before_finetune}")
            logger.log(f"微调后: {ppl_after_finetune}")

            # 计算改善百分比
            wikitext_key = 'wikitext2 (wikitext-2-raw-v1)'
            if wikitext_key in ppl_before_finetune and wikitext_key in ppl_after_finetune:
                before_val = ppl_before_finetune[wikitext_key]
                after_val = ppl_after_finetune[wikitext_key]
                improvement = (before_val - after_val) / before_val * 100
                logger.log(f"PPL 改善: {improvement:.2f}%")

    logger.log("\n" + "=" * 60)
    logger.log("✅ 完整流程完成！")
    logger.log("=" * 60)


if __name__ == "__main__":
    main()
