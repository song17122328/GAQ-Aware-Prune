#!/usr/bin/env python3
"""
带层冻结的微调测试脚本
用于极度不稳定的模型（PPL>50）

策略：只微调最重要的几层，冻结其他层
"""

import os
import sys
import argparse
import torch
import gc
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from LLMPruner.trainer.finetuner import FineTuner
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.utils.logger import LoggerWithDepth


def freeze_layers(model, freeze_strategy='most'):
    """
    冻结部分层

    Args:
        model: 模型
        freeze_strategy: 冻结策略
            - 'most': 冻结大部分层，只训练最后几层
            - 'half': 冻结前半部分
            - 'ends': 冻结首尾，只训练中间层
    """
    num_layers = len(model.model.layers)

    if freeze_strategy == 'most':
        # 只训练最后5层
        freeze_range = range(0, num_layers - 5)
        train_range = range(num_layers - 5, num_layers)

    elif freeze_strategy == 'half':
        # 冻结前半部分
        mid = num_layers // 2
        freeze_range = range(0, mid)
        train_range = range(mid, num_layers)

    elif freeze_strategy == 'ends':
        # 冻结前3层和后3层，训练中间层
        freeze_range = list(range(0, 3)) + list(range(num_layers - 3, num_layers))
        train_range = range(3, num_layers - 3)

    else:
        raise ValueError(f"未知策略: {freeze_strategy}")

    # 冻结指定层
    frozen_count = 0
    for layer_idx in freeze_range:
        for param in model.model.layers[layer_idx].parameters():
            param.requires_grad = False
            frozen_count += param.numel()

    # 计算可训练参数
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    return {
        'frozen_layers': list(freeze_range),
        'trainable_layers': list(train_range),
        'frozen_params': frozen_count,
        'trainable_params': trainable_count,
        'total_params': total_count,
        'trainable_ratio': trainable_count / total_count
    }


def main():
    parser = argparse.ArgumentParser(description='带层冻结的微调测试')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='剪枝后的模型路径')
    parser.add_argument('--save_name', type=str, default='test_frozen_finetune',
                       help='保存名称')

    # 冻结策略
    parser.add_argument('--freeze_strategy', type=str, default='most',
                       choices=['most', 'half', 'ends'],
                       help='冻结策略')

    # 微调参数（使用极低学习率）
    parser.add_argument('--lr', type=float, default=1e-7,
                       help='学习率（默认极低）')
    parser.add_argument('--epochs', type=int, default=1,
                       help='训练轮数')
    parser.add_argument('--samples', type=int, default=200,
                       help='训练样本数')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度')
    parser.add_argument('--grad_accum', type=int, default=2,
                       help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=0.1,
                       help='梯度裁剪（默认极强）')
    parser.add_argument('--warmup_steps', type=int, default=5,
                       help='预热步数')

    # 测试参数
    parser.add_argument('--test_before', action='store_true',
                       help='微调前测试PPL')
    parser.add_argument('--test_after', action='store_true',
                       help='微调后测试PPL')
    parser.add_argument('--save_model', action='store_true',
                       help='保存微调后的模型')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建日志
    logger = LoggerWithDepth(
        env_name=args.save_name,
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # ==================== 步骤1: 加载模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤1: 加载剪枝后的模型")
    logger.log("=" * 60)

    checkpoint = torch.load(args.model_path, weights_only=False)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    model.to(device)

    logger.log("✅ 模型加载成功")

    # ==================== 步骤2: 冻结层 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤2: 冻结部分层")
    logger.log("=" * 60)

    freeze_info = freeze_layers(model, args.freeze_strategy)

    logger.log(f"冻结策略: {args.freeze_strategy}")
    logger.log(f"冻结层: {freeze_info['frozen_layers'][:5]}... (共{len(freeze_info['frozen_layers'])}层)")
    logger.log(f"训练层: {freeze_info['trainable_layers'][:5]}... (共{len(freeze_info['trainable_layers'])}层)")
    logger.log(f"冻结参数: {freeze_info['frozen_params']:,}")
    logger.log(f"可训练参数: {freeze_info['trainable_params']:,}")
    logger.log(f"可训练比例: {freeze_info['trainable_ratio']*100:.2f}%")

    # ==================== 步骤3: 微调前评估 ====================
    if args.test_before:
        logger.log("\n" + "=" * 60)
        logger.log("步骤3: 微调前评估")
        logger.log("=" * 60)

        model.eval()
        ppl_before = PPLMetric(model, tokenizer, ['wikitext2'],
                              seq_len=128, device=device)
        logger.log(f"微调前 PPL: {ppl_before}")

    # ==================== 步骤4: 执行微调 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤4: 执行微调（带层冻结）")
    logger.log("=" * 60)

    finetuner = FineTuner(
        model,
        tokenizer,
        device=device,
        logger=logger,
        use_lora=False
    )

    try:
        finetune_stats = finetuner.finetune(
            dataset_name='wikitext',
            num_samples=args.samples,
            seq_len=args.seq_len,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=args.warmup_steps,
            weight_decay=0.0,  # 不使用权重衰减
            split='train'
        )

        logger.log("\n✅ 微调成功完成！")
        logger.log(f"最终Loss: {finetune_stats['final_loss']:.4f}")

    except Exception as e:
        logger.log(f"\n❌ 微调失败: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return

    # ==================== 步骤5: 微调后评估 ====================
    if args.test_after:
        logger.log("\n" + "=" * 60)
        logger.log("步骤5: 微调后评估")
        logger.log("=" * 60)

        model.eval()
        ppl_after = PPLMetric(model, tokenizer, ['wikitext2'],
                             seq_len=128, device=device)
        logger.log(f"微调后 PPL: {ppl_after}")

        if args.test_before:
            logger.log(f"\nPPL对比:")
            logger.log(f"  微调前: {ppl_before['wikitext2 (wikitext-2-raw-v1)']:.2f}")
            logger.log(f"  微调后: {ppl_after['wikitext2 (wikitext-2-raw-v1)']:.2f}")

    # ==================== 步骤6: 保存模型 ====================
    if args.save_model:
        logger.log("\n" + "=" * 60)
        logger.log("步骤6: 保存微调后的模型")
        logger.log("=" * 60)

        save_path = os.path.join(logger.log_dir, 'pytorch_model_finetuned_frozen.bin')

        # 解冻所有层再保存
        for param in model.parameters():
            param.requires_grad = True

        save_dict = {
            'model': model,
            'tokenizer': tokenizer,
            'freeze_info': freeze_info,
            'finetune_stats': finetune_stats,
            'config': args.__dict__
        }

        model.half()
        torch.save(save_dict, save_path)
        logger.log(f"✅ 模型已保存到: {save_path}")

    logger.log("\n" + "=" * 60)
    logger.log("✅ 测试完成！")
    logger.log("=" * 60)


if __name__ == "__main__":
    main()
