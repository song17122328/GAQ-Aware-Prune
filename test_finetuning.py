#!/usr/bin/env python3
"""
独立的微调测试脚本
用于验证微调功能是否正常工作

使用方法：
python test_finetuning.py --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin
"""

import os
import sys
import argparse
import torch
import gc
from transformers import AutoTokenizer

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from LLMPruner.trainer.finetuner import FineTuner
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.utils.logger import LoggerWithDepth


def main():
    parser = argparse.ArgumentParser(description='微调测试脚本')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='剪枝后的模型路径（.bin文件）')
    parser.add_argument('--save_name', type=str, default='test_finetune',
                       help='保存名称')

    # 微调参数
    parser.add_argument('--method', type=str, default='full',
                       choices=['full', 'lora'],
                       help='微调方法')
    parser.add_argument('--use_lora', action='store_true',
                       help='使用LoRA微调（等同于--method lora）')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=1,
                       help='训练轮数')
    parser.add_argument('--samples', type=int, default=100,
                       help='训练样本数（测试用小值）')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='batch size')
    parser.add_argument('--seq_len', type=int, default=256,
                       help='序列长度')
    parser.add_argument('--grad_accum', type=int, default=2,
                       help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--warmup_steps', type=int, default=10,
                       help='预热步数')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')

    # LoRA专用参数
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA秩（越大效果越好但参数越多，建议4-16）')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA缩放系数（通常设为r的2倍）')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout率')
    parser.add_argument('--lora_target_attention', action='store_true', default=False,
                       help='LoRA是否应用到Attention层（q,k,v,o）')
    parser.add_argument('--lora_target_mlp', action='store_true', default=False,
                       help='LoRA是否应用到MLP层（gate,up,down）')

    # 测试参数
    parser.add_argument('--test_before', action='store_true',
                       help='微调前测试PPL')
    parser.add_argument('--test_after', action='store_true',
                       help='微调后测试PPL')
    parser.add_argument('--save_model', action='store_true',
                       help='保存微调后的模型')

    args = parser.parse_args()

    # 自动选择GPU
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

    if not os.path.exists(args.model_path):
        logger.log(f"❌ 模型文件不存在: {args.model_path}")
        return

    logger.log(f"从 {args.model_path} 加载模型...")
    checkpoint = torch.load(args.model_path, weights_only=False)

    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']

    logger.log("✅ 模型加载成功")
    logger.log(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 确保模型在正确的设备上
    model.to(device)

    # ==================== 步骤2: 微调前评估（可选） ====================
    if args.test_before:
        logger.log("\n" + "=" * 60)
        logger.log("步骤2: 微调前评估")
        logger.log("=" * 60)

        model.eval()
        ppl_before = PPLMetric(model, tokenizer, ['wikitext2'],
                              seq_len=128, device=device)
        logger.log(f"微调前 PPL: {ppl_before}")

    # ==================== 步骤3: 执行微调 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤3: 执行微调")
    logger.log("=" * 60)

    # 创建微调器
    use_lora = (args.method == 'lora' or args.use_lora)

    finetuner = FineTuner(
        model,
        tokenizer,
        device=device,
        logger=logger,
        use_lora=use_lora,
        lora_r=args.lora_r if use_lora else 8,
        lora_alpha=args.lora_alpha if use_lora else 16,
        lora_dropout=args.lora_dropout if use_lora else 0.05,
        lora_target_attention=args.lora_target_attention if use_lora else True,
        lora_target_mlp=args.lora_target_mlp if use_lora else True
    )

    # 执行微调
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
            weight_decay=args.weight_decay,
            split='train'
        )

        logger.log("\n✅ 微调成功完成！")
        logger.log(f"\n微调统计:")
        logger.log(f"  方法: {finetune_stats['method']}")
        logger.log(f"  总步数: {finetune_stats['total_steps']}")
        logger.log(f"  最终Loss: {finetune_stats['final_loss']:.4f}")

    except Exception as e:
        logger.log(f"\n❌ 微调过程出错:")
        logger.log(f"  错误类型: {type(e).__name__}")
        logger.log(f"  错误信息: {str(e)}")
        import traceback
        logger.log(f"\n详细错误信息:")
        logger.log(traceback.format_exc())
        return

    # ==================== 步骤4: 微调后评估（可选） ====================
    if args.test_after:
        logger.log("\n" + "=" * 60)
        logger.log("步骤4: 微调后评估")
        logger.log("=" * 60)

        model.eval()
        ppl_after = PPLMetric(model, tokenizer, ['wikitext2'],
                             seq_len=128, device=device)
        logger.log(f"微调后 PPL: {ppl_after}")

        if args.test_before:
            logger.log(f"\nPPL对比:")
            logger.log(f"  微调前: {ppl_before['wikitext2 (wikitext-2-raw-v1)']:.2f}")
            logger.log(f"  微调后: {ppl_after['wikitext2 (wikitext-2-raw-v1)']:.2f}")

    # ==================== 步骤5: 保存模型（可选） ====================
    if args.save_model:
        logger.log("\n" + "=" * 60)
        logger.log("步骤5: 保存微调后的模型")
        logger.log("=" * 60)

        save_path = os.path.join(logger.log_dir, 'pytorch_model_finetuned.bin')

        finetuner.save_finetuned_model(
            save_path=save_path,
            layer_pruning_rates=checkpoint.get('layer_pruning_rates', {}),
            layer_importance=checkpoint.get('layer_importance', {}),
            finetune_stats=finetune_stats,
            extra_info=args.__dict__
        )

    logger.log("\n" + "=" * 60)
    logger.log("✅ 测试完成！")
    logger.log("=" * 60)


if __name__ == "__main__":
    main()
