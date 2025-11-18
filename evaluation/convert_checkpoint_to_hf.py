#!/usr/bin/env python3
"""
将剪枝后的checkpoint转换为HuggingFace格式

用法:
    python evaluation/convert_checkpoint_to_hf.py \
        --checkpoint prune_log/ours/pytorch_model.bin \
        --output_dir models/ours_hf

这样转换后的模型可以用于Zero-shot评估
"""

import argparse
import torch
import os
import shutil
from pathlib import Path


def convert_checkpoint_to_hf(checkpoint_path: str, output_dir: str):
    """
    转换checkpoint为HuggingFace格式

    Args:
        checkpoint_path: checkpoint文件路径 (.bin)
        output_dir: 输出目录
    """
    print(f"加载checkpoint: {checkpoint_path}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError("不支持的checkpoint格式。需要包含'model'键的字典格式")

    model = checkpoint['model']
    tokenizer = checkpoint.get('tokenizer')
    config_dict = checkpoint.get('config', {})

    print(f"✓ Checkpoint加载完成")
    print(f"  模型类型: {type(model).__name__}")
    if tokenizer:
        print(f"  Tokenizer: ✓")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n保存到HuggingFace格式: {output_dir}")

    # 保存模型
    print("  保存模型...")
    model.save_pretrained(output_dir)

    # 保存tokenizer
    if tokenizer:
        print("  保存tokenizer...")
        tokenizer.save_pretrained(output_dir)
    else:
        # 如果没有tokenizer，尝试从base_model复制
        base_model = config_dict.get('base_model')
        if base_model and os.path.exists(base_model):
            print(f"  从base_model复制tokenizer: {base_model}")
            from transformers import AutoTokenizer
            base_tokenizer = AutoTokenizer.from_pretrained(base_model)
            base_tokenizer.save_pretrained(output_dir)
        else:
            print("  ⚠️  警告: 没有找到tokenizer，可能需要手动复制")

    # 保存配置信息
    config_path = os.path.join(output_dir, 'pruning_config.txt')
    with open(config_path, 'w') as f:
        f.write("剪枝配置信息\n")
        f.write("=" * 60 + "\n\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"  保存剪枝配置: pruning_config.txt")

    print(f"\n✓ 转换完成！")
    print(f"\n现在可以使用HuggingFace格式进行评估:")
    print(f"  python evaluation/run_evaluation.py \\")
    print(f"      --model_path {output_dir} \\")
    print(f"      --metrics zeroshot \\")
    print(f"      --output results.json")


def main():
    parser = argparse.ArgumentParser(description='转换checkpoint为HuggingFace格式')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint文件路径 (.bin)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    args = parser.parse_args()

    convert_checkpoint_to_hf(args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()
