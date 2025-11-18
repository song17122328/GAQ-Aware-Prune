#!/usr/bin/env python3
"""
测试模型是否正确加载到GPU

用法：
    python test_gpu_loading.py --model_path prune_log/xxx/pytorch_model.bin
"""

import argparse
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.utils.model_loader import load_model_and_tokenizer
from evaluation.utils.get_best_gpu import get_best_gpu


def check_model_device(model):
    """检查模型参数所在的设备"""
    devices = set()
    for name, param in model.named_parameters():
        devices.add(str(param.device))
        if len(devices) <= 5:  # 只打印前5个参数的设备
            print(f"  参数 {name[:50]:50s} -> 设备: {param.device}")

    print(f"\n模型参数分布在以下设备上: {devices}")

    # 检查是否全部在GPU上
    all_cuda = all('cuda' in d for d in devices)
    if all_cuda:
        print("✓ 所有参数都在GPU上")
        return True
    else:
        print("✗ 存在CPU上的参数！")
        return False


def main():
    parser = argparse.ArgumentParser(description='测试模型GPU加载')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径（HF目录或.bin文件）')
    parser.add_argument('--auto_select_gpu', action='store_true',
                       help='自动选择GPU')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')

    args = parser.parse_args()

    # 自动选择GPU
    if args.auto_select_gpu:
        gpu_id = get_best_gpu()
        args.device = f'cuda:{gpu_id}'
        print(f"✓ 自动选择GPU: {args.device}\n")

    print("="*80)
    print(f"测试模型加载到GPU")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"目标设备: {args.device}")
    print()

    # 加载模型
    print("步骤1: 加载模型...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        force_single_device=True
    )

    print("\n步骤2: 检查模型参数设备...")
    all_on_gpu = check_model_device(model)

    # 检查CUDA内存占用
    if args.device.startswith('cuda'):
        print(f"\n步骤3: 检查GPU显存占用...")
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0

        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)    # GB

        print(f"  GPU {device_id} 显存占用:")
        print(f"    已分配: {allocated:.2f} GB")
        print(f"    已保留: {reserved:.2f} GB")

        if allocated > 0.1:
            print(f"  ✓ GPU有显存占用")
        else:
            print(f"  ✗ GPU显存占用几乎为0，模型可能仍在CPU上")

    # 测试前向传播
    print(f"\n步骤4: 测试前向传播...")
    try:
        inputs = tokenizer("Hello, world!", return_tensors='pt')

        # 检查输入张量的设备
        print(f"  输入张量设备: {inputs['input_ids'].device}")

        # 移动输入到目标设备
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        print(f"  移动后输入设备: {inputs['input_ids'].device}")

        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"  输出张量设备: {outputs.logits.device}")
        print(f"  ✓ 前向传播成功")

    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")

    print("\n" + "="*80)
    if all_on_gpu:
        print("结论: ✓ 模型已正确加载到GPU")
    else:
        print("结论: ✗ 模型未完全加载到GPU，请检查代码")
    print("="*80)


if __name__ == '__main__':
    main()
