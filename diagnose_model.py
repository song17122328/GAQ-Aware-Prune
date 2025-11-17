#!/usr/bin/env python3
"""
模型健康诊断脚本
用于检查剪枝后的模型是否存在数值问题
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def check_model_health(model_path):
    """
    检查模型权重的健康状态

    返回：
        bool: 模型是否健康
        dict: 诊断信息
    """
    print("=" * 60)
    print("开始模型健康检查")
    print("=" * 60)

    # 加载模型
    print(f"\n1. 加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在！")
        return False, {}

    checkpoint = torch.load(model_path, weights_only=False)
    model = checkpoint['model']

    print("✅ 模型加载成功")

    # 检查项
    diagnostics = {
        'total_params': 0,
        'nan_params': 0,
        'inf_params': 0,
        'zero_params': 0,
        'very_large_params': 0,
        'problematic_layers': [],
        'weight_stats': {},
    }

    # 遍历所有参数
    print("\n2. 检查模型权重...")

    for name, param in model.named_parameters():
        diagnostics['total_params'] += param.numel()

        # 检查NaN
        nan_count = torch.isnan(param).sum().item()
        if nan_count > 0:
            diagnostics['nan_params'] += nan_count
            diagnostics['problematic_layers'].append(f"{name}: {nan_count} NaN值")

        # 检查Inf
        inf_count = torch.isinf(param).sum().item()
        if inf_count > 0:
            diagnostics['inf_params'] += inf_count
            diagnostics['problematic_layers'].append(f"{name}: {inf_count} Inf值")

        # 检查零值（过多可能有问题）
        zero_count = (param == 0).sum().item()
        zero_ratio = zero_count / param.numel()
        if zero_ratio > 0.9:  # 超过90%是零
            diagnostics['zero_params'] += zero_count
            diagnostics['problematic_layers'].append(
                f"{name}: {zero_ratio*100:.1f}% 零值"
            )

        # 检查过大的值
        very_large = (param.abs() > 1e4).sum().item()
        if very_large > 0:
            diagnostics['very_large_params'] += very_large
            max_val = param.abs().max().item()
            diagnostics['problematic_layers'].append(
                f"{name}: {very_large} 个过大值 (最大={max_val:.2e})"
            )

        # 统计每层的权重分布
        if 'layers' in name and ('weight' in name or 'bias' in name):
            layer_stats = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'abs_mean': param.abs().mean().item(),
            }
            diagnostics['weight_stats'][name] = layer_stats

    # 打印诊断结果
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)

    print(f"\n总参数量: {diagnostics['total_params']:,}")

    is_healthy = True

    # NaN检查
    if diagnostics['nan_params'] > 0:
        print(f"❌ 发现 {diagnostics['nan_params']:,} 个NaN值")
        is_healthy = False
    else:
        print(f"✅ 无NaN值")

    # Inf检查
    if diagnostics['inf_params'] > 0:
        print(f"❌ 发现 {diagnostics['inf_params']:,} 个Inf值")
        is_healthy = False
    else:
        print(f"✅ 无Inf值")

    # 零值检查
    zero_ratio = diagnostics['zero_params'] / diagnostics['total_params']
    if zero_ratio > 0.3:  # 超过30%可能有问题
        print(f"⚠️  过多零值: {zero_ratio*100:.1f}%")
    else:
        print(f"✅ 零值比例正常: {zero_ratio*100:.1f}%")

    # 过大值检查
    if diagnostics['very_large_params'] > 0:
        print(f"⚠️  发现 {diagnostics['very_large_params']:,} 个过大值 (>1e4)")
    else:
        print(f"✅ 无异常大的值")

    # 问题层列表
    if diagnostics['problematic_layers']:
        print(f"\n❌ 发现 {len(diagnostics['problematic_layers'])} 个问题层:")
        for issue in diagnostics['problematic_layers'][:10]:  # 只显示前10个
            print(f"  - {issue}")
        if len(diagnostics['problematic_layers']) > 10:
            print(f"  ... 还有 {len(diagnostics['problematic_layers']) - 10} 个问题")
    else:
        print(f"\n✅ 所有层看起来正常")

    # 权重统计摘要
    print(f"\n权重统计摘要（前5层）:")
    for i, (name, stats) in enumerate(list(diagnostics['weight_stats'].items())[:5]):
        print(f"  {name}:")
        print(f"    均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}")
        print(f"    范围=[{stats['min']:.6f}, {stats['max']:.6f}]")

    # 最终判断
    print("\n" + "=" * 60)
    if is_healthy:
        print("✅ 模型健康状态: 良好")
        print("   可以进行微调")
    else:
        print("❌ 模型健康状态: 异常")
        print("   建议：")
        print("   1. 重新进行剪枝")
        print("   2. 检查剪枝率是否过高")
        print("   3. 检查剪枝过程中是否有错误")
    print("=" * 60)

    return is_healthy, diagnostics


def test_forward_pass(model_path, device='cuda'):
    """
    测试模型前向传播是否正常
    """
    print("\n" + "=" * 60)
    print("3. 测试前向传播")
    print("=" * 60)

    checkpoint = torch.load(model_path, weights_only=False)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']

    model.to(device)
    model.eval()

    # 创建测试输入
    test_text = "Hello, this is a test."
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 检查输出
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()

        if has_nan:
            print("❌ 输出包含NaN值")
            return False
        elif has_inf:
            print("❌ 输出包含Inf值")
            return False
        else:
            print("✅ 前向传播正常")
            print(f"   输出shape: {logits.shape}")
            print(f"   输出范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            return True

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='模型健康诊断')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--test_forward', action='store_true',
                       help='测试前向传播')

    args = parser.parse_args()

    # 健康检查
    is_healthy, diagnostics = check_model_health(args.model_path)

    # 前向传播测试
    if args.test_forward and is_healthy:
        forward_ok = test_forward_pass(args.model_path)
        if not forward_ok:
            is_healthy = False

    # 返回状态码
    sys.exit(0 if is_healthy else 1)


if __name__ == "__main__":
    main()
