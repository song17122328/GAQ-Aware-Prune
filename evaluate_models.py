#!/usr/bin/env python3
"""
模型对比评估脚本
用于评估原模型、剪枝后模型、微调后模型的PPL

使用方法：
python evaluate_models.py \
    --original_model /path/to/Llama-3-8B-Instruct \
    --pruned_model prune_log/experiment/pytorch_model.bin \
    --finetuned_model prune_log/experiment/pytorch_model_finetuned.bin \
    --seq_len 128
"""

import os
import sys
import argparse
import torch
import gc
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from LLMPruner.evaluator.ppl import PPLMetric


def load_model(model_path, model_type='huggingface', device='cuda'):
    """
    加载模型

    Args:
        model_path: 模型路径
        model_type: 'huggingface' 或 'checkpoint'
        device: 设备

    Returns:
        (model, tokenizer)
    """
    print(f"\n{'=' * 60}")
    print(f"加载 {model_type} 模型: {model_path}")
    print('=' * 60)

    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return None, None

    try:
        if model_type == 'huggingface':
            # 加载 HuggingFace 原始模型
            print("从 HuggingFace 格式加载...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device
            )
        else:
            # 加载 checkpoint（剪枝后或微调后）
            print("从 checkpoint 加载...")
            checkpoint = torch.load(model_path, weights_only=False)
            model = checkpoint['model']
            tokenizer = checkpoint['tokenizer']
            model.to(device)

        print(f"✅ 模型加载成功")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_ppl(model, tokenizer, datasets, seq_len, device='cuda'):
    """
    评估PPL

    Args:
        model: 模型
        tokenizer: tokenizer
        datasets: 数据集列表
        seq_len: 序列长度
        device: 设备

    Returns:
        PPL结果字典
    """
    if model is None or tokenizer is None:
        return None

    print(f"\n使用 seq_len={seq_len} 计算PPL...")
    model.eval()

    try:
        ppl_metric = PPLMetric(
            model,
            tokenizer,
            datasets=datasets,
            seq_len=seq_len,
            device=device
        )
        return ppl_metric
    except Exception as e:
        print(f"❌ PPL计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison_table(results):
    """
    打印对比表格

    Args:
        results: {model_name: {dataset: ppl_value}}
    """
    print("\n" + "=" * 80)
    print("PPL 对比结果")
    print("=" * 80)

    # 获取所有数据集
    all_datasets = set()
    for model_results in results.values():
        if model_results:
            all_datasets.update(model_results.keys())

    # 打印表头
    print(f"\n{'模型':<30} | {'PPL':>15} | {'参数量':>20} | {'变化':>15}")
    print("-" * 80)

    # 按顺序：原模型 -> 剪枝后 -> 微调后
    model_order = ['原始模型', '剪枝后模型', '微调后模型']

    for dataset in sorted(all_datasets):
        print(f"\n数据集: {dataset}")
        print("-" * 80)

        original_ppl = None
        pruned_ppl = None

        for model_name in model_order:
            if model_name not in results:
                continue

            model_data = results[model_name]
            if model_data is None or dataset not in model_data['ppl']:
                ppl_str = "N/A"
                change_str = "-"
            else:
                ppl_value = model_data['ppl'][dataset]
                ppl_str = f"{ppl_value:.2f}"

                # 计算变化
                if model_name == '原始模型':
                    original_ppl = ppl_value
                    change_str = "基准"
                elif model_name == '剪枝后模型':
                    pruned_ppl = ppl_value
                    if original_ppl:
                        change_pct = ((ppl_value - original_ppl) / original_ppl) * 100
                        change_str = f"+{change_pct:.1f}%" if change_pct > 0 else f"{change_pct:.1f}%"
                    else:
                        change_str = "-"
                else:  # 微调后
                    if pruned_ppl:
                        # 相对于剪枝后的改善
                        change_pct = ((ppl_value - pruned_ppl) / pruned_ppl) * 100
                        change_str = f"{change_pct:.1f}% (vs剪枝)"
                    elif original_ppl:
                        # 相对于原模型
                        change_pct = ((ppl_value - original_ppl) / original_ppl) * 100
                        change_str = f"{change_pct:.1f}% (vs原模型)"
                    else:
                        change_str = "-"

            param_count = model_data.get('param_count', 0) if model_data else 0
            param_str = f"{param_count:,}" if param_count > 0 else "N/A"

            print(f"{model_name:<30} | {ppl_str:>15} | {param_str:>20} | {change_str:>15}")

    print("=" * 80)


def save_results(results, output_path):
    """
    保存结果到JSON文件

    Args:
        results: 评估结果
        output_path: 输出路径
    """
    try:
        # 转换为可序列化的格式
        serializable_results = {}
        for model_name, data in results.items():
            if data is None:
                serializable_results[model_name] = None
            else:
                serializable_results[model_name] = {
                    'ppl': {k: float(v) for k, v in data['ppl'].items()},
                    'param_count': int(data['param_count'])
                }

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 结果已保存到: {output_path}")

    except Exception as e:
        print(f"❌ 保存结果失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='模型对比评估脚本')

    # 模型路径
    parser.add_argument('--original_model', type=str, default=None,
                       help='原始模型路径（HuggingFace格式）')
    parser.add_argument('--pruned_model', type=str, default=None,
                       help='剪枝后模型路径（.bin checkpoint）')
    parser.add_argument('--finetuned_model', type=str, default=None,
                       help='微调后模型路径（.bin checkpoint）')

    # 评估参数
    parser.add_argument('--datasets', type=str, nargs='+', default=['wikitext2'],
                       help='评估数据集列表（默认：wikitext2）')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度（默认：128）')

    # 输出选项
    parser.add_argument('--save_results', type=str, default=None,
                       help='保存结果到JSON文件（可选）')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='结果输出目录（默认：evaluation_results）')

    args = parser.parse_args()

    # 检查至少提供一个模型
    if not any([args.original_model, args.pruned_model, args.finetuned_model]):
        print("❌ 错误：至少需要提供一个模型路径")
        print("使用 --help 查看帮助")
        return

    # 自动选择GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    results = {}

    # ==================== 评估原始模型 ====================
    if args.original_model:
        print("\n" + "=" * 80)
        print("步骤 1: 评估原始模型")
        print("=" * 80)

        model, tokenizer = load_model(args.original_model, 'huggingface', device)

        if model is not None:
            param_count = sum(p.numel() for p in model.parameters())
            ppl_result = evaluate_ppl(model, tokenizer, args.datasets, args.seq_len, device)

            if ppl_result:
                print(f"\n原始模型 PPL: {ppl_result}")
                results['原始模型'] = {
                    'ppl': dict(ppl_result),
                    'param_count': param_count
                }

            # 清理显存
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    # ==================== 评估剪枝后模型 ====================
    if args.pruned_model:
        print("\n" + "=" * 80)
        print("步骤 2: 评估剪枝后模型")
        print("=" * 80)

        model, tokenizer = load_model(args.pruned_model, 'checkpoint', device)

        if model is not None:
            param_count = sum(p.numel() for p in model.parameters())
            ppl_result = evaluate_ppl(model, tokenizer, args.datasets, args.seq_len, device)

            if ppl_result:
                print(f"\n剪枝后模型 PPL: {ppl_result}")
                results['剪枝后模型'] = {
                    'ppl': dict(ppl_result),
                    'param_count': param_count
                }

            # 清理显存
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    # ==================== 评估微调后模型 ====================
    if args.finetuned_model:
        print("\n" + "=" * 80)
        print("步骤 3: 评估微调后模型")
        print("=" * 80)

        model, tokenizer = load_model(args.finetuned_model, 'checkpoint', device)

        if model is not None:
            param_count = sum(p.numel() for p in model.parameters())
            ppl_result = evaluate_ppl(model, tokenizer, args.datasets, args.seq_len, device)

            if ppl_result:
                print(f"\n微调后模型 PPL: {ppl_result}")
                results['微调后模型'] = {
                    'ppl': dict(ppl_result),
                    'param_count': param_count
                }

            # 清理显存
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    # ==================== 生成对比报告 ====================
    if results:
        print_comparison_table(results)

        # 保存结果
        if args.save_results:
            output_path = args.save_results
        else:
            # 自动生成文件名
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(args.output_dir, f'evaluation_{timestamp}.json')

        save_results(results, output_path)
    else:
        print("\n❌ 没有成功评估任何模型")

    print("\n" + "=" * 80)
    print("✅ 评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
