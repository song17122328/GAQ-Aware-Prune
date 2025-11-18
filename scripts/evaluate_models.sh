#!/bin/bash
# 评估所有模型的示例脚本

PYTHON="/data/home/yuanxiaosong/miniconda3/bin/python"
DEVICE="cuda"

echo "======================================"
echo "评估所有模型"
echo "======================================"

# 1. 评估原始模型
echo "[1/3] 评估原始模型..."
$PYTHON evaluation/run_evaluation.py \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics ppl,speed,memory \
    --output results/original.json \
    --device $DEVICE

# 2. 评估你的最优配置
echo "[2/3] 评估最优配置..."
$PYTHON evaluation/run_evaluation.py \
    --model_path prune_log/ppl_search_*_ratio_0.7_9.3_freeze_8/pytorch_model.bin \
    --metrics ppl,speed,memory \
    --output results/ours.json \
    --device $DEVICE

# 3. 评估Baseline (Uniform)
echo "[3/3] 评估Baseline..."
$PYTHON evaluation/run_evaluation.py \
    --model_path prune_log/baseline_uniform/pytorch_model.bin \
    --metrics ppl,speed,memory \
    --output results/baseline_uniform.json \
    --device $DEVICE

# 4. 生成对比表格
echo "======================================"
echo "生成对比表格"
echo "======================================"
$PYTHON evaluation/run_evaluation.py \
    --compare \
    --model_paths results/original.json,results/ours.json,results/baseline_uniform.json \
    --output results/comparison.md

echo "✓ 评估完成！结果保存在 results/ 目录"
