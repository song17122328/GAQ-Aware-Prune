#!/bin/bash
# 测试不同数据集对剪枝效果的影响

MODEL="/newdata/LLMs/Llama-3-8B-Instruct"
PRUNING_RATIO=0.25
NUM_SAMPLES=128
LAYER_START=0
LAYER_END=32

echo "=========================================="
echo "测试数据集对剪枝效果的影响"
echo "=========================================="
echo ""

# 实验 1: WikiText2 剪枝 + WikiText2 评估（一致）
echo "[实验 1] WikiText2 → WikiText2（数据一致）"
python llama3_global_pruning.py \
    --base_model $MODEL \
    --save_ckpt_log_name exp1_wiki_to_wiki \
    --pruning_ratio $PRUNING_RATIO \
    --importance_method taylor \
    --dataset wikitext2 \
    --num_samples $NUM_SAMPLES \
    --layer_start $LAYER_START \
    --layer_end $LAYER_END \
    --test_after_prune \
    --save_model

echo ""
echo "=========================================="
echo ""

# 实验 2: C4 剪枝 + WikiText2 评估（不一致）
echo "[实验 2] C4 → WikiText2（数据不一致）"
python llama3_global_pruning.py \
    --base_model $MODEL \
    --save_ckpt_log_name exp2_c4_to_wiki \
    --pruning_ratio $PRUNING_RATIO \
    --importance_method taylor \
    --dataset c4 \
    --num_samples $NUM_SAMPLES \
    --layer_start $LAYER_START \
    --layer_end $LAYER_END \
    --test_after_prune \
    --save_model

echo ""
echo "=========================================="
echo "实验完成！"
echo ""
echo "预期结果："
echo "  实验 1 (Wiki→Wiki): PPL 较低（数据一致）"
echo "  实验 2 (C4→Wiki): PPL 较高（数据不匹配）"
echo ""
echo "请检查日志对比 PPL 差异"
echo "=========================================="
