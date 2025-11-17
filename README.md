# GAQ-Aware-Prune

GQA（Grouped Query Attention）感知的 LLaMA-3 模型剪枝工具

## 项目简介

本项目实现了针对 LLaMA-3 模型的智能剪枝方案：
- **GQA感知剪枝**：保持 4:1 的 Q:KV 头比例
- **非均衡层级剪枝**：重要层剪枝少，不重要层剪枝多
- **LoRA微调恢复**：剪枝后使用 LoRA 高效恢复性能

---

## 快速开始

### 1. 剪枝模型

#### 剪枝 Attention 和 MLP（推荐）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_25pct \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model
```

#### 只剪枝 MLP（保留Attention）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_mlp_only_25pct \
    --pruning_ratio 0.25 \
    --no_prune_attention \
    --prune_mlp \
    --save_model
```

#### 只剪枝 Attention（保留MLP）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_attn_only_25pct \
    --pruning_ratio 0.25 \
    --save_model
```

**关键参数**：
- `--base_model`: 原始模型路径
- `--pruning_ratio`: 剪枝率（0.25 = 剪掉25%参数）
- `--prune_attention`: 是否剪枝Attention层（默认True）
- `--no_prune_attention`: 禁用Attention剪枝（用于只剪MLP）
- `--prune_mlp`: 是否剪枝MLP层（默认False）
- `--save_model`: 保存剪枝后的模型

**输出**：`prune_log/{experiment_name}/pytorch_model.bin`

---

### 2. LoRA 微调

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_25pct/pytorch_model.bin \
    --save_name lora_finetune \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_attention \
    --lora_target_mlp \
    --lr 2e-4 \
    --samples 1000 \
    --epochs 3 \
    --seq_len 512 \
    --batch_size 1 \
    --grad_accum 4 \
    --test_before \
    --test_after \
    --save_model
```

**关键参数**：
- `--use_lora`: 启用 LoRA 微调
- `--lora_r`: LoRA 秩（建议 4-16）
- `--lora_target_attention`: 对 Attention 层应用 LoRA（q/k/v/o）
- `--lora_target_mlp`: 对 MLP 层应用 LoRA（gate/up/down）
- `--lr`: 学习率（LoRA 推荐 2e-4，全参数微调推荐 1e-5）
- `--seq_len`: 序列长度（建议与评估一致）

**输出**：`prune_log/lora_finetune/pytorch_model_finetuned.bin`

---

### 3. 评估对比

#### 方法A：使用独立评估脚本

```bash
python evaluate_models.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama3_pruned_25pct/pytorch_model.bin \
    --finetuned_model prune_log/lora_finetune/pytorch_model_finetuned.bin \
    --seq_len 512 \
    --save_results results.json
```

**关键参数**：
- `--original_model`: 原始模型（可选）
- `--pruned_model`: 剪枝后模型（可选）
- `--finetuned_model`: 微调后模型（可选）
- `--seq_len`: 评估序列长度（应与训练一致）
- `--save_results`: 保存结果到 JSON

#### 方法B：在主脚本中测量原模型PPL（推荐）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name experiment_with_baseline \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --test_original_ppl \
    --test_after_prune \
    --eval_seq_len 512 \
    --save_model \
    --finetune \
    --finetune_method lora
```

**新增参数**：
- `--test_original_ppl`: 剪枝前评估原模型PPL（作为baseline）
- `--eval_seq_len`: PPL评估时的序列长度（默认128，建议与微调seq_len一致）
- `--test_after_prune`: 剪枝后评估PPL（可对比原模型和微调后）

**优势**：一次运行即可得到完整的 原模型→剪枝后→微调后 的PPL对比

**输出示例**：
```
================================================================================
PPL 对比结果
================================================================================

数据集: wikitext2 (wikitext-2-raw-v1)
--------------------------------------------------------------------------------
模型                           |             PPL |               参数量 |            变化
--------------------------------------------------------------------------------
原始模型                        |           12.34 |        8,030,261,248 |            基准
剪枝后模型                      |           80.85 |        6,024,195,936 |         +555.2%
微调后模型                      |           35.82 |        6,024,195,936 |  -55.7% (vs剪枝)
================================================================================
```

---

## 完整工作流程

### 方案A：剪枝 + LoRA 微调（推荐）

```bash
# 步骤1: 剪枝
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name exp_25pct \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model

# 步骤2: LoRA 微调
python test_finetuning.py \
    --model_path prune_log/exp_25pct/pytorch_model.bin \
    --save_name exp_25pct_lora \
    --use_lora \
    --lora_target_attention \
    --lora_target_mlp \
    --lr 2e-4 \
    --samples 1000 \
    --epochs 3 \
    --seq_len 512 \
    --save_model

# 步骤3: 评估
python evaluate_models.py \
    --pruned_model prune_log/exp_25pct/pytorch_model.bin \
    --finetuned_model prune_log/exp_25pct_lora/pytorch_model_finetuned.bin \
    --seq_len 512
```

### 方案B：一键剪枝+微调

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name exp_integrated \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_attention \
    --lora_target_mlp \
    --finetune_lr 2e-4 \
    --finetune_samples 1000 \
    --finetune_epochs 3
```

---

## 常用参数组合

### 对比不同剪枝目标（推荐实验）

```bash
# 实验1: 只剪枝 Attention
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name compare_attn_only \
    --pruning_ratio 0.25 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验2: 只剪枝 MLP
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name compare_mlp_only \
    --pruning_ratio 0.25 \
    --no_prune_attention \
    --prune_mlp \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验3: 同时剪枝 Attention + MLP
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name compare_both \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --test_original_ppl \
    --test_after_prune \
    --save_model
```

### 快速测试（Debug）

```bash
# 剪枝测试（只处理部分层）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name debug_test \
    --pruning_ratio 0.25 \
    --layer_start 10 \
    --layer_end 15 \
    --layer_importance_samples 10

# 微调测试（少量样本）
python test_finetuning.py \
    --model_path prune_log/debug_test/pytorch_model.bin \
    --save_name debug_finetune \
    --use_lora \
    --lora_target_attention \
    --lora_target_mlp \
    --samples 50 \
    --epochs 1 \
    --seq_len 128
```

### 生产环境（完整训练）

```bash
# 剪枝
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name production \
    --pruning_ratio 0.25 \
    --layer_importance_method removal \
    --layer_importance_samples 50 \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model

# LoRA 微调
python test_finetuning.py \
    --model_path prune_log/production/pytorch_model.bin \
    --save_name production_lora \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_attention \
    --lora_target_mlp \
    --lr 2e-4 \
    --samples 1000 \
    --epochs 3 \
    --seq_len 512 \
    --batch_size 1 \
    --grad_accum 4 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --save_model
```

---

## 核心脚本说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| `llama3_unbalanced_pruning_gqa_aware.py` | 主剪枝脚本 | 剪枝后的模型 |
| `test_finetuning.py` | 独立微调脚本 | 微调后的模型 |
| `evaluate_models.py` | 模型对比评估 | PPL 对比报告 |
| `diagnose_model.py` | 模型健康检查 | 诊断报告 |

---

## 输出目录结构

```
prune_log/
└── {experiment_name}/
    ├── description.txt                    # 实验配置
    ├── layer_importance_config.json       # 层重要性分数
    ├── pruning_strategy.png               # 剪枝策略可视化
    ├── pytorch_model.bin                  # 剪枝后模型
    ├── pytorch_model_finetuned.bin        # 微调后模型
    └── {timestamp}/
        ├── training.log                   # 详细日志
        └── train.sh                       # 运行命令备份
```

---

## 参数调优指南

### 剪枝率选择

| 剪枝率 | 参数减少 | PPL 影响 | 推荐场景 |
|--------|----------|----------|----------|
| 15% | 少 | 小（易恢复） | 保守剪枝 |
| 25% | 中等 | 中等 | **推荐** |
| 35% | 多 | 大（难恢复） | 激进剪枝 |

### LoRA 配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lora_r` | 8 | 秩（4-16，越大越强但参数越多） |
| `lora_alpha` | 16 | 缩放系数（通常为 r 的 2 倍） |
| `lora_dropout` | 0.05 | Dropout 率 |
| `lr` | 2e-4 | 学习率（比全参数微调高） |

### 显存优化

如果显存不足：

```bash
# 减少样本数
--samples 500 \
--seq_len 256

# 增加梯度累积
--batch_size 1 \
--grad_accum 8

# 使用更小的 LoRA 秩
--lora_r 4
```

---

## 常见问题

### Q1: 剪枝后 PPL 很高（如 80）正常吗？

**A**: 对于 25% 非均衡剪枝，PPL = 80 是正常的。关键是 GQA 比例（4:1）必须保持，否则 PPL 会飙升到几万。使用 LoRA 微调可以将 PPL 降到 35 左右。

### Q2: 微调时出现 NaN Loss 怎么办？

**A**: 使用 LoRA 微调替代全参数微调：
```bash
--use_lora \
--lora_target_attention \
--lora_target_mlp \
--lr 2e-4
```

### Q3: 如何选择 seq_len？

**A**: 训练和评估应该使用**相同的 seq_len**：
- 快速实验：128
- 生产环境：512
- 保持一致很重要！

### Q4: 如何查看日志？

**A**:
```bash
# 查看最新日志
tail -f prune_log/{experiment_name}/*/training.log

# 查看 PPL 结果
grep "PPL" prune_log/{experiment_name}/*/training.log
```

---

## 性能基准

基于 LLaMA-3-8B-Instruct (25% 剪枝)：

| 阶段 | PPL (wikitext2) | 参数量 | 备注 |
|------|-----------------|--------|------|
| 原始模型 | ~12 | 8.03B | 基准 |
| 剪枝后 | ~80 | 6.02B (-25%) | 保持 4:1 GQA |
| LoRA 微调后 | ~35 | 6.02B | 55% PPL 降低 |

---

## 进阶用法

### 批量评估多个实验

```bash
#!/bin/bash
for ratio in 15 25 35; do
    python evaluate_models.py \
        --finetuned_model prune_log/exp_${ratio}pct/pytorch_model_finetuned.bin \
        --seq_len 512 \
        --save_results results_${ratio}pct.json
done
```

### 重用层重要性分析

```bash
# 首次运行：分析并保存
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name cache_importance \
    --pruning_ratio 0.25 \
    --save_model

# 后续运行：直接使用缓存
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name exp2 \
    --pruning_ratio 0.30 \
    --skip_importance_analysis \
    --layer_importance_config prune_log/cache_importance/layer_importance_config.json \
    --save_model
```

---

## 文档

- **CLAUDE.md**: AI 助手开发指南（详细的代码库文档）
- **README.md**: 本文件（脚本使用说明）

---

## 技术细节

**核心技术**：
- Taylor 重要性：基于梯度的参数重要性评估
- GQA 感知：保持 Q:KV = 4:1 的架构约束
- 非均衡剪枝：根据层重要性动态调整剪枝率
- LoRA 微调：低秩适应，高效恢复性能

**关键模块**：
- `LLMPruner/methods/gqa_aware.py`: GQA 感知剪枝算法
- `LLMPruner/importance/layer_analyzer.py`: 层重要性分析
- `LLMPruner/trainer/finetuner.py`: LoRA/全参数微调
- `LLMPruner/evaluator/ppl.py`: 困惑度评估

---

**最后更新**: 2025-11-17
