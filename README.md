# GAQ-Aware-Prune

GQA（Grouped Query Attention）感知的 LLaMA-3 模型剪枝工具

## 项目简介

本项目实现了针对 LLaMA-3 模型的智能剪枝方案：
- **GQA感知剪枝**：保持 4:1 的 Q:KV 头比例
- **非均衡层级剪枝**：重要层剪枝少，不重要层剪枝多
- **参数量分布控制**：精确控制 Attention 和 MLP 的剪枝比例
- **LoRA微调恢复**：剪枝后使用 LoRA 高效恢复性能

---

## 快速开始

### 1. 对比不同剪枝分布（推荐：找到最佳比例）

**第一步：测试不同的 Attention:MLP 剪枝比例**

> **重要**：使用 `--test_original_ppl` 和 `--test_after_prune` 来测量 PPL，对比不同分布的效果

```bash
# 实验1: 均衡剪枝（Attention:MLP = 5:5，x+y=10）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_5_5 \
    --pruning_ratio 0.25 \
    --pruning_distribution 5:5 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验2: 根据实际参数量分布（Attention:MLP = 2:8）
# LLaMA-3-8B 的 Attention 约占 19.2%，MLP 约占 80.8%
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_2_8 \
    --pruning_ratio 0.25 \
    --pruning_distribution 2:8 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_1_9 \
    --pruning_ratio 0.25 \
    --pruning_distribution 1:9 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验3: MLP 占主导（Attention:MLP = 3:7）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_3_7 \
    --pruning_ratio 0.25 \
    --pruning_distribution 3:7 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验4: Attention 占主导（Attention:MLP = 7:3）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_7_3 \
    --pruning_ratio 0.25 \
    --pruning_distribution 7:3 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验5: 只剪枝 MLP（Attention:MLP = 0:10）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_0_10 \
    --pruning_ratio 0.25 \
    --pruning_distribution 0:10 \
    --test_original_ppl \
    --test_after_prune \
    --save_model

# 实验6: 只剪枝 Attention（Attention:MLP = 10:0）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name dist_10_0 \
    --pruning_ratio 0.25 \
    --pruning_distribution 10:0 \
    --test_original_ppl \
    --test_after_prune \
    --save_model
```

**第二步：对比 PPL 结果**

查看日志文件对比各实验的 PPL：
```bash
# 查看所有实验的 PPL（从日志中提取）
for exp in dist_5_5 dist_2_8 dist_3_7 dist_7_3 dist_0_10 dist_10_0; do
    echo "=== $exp ==="
    grep "原始模型 PPL\|剪枝后 PPL" prune_log/$exp/*/training.log | head -2
done
```

**第三步：选择最佳分布并微调**

根据 PPL 对比结果，选择最佳分布进行微调：
```bash
# 假设 dist_2_8 效果最好
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name best_dist_2_8_finetuned \
    --pruning_ratio 0.25 \
    --pruning_distribution 2:8 \
    --test_original_ppl \
    --test_after_prune \
    --save_model \
    --finetune \
    --finetune_method lora \
    --finetune_lr 2e-4 \
    --finetune_epochs 3 \
    --finetune_samples 1000 \
    --finetune_seq_len 512
```

**工作原理**：
- `--pruning_distribution x:y` 表示 Attention 和 MLP **实际剪掉的参数量之比**（x+y=10）
- 例如：`--pruning_ratio 0.25 --pruning_distribution 2:8`
  - LLaMA-3-8B: Attention 占 19.2%，MLP 占 80.8%
  - 总剪枝量 = Attention+MLP 的 25%
  - Attention 剪枝量 = 总剪枝量 × 2/10 = 20%
  - MLP 剪枝量 = 总剪枝量 × 8/10 = 80%
- 脚本会自动根据层重要性分配每层的剪枝率

**关键参数**：
- `--base_model`: 原始模型路径
- `--pruning_ratio`: 总体剪枝率（0.25 = 剪掉 Attention+MLP 总参数的 25%）
- `--pruning_distribution`: Attention:MLP 的剪枝参数量比例（x:y 格式，x+y=10）
- `--test_original_ppl`: 评估原始模型 PPL（作为 baseline）
- `--test_after_prune`: 评估剪枝后 PPL（对比不同分布）
- `--save_model`: 保存剪枝后的模型

**输出**：
- 剪枝模型：`prune_log/{experiment_name}/pytorch_model.bin`
- 详细日志：`prune_log/{experiment_name}/{timestamp}/training.log`
- 剪枝配置：`prune_log/{experiment_name}/pruning_strategy_config.json`

---

### 2. 剪枝模型（不测试 PPL）

如果已知最佳分布，可以直接剪枝：

```bash
# 使用最佳分布 2:8 进行剪枝
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_2_8 \
    --pruning_ratio 0.25 \
    --pruning_distribution 2:8 \
    --save_model
```

---

### 3. LoRA 微调

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_balanced/pytorch_model.bin \
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
    --pruned_model prune_log/llama3_pruned_balanced/pytorch_model.bin \
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
    --pruning_distribution 5:5 \
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
性能对比总结
================================================================================
原始模型: {'wikitext2 (wikitext-2-raw-v1)': 12.34}
剪枝后（微调前）: {'wikitext2 (wikitext-2-raw-v1)': 80.85}
微调后: {'wikitext2 (wikitext-2-raw-v1)': 35.82}
微调改善（vs剪枝后）: 55.7%
相对原模型退化: +190.2%
```

---

## 完整工作流程（一键运行）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_full_pipeline \
    --pruning_ratio 0.25 \
    --pruning_distribution 5:5 \
    --test_original_ppl \
    --test_after_prune \
    --eval_seq_len 512 \
    --save_model \
    --finetune \
    --finetune_method lora \
    --finetune_lr 2e-4 \
    --finetune_epochs 3 \
    --finetune_samples 1000 \
    --finetune_seq_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_attention \
    --lora_target_mlp
```

这将自动执行：
1. 评估原模型 PPL
2. 层重要性分析
3. 计算 Attention 和 MLP 各层剪枝率
4. 执行剪枝（保存模型）
5. 评估剪枝后 PPL
6. LoRA 微调
7. 评估微调后 PPL
8. 输出完整对比报告

---

## 常用参数组合

### 批量对比不同剪枝分布（Shell脚本）

使用 Shell 脚本自动化对比不同分布：

```bash
#!/bin/bash
# 对比不同的 Attention:MLP 剪枝分布（x+y=10）

MODEL_PATH="/newdata/LLMs/Llama-3-8B-Instruct"
PRUNING_RATIO=0.25

# 定义要测试的分布
distributions=("10:0" "7:3" "5:5" "3:7" "2:8" "0:10")

for dist in "${distributions[@]}"; do
    # 将 : 替换为 _ 作为实验名称
    exp_name="dist_${dist//:/_}"

    echo "====================================="
    echo "Running experiment: $exp_name"
    echo "Distribution: $dist"
    echo "====================================="

    python llama3_unbalanced_pruning_gqa_aware.py \
        --base_model $MODEL_PATH \
        --save_ckpt_log_name $exp_name \
        --pruning_ratio $PRUNING_RATIO \
        --pruning_distribution $dist \
        --test_original_ppl \
        --test_after_prune \
        --save_model
done

# 汇总所有实验的 PPL 结果
echo ""
echo "====================================="
echo "PPL Summary"
echo "====================================="
for dist in "${distributions[@]}"; do
    exp_name="dist_${dist//:/_}"
    echo ""
    echo "=== $exp_name (Attention:MLP = $dist) ==="
    grep "原始模型 PPL\|剪枝后 PPL" prune_log/$exp_name/*/training.log | head -2
done
```

**推荐的测试分布**（x+y=10）：
| 分布 | Attention | MLP | 说明 |
|------|-----------|-----|------|
| 10:0 | 100% | 0% | 只剪 Attention，保留所有 MLP |
| 7:3 | 70% | 30% | Attention 占主导 |
| 5:5 | 50% | 50% | 均衡剪枝 |
| 3:7 | 30% | 70% | MLP 占主导 |
| 2:8 | 20% | 80% | 根据 LLaMA-3 实际参数量分布 |
| 0:10 | 0% | 100% | 只剪 MLP，保留所有 Attention |

### 快速测试（Debug）

```bash
# 剪枝测试（只处理部分层）
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name debug_test \
    --pruning_ratio 0.25 \
    --pruning_distribution 5:5 \
    --layer_start 10 \
    --layer_end 15 \
    --layer_importance_samples 10

# 微调测试（少量样本）
python test_finetuning.py \
    --model_path prune_log/debug_test/pytorch_model.bin \
    --save_name debug_finetune \
    --use_lora \
    --samples 100 \
    --epochs 1 \
    --test_after
```

---

## 核心参数说明

### 剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pruning_ratio` | float | 0.25 | 总体剪枝率（0.25 = 剪掉25%参数） |
| `--pruning_distribution` | str | "5:5" | Attention:MLP 剪枝参数量比例 |
| `--pruning_strategy` | str | "inverse" | 剪枝策略：inverse/proportional/uniform |
| `--layer_importance_weight` | float | 1.0 | 层间剪枝率差异系数（越大差异越明显） |
| `--min_pruning_rate` | float | 0.15 | 单层最小剪枝率 |
| `--max_pruning_rate` | float | 0.5 | 单层最大剪枝率 |

### 层重要性评估

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--layer_importance_method` | str | "removal" | 评估方法：removal/activation |
| `--layer_importance_samples` | int | 50 | 用于评估层重要性的样本数 |
| `--skip_importance_analysis` | flag | False | 跳过分析，从文件加载 |

### 通道/头重要性评估

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--channel_importance_samples` | int | 10 | 用于计算Taylor重要性的样本数 |
| `--taylor_seq_len` | int | 128 | Taylor重要性计算时的序列长度 |

### 微调参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--finetune` | flag | False | 是否进行微调 |
| `--finetune_method` | str | "full" | 微调方法：full/lora |
| `--finetune_lr` | float | 1e-5 | 学习率（LoRA: 2e-4, 全参数: 1e-5） |
| `--finetune_epochs` | int | 1 | 微调轮数 |
| `--finetune_samples` | int | 500 | 微调样本数 |
| `--lora_r` | int | 8 | LoRA秩（建议4-16） |
| `--lora_alpha` | int | 16 | LoRA缩放系数（通常为r的2倍） |

### 评估参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_original_ppl` | flag | False | 剪枝前评估原模型PPL |
| `--test_after_prune` | flag | False | 剪枝后评估PPL |
| `--eval_seq_len` | int | 128 | PPL评估序列长度 |

---

## 输出文件说明

```
prune_log/{experiment_name}/
├── description.txt                         # 完整配置参数
├── pruning_strategy_config.json            # 剪枝策略详细配置
│   ├── attention_pruning_rates            # Attention各层剪枝率
│   ├── mlp_pruning_rates                  # MLP各层剪枝率
│   ├── layer_importance                   # 层重要性分数
│   ├── pruning_distribution               # 剪枝分布比例
│   └── target_pruning_ratio               # 目标剪枝率
├── pruning_strategy.png                    # 剪枝策略可视化
├── pytorch_model.bin                       # 剪枝后模型
├── pytorch_model_finetuned.bin             # 微调后模型（如果启用）
└── {timestamp}/
    ├── training.log                        # 详细日志
    └── train.sh                            # 运行命令备份
```

---

## 日志解读

### 参数统计输出

```
参与剪枝的层范围: [0, 32)
Attention 总参数量: 536,870,912 (40.0%)
MLP 总参数量: 805,306,368 (60.0%)
可剪枝总参数量: 1,342,177,280

总目标剪枝参数量: 335,544,320 (25.0%)
  -> Attention 目标剪枝量: 167,772,160 (31.25% of Attention)
  -> MLP 目标剪枝量: 167,772,160 (20.83% of MLP)
```

**说明**：
- Attention 和 MLP 的参数量不同，所以即使 5:5 分配，实际的剪枝率也不同
- 这就是为什么需要基于参数量的精确分配

### 剪枝执行输出

```
处理 Layer 0
  Attention 剪枝率: 0.31%, MLP 剪枝率: 0.21%
  结果: Attention: 32Q:8KV → 28Q:7KV, MLP: 14336→11392
```

---

## 典型场景与参数推荐

| 场景 | pruning_ratio | pruning_distribution | 说明 |
|------|---------------|----------------------|------|
| **推荐：根据模型实际分布** | 0.25 | **2:8** | LLaMA-3-8B 实际参数量分布（Attention 19.2%, MLP 80.8%） |
| 探索最佳分布 | 0.25 | 5:5, 7:3, 3:7, 2:8 | 对比不同分布找最优 |
| MLP 占主导 | 0.25 | 2:8 或 3:7 | 更多保留 Attention |
| 均衡剪枝 | 0.25 | 5:5 | Attention 和 MLP 剪掉相同参数量 |
| Attention 占主导 | 0.25 | 7:3 或 8:2 | 更多保留 MLP |
| 极端测试：只剪 MLP | 0.25 | 0:10 | 保留所有 Attention |
| 极端测试：只剪 Attention | 0.25 | 10:0 | 保留所有 MLP |
| 激进剪枝 | 0.40 | 2:8 | 高剪枝率 |
| 保守剪枝 | 0.15 | 2:8 | 低剪枝率 |

**注意**：所有分布 x:y 满足 x+y=10

---

## FAQ

### Q1: 如何选择 pruning_distribution？

**建议**：先运行多个实验对比不同分布的 PPL：

```bash
# 推荐测试 6 种分布（x+y=10）
for dist in "10:0" "7:3" "5:5" "3:7" "2:8" "0:10"; do
    python llama3_unbalanced_pruning_gqa_aware.py \
        --base_model /newdata/LLMs/Llama-3-8B-Instruct \
        --save_ckpt_log_name dist_${dist//:/_} \
        --pruning_ratio 0.25 \
        --pruning_distribution $dist \
        --test_original_ppl \
        --test_after_prune \
        --save_model
done

# 查看结果对比
for dist in "10:0" "7:3" "5:5" "3:7" "2:8" "0:10"; do
    echo "=== Attention:MLP = $dist ==="
    grep "剪枝后 PPL" prune_log/dist_${dist//:/_}/*/training.log
done
```

**特别推荐**：2:8 分布与 LLaMA-3-8B 的实际参数量分布接近（Attention 19.2%, MLP 80.8%）

### Q2: pruning_distribution 和实际剪枝率的关系？

- `pruning_distribution x:y` 控制的是**参数量**的分配比例（x+y=10）
- 例如：`--pruning_distribution 2:8` 表示：
  - Attention 剪掉的参数量 = 总剪枝量 × 2/10 = 20%
  - MLP 剪掉的参数量 = 总剪枝量 × 8/10 = 80%
- 由于 Attention 和 MLP 的总参数量不同，实际的剪枝率也不同：
  - 如果 Attention 占 19.2%，MLP 占 80.8%
  - 使用 2:8 分布时：
    - Attention 剪枝率 ≈ 26%
    - MLP 剪枝率 ≈ 25%

### Q3: 为什么推荐 2:8 分布？

对于 LLaMA-3-8B：
- Attention 参数量约占 19.2%
- MLP 参数量约占 80.8%
- 使用 2:8 分布时，Attention 和 MLP 的剪枝率接近，更加均衡

### Q4: 如何复现某个实验？

所有配置都保存在 `pruning_strategy_config.json`，包括：
- 各层的 Attention 和 MLP 剪枝率
- 层重要性分数
- 剪枝分布比例
- 总参数量和剪枝参数量

---

## 项目结构

```
GAQ-Aware-Prune/
├── llama3_unbalanced_pruning_gqa_aware.py  # 主剪枝脚本
├── test_finetuning.py                      # 独立微调脚本
├── evaluate_models.py                      # 模型评估对比脚本
├── README.md                               # 本文档
├── CLAUDE.md                               # AI助手开发指南
├── PARAMETERS_GUIDE.md                     # 参数选择详细指南
└── LLMPruner/                              # 剪枝库
    ├── methods/                            # 剪枝方法
    ├── importance/                         # 重要性分析
    ├── evaluator/                          # 模型评估
    ├── trainer/                            # 微调模块
    └── datasets/                           # 数据加载
```

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{gaq_aware_prune,
  title={GAQ-Aware-Prune: GQA-Aware Structured Pruning for LLaMA-3},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/GAQ-Aware-Prune}}
}
```

---

## License

MIT License
