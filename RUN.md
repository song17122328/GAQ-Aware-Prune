# 启动命令文档

本文档记录如何运行 GQA-Aware 剪枝流程。

---

## 快速启动

### 基本命令（推荐）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_gqa_aware_pruned_v3 \
    --pruning_ratio 0.25 \
    --importance_method removal \
    --importance_samples 50 \
    --pruning_strategy inverse \
    --alpha 1.0 \
    --min_pruning_rate 0.15 \
    --max_pruning_rate 0.5 \
    --layer_start 0 \
    --layer_end 32 \
    --num_examples 10 \
    --head_dim 128 \
    --gqa_ratio 4 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --max_seq_len 128
```

**注意**: 设备会自动选择最优 GPU，无需手动指定

---

## 参数说明

### 必需参数

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--base_model` | `/newdata/LLMs/Llama-3-8B-Instruct` | 原始模型路径 |
| `--save_ckpt_log_name` | `llama3_gqa_aware_pruned_v3` | 日志和模型保存目录名 |

### 剪枝参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pruning_ratio` | `0.25` | 整体平均剪枝率（25%）|
| `--min_pruning_rate` | `0.15` | 最小剪枝率（至少剪1个GQA组，对应8个KV heads的12.5%）|
| `--max_pruning_rate` | `0.5` | 最大剪枝率（50%）|

### 层重要性评估

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--importance_method` | `removal` | 评估方法：`removal`（逐层移除法）或 `activation`（激活值法）|
| `--importance_samples` | `50` | 用于评估层重要度的样本数量 |
| `--skip_importance_analysis` | - | 跳过层重要度分析，使用已保存的配置（加快调试）|
| `--importance_config` | `layer_importance_config.json` | 层重要度配置文件路径 |

### 非均衡剪枝策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pruning_strategy` | `inverse` | `inverse`（重要层剪少）/ `proportional`（重要层剪多）/ `uniform`（均匀剪）|
| `--alpha` | `1.0` | 重要性权重系数，越大差异越明显（0.5-3.0）|

### 剪枝范围

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layer_start` | `0` | 剪枝起始层索引 |
| `--layer_end` | `32` | 剪枝结束层索引（Llama-3-8B 共32层）|

### GQA 配置（固定，不建议修改）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--head_dim` | `128` | 每个 attention head 的维度（Llama-3 固定）|
| `--gqa_ratio` | `4` | Q:KV 比例（Llama-3 为 4:1）|

### 设备选择

**自动选择**: 程序会自动选择显存最多的GPU，无需手动指定设备。

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_examples` | `10` | Taylor 重要性评估的样本数 |
| `--max_seq_len` | `128` | PPL 评估最大序列长度 |
| `--prune_mlp` | - | 是否也剪枝 MLP（使用 Taylor importance）|
| `--save_model` | - | 是否保存模型 |
| `--test_after_prune` | - | 剪枝后是否立即评估 PPL |

---

## 常用场景

### 场景 1：标准剪枝（25% 参数）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_25pct \
    --pruning_ratio 0.25 \
    --importance_method removal \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**预期结果**：
- 参数减少：~20-25%
- PPL 退化：<5%
- 所有层保持 4:1 GQA 比例

---

### 场景 2：激进剪枝（30% 参数）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_30pct \
    --pruning_ratio 0.30 \
    --min_pruning_rate 0.20 \
    --max_pruning_rate 0.60 \
    --importance_method removal \
    --pruning_strategy inverse \
    --alpha 1.5 \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**注意**：更高的剪枝率可能导致 PPL 退化增加。

---

### 场景 3：仅剪枝 Attention（保留 MLP）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_attn_only \
    --pruning_ratio 0.25 \
    --importance_method removal \
    --pruning_strategy inverse \
    --save_model \
    --test_after_prune
```

**说明**：移除 `--prune_mlp` 标志

---

### 场景 4：仅剪枝中间层（保护首尾层）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_middle_layers \
    --pruning_ratio 0.25 \
    --layer_start 5 \
    --layer_end 27 \
    --importance_method removal \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**说明**：跳过前5层和后5层

---

### 场景 5：两阶段剪枝（重用层重要性分析）

#### 阶段 1：分析层重要性（耗时 ~30分钟）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name importance_analysis \
    --importance_method removal \
    --importance_samples 100
```

**说明**：不加 `--save_model`，只分析层重要性

#### 阶段 2：使用保存的重要性快速剪枝（耗时 ~10分钟）

```bash
# 尝试 25% 剪枝
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_25pct \
    --skip_importance_analysis \
    --importance_config prune_log/importance_analysis/layer_importance_config.json \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune

# 尝试 30% 剪枝（使用相同的重要性分析）
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_30pct \
    --skip_importance_analysis \
    --importance_config prune_log/importance_analysis/layer_importance_config.json \
    --pruning_ratio 0.30 \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**优点**：避免重复计算层重要性，快速尝试不同剪枝率

---

### 场景 6：调试模式（快速验证）

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name debug_test \
    --pruning_ratio 0.25 \
    --importance_method removal \
    --importance_samples 10 \
    --num_examples 5 \
    --layer_start 10 \
    --layer_end 15 \
    --test_after_prune
```

**说明**：
- 减少样本数（10个样本评估层重要性，5个样本计算梯度）
- 只剪枝 5 层（10-15）
- 快速验证流程是否正常

---

## 输出文件

### 日志和模型保存位置

```
prune_log/
└── {save_ckpt_log_name}/
    ├── description.txt                    # 配置参数
    ├── pytorch_model.bin                  # 最佳模型（如果使用 --save_model）
    ├── layer_importance_config.json       # 层重要性配置
    ├── pruning_strategy.png               # 剪枝策略可视化
    └── {timestamp}/
        ├── description.txt                # 本次运行配置
        ├── training.log                   # 详细日志
        └── train.sh                       # 运行命令备份
```

### 查看结果

```bash
# 查看日志
cat prune_log/llama3_gqa_aware_pruned_v3/*/training.log

# 查看配置
cat prune_log/llama3_gqa_aware_pruned_v3/description.txt

# 查看层重要性
cat prune_log/llama3_gqa_aware_pruned_v3/layer_importance_config.json

# 查看剪枝策略图
# （需要图片查看器）
eog prune_log/llama3_gqa_aware_pruned_v3/pruning_strategy.png
```

---

## 性能优化

### 减少内存占用

如果遇到 CUDA OOM 错误：

```bash
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_low_mem \
    --pruning_ratio 0.25 \
    --importance_samples 20 \      # 减少到 20
    --num_examples 5 \             # 减少到 5
    --max_seq_len 64 \             # 减少到 64
    --prune_mlp \
    --save_model
```

### 加速调试

```bash
# 1. 首先分析层重要性（只需运行一次）
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name layer_analysis \
    --importance_method removal \
    --importance_samples 50

# 2. 后续直接使用保存的分析结果
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name quick_prune \
    --skip_importance_analysis \
    --importance_config prune_log/layer_analysis/layer_importance_config.json \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

---

## 环境变量

如果模型路径固定，可以使用环境变量：

```bash
# 设置环境变量
export MODEL_PATH="/newdata/LLMs/Llama-3-8B-Instruct"

# 使用环境变量
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model $MODEL_PATH \
    --save_ckpt_log_name llama3_pruned \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

或创建配置文件 `config.sh`:

```bash
#!/bin/bash
# config.sh

export MODEL_PATH="/newdata/LLMs/Llama-3-8B-Instruct"
export SAVE_NAME="llama3_gqa_aware_pruned_v3"
export PRUNING_RATIO=0.25
```

然后使用：

```bash
source config.sh

python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model $MODEL_PATH \
    --save_ckpt_log_name $SAVE_NAME \
    --pruning_ratio $PRUNING_RATIO \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

---

## 故障排除

### 1. CUDA Out of Memory

**解决方案**：
```bash
--importance_samples 20 \
--num_examples 5 \
--max_seq_len 64
```

### 2. 数据集下载失败

**解决方案**：
- 检查网络连接
- 使用代理：`export HF_ENDPOINT=https://hf-mirror.com`
- 手动下载数据集到 `~/.cache/huggingface/datasets/`

### 3. Tokenizer Padding Token 错误

**解决方案**：已在代码中自动修复，无需手动处理

### 4. PPL 为 NaN 或异常高

**检查**：
- 确认模型路径正确
- 检查剪枝率是否过高（建议 ≤0.3）
- 查看日志中的层级剪枝情况

---

## 预期结果

### 标准配置（25% 剪枝）

- **参数减少**：20-25%
- **PPL 退化**：<5% (vs 基线)
- **运行时间**：~1-2 小时（取决于硬件）
- **内存需求**：~24GB VRAM

### 对比旧方法

| 方法 | PPL (WikiText-2) | 参数减少 | GQA 保持 |
|------|------------------|----------|----------|
| 旧方法 (torch_pruning) | 718,107 ❌ | 25% | ❌ 破坏 |
| 新方法 (GQA-aware) | ~12-13 ✅ | 25% | ✅ 保持 4:1 |

---

## 更多信息

- 详细文档：`CLAUDE.md`
- 数据集说明：`DATASETS.md`
- LLMPruner 模块：`LLMPruner/README.md`
- 原始中文文档：`读我.md`
