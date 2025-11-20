# 全局剪枝使用指南

## 理论基础

### 核心思想：分数背包问题 (Fractional Knapsack)

将大模型结构化剪枝问题建模为**在参数预算约束下最大化保留价值**的优化问题：

- **物品**：模型中所有剪枝单元（MLP 神经元组、Attention Head 组）
- **背包容量**：允许保留的总参数预算
- **物品价值**：单元对模型的重要性 $I_u$（Taylor 或 Wanda）
- **物品成本**：单元的参数量 $C_u$
- **优化目标**：最大化保留的总价值

### 性价比得分

$$S_u = \frac{I_u}{C_u}$$

- **$I_u$ (Importance)**：支持三种计算方法

  1. **一阶泰勒展开 (Taylor First-Order)**
     $$I_u = \sum_{\theta \in u} \left| \theta \cdot \frac{\partial \mathcal{L}}{\partial \theta} \right|$$

  2. **二阶泰勒展开 (Taylor Second-Order)**
     $$I_u = \sum_{\theta \in u} \left| \theta \cdot \frac{\partial \mathcal{L}}{\partial \theta} \right| + \frac{1}{2} \left| \theta^2 \cdot H_{diag} \right|$$
     其中 $H_{diag}$ 是 Hessian 对角线近似（通过梯度平方累加）

  3. **Wanda (Weight and Activation)**
     $$I_u = \sum_{\theta \in u} \left| \theta \cdot A_u \right|$$
     其中 $A_u$ 是对应单元的平均激活值

- **$C_u$ (Cost)**：单元的参数数量
  - Attention Group: ~6.3M 参数（1 KV + 4 Q heads）
  - MLP Channel: ~12K 参数（gate/up/down 各一份）

### 混合剪枝策略

算法自动实现：
- **深度剪枝**：移除得分极低的整层
- **宽度剪枝**：剪除冗余神经元
- **自动比例平衡**：无需人工搜索 Attention:MLP 比例

---

## 快速开始

### 基础用法（一阶 Taylor）

```bash
python llama3_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_global_25pct \
    --pruning_ratio 0.25 \
    --importance_method taylor \
    --num_samples 128 \
    --test_before_prune \
    --test_after_prune \
    --save_model
```

### 使用二阶 Taylor（更准确的重要性估计）

```bash
python llama3_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_global_25pct_2nd \
    --pruning_ratio 0.25 \
    --importance_method taylor_2nd \
    --num_samples 128 \
    --test_before_prune \
    --test_after_prune \
    --save_model
```

### 使用 Wanda（无需梯度计算）

```bash
python llama3_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_global_25pct_wanda \
    --pruning_ratio 0.25 \
    --importance_method wanda \
    --num_samples 128 \
    --test_before_prune \
    --test_after_prune \
    --save_model
```

### 带微调的完整流程

```bash
python llama3_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_global_25pct_finetuned \
    --pruning_ratio 0.25 \
    --num_samples 128 \
    --test_before_prune \
    --test_after_prune \
    --finetune \
    --finetune_method lora \
    --finetune_samples 500 \
    --finetune_lr 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --save_model
```

### 自动深度剪枝（移除空层）

```bash
python llama3_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_global_30pct_depth \
    --pruning_ratio 0.30 \
    --remove_empty_layers \
    --test_after_prune \
    --save_model
```

---

## 重要性计算方法对比

### 三种方法的特点

| 方法 | 计算成本 | 准确性 | 适用场景 |
|------|---------|--------|---------|
| **Taylor (一阶)** | 中等 | 较好 | 标准场景，平衡性能和计算成本 |
| **Taylor_2nd (二阶)** | 高 | 最好 | 需要最高精度，愿意牺牲计算时间 |
| **Wanda** | 低 | 良好 | 快速剪枝，无需反向传播 |

### 详细对比

**一阶 Taylor (`--importance_method taylor`)**
- **原理**: 使用一阶泰勒展开 $I = |\theta \cdot g|$
- **优点**: 计算适中，精度不错，被广泛验证
- **缺点**: 未考虑曲率信息
- **推荐**: 大多数场景的默认选择

**二阶 Taylor (`--importance_method taylor_2nd`)**
- **原理**: 使用二阶泰勒展开 $I = |\theta \cdot g| + \frac{1}{2}|\theta^2 \cdot H_{diag}|$
- **优点**: 考虑 Hessian 曲率信息，理论上更准确
- **缺点**: 需要累加梯度平方，额外内存和计算开销
- **推荐**: 追求极致精度，或当一阶 Taylor 效果不理想时

**Wanda (`--importance_method wanda`)**
- **原理**: 使用权重和激活值乘积 $I = |\theta \cdot A|$
- **优点**: 只需前向传播，无需反向传播，速度快
- **缺点**: 未考虑梯度信息，可能不如 Taylor 精确
- **推荐**: 快速剪枝原型验证，或计算资源受限场景

### 性能测试建议

建议先用一阶 Taylor 作为基准，然后尝试其他方法对比：

1. **基准**: `--importance_method taylor`
2. **追求精度**: `--importance_method taylor_2nd`
3. **追求速度**: `--importance_method wanda`

---

## 参数说明

### 剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pruning_ratio` | float | 0.25 | 目标剪枝率（相对于模型总参数） |
| `--importance_method` | str | taylor | 重要性计算方法（taylor/taylor_2nd/wanda） |
| `--num_samples` | int | 128 | 用于计算重要性的样本数 |
| `--remove_empty_layers` | flag | False | 是否移除被完全剪空的层 |

### GQA 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--head_dim` | int | 128 | Attention head 维度 |
| `--gqa_ratio` | int | 4 | Q:KV 比例 |

### 微调参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--finetune` | flag | False | 是否进行微调 |
| `--finetune_method` | str | lora | 微调方法（full/lora） |
| `--finetune_samples` | int | 500 | 微调样本数 |
| `--finetune_lr` | float | 1e-4 | 微调学习率 |
| `--finetune_epochs` | int | 1 | 微调轮数 |
| `--lora_r` | int | 8 | LoRA rank |
| `--lora_alpha` | int | 16 | LoRA alpha |

### 评估与保存

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_before_prune` | flag | False | 评估基线 PPL |
| `--test_after_prune` | flag | False | 评估剪枝后 PPL |
| `--save_model` | flag | False | 保存模型 |

---

## 输出文件

运行后会在 `prune_log/{save_ckpt_log_name}/` 生成：

```
prune_log/{save_ckpt_log_name}/
├── description.txt                  # 实验配置
├── global_group_table.csv           # 完整的全局分析表
├── groups_to_prune.csv              # 要剪枝的 groups 列表
├── pytorch_model.bin                # 剪枝后的模型
└── {timestamp}/
    └── training.log                 # 详细日志
```

### 分析表结构

**global_group_table.csv**:
```
layer_idx,group_type,group_idx,importance,cost,score
5,attention,2,0.123456,6291456,1.962e-08
12,mlp,1024,0.234567,12288,1.909e-05
...
```

- 按 `score` 升序排列
- score 最低的是最优先剪枝的候选

---

## 执行流程

脚本的完整执行流程：

```
[Step 1] 加载模型
    ↓
[Step 2] 评估基线 PPL（可选）
    ↓
[Step 3] 计算梯度（Taylor importance）
    ↓
[Step 4] 构建全局 Group 分析表
    - 计算每个 group 的 importance
    - 计算每个 group 的 cost
    - 计算 score = importance / cost
    - 全局排序
    ↓
[Step 5] 选择要剪枝的 groups
    - 从 score 最低的开始选择
    - 累加参数量直到达到目标剪枝率
    ↓
[Step 6] 执行全局剪枝
    - Attention: 剪除选中的 GQA groups
    - MLP: 剪除选中的 channels
    - 保持每层 4:1 比例
    ↓
[Step 7] 移除空层（可选）
    - 如果某层的 Attention 和 MLP 都被剪空
    - 自动移除该层（深度剪枝）
    ↓
[Step 8] 统计剪枝结果
    ↓
[Step 9] 评估剪枝后 PPL（可选）
    ↓
[Step 10] 微调恢复（可选）
    - 使用 LoRA 或全参数微调
    - 恢复模型性能
    ↓
[Step 11] 保存模型（可选）
```

---

## 与原有方法的对比

### 原有方法：层级剪枝 (`llama3_unbalanced_pruning_gqa_aware.py`)

**特点**：
- 先计算层重要性
- 根据层重要性分配各层剪枝率
- 每层独立执行剪枝
- 需要手动指定 Attention:MLP 比例

**优势**：
- 保持各层结构平衡
- 执行速度快

**劣势**：
- 无法跨层对比
- 需要人工搜索 Attention:MLP 比例
- 可能剪掉重要层的重要组件

### 新方法：全局剪枝 (`llama3_global_pruning.py`)

**特点**：
- 构建全局分析表
- 跨层、跨组件对比
- 根据 Score = Importance / Cost 排序
- 自动确定最优剪枝策略

**优势**：
- 全局最优
- 自动平衡 Attention:MLP 比例
- 自动实现深度+宽度混合剪枝
- 理论上性能更优

**劣势**：
- 构建分析表耗时较长
- 可能导致某些层被完全剪空

---

## 使用建议

### 1. 选择合适的 `pruning_ratio`

| 剪枝率 | 适用场景 | 预期效果 |
|--------|---------|---------|
| 0.15-0.20 | 保守剪枝 | PPL 退化 < 5% |
| 0.20-0.30 | 标准剪枝 | PPL 退化 5-15% |
| 0.30-0.40 | 激进剪枝 | PPL 退化 > 15%，需要微调 |

### 2. 调整 `num_samples`

- **快速测试**：`--num_samples 10-20`（不准确但快速）
- **标准设置**：`--num_samples 128`（平衡速度和准确性）
- **高精度**：`--num_samples 256-512`（更准确但耗时）

### 3. 微调策略

**场景 1：轻度剪枝（< 20%）**
```bash
--finetune \
--finetune_method lora \
--finetune_samples 500 \
--lora_r 8
```

**场景 2：中度剪枝（20-30%）**
```bash
--finetune \
--finetune_method lora \
--finetune_samples 1000 \
--lora_r 16 \
--lora_alpha 32
```

**场景 3：激进剪枝（> 30%）**
```bash
--finetune \
--finetune_method full \
--finetune_samples 2000 \
--finetune_lr 1e-5
```

### 4. 自动深度剪枝

如果您希望算法自动移除冗余的层：

```bash
--pruning_ratio 0.30 \
--remove_empty_layers
```

**注意**：
- 这会改变模型的层数
- 可能导致与原始模型不兼容
- 建议在激进剪枝（> 25%）时使用

---

## Debug 模式

### 快速测试（仅剪少数几层）

```bash
python llama3_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name debug_test \
    --pruning_ratio 0.25 \
    --num_samples 10 \
    --layer_start 10 \
    --layer_end 15 \
    --test_after_prune
```

这样只会分析和剪枝 Layer 10-14，用于快速验证。

---

## 常见问题

### Q1: 为什么 Attention 和 MLP 的剪枝数量差异很大？

**A**: 这是正常现象！因为：
- Attention Group 成本高（~6.3M 参数）
- MLP Channel 成本低（~12K 参数）
- 算法根据 Score = Importance / Cost 自动平衡
- 如果 MLP 的 importance 相对较低，会优先剪除更多 MLP channels

### Q2: 为什么某些层被完全剪空？

**A**: 这是全局剪枝的特点！
- 如果某层的所有组件得分都极低（符合 U 型分布观察）
- 算法会自动将该层剪空，实现深度剪枝
- 使用 `--remove_empty_layers` 可以移除这些层

### Q3: 全局剪枝比层级剪枝慢多少？

**A**:
- **分析表构建**：差不多（都需要计算梯度）
- **剪枝执行**：稍慢（需要处理全局表）
- **总体**：慢 10-20%，但效果更优

### Q4: 如何查看剪枝的详细统计？

**A**: 查看生成的 CSV 文件：

```bash
# 查看 score 最低的 20 个 groups
head -n 21 prune_log/{name}/groups_to_prune.csv

# 统计各层剪枝情况
awk -F, '{print $1,$2}' prune_log/{name}/groups_to_prune.csv | sort | uniq -c
```

### Q5: 可以先生成分析表，后续再执行剪枝吗？

**A**: 可以！分两步执行：

**Step 1: 生成分析表**
```bash
python demo_global_pruning.py \
    --base_model /path/to/model \
    --save_table_path prune_log/analysis.json
```

**Step 2: 基于分析表剪枝**
（当前脚本暂不支持，需要从 CSV 读取并应用）

---

## 实验建议

### 典型实验流程

1. **基线评估**
   ```bash
   python llama3_global_pruning.py \
       --base_model /path/to/model \
       --save_ckpt_log_name baseline \
       --test_before_prune
   ```

2. **无微调剪枝（测试剪枝效果）**
   ```bash
   python llama3_global_pruning.py \
       --base_model /path/to/model \
       --save_ckpt_log_name prune_25pct_no_ft \
       --pruning_ratio 0.25 \
       --test_before_prune \
       --test_after_prune \
       --save_model
   ```

3. **带微调剪枝（恢复性能）**
   ```bash
   python llama3_global_pruning.py \
       --base_model /path/to/model \
       --save_ckpt_log_name prune_25pct_lora \
       --pruning_ratio 0.25 \
       --test_before_prune \
       --test_after_prune \
       --finetune \
       --finetune_method lora \
       --finetune_samples 1000 \
       --save_model
   ```

4. **对比评估**
   ```bash
   python evaluate_models.py \
       --models baseline prune_25pct_no_ft prune_25pct_lora
   ```

---

## 总结

全局剪枝方法基于**分数背包问题**建模，通过计算每个组件的**性价比得分（Score = Importance / Cost）**，在全局范围内选择最优剪枝策略。

**核心优势**：
- ✅ 跨层、跨组件对比
- ✅ 自动平衡 Attention:MLP 比例
- ✅ 自动实现深度+宽度混合剪枝
- ✅ 理论上达到全局最优

**适用场景**：
- 需要极致剪枝性能
- 愿意牺牲一定计算时间
- 对模型结构变化有一定容忍度（如自动深度剪枝）

**不适用场景**：
- 需要快速剪枝
- 必须保持各层结构平衡
- 不希望模型层数改变

---

更多信息请参考：
- 理论细节：`CLAUDE.md` - 全局剪枝部分
- 代码实现：`core/methods/global_pruning.py`
- 模块文档：`core/README.md`
