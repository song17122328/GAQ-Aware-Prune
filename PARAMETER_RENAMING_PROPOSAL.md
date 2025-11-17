# 参数命名优化方案

## 🎯 当前问题

部分参数名字不够清晰，无法直接看出其用途。例如：
- `--importance_samples` vs `--num_examples` - 两者都是样本数，但用途不同
- `--alpha` - 太抽象
- `--max_seq_len` - 不明确是哪个阶段的序列长度

---

## ✨ 建议的重命名方案

### 核心评估参数重命名

| 旧参数名 | 新参数名 | 说明 | 优势 |
|---------|---------|------|------|
| `--importance_samples` | `--layer_importance_samples` | 层级重要性评估样本数 | ✅ 明确是"层级" |
| `--num_examples` | `--head_importance_samples` | 头/通道重要性评估样本数 | ✅ 明确是"头级别" |
| `--importance_method` | `--layer_importance_method` | 层重要性评估方法 | ✅ 明确是"层级" |
| `--importance_config` | `--layer_importance_config` | 层重要性配置文件 | ✅ 明确是"层级" |
| `--max_seq_len` | `--taylor_seq_len` | Taylor计算时的序列长度 | ✅ 明确用途 |

### 剪枝策略参数重命名

| 旧参数名 | 新参数名 | 说明 | 优势 |
|---------|---------|------|------|
| `--alpha` | `--importance_weight` | 重要性权重系数 | ✅ 更直观 |
| `--min_pruning_rate` | `--layer_min_pruning_rate` | 单层最小剪枝率 | ✅ 明确是"层级" |
| `--max_pruning_rate` | `--layer_max_pruning_rate` | 单层最大剪枝率 | ✅ 明确是"层级" |

---

## 📊 重命名前后对比

### 示例：标准实验

**旧命名**（不直观）:
```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --importance_samples 50 \        # ❓ 什么的重要性？
    --num_examples 10 \              # ❓ 什么的样本？
    --importance_method removal \    # ❓ 什么的方法？
    --alpha 1.0 \                    # ❓ alpha是什么？
    --max_seq_len 64                 # ❓ 哪个阶段的序列长度？
```

**新命名**（一目了然）:
```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --layer_importance_samples 50 \      # ✅ 层级重要性评估样本数
    --head_importance_samples 10 \       # ✅ 头级别重要性评估样本数
    --layer_importance_method removal \  # ✅ 层级重要性评估方法
    --importance_weight 1.0 \            # ✅ 重要性权重系数
    --taylor_seq_len 64                  # ✅ Taylor计算序列长度
```

---

## 🔧 实施方案

### 方案A：完全替换（破坏性变更）

**优点**: 参数名最清晰
**缺点**: 会破坏现有脚本

**实施**:
1. 直接修改参数名
2. 更新所有文档
3. 版本号升级到 2.0

### 方案B：兼容性重命名（推荐）⭐

**优点**: 向后兼容，新旧参数都能用
**缺点**: 代码稍微复杂一点

**实施**:
```python
# 新参数（推荐使用）
parser.add_argument('--layer_importance_samples', type=int, default=50,
                   dest='layer_importance_samples',
                   help='层级重要性评估样本数')

# 旧参数（保持兼容）
parser.add_argument('--importance_samples', type=int,
                   dest='layer_importance_samples',
                   help='(已废弃，请使用 --layer_importance_samples)')

# 在参数解析后添加警告
if '--importance_samples' in sys.argv:
    logger.log("⚠️ --importance_samples 已废弃，请使用 --layer_importance_samples")
```

---

## 📝 详细重命名建议

### 1. 层级重要性相关

```python
# 评估样本数
--layer_importance_samples    # 替代 --importance_samples
含义: 用于评估每一层重要性的样本数量
示例: --layer_importance_samples 50

# 评估方法
--layer_importance_method     # 替代 --importance_method
含义: 层重要性评估方法（removal或activation）
示例: --layer_importance_method removal

# 配置文件
--layer_importance_config     # 替代 --importance_config
含义: 层重要性配置文件路径
示例: --layer_importance_config prune_log/exp/layer_importance_config.json
```

### 2. 头/通道级别重要性相关

```python
# 评估样本数
--head_importance_samples     # 替代 --num_examples
含义: 用于计算头/通道Taylor重要性的样本数量
备选: --taylor_samples, --channel_importance_samples
示例: --head_importance_samples 10

# Taylor序列长度
--taylor_seq_len             # 替代 --max_seq_len
含义: Taylor重要性计算时的序列长度
示例: --taylor_seq_len 64
```

### 3. 剪枝策略相关

```python
# 重要性权重
--importance_weight          # 替代 --alpha
含义: 层重要性权重系数，控制层间剪枝率差异
备选: --pruning_alpha, --layer_importance_alpha
示例: --importance_weight 1.0

# 层级剪枝率范围
--layer_min_pruning_rate     # 替代 --min_pruning_rate
含义: 单层最小剪枝率
示例: --layer_min_pruning_rate 0.15

--layer_max_pruning_rate     # 替代 --max_pruning_rate
含义: 单层最大剪枝率
示例: --layer_max_pruning_rate 0.5
```

---

## 🎨 命名规范总结

### 命名原则

1. **明确层级**: 如果参数是层级的，加 `layer_` 前缀
2. **明确对象**: 如果是头/通道级别，加 `head_` 或 `channel_` 前缀
3. **明确用途**: 用完整的词而非缩写（`importance` 而非 `imp`）
4. **避免歧义**: 不要用 `num_examples` 这种泛泛的名字

### 层级结构

```
层级 (Layer Level):
  --layer_importance_samples
  --layer_importance_method
  --layer_importance_config
  --layer_min_pruning_rate
  --layer_max_pruning_rate

头/通道级 (Head/Channel Level):
  --head_importance_samples
  --taylor_seq_len

全局:
  --pruning_ratio           # 总体剪枝率
  --pruning_strategy        # 剪枝策略
  --importance_weight       # 重要性权重
```

---

## 🚀 迁移指南

### 用户迁移步骤

**旧脚本**:
```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --importance_samples 50 \
    --num_examples 10 \
    --alpha 1.0
```

**新脚本**:
```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --layer_importance_samples 50 \
    --head_importance_samples 10 \
    --importance_weight 1.0
```

### 兼容性说明

如果采用方案B（兼容性重命名），旧参数仍然可用：
```bash
# 这两个命令等价
python script.py --importance_samples 50
python script.py --layer_importance_samples 50

# 但会收到警告：
# ⚠️ --importance_samples 已废弃，请使用 --layer_importance_samples
```

---

## 📊 其他可能需要改进的参数

### 建议考虑

| 当前参数 | 是否需要改 | 建议 |
|---------|-----------|------|
| `--save_ckpt_log_name` | 可选 | `--experiment_name` 更简洁 |
| `--prune_mlp` | ✅ 清晰 | 保持不变 |
| `--head_dim` | ✅ 清晰 | 保持不变 |
| `--gqa_ratio` | ✅ 清晰 | 保持不变 |
| `--layer_start` | ✅ 清晰 | 保持不变 |
| `--layer_end` | ✅ 清晰 | 保持不变 |

### 微调相关参数

当前命名已经比较清晰，建议保持：
```bash
--finetune                    # ✅ 清晰
--finetune_method             # ✅ 清晰
--finetune_lr                 # ✅ 清晰
--finetune_samples            # ✅ 清晰
--lora_r                      # ✅ 清晰
--lora_alpha                  # ✅ 清晰（LoRA社区通用术语）
```

---

## 🎯 推荐行动

### 优先级1：核心重命名（高影响）

必须改的参数：
```python
--importance_samples → --layer_importance_samples
--num_examples → --head_importance_samples
```

### 优先级2：辅助重命名（中等影响）

建议改的参数：
```python
--max_seq_len → --taylor_seq_len
--alpha → --importance_weight
--importance_method → --layer_importance_method
--importance_config → --layer_importance_config
```

### 优先级3：可选重命名（低影响）

可以改但不紧迫：
```python
--min_pruning_rate → --layer_min_pruning_rate
--max_pruning_rate → --layer_max_pruning_rate
```

---

## ✅ 实施检查清单

如果决定重命名，需要修改：

- [ ] `llama3_unbalanced_pruning_gqa_aware.py` - 主脚本参数定义
- [ ] `README.md` - 所有示例命令
- [ ] `CLAUDE.md` - 所有示例命令
- [ ] `PARAMETERS_GUIDE.md` - 参数说明
- [ ] 所有示例脚本和文档中的命令

---

**建议**: 采用**方案B（兼容性重命名）**，这样可以：
1. 新用户使用清晰的参数名
2. 旧脚本仍然能运行
3. 逐步迁移，减少破坏性

**最后更新**: 2025-11-17
