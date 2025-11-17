# 修复NaN Loss问题指南

## 🚨 问题描述

您遇到的问题：
- ✅ 剪枝后PPL = 80.85（**异常高**，正常应为10-15）
- ❌ 微调时Loss立即变成NaN
- ❌ 微调后PPL也是NaN，模型完全损坏

## 🔍 根本原因

**剪枝后PPL=80.85已经说明模型有严重问题**。正常的剪枝应该：
- 25%剪枝率：PPL 11-13
- 30%剪枝率：PPL 13-15
- **>50 的PPL**: 模型已经严重损坏

可能的原因：
1. ❌ 剪枝率过高（超过模型承受能力）
2. ❌ 剪枝过程出现错误
3. ❌ 某些层被过度剪枝
4. ❌ GQA比例未正确维护

---

## 🛠️ 解决方案

### 步骤1：诊断当前模型

首先运行诊断脚本检查模型是否真的损坏：

```bash
python diagnose_model.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --test_forward
```

**预期输出**：
```
============================================================
诊断结果
============================================================

总参数量: 6,363,025,408
✅ 无NaN值
✅ 无Inf值
✅ 零值比例正常: 15.2%
✅ 无异常大的值

✅ 所有层看起来正常

============================================================
3. 测试前向传播
============================================================
✅ 前向传播正常
   输出shape: torch.Size([1, 7, 128256])
   输出范围: [-15.23, 18.45]
```

**如果发现NaN/Inf**：模型在剪枝时就已经损坏，需要重新剪枝。

---

### 步骤2：重新剪枝（降低剪枝率）

**原因**：您的PPL=80.85说明25%的剪枝率太高了。

**解决方案**：降低到15-20%

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_15pct \
    --pruning_ratio 0.15 \
    --importance_method removal \
    --importance_samples 50 \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**关键参数**：
- `--pruning_ratio 0.15` - 降低到15%（从25%）
- `--test_after_prune` - 立即检查PPL

**预期剪枝后PPL**：应该在 10-12 之间。

---

### 步骤3：如果仍然PPL过高，进一步调整

#### 3.1 只剪枝Attention，不剪枝MLP

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_attn_only_20pct \
    --pruning_ratio 0.20 \
    --importance_method removal \
    --pruning_strategy inverse \
    --save_model \
    --test_after_prune
```

**注意**：移除了 `--prune_mlp`

#### 3.2 保护首尾层

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_middle_layers \
    --pruning_ratio 0.20 \
    --layer_start 3 \
    --layer_end 29 \
    --importance_method removal \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**说明**：跳过前3层和后3层（这些层通常更重要）

#### 3.3 使用更保守的剪枝策略

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_conservative \
    --pruning_ratio 0.20 \
    --importance_method removal \
    --pruning_strategy inverse \
    --alpha 1.5 \
    --min_pruning_rate 0.10 \
    --max_pruning_rate 0.35 \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**关键参数**：
- `--alpha 1.5` - 增加重要性权重（更保守）
- `--min_pruning_rate 0.10` - 最少剪10%
- `--max_pruning_rate 0.35` - 最多剪35%

---

### 步骤4：确认剪枝成功后再微调

**检查点**：
- ✅ 剪枝后PPL < 15
- ✅ GQA比例保持4:1
- ✅ 无错误信息

然后进行微调：

#### 4.1 使用极低学习率微调

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_15pct/pytorch_model.bin \
    --save_name finetune_ultra_safe \
    --lr 5e-7 \
    --samples 500 \
    --epochs 2 \
    --seq_len 256 \
    --grad_accum 4 \
    --max_grad_norm 0.5 \
    --warmup_steps 20 \
    --test_before \
    --test_after
```

**关键参数**：
- `--lr 5e-7` - 极低学习率（比默认的1e-5低20倍）
- `--max_grad_norm 0.5` - 强梯度裁剪
- `--seq_len 256` - 短序列（减少显存压力）

#### 4.2 完整流程（剪枝+微调）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_complete_safe \
    --pruning_ratio 0.15 \
    --importance_method removal \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method full \
    --finetune_lr 5e-7 \
    --finetune_epochs 2 \
    --finetune_samples 1000 \
    --finetune_grad_accum 4 \
    --finetune_max_grad_norm 0.5 \
    --finetune_warmup_steps 50
```

---

## 📊 PPL基准参考

| 剪枝率 | 剪枝后PPL（正常） | 剪枝后PPL（异常） | 是否能微调 |
|-------|----------------|-----------------|----------|
| 15% | 10-11 | >30 | ✅ 可以 |
| 20% | 11-13 | >40 | ✅ 可以 |
| 25% | 12-15 | >50 | ⚠️  需要降低学习率 |
| 30% | 14-18 | >70 | ❌ 建议重新剪枝 |
| >30% | 16-25 | >100 | ❌ 必须重新剪枝 |

**您的情况**：25%剪枝后PPL=80.85 → **严重异常**，必须重新剪枝。

---

## 🔧 新增的安全检查

现在微调脚本包含以下安全机制：

### 1. 微调前模型健康检查

```
检查模型权重健康状态...
✅ 模型权重正常
```

如果检测到问题：
```
❌ 模型存在数值问题:
  NaN参数数量: 1,234
  Inf参数数量: 567

建议:
  1. 重新运行剪枝流程
  2. 检查剪枝率是否过高
  3. 运行诊断脚本: python diagnose_model.py --model_path <path>
```

### 2. 训练中Loss监控

每个batch都会检查Loss：
```python
if torch.isnan(loss) or torch.isinf(loss) or loss_value > 1e6:
    # 立即停止并给出详细建议
```

---

## 🎯 推荐的完整流程

### 方案A：保守剪枝（推荐）⭐

```bash
# 步骤1：15%剪枝
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_safe_15pct \
    --pruning_ratio 0.15 \
    --importance_method removal \
    --pruning_strategy inverse \
    --alpha 1.2 \
    --prune_mlp \
    --save_model \
    --test_after_prune

# 步骤2：检查PPL（应该<12）
grep "剪枝后 PPL" prune_log/llama3_safe_15pct/*/training.log

# 步骤3：如果PPL正常，进行微调
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_safe_15pct_finetuned \
    --pruning_ratio 0.15 \
    --skip_importance_analysis \
    --importance_config prune_log/llama3_safe_15pct/layer_importance_config.json \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_lr 5e-7 \
    --finetune_epochs 2 \
    --finetune_samples 1000 \
    --finetune_grad_accum 4 \
    --finetune_max_grad_norm 0.5
```

### 方案B：只剪Attention（更安全）

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_attn_20pct \
    --pruning_ratio 0.20 \
    --importance_method removal \
    --pruning_strategy inverse \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_lr 1e-6 \
    --finetune_epochs 2 \
    --finetune_samples 1000
```

**注意**：没有 `--prune_mlp`

---

## 📝 成功标志

### 剪枝成功：
```
剪枝后 PPL:   wikitext2 (wikitext-2-raw-v1): 11.23
GQA比例验证: ✅ 所有层保持4:1
实际剪枝率: 15.12%
```

### 微调成功：
```
检查模型权重健康状态...
✅ 模型权重正常

开始第 1/2 轮微调...
  进度: 10% | 平均Loss: 2.1234 | LR: 2.50e-07
  进度: 20% | 平均Loss: 1.9876 | LR: 5.00e-07
  ...
  进度: 100% | 平均Loss: 1.6543 | LR: 4.75e-07
✅ 第 1 轮完成，平均Loss: 1.6543

微调后 PPL:   wikitext2 (wikitext-2-raw-v1): 10.87
```

**关键指标**：
- ✅ Loss从~2.0下降到~1.6（正常）
- ✅ 无NaN或Inf
- ✅ PPL下降（从11.23到10.87）

---

## ❓ 常见问题

### Q1: 为什么我的PPL这么高（80.85）？

**A**: 可能的原因：
1. 剪枝率25%对您的模型来说太高
2. 某些关键层被过度剪枝
3. MLP剪枝可能过于激进

**解决**：降低到15-20%，先不剪MLP

### Q2: 降低学习率后还是NaN怎么办？

**A**: 说明剪枝后的模型已经无法恢复，必须：
1. 重新剪枝，使用更低的剪枝率（10-15%）
2. 跳过问题层（使用`--layer_start`和`--layer_end`）
3. 只剪Attention，不剪MLP

### Q3: 如何判断剪枝率是否合适？

**A**: 看剪枝后的PPL：
- PPL < 15: ✅ 很好，可以继续
- PPL 15-25: ⚠️  可接受，但要小心微调
- PPL 25-50: ❌ 太高了，降低剪枝率
- PPL > 50: ❌ 完全损坏，必须重新剪枝

### Q4: 微调需要多长时间？

**A**: 取决于配置：
- 500样本，1轮：~30分钟
- 1000样本，2轮：~1-1.5小时
- 2000样本，3轮：~2-3小时

---

## 🔧 调试技巧

### 1. 实时监控Loss

```bash
# 在另一个终端
tail -f prune_log/llama3_*/*/training.log | grep "Loss"
```

### 2. 检查剪枝统计

```bash
grep -E "(剪枝率|PPL|GQA)" prune_log/llama3_*/*/training.log
```

### 3. 对比不同配置

```bash
# 运行多个配置
for ratio in 0.15 0.20 0.25; do
    python llama3_unbalanced_pruning_gqa_aware.py \
        --pruning_ratio $ratio \
        --test_after_prune \
        --save_ckpt_log_name test_ratio_${ratio}
done

# 对比结果
grep "剪枝后 PPL" prune_log/test_ratio_*/*/training.log
```

---

## 📞 获取帮助

如果以上方法都无法解决，请提供：
1. 完整的剪枝日志
2. `diagnose_model.py`的输出
3. 剪枝后的PPL值
4. 使用的具体命令

---

**最后更新**: 2025-11-17
**版本**: 1.1
