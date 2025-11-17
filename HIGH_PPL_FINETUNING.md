# 高PPL模型微调指南（针对非均衡剪枝）

## 📊 您的情况

- ✅ 模型权重健康（无NaN/Inf）
- ✅ 前向传播正常
- ✅ GQA 4:1比例保持正确
- ⚠️  PPL=80.85（非均衡剪枝的正常结果）
- ❌ 微调时Loss立即变成NaN

## 🔍 原因分析

**模型处于极度不稳定的临界状态**：
- 虽然权重本身正常，但PPL=80说明模型已经非常脆弱
- 前向传播可以，但反向传播时梯度极易爆炸
- 需要**极低学习率 + 极强梯度裁剪**才能稳定训练

---

## 🎯 推荐方案

### 方案1：极低学习率（最简单）⭐

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name ultra_low_lr \
    --lr 1e-7 \
    --samples 200 \
    --epochs 1 \
    --seq_len 128 \
    --batch_size 1 \
    --grad_accum 2 \
    --max_grad_norm 0.1 \
    --warmup_steps 5 \
    --weight_decay 0.0 \
    --test_before \
    --test_after
```

**关键配置**：
- `lr=1e-7` - 比默认低**100倍**
- `max_grad_norm=0.1` - 比默认低**10倍**
- `seq_len=128` - 减少显存压力
- `samples=200` - 先小规模测试

**预期结果**：
```
进度: 10% | 平均Loss: 4.3821 | LR: 2.00e-08
进度: 20% | 平均Loss: 4.1234 | LR: 4.00e-08
...
进度: 100% | 平均Loss: 3.8765 | LR: 9.50e-08
✅ 第 1 轮完成，平均Loss: 3.8765
```

如果Loss从~4.0下降到~3.8且无NaN，说明成功了！

---

### 方案2：渐进式提高学习率（更安全）

#### 阶段1：lr=1e-7（热身）

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name stage1_1e7 \
    --lr 1e-7 \
    --samples 500 \
    --epochs 1 \
    --seq_len 128 \
    --max_grad_norm 0.1 \
    --test_before \
    --test_after \
    --save_model
```

**检查点**：
- ✅ Loss下降（从4.0到3.5左右）
- ✅ 无NaN
- ✅ PPL略有下降（80→75左右）

#### 阶段2：lr=5e-7（加速）

```bash
python test_finetuning.py \
    --model_path prune_log/stage1_1e7/pytorch_model_finetuned.bin \
    --save_name stage2_5e7 \
    --lr 5e-7 \
    --samples 500 \
    --epochs 1 \
    --seq_len 256 \
    --max_grad_norm 0.3 \
    --test_before \
    --test_after \
    --save_model
```

#### 阶段3：lr=1e-6（正常微调）

```bash
python test_finetuning.py \
    --model_path prune_log/stage2_5e7/pytorch_model_finetuned.bin \
    --save_name stage3_1e6 \
    --lr 1e-6 \
    --samples 1000 \
    --epochs 2 \
    --seq_len 512 \
    --max_grad_norm 0.5 \
    --test_before \
    --test_after \
    --save_model
```

**预期最终结果**：
- PPL从80降到50-60左右（30-40%改善）

---

### 方案3：冻结层微调（最稳定）

只训练最重要的层，冻结其他层：

```bash
python test_finetuning_frozen.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name frozen_most \
    --freeze_strategy most \
    --lr 1e-7 \
    --samples 500 \
    --epochs 2 \
    --seq_len 256 \
    --max_grad_norm 0.1 \
    --test_before \
    --test_after \
    --save_model
```

**冻结策略**：
- `most`: 只训练最后5层（最稳定）
- `half`: 训练后半部分层
- `ends`: 冻结首尾，训练中间层

**优势**：
- 只训练10-20%的参数
- 更稳定（不易NaN）
- 更快（每步时间减少）

---

## 📈 学习率对照表

基于您的PPL=80的情况：

| 学习率 | 梯度裁剪 | 预期效果 | 推荐场景 |
|--------|---------|---------|---------|
| 1e-7 | 0.1 | ✅ 最稳定 | 首次尝试 |
| 5e-7 | 0.3 | ✅ 稳定 | 第2阶段 |
| 1e-6 | 0.5 | ⚠️  可能NaN | 第3阶段 |
| 5e-6 | 1.0 | ❌ 很可能NaN | 不推荐 |
| 1e-5 | 1.0 | ❌ 必然NaN | 绝对不行 |

---

## 🎛️ 参数调优指南

### 如果仍然NaN：

#### 策略A：进一步降低学习率
```bash
--lr 5e-8  # 从1e-7降到5e-8
--max_grad_norm 0.05  # 从0.1降到0.05
```

#### 策略B：使用更小的batch
```bash
--seq_len 64  # 从128降到64
--batch_size 1
--grad_accum 1  # 不使用梯度累积
```

#### 策略C：冻结更多层
```python
# 修改 test_finetuning_frozen.py
# 只训练最后3层而不是5层
freeze_range = range(0, num_layers - 3)
```

### 如果训练太慢：

#### 策略A：逐步提高学习率
```bash
# 如果1e-7成功，尝试2e-7
--lr 2e-7
```

#### 策略B：增加batch size
```bash
--batch_size 1
--grad_accum 4  # 有效batch size = 4
```

---

## 🔍 监控指标

### 健康的训练：

```
进度: 10% | 平均Loss: 4.2345 | LR: 2.00e-08
进度: 20% | 平均Loss: 4.1123 | LR: 4.00e-08
进度: 30% | 平均Loss: 3.9876 | LR: 6.00e-08
进度: 40% | 平均Loss: 3.8654 | LR: 8.00e-08
...
```

**关键特征**：
- ✅ Loss持续下降（4.2→3.8）
- ✅ 下降平滑，无剧烈波动
- ✅ 无NaN或Inf

### 不健康的训练：

```
进度: 10% | 平均Loss: 4.2345 | LR: 2.00e-05
进度: 20% | 平均Loss: nan | LR: 4.00e-05  ← 立即NaN
```

或：

```
进度: 10% | 平均Loss: 4.2345 | LR: 2.00e-06
进度: 20% | 平均Loss: 5.6789 | LR: 4.00e-06  ← Loss上升
进度: 30% | 平均Loss: 8.9012 | LR: 6.00e-06  ← 持续上升
进度: 40% | 平均Loss: nan | LR: 8.00e-06  ← 最终NaN
```

---

## 📊 预期改善幅度

基于PPL=80的起点：

| 方案 | 训练时间 | 参数量 | 预期PPL | 改善幅度 |
|------|---------|--------|---------|---------|
| 极低lr（1e-7） | ~30分钟 | 100% | 70-75 | 6-13% |
| 渐进式3阶段 | ~2小时 | 100% | 50-60 | 25-38% |
| 冻结层 | ~20分钟 | 10-20% | 65-70 | 13-19% |

---

## 🛠️ 完整测试流程

### 步骤1：快速验证（5分钟）

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name quick_test \
    --lr 1e-7 \
    --samples 50 \
    --epochs 1 \
    --seq_len 64 \
    --max_grad_norm 0.1
```

**检查点**：
- ✅ 如果Loss下降且无NaN → 继续步骤2
- ❌ 如果还是NaN → 降低到5e-8或使用方案3

### 步骤2：小规模训练（30分钟）

```bash
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name small_scale \
    --lr 1e-7 \
    --samples 500 \
    --epochs 1 \
    --seq_len 128 \
    --max_grad_norm 0.1 \
    --test_before \
    --test_after \
    --save_model
```

**检查点**：
- ✅ PPL下降（80→75左右）→ 继续步骤3
- ⚠️  PPL几乎不变 → 提高到2e-7
- ❌ PPL上升 → 保持1e-7，增加样本数

### 步骤3：渐进式完整训练（2小时）

使用方案2的3阶段训练。

---

## 💡 高级技巧

### 技巧1：混合精度训练

如果显存充足，使用FP32可能更稳定：

```python
# 在test_finetuning.py中修改
model = model.float()  # 改用FP32而不是FP16
```

### 技巧2：层级学习率

不同层使用不同学习率（需要修改代码）：

```python
param_groups = [
    {'params': model.model.layers[:16].parameters(), 'lr': 5e-8},  # 前半部分
    {'params': model.model.layers[16:].parameters(), 'lr': 1e-7},  # 后半部分
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
```

### 技巧3：分段微调

先微调Attention，再微调MLP：

```python
# 第1阶段：只训练Attention
for layer in model.model.layers:
    for name, param in layer.named_parameters():
        if 'mlp' in name:
            param.requires_grad = False

# 微调...

# 第2阶段：只训练MLP
for layer in model.model.layers:
    for name, param in layer.named_parameters():
        if 'mlp' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
```

---

## ❓ 常见问题

### Q1: 为什么1e-7这么低的学习率？

**A**: 您的PPL=80意味着模型处于极度不稳定状态：
- 正常模型PPL~10，可以用1e-5
- 您的模型PPL~80（高8倍），需要低得多的学习率
- 1e-7约是正常值的1/100，对应不稳定度的平方

### Q2: Loss从4.0降到3.8算成功吗？

**A**: ✅ 是的！对于PPL=80的模型：
- Loss 4.0 → 3.8 = 5%改善
- 对应PPL 80 → 70-75 = 6-13%改善
- 这是很好的进步

### Q3: 能用LoRA吗？

**A**: ⚠️  不推荐：
- LoRA只调整部分权重
- 您的模型需要更全面的调整
- 全参数微调更适合

### Q4: 训练多少轮合适？

**A**: 取决于Loss下降趋势：
```
1轮：Loss 4.0 → 3.8  ✅ 继续
2轮：Loss 3.8 → 3.6  ✅ 继续
3轮：Loss 3.6 → 3.5  ⚠️  收益递减
4轮：Loss 3.5 → 3.5  ❌ 停止（过拟合）
```

### Q5: 什么时候停止？

**停止信号**：
1. Loss不再下降
2. PPL不再改善或开始上升
3. 在验证集上表现变差

---

## 🎯 推荐的最佳实践

### 完整3阶段流程

```bash
# 阶段1：热身（lr=1e-7）
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name stage1 \
    --lr 1e-7 \
    --samples 500 \
    --epochs 1 \
    --seq_len 128 \
    --max_grad_norm 0.1 \
    --save_model

# 检查PPL（应该从80降到~75）
grep "微调后 PPL" prune_log/stage1/*/training.log

# 阶段2：加速（lr=5e-7）
python test_finetuning.py \
    --model_path prune_log/stage1/pytorch_model_finetuned.bin \
    --save_name stage2 \
    --lr 5e-7 \
    --samples 1000 \
    --epochs 1 \
    --seq_len 256 \
    --max_grad_norm 0.3 \
    --save_model

# 检查PPL（应该从75降到~60）

# 阶段3：精细调整（lr=1e-6）
python test_finetuning.py \
    --model_path prune_log/stage2/pytorch_model_finetuned.bin \
    --save_name stage3 \
    --lr 1e-6 \
    --samples 1000 \
    --epochs 2 \
    --seq_len 512 \
    --max_grad_norm 0.5 \
    --test_before \
    --test_after \
    --save_model

# 最终PPL应该在50-60之间
```

---

## 📞 如果还有问题

如果以上所有方法都失败：

1. **尝试更低的学习率**：5e-8甚至1e-8
2. **尝试冻结层方法**：只训练最后3层
3. **检查是否是数据问题**：尝试不同的数据集
4. **考虑量化**：使用4-bit或8-bit量化可能更稳定

---

**最后更新**: 2025-11-17
**针对**: PPL>50的高度不稳定模型
