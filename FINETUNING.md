# 微调方法改进指南

## 改进概述

我们对微调模块进行了全面改进，支持两种微调方式：

### 1. 改进的全参数微调 ⭐ **推荐**
- **适用场景**：剪枝后的模型（默认推荐）
- **优势**：性能恢复最彻底，适合结构被改变的剪枝模型
- **新增功能**：
  - ✅ 梯度累积（模拟更大batch size）
  - ✅ 梯度裁剪（防止梯度爆炸）
  - ✅ 学习率调度器（线性warmup + decay）
  - ✅ 权重衰减（L2正则化）
  - ✅ 详细的训练日志

### 2. LoRA微调（备选）
- **适用场景**：显存受限的环境
- **优势**：参数少，训练快，显存占用低
- **注意事项**：
  - 需要安装PEFT库：`pip install peft`
  - 对于剪枝后的模型，性能恢复可能不如全参数微调
  - LoRA会自动应用到剪枝后的attention层

---

## 快速开始

### 方案1：改进的全参数微调（推荐）⭐

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_finetuned \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method full \
    --finetune_lr 1e-5 \
    --finetune_epochs 2 \
    --finetune_samples 1000 \
    --finetune_batch_size 1 \
    --finetune_grad_accum 4 \
    --finetune_max_grad_norm 1.0 \
    --finetune_weight_decay 0.01 \
    --finetune_warmup_steps 50
```

**预期结果**：
- PPL从剪枝后的 ~12-13 恢复到 ~11-12
- 训练时间：约1-2小时（取决于硬件）
- 显存需求：约20-24GB VRAM

### 方案2：LoRA微调（低显存）

```bash
# 首先安装PEFT库
pip install peft

# 运行LoRA微调
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_lora \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method lora \
    --finetune_lr 2e-4 \
    --finetune_epochs 3 \
    --finetune_samples 1000 \
    --finetune_batch_size 1 \
    --finetune_grad_accum 8
```

**预期结果**：
- 训练参数：仅约0.5-1%的原始参数
- 训练时间：约30-60分钟
- 显存需求：约12-16GB VRAM

---

## 参数详解

### 基础微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--finetune` | `False` | 是否启用微调（必须） |
| `--finetune_method` | `full` | 微调方法：`full`（全参数）或 `lora` |
| `--finetune_lr` | `1e-5` | 学习率（LoRA建议用2e-4） |
| `--finetune_epochs` | `1` | 训练轮数 |
| `--finetune_samples` | `500` | 训练样本数 |
| `--finetune_batch_size` | `1` | 每步batch大小 |
| `--finetune_seq_len` | `512` | 序列长度 |

### 高级参数（新增）⭐

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--finetune_grad_accum` | `4` | 梯度累积步数 |
| `--finetune_max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `--finetune_weight_decay` | `0.01` | 权重衰减（L2正则） |
| `--finetune_warmup_steps` | `0` | 学习率预热步数 |

**有效Batch Size = batch_size × grad_accum**
- 例如：`batch_size=1, grad_accum=4` → 有效batch size=4

---

## 方法对比

### 全参数微调 vs LoRA

| 对比维度 | 全参数微调 | LoRA微调 |
|---------|-----------|---------|
| **性能恢复** | ⭐⭐⭐⭐⭐ 最好 | ⭐⭐⭐ 较好 |
| **训练速度** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 最快 |
| **显存需求** | ⭐⭐⭐ 中等（20-24GB） | ⭐⭐⭐⭐⭐ 最低（12-16GB） |
| **适用剪枝** | ⭐⭐⭐⭐⭐ 完全兼容 | ⭐⭐⭐ 需要适配 |
| **实现复杂度** | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐ 需要PEFT |

### 推荐选择

1. **优先选择全参数微调**（如果显存足够）
   - 剪枝后参数已减少20-30%，全参数微调可接受
   - 剪枝改变了结构，需要充分调整
   - 梯度累积和裁剪使训练稳定

2. **备选LoRA微调**（显存受限时）
   - 显存不足20GB时
   - 需要快速验证效果时
   - 可以先用LoRA快速测试，再用全参数精调

---

## 使用场景

### 场景1：标准微调（推荐配置）⭐

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/model \
    --save_ckpt_log_name standard_finetune \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method full \
    --finetune_lr 1e-5 \
    --finetune_epochs 2 \
    --finetune_samples 1000 \
    --finetune_grad_accum 4 \
    --finetune_warmup_steps 50
```

**说明**：
- 有效batch size = 4
- 预热50步后线性衰减
- 梯度裁剪防止爆炸

### 场景2：激进剪枝 + 多轮微调

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/model \
    --save_ckpt_log_name aggressive_prune \
    --pruning_ratio 0.30 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method full \
    --finetune_lr 2e-5 \
    --finetune_epochs 3 \
    --finetune_samples 2000 \
    --finetune_grad_accum 8 \
    --finetune_warmup_steps 100 \
    --finetune_weight_decay 0.02
```

**说明**：
- 更高的剪枝率需要更充分的微调
- 更大的有效batch size（8）
- 更多样本和轮数

### 场景3：低显存环境（LoRA）

```bash
# 确保安装PEFT
pip install peft

python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/model \
    --save_ckpt_log_name lora_finetune \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method lora \
    --finetune_lr 2e-4 \
    --finetune_epochs 3 \
    --finetune_samples 1500 \
    --finetune_grad_accum 8
```

**说明**：
- LoRA学习率通常更高（2e-4）
- 更多轮数补偿参数少
- 梯度累积减少显存占用

### 场景4：快速验证

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/model \
    --save_ckpt_log_name quick_test \
    --pruning_ratio 0.25 \
    --importance_samples 20 \
    --layer_start 10 \
    --layer_end 15 \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_samples 100 \
    --finetune_epochs 1 \
    --finetune_batch_size 1 \
    --finetune_grad_accum 2
```

**说明**：
- 只剪枝5层
- 少量样本快速验证
- 适合调试和测试

---

## 改进详解

### 1. 梯度累积（Gradient Accumulation）

**作用**：模拟更大的batch size，在显存受限时非常有用

**原理**：
```python
# 传统方法（batch_size=4需要4倍显存）
for batch in data:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 梯度累积（batch_size=1, grad_accum=4，显存需求不变）
for i, batch in enumerate(data):
    loss = model(batch) / grad_accum  # 除以累积步数
    loss.backward()  # 累积梯度
    if (i + 1) % grad_accum == 0:
        optimizer.step()  # 每N步更新一次
        optimizer.zero_grad()
```

**配置建议**：
- GPU 24GB：`grad_accum=4-8`
- GPU 16GB：`grad_accum=8-16`
- GPU 40GB：`grad_accum=2-4`

### 2. 梯度裁剪（Gradient Clipping）

**作用**：防止梯度爆炸，stabilize训练

**原理**：
```python
# 裁剪梯度范数到max_grad_norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

**配置建议**：
- 通常使用：`1.0`
- 如果loss震荡：降低到 `0.5`
- 如果训练太慢：提高到 `2.0`

### 3. 学习率调度器（LR Scheduler）

**作用**：动态调整学习率，提高收敛效果

**策略**：线性warmup + 线性decay
```
LR
 ^
 |    /\
 |   /  \___
 |  /       \___
 | /            \___
 +-------------------> Steps
   warmup  decay
```

**配置建议**：
- `warmup_steps`：总步数的5-10%
- 例如：总共1000步 → `warmup_steps=50-100`

### 4. 权重衰减（Weight Decay）

**作用**：L2正则化，防止过拟合

**配置建议**：
- 默认：`0.01`
- 小数据集：`0.02-0.05`（更强正则）
- 大数据集：`0.001-0.01`（较弱正则）

---

## 训练日志解读

### 全参数微调日志示例

```
微调配置:
  模式: 全参数微调
  学习率: 1e-05
  轮数: 2
  样本数: 1000
  Batch size: 1
  梯度累积步数: 4
  有效Batch size: 4
  序列长度: 512
  梯度裁剪: 1.0
  权重衰减: 0.01
  学习率调度: 启用
  预热步数: 50

开始第 1/2 轮微调...
  进度: 10% | 平均Loss: 2.3456 | LR: 2.00e-06
  进度: 20% | 平均Loss: 2.1234 | LR: 4.00e-06
  ...
  进度: 100% | 平均Loss: 1.8765 | LR: 9.50e-06
✅ 第 1 轮完成，平均Loss: 1.8765
```

**关键指标**：
- ✅ **Loss下降**：从 2.3 → 1.8（正常）
- ✅ **LR变化**：从 2e-6 → 9.5e-6（warmup正常）
- ❌ **Loss NaN**：检查学习率是否过高

### LoRA微调日志示例

```
配置LoRA (r=8, alpha=16, dropout=0.05)...
trainable params: 4,194,304 || all params: 6,024,195,936 || trainable%: 0.0696

微调配置:
  模式: LoRA微调
  ...

✅ 微调完成！
合并LoRA权重到模型...
✅ LoRA权重已合并
```

**关键指标**：
- ✅ **训练参数**：~0.07%（正常）
- ✅ **权重合并成功**：最终模型包含LoRA调整

---

## 故障排除

### 问题1：CUDA Out of Memory

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：

1. **减少batch size**（已经是1的话）：
   ```bash
   --finetune_seq_len 256  # 从512降到256
   ```

2. **增加梯度累积**（减少显存占用）：
   ```bash
   --finetune_grad_accum 8  # 从4增加到8
   ```

3. **使用LoRA**：
   ```bash
   --finetune_method lora
   ```

4. **减少样本数**：
   ```bash
   --finetune_samples 500  # 从1000降到500
   ```

### 问题2：Loss为NaN或Inf

**症状**：训练日志显示 `Loss: nan`

**可能原因**：
1. 学习率过高
2. 梯度爆炸
3. 模型权重已损坏

**解决方案**：

1. **降低学习率**：
   ```bash
   --finetune_lr 5e-6  # 从1e-5降到5e-6
   ```

2. **启用/降低梯度裁剪**：
   ```bash
   --finetune_max_grad_norm 0.5  # 从1.0降到0.5
   ```

3. **检查剪枝后的模型**：
   ```bash
   # 确保剪枝后PPL是合理的（<20）
   grep "剪枝后 PPL" logs/training.log
   ```

### 问题3：训练很慢

**症状**：每个epoch需要数小时

**解决方案**：

1. **减少样本数**：
   ```bash
   --finetune_samples 500  # 从1000降到500
   ```

2. **减少序列长度**：
   ```bash
   --finetune_seq_len 256  # 从512降到256
   ```

3. **使用LoRA**（更快）：
   ```bash
   --finetune_method lora
   ```

### 问题4：PEFT库未安装（LoRA）

**症状**：`⚠️ PEFT库未安装，回退到全参数微调`

**解决方案**：
```bash
pip install peft
```

或手动指定版本：
```bash
pip install peft==0.7.0
```

### 问题5：微调后PPL没有改善

**症状**：微调后PPL仍然很高或甚至变差

**可能原因**：
1. 学习率不合适
2. 训练样本/轮数不够
3. 过拟合

**解决方案**：

1. **调整学习率**：
   ```bash
   # 试试更高的学习率
   --finetune_lr 2e-5

   # 或者更低的
   --finetune_lr 5e-6
   ```

2. **增加训练量**：
   ```bash
   --finetune_epochs 3 \
   --finetune_samples 2000
   ```

3. **增加权重衰减**（防止过拟合）：
   ```bash
   --finetune_weight_decay 0.05
   ```

---

## 最佳实践

### 1. 选择合适的微调方法

```
显存 >= 24GB → 全参数微调（推荐）
显存 16-24GB → 全参数微调 + 梯度累积
显存 < 16GB → LoRA微调
```

### 2. 学习率设置

| 微调方法 | 学习率范围 | 推荐值 |
|---------|-----------|-------|
| 全参数微调 | 1e-6 ~ 5e-5 | 1e-5 |
| LoRA微调 | 1e-4 ~ 5e-4 | 2e-4 |

### 3. 训练量设置

| 剪枝率 | 推荐样本数 | 推荐轮数 |
|-------|----------|---------|
| 15-20% | 500-1000 | 1-2 |
| 20-25% | 1000-1500 | 2-3 |
| 25-30% | 1500-2000 | 3-4 |
| > 30% | 2000+ | 4+ |

### 4. 渐进式微调策略

对于激进剪枝（>30%），建议分阶段微调：

**阶段1：快速恢复**
```bash
--finetune_lr 2e-5 \
--finetune_epochs 2 \
--finetune_samples 1000
```

**阶段2：精细调整**
```bash
# 加载阶段1的模型，继续微调
--finetune_lr 5e-6 \
--finetune_epochs 2 \
--finetune_samples 1000 \
--finetune_weight_decay 0.02
```

---

## 总结

### 关键改进点

1. ✅ **梯度累积** - 在有限显存下模拟大batch
2. ✅ **梯度裁剪** - 稳定训练，防止爆炸
3. ✅ **学习率调度** - warmup + decay提高收敛
4. ✅ **权重衰减** - 防止过拟合
5. ✅ **LoRA支持** - 低显存环境备选方案

### 推荐配置

**标准配置**（显存充足）：
```bash
--finetune_method full \
--finetune_lr 1e-5 \
--finetune_epochs 2 \
--finetune_samples 1000 \
--finetune_grad_accum 4 \
--finetune_warmup_steps 50
```

**低显存配置**：
```bash
--finetune_method lora \
--finetune_lr 2e-4 \
--finetune_epochs 3 \
--finetune_samples 1500 \
--finetune_grad_accum 8
```

---

## 参考资源

- [PEFT库文档](https://huggingface.co/docs/peft)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [梯度累积详解](https://kozodoi.me/blog/20210219/gradient-accumulation)

**更新日期**：2025-11-17
