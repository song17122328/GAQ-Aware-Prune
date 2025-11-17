# 微调功能测试指南

## 问题修复

已修复的问题：
1. ✅ 导入错误：`get_linear_schedule_with_warmup` 应从 `transformers` 导入
2. ✅ 兼容性：添加了备用的学习率调度器（旧版transformers）
3. ✅ 空指针：修复了 scheduler 为 None 时的错误

## 快速测试微调功能

### 步骤1：使用独立脚本测试

```bash
# 简单测试（100样本，1轮，快速验证）
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name test_finetune_simple \
    --samples 100 \
    --epochs 1 \
    --seq_len 256 \
    --test_before \
    --test_after

# 完整测试（1000样本，2轮，带保存）
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name test_finetune_full \
    --method full \
    --lr 1e-5 \
    --samples 1000 \
    --epochs 2 \
    --seq_len 512 \
    --grad_accum 4 \
    --warmup_steps 50 \
    --test_before \
    --test_after \
    --save_model

# LoRA测试（需要先安装PEFT）
pip install peft

python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --save_name test_finetune_lora \
    --method lora \
    --lr 2e-4 \
    --samples 500 \
    --epochs 2 \
    --seq_len 512 \
    --grad_accum 8 \
    --test_before \
    --test_after \
    --save_model
```

### 步骤2：验证主脚本

确认独立测试成功后，使用完整流程：

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_test_complete \
    --pruning_ratio 0.25 \
    --importance_samples 20 \
    --num_examples 5 \
    --layer_start 10 \
    --layer_end 15 \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_method full \
    --finetune_samples 100 \
    --finetune_epochs 1 \
    --finetune_grad_accum 2
```

## 命令行参数说明

### test_finetuning.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | **必需** | 剪枝后的模型路径 |
| `--save_name` | `test_finetune` | 保存名称 |
| `--method` | `full` | 微调方法（full/lora） |
| `--lr` | `1e-5` | 学习率 |
| `--epochs` | `1` | 训练轮数 |
| `--samples` | `100` | 训练样本数 |
| `--batch_size` | `1` | batch大小 |
| `--seq_len` | `256` | 序列长度 |
| `--grad_accum` | `2` | 梯度累积步数 |
| `--max_grad_norm` | `1.0` | 梯度裁剪 |
| `--warmup_steps` | `10` | 预热步数 |
| `--weight_decay` | `0.01` | 权重衰减 |
| `--test_before` | `False` | 微调前测试PPL |
| `--test_after` | `False` | 微调后测试PPL |
| `--save_model` | `False` | 保存微调后模型 |

## 预期日志输出

### 成功的微调日志

```
============================================================
步骤3: 执行微调
============================================================
✅ 微调数据加载完成，shape: torch.Size([100, 256])
全参数微调模式：训练 6,363,025,408 个参数

微调配置:
  模式: 全参数微调
  学习率: 1e-05
  轮数: 1
  样本数: 100
  Batch size: 1
  梯度累积步数: 2
  有效Batch size: 2
  序列长度: 256
  梯度裁剪: 1.0
  权重衰减: 0.01
  学习率调度: 启用
  预热步数: 10

开始第 1/1 轮微调...
  进度: 10% | 平均Loss: 2.3456 | LR: 2.00e-06
  进度: 20% | 平均Loss: 2.1234 | LR: 4.00e-06
  进度: 30% | 平均Loss: 2.0123 | LR: 6.00e-06
  ...
  进度: 100% | 平均Loss: 1.8765 | LR: 9.50e-06
✅ 第 1 轮完成，平均Loss: 1.8765

✅ 微调完成！

✅ 微调成功完成！

微调统计:
  方法: full
  总步数: 50
  最终Loss: 1.8765
```

### 如果遇到错误

脚本会详细输出错误信息：

```
❌ 微调过程出错:
  错误类型: RuntimeError
  错误信息: CUDA out of memory

详细错误信息:
Traceback (most recent call last):
  ...
```

## 常见问题

### Q1: CUDA Out of Memory

**解决方案**：
```bash
# 减少序列长度
--seq_len 128

# 增加梯度累积
--grad_accum 4

# 减少样本数
--samples 50
```

### Q2: 找不到模型文件

**解决方案**：
```bash
# 检查模型路径是否正确
ls -lh prune_log/llama3_pruned_finetuned/pytorch_model.bin

# 或使用绝对路径
--model_path /absolute/path/to/pytorch_model.bin
```

### Q3: transformers版本问题

**解决方案**：
```bash
# 升级transformers
pip install --upgrade transformers

# 如果无法升级，脚本会自动使用备用调度器
```

### Q4: LoRA导入失败

**解决方案**：
```bash
# 安装PEFT
pip install peft

# 如果PEFT不可用，会自动回退到全参数微调
```

## 性能基准

基于测试的预期结果：

| 配置 | 样本数 | 轮数 | 时间 | PPL改善 |
|------|--------|------|------|---------|
| 快速测试 | 100 | 1 | ~5分钟 | 小幅改善 |
| 标准配置 | 1000 | 2 | ~1小时 | 明显改善 |
| 完整配置 | 2000 | 3 | ~2小时 | 最佳效果 |

## 调试技巧

### 1. 逐步测试

```bash
# 第1步：只测试数据加载
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --samples 10 \
    --epochs 1

# 第2步：测试微调前评估
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --samples 10 \
    --test_before

# 第3步：完整流程
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_finetuned/pytorch_model.bin \
    --samples 100 \
    --test_before \
    --test_after
```

### 2. 查看日志

```bash
# 查看最新的日志
ls -lt prune_log/test_finetune_*/

# 查看训练日志
cat prune_log/test_finetune_simple/*/training.log

# 搜索错误
grep -i "error\|失败\|❌" prune_log/test_finetune_simple/*/training.log
```

### 3. 监控显存

```bash
# 在另一个终端监控GPU
watch -n 1 nvidia-smi
```

## 成功标志

微调成功的标志：
- ✅ 日志显示 "✅ 微调完成！"
- ✅ Loss从初始值（~2.0-3.0）下降到较低值（~1.5-2.0）
- ✅ 没有NaN或Inf的loss
- ✅ 如果启用test_after，PPL应该比test_before更低
- ✅ 学习率正常变化（warmup阶段递增，后期递减）

## 下一步

成功测试微调后：

1. ✅ 使用主脚本进行完整的剪枝+微调流程
2. ✅ 调整超参数以获得最佳效果
3. ✅ 在实际任务上评估微调后的模型

详细使用指南请参考：
- `FINETUNING.md` - 完整的微调指南
- `RUN.md` - 完整流程使用指南
- `CLAUDE.md` - 开发者指南
