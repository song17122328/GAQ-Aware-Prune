# 模型评估脚本使用指南

本文档介绍如何使用 `evaluate_models.py` 脚本对比原始模型、剪枝后模型和微调后模型的PPL性能。

---

## 快速开始

### 基本用法

```bash
python evaluate_models.py \
    --original_model /path/to/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama3_pruned/pytorch_model.bin \
    --finetuned_model prune_log/llama3_pruned/pytorch_model_finetuned.bin \
    --seq_len 128
```

### 最简用法（只评估部分模型）

```bash
# 只评估剪枝后和微调后
python evaluate_models.py \
    --pruned_model prune_log/llama3_pruned/pytorch_model.bin \
    --finetuned_model prune_log/llama3_pruned/pytorch_model_finetuned.bin
```

---

## 命令行参数

### 模型路径（至少提供一个）

| 参数 | 说明 | 格式 | 示例 |
|------|------|------|------|
| `--original_model` | 原始模型路径 | HuggingFace格式 | `/data/models/Llama-3-8B-Instruct` |
| `--pruned_model` | 剪枝后模型 | .bin checkpoint | `prune_log/exp/pytorch_model.bin` |
| `--finetuned_model` | 微调后模型 | .bin checkpoint | `prune_log/exp/pytorch_model_finetuned.bin` |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | `wikitext2` | 评估数据集（支持多个） |
| `--seq_len` | `128` | 序列长度 |

### 输出选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_results` | `None` | 指定保存结果的JSON文件路径 |
| `--output_dir` | `evaluation_results` | 结果输出目录（自动命名时使用） |

---

## 使用场景

### 场景1：完整评估（原模型 + 剪枝 + 微调）

```bash
python evaluate_models.py \
    --original_model /data/models/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama3_pruned_25pct/pytorch_model.bin \
    --finetuned_model prune_log/llama3_pruned_25pct/pytorch_model_finetuned.bin \
    --seq_len 128 \
    --save_results results.json
```

**输出示例**：
```
================================================================================
PPL 对比结果
================================================================================

模型                           |             PPL |               参数量 |            变化
--------------------------------------------------------------------------------

数据集: wikitext2 (wikitext-2-raw-v1)
--------------------------------------------------------------------------------
原始模型                        |           12.34 |        8,030,261,248 |            基准
剪枝后模型                      |           80.85 |        6,024,195,936 |         +555.2%
微调后模型                      |           35.82 |        6,024,195,936 |  -55.7% (vs剪枝)
================================================================================
```

### 场景2：只对比剪枝前后（无原模型）

如果原模型太大或不方便加载：

```bash
python evaluate_models.py \
    --pruned_model prune_log/llama3_pruned/pytorch_model.bin \
    --finetuned_model prune_log/llama3_pruned/pytorch_model_finetuned.bin \
    --seq_len 512
```

### 场景3：评估多个数据集

```bash
python evaluate_models.py \
    --finetuned_model prune_log/llama3_pruned/pytorch_model_finetuned.bin \
    --datasets wikitext2 ptb c4 \
    --seq_len 128
```

### 场景4：对比不同seq_len的影响

```bash
# seq_len=128
python evaluate_models.py \
    --finetuned_model prune_log/llama3_pruned/pytorch_model_finetuned.bin \
    --seq_len 128 \
    --save_results eval_128.json

# seq_len=512
python evaluate_models.py \
    --finetuned_model prune_log/llama3_pruned/pytorch_model_finetuned.bin \
    --seq_len 512 \
    --save_results eval_512.json
```

---

## 输出说明

### 终端输出

脚本会打印清晰的对比表格：

```
模型                           |             PPL |               参数量 |            变化
--------------------------------------------------------------------------------
原始模型                        |           12.34 |        8,030,261,248 |            基准
剪枝后模型                      |           80.85 |        6,024,195,936 |         +555.2%
微调后模型                      |           35.82 |        6,024,195,936 |  -55.7% (vs剪枝)
```

**变化列说明**：
- **原始模型**："基准" - 作为参考基准
- **剪枝后模型**：相对于原始模型的PPL变化百分比
- **微调后模型**：相对于剪枝后模型的PPL变化（如果有原模型，也会显示相对原模型的变化）

### JSON输出

如果使用 `--save_results`，会生成JSON文件：

```json
{
  "timestamp": "2025-11-17T17:30:00.123456",
  "results": {
    "原始模型": {
      "ppl": {
        "wikitext2 (wikitext-2-raw-v1)": 12.34
      },
      "param_count": 8030261248
    },
    "剪枝后模型": {
      "ppl": {
        "wikitext2 (wikitext-2-raw-v1)": 80.85
      },
      "param_count": 6024195936
    },
    "微调后模型": {
      "ppl": {
        "wikitext2 (wikitext-2-raw-v1)": 35.82
      },
      "param_count": 6024195936
    }
  }
}
```

---

## 典型工作流程

### 完整实验评估流程

```bash
# 步骤1: 剪枝
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /data/models/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_25pct \
    --pruning_ratio 0.25 \
    --save_model

# 步骤2: 微调
python test_finetuning.py \
    --model_path prune_log/llama3_pruned_25pct/pytorch_model.bin \
    --save_name llama3_finetuned \
    --use_lora \
    --lora_target_attention \
    --lora_target_mlp \
    --lr 2e-4 \
    --samples 1000 \
    --epochs 3 \
    --seq_len 512 \
    --save_model

# 步骤3: 对比评估
python evaluate_models.py \
    --original_model /data/models/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama3_pruned_25pct/pytorch_model.bin \
    --finetuned_model prune_log/llama3_finetuned/pytorch_model_finetuned.bin \
    --seq_len 512 \
    --save_results experiment_results.json
```

---

## 注意事项

### 1. 显存管理

- 脚本会**依次加载模型**，不会同时加载多个模型到显存
- 每个模型评估完后会自动清理显存（`torch.cuda.empty_cache()`）
- 如果评估原始8B模型，建议至少有24GB显存

### 2. seq_len 一致性

**重要**：应该使用与训练/微调时相同的 `seq_len`

```bash
# 如果微调时用的 seq_len=512
python test_finetuning.py --seq_len 512 ...

# 评估时也应该用 seq_len=512
python evaluate_models.py --seq_len 512 ...
```

### 3. 数据集选择

支持的数据集（通过PPLMetric）：
- `wikitext2` (默认，推荐)
- `ptb`
- `c4`

多数据集评估：
```bash
python evaluate_models.py \
    --datasets wikitext2 ptb \
    --pruned_model ...
```

### 4. 模型路径格式

- **原始模型** (`--original_model`)：必须是HuggingFace格式目录（包含config.json、model.safetensors等）
- **剪枝/微调模型** (`--pruned_model`, `--finetuned_model`)：必须是通过本项目保存的.bin checkpoint文件

---

## 故障排查

### 问题1: "模型路径不存在"

**原因**：路径错误或文件不存在

**解决**：
```bash
# 检查路径
ls -lh prune_log/llama3_pruned/pytorch_model.bin

# 使用绝对路径
python evaluate_models.py \
    --pruned_model /absolute/path/to/pytorch_model.bin
```

### 问题2: CUDA Out of Memory

**原因**：显存不足

**解决方案A**：逐个评估
```bash
# 分别运行，不要同时评估多个
python evaluate_models.py --pruned_model ...
python evaluate_models.py --finetuned_model ...
```

**解决方案B**：减少seq_len
```bash
python evaluate_models.py --seq_len 64 ...
```

### 问题3: PPL值异常（NaN或极大值）

**原因**：模型可能损坏或不兼容

**检查**：
```bash
# 使用diagnose_model.py检查模型健康状态
python diagnose_model.py \
    --model_path prune_log/llama3_pruned/pytorch_model.bin
```

---

## 批量评估脚本

如果需要评估多个实验：

```bash
#!/bin/bash
# batch_evaluate.sh

experiments=(
    "llama3_pruned_15pct"
    "llama3_pruned_25pct"
    "llama3_pruned_35pct"
)

for exp in "${experiments[@]}"; do
    echo "评估实验: $exp"
    python evaluate_models.py \
        --pruned_model prune_log/${exp}/pytorch_model.bin \
        --finetuned_model prune_log/${exp}/pytorch_model_finetuned.bin \
        --seq_len 128 \
        --save_results evaluation_results/${exp}_results.json
done

echo "所有实验评估完成！"
```

运行：
```bash
chmod +x batch_evaluate.sh
./batch_evaluate.sh
```

---

## 进阶用法

### 自定义评估数据集

修改脚本中的数据集列表，或通过命令行指定：

```bash
python evaluate_models.py \
    --datasets wikitext2 ptb c4 \
    --pruned_model ...
```

### 导出结果到CSV

可以用Python脚本转换JSON到CSV：

```python
import json
import csv

with open('results.json') as f:
    data = json.load(f)

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'PPL', 'Parameters'])

    for model, info in data['results'].items():
        for dataset, ppl in info['ppl'].items():
            writer.writerow([model, dataset, ppl, info['param_count']])
```

---

## 总结

**推荐使用方式**：

1. **完整对比**（有原模型）：
   ```bash
   python evaluate_models.py \
       --original_model /path/to/original \
       --pruned_model prune_log/exp/pytorch_model.bin \
       --finetuned_model prune_log/exp/pytorch_model_finetuned.bin \
       --seq_len 512
   ```

2. **快速对比**（只看剪枝+微调效果）：
   ```bash
   python evaluate_models.py \
       --pruned_model prune_log/exp/pytorch_model.bin \
       --finetuned_model prune_log/exp/pytorch_model_finetuned.bin
   ```

3. **保存结果以便后续分析**：
   ```bash
   python evaluate_models.py \
       --finetuned_model prune_log/exp/pytorch_model_finetuned.bin \
       --save_results my_experiment_$(date +%Y%m%d).json
   ```

---

**最后更新**: 2025-11-17
