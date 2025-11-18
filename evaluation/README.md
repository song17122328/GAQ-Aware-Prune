# Evaluation 模块使用指南

GAQ-Aware-Prune 的统一评估模块，支持性能和效率指标的全面评估。

---

## 📁 文件结构

```
evaluation/
├── metrics/              # 指标模块
│   ├── performance.py    # 性能指标（PPL, Zero-shot）
│   └── efficiency.py     # 效率指标（速度、内存）
│
├── utils/               # 工具函数
│   ├── model_loader.py  # 模型加载
│   └── result_parser.py # 结果解析
│
├── run_evaluation.py    # ⭐ 统一入口脚本
└── README.md           # 本文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础评估（PPL、速度、内存）
pip install torch transformers datasets

# Zero-shot/Few-shot评估（可选）
pip install lm-eval
```

---

### 2. 评估单个模型

```bash
# 评估所有指标
python evaluation/run_evaluation.py \
    --model_path prune_log/ours_optimal/pytorch_model.bin \
    --metrics all \
    --output results/ours.json

# 只评估PPL和速度
python evaluation/run_evaluation.py \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics ppl,speed \
    --output results/original.json

# 自定义评估配置
python evaluation/run_evaluation.py \
    --model_path prune_log/ours_optimal/pytorch_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --ppl_datasets wikitext2,ptb,c4 \
    --zeroshot_tasks hellaswag,piqa,winogrande \
    --speed_samples 100 \
    --output results/ours_full.json
```

---

### 3. 对比多个模型

```bash
# 首先评估各个模型（生成.json文件）
python evaluation/run_evaluation.py --model_path ... --output results/model1.json
python evaluation/run_evaluation.py --model_path ... --output results/model2.json

# 然后生成对比表格
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/model1.json,results/model2.json,results/model3.json \
    --output results/comparison_table.md
```

**输出示例** (`comparison_table.md`):
```markdown
| Metric | Original | Ours | Baseline_Uniform |
|---|---|---|---|
| Parameters (B) | 8.03 | 6.02 | 6.02 |
| PPL (WikiText-2) | 12.34 | 38.46 | 85.30 |
| Avg Zero-shot Acc (%) | 78.50 | 75.20 | 65.10 |
| Throughput (tokens/s) | 125.3 | 168.7 | 169.2 |
| GPU Memory (MB) | 16384 | 12288 | 12288 |
```

---

## 📊 支持的评估指标

### 性能指标

| 指标 | 说明 | 参数 |
|------|------|------|
| `ppl` | 多数据集PPL | `--ppl_datasets wikitext2,ptb,c4` |
| `zeroshot` | Zero-shot准确率 | `--zeroshot_tasks hellaswag,piqa,...` |
| `fewshot` | Few-shot准确率（可选）| 默认MMLU 5-shot |

**支持的PPL数据集**: `wikitext2`, `ptb`, `c4`

**支持的Zero-shot任务**:
- `hellaswag` - 常识推理
- `piqa` - 物理常识
- `winogrande` - 代词消歧
- `arc_easy` / `arc_challenge` - 科学问答
- `boolq` - 是非问答

---

### 效率指标

| 指标 | 说明 | 自动测量 |
|------|------|---------|
| `speed` | 推理速度（吞吐量、延迟）| batch_size=1,4,8 |
| `memory` | 显存占用 | 模型加载+推理峰值 |
| `efficiency` | 全面效率评估 | 包含speed+memory |

---

## 🔧 高级用法

### 单独使用各模块

#### 1. 性能指标

```python
from evaluation.metrics.performance import evaluate_ppl, evaluate_zeroshot
from evaluation.utils.model_loader import load_model_and_tokenizer

# 加载模型
model, tokenizer = load_model_and_tokenizer('/path/to/model')

# 评估PPL
ppl_results = evaluate_ppl(model, tokenizer, datasets=['wikitext2', 'ptb'])
print(ppl_results)  # {'wikitext2 (wikitext-2-raw-v1)': 38.46, ...}

# 评估Zero-shot（需要HF格式模型）
zeroshot_results = evaluate_zeroshot('/path/to/model', tasks=['hellaswag', 'piqa'])
print(zeroshot_results)
```

---

#### 2. 效率指标

```python
from evaluation.metrics.efficiency import evaluate_efficiency
from evaluation.utils.model_loader import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer('/path/to/model')

# 全面效率评估
efficiency_results = evaluate_efficiency(
    model, tokenizer,
    num_samples=100,
    batch_sizes=[1, 4, 8]
)

print(f"参数量: {efficiency_results['model_info']['total_params_B']:.2f}B")
print(f"吞吐量: {efficiency_results['speed']['batch_size_1']['throughput_tokens_per_sec']:.1f} tokens/s")
print(f"显存: {efficiency_results['memory']['model_memory_mb']:.1f} MB")
```

---

#### 3. 模型加载工具

```python
from evaluation.utils.model_loader import load_model_and_tokenizer, get_model_info

# 加载模型（自动识别HF目录或checkpoint）
model, tokenizer = load_model_and_tokenizer(
    'prune_log/xxx/pytorch_model.bin',
    device='cuda',
    torch_dtype=torch.float16
)

# 获取模型详细信息
info = get_model_info(model)
print(f"总参数: {info['total_params_B']:.2f}B")
print(f"Attention参数: {info['attention_params_M']:.1f}M ({info['attention_ratio']*100:.1f}%)")
print(f"MLP参数: {info['mlp_params_M']:.1f}M ({info['mlp_ratio']*100:.1f}%)")
```

---

## 📝 输出格式

### JSON结果文件

```json
{
  "model_path": "prune_log/ours/pytorch_model.bin",
  "timestamp": "2025-11-18T12:00:00",
  "metrics": {
    "model_info": {
      "total_params": 6024195936,
      "total_params_B": 6.02,
      "attention_params_M": 1024.5,
      "mlp_params_M": 4999.7
    },
    "ppl": {
      "wikitext2 (wikitext-2-raw-v1)": 38.46,
      "ptb": 42.31
    },
    "zeroshot": {
      "hellaswag": {"accuracy": 0.752},
      "piqa": {"accuracy": 0.768}
    },
    "avg_zeroshot_acc": 0.760,
    "efficiency": {
      "speed": {
        "batch_size_1": {
          "throughput_tokens_per_sec": 168.7,
          "latency_ms_per_token": 5.93
        }
      },
      "memory": {
        "model_memory_mb": 12288.5,
        "inference_peak_mb": 14563.2
      }
    }
  }
}
```

---

## 🎯 常见使用场景

### 场景1: 论文实验 - 完整评估

```bash
# 评估所有模型的所有指标
for model in original ours baseline1 baseline2; do
    python evaluation/run_evaluation.py \
        --model_path models/${model}/pytorch_model.bin \
        --metrics ppl,zeroshot,speed,memory \
        --output results/${model}.json
done

# 生成对比表格
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/original.json,results/ours.json,results/baseline1.json,results/baseline2.json \
    --output paper_table.md
```

---

### 场景2: 快速验证 - 只测PPL

```bash
python evaluation/run_evaluation.py \
    --model_path prune_log/test/pytorch_model.bin \
    --metrics ppl \
    --ppl_datasets wikitext2 \
    --output quick_test.json
```

---

### 场景3: 性能深入分析 - 多数据集PPL

```bash
python evaluation/run_evaluation.py \
    --model_path prune_log/ours/pytorch_model.bin \
    --metrics ppl \
    --ppl_datasets wikitext2,ptb,c4 \
    --output ppl_analysis.json
```

---

## ⚠️ 注意事项

### 1. Zero-shot评估限制

**问题**: Zero-shot需要HF格式模型，不支持直接加载`.bin` checkpoint

**解决方案**:
```bash
# 方法1: 剪枝时保存为完整HF格式
python llama3_unbalanced_pruning_gqa_aware.py \
    --save_model \
    --save_ckpt_log_name my_model  # 保存到prune_log/my_model/

# 方法2: 手动转换checkpoint
# （需要额外脚本，暂未提供）
```

---

### 2. 显存不足

**问题**: 评估时OOM

**解决方案**:
```bash
# 减少速度测试样本数
--speed_samples 20

# 或只测试小batch size
# (修改代码中的batch_sizes参数)
```

---

### 3. lm-eval安装问题

**问题**: `ModuleNotFoundError: No module named 'lm_eval'`

**解决方案**:
```bash
pip install lm-eval

# 如果需要最新版
pip install git+https://github.com/EleutherAI/lm-evaluation-harness
```

---

## 📚 参考资料

- [lm-evaluation-harness文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- 项目主README: `../README.md`

---

## 🤝 贡献

如需添加新的评估指标或改进现有功能，请：
1. 在对应的`metrics/`模块中添加功能
2. 更新`run_evaluation.py`以支持新指标
3. 更新本README
