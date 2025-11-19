# 评估模块 (Evaluation Module)

统一的模型评估套件，用于评估剪枝模型的性能、质量和效率。

## 目录结构

```
evaluation/
├── README.md                    # 本文档
├── __init__.py                  # 包初始化
├── run_evaluation.py            # 统一评估入口
├── download_datasets.py         # 数据集下载脚本
│
├── metrics/                     # 评估指标
│   ├── __init__.py
│   ├── ppl.py                   # PPL 困惑度评估
│   ├── zeroshot.py              # 自定义 Zero-shot 评估器
│   ├── performance.py           # lm-eval 接口（备用）
│   └── efficiency.py            # 效率评估（速度、显存）
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── model_loader.py          # 模型加载器
│   └── get_best_gpu.py          # GPU 选择
│
├── tasks/                       # 任务目录（保留）
│   └── __init__.py
│
└── docs/                        # 文档
    └── dataset_download.md      # 数据集下载说明
```

## 快速开始

### 1. 下载数据集

首次使用前需要下载评估数据集：

```bash
python evaluation/download_datasets.py
```

这会下载以下数据集到 `data/zeroshot/` 目录：
- BoolQ, PIQA, HellaSwag, Winogrande
- ARC-Easy, ARC-Challenge, OpenBookQA

### 2. 运行评估

#### 基本用法

```bash
# 评估 PPL 和速度
python evaluation/run_evaluation.py \
    --model_path prune_log/experiment/pytorch_model.bin \
    --metrics ppl,speed \
    --output results/model_eval.json

# 评估 Zero-shot 准确率
python evaluation/run_evaluation.py \
    --model_path prune_log/experiment/pytorch_model.bin \
    --metrics zeroshot \
    --output results/zeroshot_eval.json

# 全部评估
python evaluation/run_evaluation.py \
    --model_path prune_log/experiment/pytorch_model.bin \
    --metrics all \
    --output results/full_eval.json
```

#### 自动选择 GPU

```bash
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl,zeroshot \
    --auto_select_gpu \
    --output results.json
```

#### 指定 GPU

```bash
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl \
    --device cuda:2 \
    --output results.json
```

### 3. 对比多个模型

```bash
# 先评估各个模型
python evaluation/run_evaluation.py --model_path model1.bin --metrics all --output results/model1.json
python evaluation/run_evaluation.py --model_path model2.bin --metrics all --output results/model2.json

# 生成对比表格
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/model1.json,results/model2.json \
    --output comparison.md
```

## 评估指标详解

### PPL (困惑度)

衡量语言模型的语言建模能力，越低越好。

```bash
--metrics ppl
--ppl_datasets wikitext2,ptb  # 默认数据集
```

**输出示例**:
```json
{
  "ppl": {
    "wikitext2": 12.34,
    "ptb": 45.67
  }
}
```

### Zero-shot 准确率

衡量模型在未见任务上的推理能力。

```bash
--metrics zeroshot
--zeroshot_tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
--zeroshot_batch_size 8  # 批处理大小（加速评估）
```

**支持的任务**:
| 任务 | 描述 | 选项数 |
|------|------|--------|
| boolq | 是非问答 | 2 |
| piqa | 物理常识推理 | 2 |
| hellaswag | 常识推理 | 4 |
| winogrande | 代词消歧 | 2 |
| arc_easy | 科学问答（简单） | 4 |
| arc_challenge | 科学问答（困难） | 4 |
| openbookqa | 科学推理 | 4 |

**输出示例**:
```json
{
  "zeroshot": {
    "boolq": {"accuracy": 0.78, "correct": 780, "total": 1000},
    "piqa": {"accuracy": 0.79, "correct": 1440, "total": 1838},
    ...
  },
  "avg_zeroshot_acc": 0.65
}
```

### 效率评估

衡量模型的推理速度和显存占用。

```bash
--metrics speed,memory
# 或
--metrics efficiency  # 同时测速度和显存
--speed_samples 50    # 测速样本数
```

**输出示例**:
```json
{
  "efficiency": {
    "throughput": {
      "batch_1": 45.2,   // tokens/sec
      "batch_4": 120.5
    },
    "latency_ms": {
      "batch_1": 22.1,
      "batch_4": 33.2
    },
    "memory": {
      "peak_mb": 15234,
      "allocated_mb": 14500
    }
  }
}
```

## 自定义 Zero-shot 评估器

本项目使用自定义 Zero-shot 评估器，不依赖 lm-eval 库，具有以下优势：

1. **完全离线** - 无需网络访问
2. **批处理支持** - 显著提升评估速度
3. **更好控制** - 可自定义数据格式和评估逻辑

### 直接使用评估器

```bash
python evaluation/metrics/zeroshot.py \
    --model_path your_model.bin \
    --tasks piqa,boolq \
    --batch_size 16 \
    --device cuda:0
```

### 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--batch_size` | 8 | 批处理大小 |
| `--no_batch` | False | 禁用批处理 |
| `--data_dir` | `data/zeroshot` | 数据目录 |

### 评估原理

1. 对每个问题，构造 (context, choice) 对
2. 计算每个 choice 的 log-likelihood
3. 选择 log-likelihood 最高的作为预测答案
4. 与标签对比计算准确率

## 数据目录结构

```
data/
└── zeroshot/
    ├── boolq/
    │   └── validation.jsonl
    ├── piqa/
    │   └── validation.jsonl
    ├── hellaswag/
    │   └── validation.jsonl
    ├── winogrande/
    │   └── validation.jsonl
    ├── arc_easy/
    │   └── validation.jsonl
    ├── arc_challenge/
    │   └── validation.jsonl
    └── openbookqa/
        └── validation.jsonl
```

每个 JSONL 文件包含验证集样本，格式因任务而异。

## API 使用

### 在代码中调用

```python
from evaluation.run_evaluation import evaluate_single_model
from evaluation.metrics.zeroshot import evaluate_zeroshot_custom
from evaluation.utils.model_loader import load_model_and_tokenizer

# 方式1: 使用统一接口
results = evaluate_single_model(
    model_path='prune_log/experiment/pytorch_model.bin',
    metrics=['ppl', 'zeroshot'],
    device='cuda:0',
    zeroshot_batch_size=16
)

# 方式2: 直接调用评估器
model, tokenizer = load_model_and_tokenizer(model_path, device='cuda:0')
zeroshot_results = evaluate_zeroshot_custom(
    model, tokenizer,
    tasks=['piqa', 'boolq'],
    device='cuda:0',
    batch_size=16
)
```

### 加载模型

```python
from evaluation.utils.model_loader import load_model_and_tokenizer

# 加载 checkpoint
model, tokenizer = load_model_and_tokenizer(
    'prune_log/experiment/pytorch_model.bin',
    device='cuda:0',
    force_single_device=True  # 直接加载到目标 GPU
)

# 加载 HuggingFace 模型
model, tokenizer = load_model_and_tokenizer(
    '/path/to/hf_model',
    device='cuda:0'
)
```

## 命令行参数完整说明

```bash
python evaluation/run_evaluation.py --help
```

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_path` | str | 必需 | 模型路径 |
| `--metrics` | str | `ppl,speed` | 评估指标，逗号分隔 |
| `--output` | str | 必需 | 输出文件路径 |
| `--device` | str | `cuda` | 设备 |
| `--auto_select_gpu` | flag | False | 自动选择空闲 GPU |
| `--ppl_datasets` | str | `wikitext2,ptb` | PPL 数据集 |
| `--zeroshot_tasks` | str | 全部7个 | Zero-shot 任务 |
| `--zeroshot_batch_size` | int | 8 | Zero-shot 批大小 |
| `--speed_samples` | int | 50 | 速度测试样本数 |
| `--use_lm_eval` | flag | False | 使用 lm-eval（备用） |
| `--compare` | flag | False | 对比模式 |
| `--model_paths` | str | - | 对比模式的结果文件 |

## 常见问题

### Q: 评估速度很慢？

A: 增加 `--zeroshot_batch_size`（如 16 或 32），但需要更多显存。

### Q: 显存不足？

A: 减少 `--zeroshot_batch_size`（如 4 或 2）。

### Q: 数据集文件不存在？

A: 运行 `python evaluation/download_datasets.py` 下载数据。

### Q: 想使用 lm-eval？

A: 添加 `--use_lm_eval` 参数（可能有网络问题）。

## 更新日志

- **v2.1** (2024-11): 添加批处理支持，显著提升 Zero-shot 评估速度
- **v2.0** (2024-11): 自定义 Zero-shot 评估器，完全离线运行
- **v1.0** (2024-10): 初始版本，基于 lm-eval

## 参考

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
