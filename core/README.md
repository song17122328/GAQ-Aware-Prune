# core 工具模块

LLM剪枝所需的辅助工具模块集合。

## 模块结构

```
core/
├── __init__.py
├── datasets/              # 数据集加载模块
│   ├── __init__.py
│   └── example_samples.py # 样本数据加载
├── evaluator/            # 评估模块
│   ├── __init__.py
│   └── ppl.py           # 困惑度评估
└── utils/               # 工具模块
    ├── logger.py        # 日志工具
    └── get_best_gpu.py  # GPU选择工具
```

## 使用说明

### 1. 数据集模块 (`datasets`)

#### 加载样本数据用于梯度计算

```python
from core.datasets.example_samples import get_examples
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/newdata/LLMs/Llama-3-8B-Instruct")

# 从wikitext数据集加载10个样本，每个长度为64
examples = get_examples('wikitext', tokenizer, num_samples=10, seq_len=64)
examples = examples.to('cuda')  # 移动到GPU

# 用于模型前向传播
loss = model(examples, labels=examples).loss
```

#### 支持的数据集

- `wikitext` / `wikitext2` / `wikitext-2` - WikiText-2（唯一支持）

#### 其他函数

```python
# 从自定义文本创建样本
from core.datasets.example_samples import get_examples_from_text

texts = ["Hello world", "This is a test"]
examples = get_examples_from_text(texts, tokenizer, seq_len=128)

# 获取校准数据（用于量化）
from core.datasets.example_samples import get_calibration_data

calib_data = get_calibration_data('wikitext', tokenizer, num_samples=128)
```

### 2. 评估模块 (`evaluator`)

#### 计算困惑度 (Perplexity)

```python
from evaluation.metrics.ppl import PPLMetric
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/newdata/LLMs/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/newdata/LLMs/Llama-3-8B-Instruct")

# 评估 wikitext2 数据集
ppl_metric = PPLMetric(
    model,
    tokenizer,
    datasets=['wikitext2'],
    seq_len=128,
    device='cuda'
)

# 查看结果
print(ppl_metric)
# 输出:
#   wikitext2 (wikitext-2-raw-v1): 12.34

# 字典式访问
wikitext_ppl = ppl_metric['wikitext2 (wikitext-2-raw-v1)']
print(f"WikiText-2 PPL: {wikitext_ppl:.2f}")

# 使用 get 方法（带默认值）
ppl = ppl_metric.get('wikitext2 (wikitext-2-raw-v1)', 'N/A')
```

#### 快捷函数

```python
from evaluation.metrics.ppl import evaluate_perplexity

# 评估单个数据集
ppl = evaluate_perplexity(model, tokenizer, 'wikitext2', seq_len=128)
print(f"PPL: {ppl:.2f}")
```

### 3. 工具模块 (`utils`)

#### 日志工具

```python
from core.utils.logger import LoggerWithDepth

logger = LoggerWithDepth(
    env_name='my_experiment',
    config={'lr': 0.001, 'batch_size': 32},
    root_dir='logs',
    setup_sublogger=True
)

logger.log("实验开始...")
logger.log(f"参数: {config}")
```

输出将保存到：
- `logs/my_experiment/description.txt` - 配置信息
- `logs/my_experiment/{timestamp}/training.log` - 训练日志
- `logs/my_experiment/pytorch_model.bin` - 最佳检查点

#### GPU选择工具

```python
from core.utils.get_best_gpu import get_best_gpu

# 自动选择显存最多的GPU
gpu_id = get_best_gpu()
device = f'cuda:{gpu_id}'
```

## 依赖项

```bash
pip install torch transformers datasets tqdm numpy matplotlib
```

## 测试

每个模块都包含测试代码：

```bash
# 测试数据集加载
python core/datasets/example_samples.py

# 测试PPL评估
python -c "from evaluation.metrics.ppl import PPLMetric"
```

## 使用示例

完整的剪枝流程示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.datasets.example_samples import get_examples
from evaluation.metrics.ppl import PPLMetric

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("/newdata/LLMs/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/newdata/LLMs/Llama-3-8B-Instruct")

# 2. 评估原始PPL
print("评估原始模型...")
baseline_ppl = PPLMetric(model, tokenizer, ['wikitext2'], seq_len=128, device='cuda')
print(f"Baseline PPL: {baseline_ppl}")

# 3. 获取样本用于计算梯度
examples = get_examples('wikitext', tokenizer, num_samples=10, seq_len=64).to('cuda')

# 4. 计算梯度
model.zero_grad()
loss = model(examples, labels=examples).loss
loss.backward()

# 5. 执行剪枝...
# (使用 gqa_aware_pruning.py 中的函数)

# 6. 评估剪枝后PPL
print("评估剪枝后模型...")
pruned_ppl = PPLMetric(model, tokenizer, ['wikitext2'], seq_len=128, device='cuda')
print(f"Pruned PPL: {pruned_ppl}")

# 7. 计算退化
degradation = (pruned_ppl['wikitext2 (wikitext-2-raw-v1)'] /
               baseline_ppl['wikitext2 (wikitext-2-raw-v1)'] - 1) * 100
print(f"PPL退化: {degradation:.2f}%")
```

## 注意事项

1. **设备管理**: 确保数据和模型在同一设备上
2. **内存优化**: 对于大模型，考虑使用较小的 `seq_len` 或 `batch_size`
3. **数据集下载**: 首次运行会自动下载数据集（需要网络连接）
4. **困惑度计算**: 使用滑动窗口方法，可能需要较长时间

## 故障排除

### 问题：CUDA OOM

**解决方案**:
- 减小 `seq_len`
- 减小 `num_samples`
- 使用 `model.half()` 转为FP16

### 问题：数据集下载失败

**解决方案**:
- 检查网络连接
- 设置 Hugging Face 镜像源
- 手动下载数据集到本地

### 问题：PPL结果为 NaN 或 Inf

**解决方案**:
- 检查模型是否已损坏
- 确认剪枝后 forward pass 正常
- 添加梯度裁剪
