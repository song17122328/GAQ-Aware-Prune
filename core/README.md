# core 工具模块

LLM剪枝所需的辅助工具模块集合。

## 模块结构

```
core/
├── __init__.py
├── methods/              # 剪枝方法模块
│   ├── __init__.py
│   ├── gqa_aware.py     # GQA感知的Taylor剪枝
│   └── global_pruning.py # 全局剪枝（基于Score=Importance/Cost）
├── importance/           # 层重要性分析模块
│   ├── __init__.py
│   └── layer_analyzer.py # 层重要性评估
├── trainer/             # 微调模块
│   ├── __init__.py
│   └── finetuner.py    # 剪枝后微调
├── datasets/            # 数据集加载模块
│   ├── __init__.py
│   └── example_samples.py # 样本数据加载
├── evaluator/          # 评估模块（已废弃，使用evaluation/）
│   ├── __init__.py
│   └── ppl.py         # 困惑度评估
└── utils/             # 工具模块
    ├── logger.py      # 日志工具
    └── get_best_gpu.py # GPU选择工具
```

## 使用说明

### 0. 剪枝方法模块 (`methods`)

提供两种剪枝策略：

#### 0.1 GQA-Aware 剪枝 (`gqa_aware.py`)

**核心思想**: 保持 GQA 架构的 4:1 比例（4个Q heads对应1个KV head），按组剪枝

**主要函数**:

```python
from core.methods.gqa_aware import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

# 1. 计算每个GQA组的Taylor importance
group_importance = compute_gqa_group_importance(layer, head_dim=128, gqa_ratio=4)
# 返回: Tensor[num_kv_heads] - 每个KV head对应的组重要性

# 2. 选择要剪枝的组（保留importance最高的）
keep_indices, prune_indices = select_gqa_groups_to_prune(
    group_importance,
    target_num_kv_heads=6  # 从8个KV heads剪到6个
)

# 3. 执行剪枝
num_q, num_kv = prune_attention_by_gqa_groups(
    layer,
    keep_indices,
    head_dim=128,
    gqa_ratio=4
)
# 剪枝后: 24Q:6KV (仍保持4:1)
```

**使用场景**: 逐层剪枝，每层根据自己的组重要性独立决定剪枝

---

#### 0.2 全局剪枝 (`global_pruning.py`)

**核心思想**: 计算每个group的 **Score = Importance / Cost**，全局排序后选择score最低的groups剪枝

**优势**:
- 跨层对比：可以比较不同层的groups
- 跨组件对比：可以比较Attention和MLP的groups
- 效率优先：优先剪掉"性价比"最低的groups

**主要函数**:

```python
from core.methods.global_pruning import (
    build_global_group_table,
    select_groups_to_prune,
    save_group_table,
    GroupInfo
)

# 1. 构建全局分析表
df = build_global_group_table(
    model=model,
    importance_method='taylor',  # 或 'wanda'
    layer_start=0,
    layer_end=32,
    head_dim=128,
    gqa_ratio=4
)
# 返回: pandas.DataFrame，包含所有groups的信息，按score排序

# 2. 选择要剪枝的groups
groups_to_prune = select_groups_to_prune(
    df=df,
    pruning_ratio=0.25,  # 剪掉25%的模型参数
    total_params=8_000_000_000
)

# 3. 保存分析表
save_group_table(df, 'prune_log/group_analysis.json')
```

**DataFrame 结构**:
```
| layer_idx | group_type | group_idx | importance | cost   | score      |
|-----------|------------|-----------|------------|--------|------------|
| 5         | attention  | 2         | 0.123456   | 524288 | 2.356e-07  |  ← score最低，优先剪枝
| 12        | mlp        | 1024      | 0.234567   | 12288  | 1.909e-05  |
| ...       | ...        | ...       | ...        | ...    | ...        |
| 15        | attention  | 4         | 9.876543   | 524288 | 1.884e-05  |  ← score最高，最不应该剪
```

**重要性计算方法**:
- **Taylor**: `importance = |weight × gradient|`（需要先计算梯度）
- **Wanda**: `importance = |weight × activation|`（需要收集激活值）

**参数成本计算**:
- **Attention Group**:
  ```
  cost = hidden_dim × (4×head_dim)  [q_proj]
       + hidden_dim × head_dim      [k_proj]
       + hidden_dim × head_dim      [v_proj]
       + (4×head_dim) × hidden_dim  [o_proj]
  ```
  对于 Llama-3-8B: `cost = 4096 × (4×128) + 4096×128 + 4096×128 + (4×128)×4096 = 6,291,456`

- **MLP Channel**:
  ```
  cost = hidden_dim  [gate_proj的一行]
       + hidden_dim  [up_proj的一行]
       + hidden_dim  [down_proj的一列]
  ```
  对于 Llama-3-8B: `cost = 4096 + 4096 + 4096 = 12,288`

**演示脚本**:
```bash
python demo_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_table_path prune_log/global_group_table.json \
    --pruning_ratio 0.25 \
    --importance_method taylor \
    --num_samples 10
```

**使用场景**: 需要全局最优剪枝策略，愿意牺牲计算时间换取更好的剪枝效果

---

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
