# 代码重构计划

## 问题1: 数据集加载重复

### 当前状况
```python
# 步骤2：层重要性评估
eval_texts = load_evaluation_data(tokenizer, 50)  # 从 wikitext2 加载文本列表

# 步骤4：Taylor importance
example_prompts = get_examples('wikitext', tokenizer, 10)  # 又从 wikitext2 加载并tokenize
```

### 优化方案
**统一数据加载，避免重复**

```python
# 在主脚本开始时，一次性加载所有需要的样本
max_samples = max(args.importance_samples, args.num_examples)
eval_texts = load_evaluation_data(tokenizer, num_samples=max_samples)

# 步骤2：使用前 importance_samples 个样本
layer_importance = analyzer.measure_layer_importance_by_removal(
    eval_texts[:args.importance_samples],
    num_layers
)

# 步骤4：使用前 num_examples 个样本，用 get_examples_from_text() 转换
example_prompts = get_examples_from_text(
    eval_texts[:args.num_examples],
    tokenizer,
    seq_len=64
).to(device)
```

**优势**：
- ✅ 只加载一次数据集
- ✅ 代码逻辑更清晰
- ✅ 复用 `get_examples_from_text()` 工具函数
- ✅ 灵活：可以让两个步骤使用相同或不同的样本

### 实施步骤
1. 修改主脚本，在步骤1后立即加载数据
2. 删除 `load_evaluation_data()` 函数（功能被 `get_examples()` 替代）
3. 使用 `get_examples_from_text()` 进行 tokenization

---

## 问题2: 代码组织优化

### 当前结构
```
GAQ-Aware-Prune/
├── gqa_aware_pruning.py           # ❌ 顶层文件
├── layer_importance.py            # ❌ 顶层文件
├── llama3_unbalanced_pruning_v3_gqa_aware.py
└── LLMPruner/                     # ❌ 名字不准确
    ├── utils/
    ├── evaluator/
    └── datasets/
```

### 优化方案1: 移动到 LLMPruner 模块内

```
GAQ-Aware-Prune/
├── llama3_unbalanced_pruning_v3_gqa_aware.py  # 主入口
└── LLMPruner/
    ├── __init__.py
    ├── methods/                   # 新增：剪枝方法
    │   ├── __init__.py
    │   └── gqa_aware.py          # 从 gqa_aware_pruning.py 移动
    ├── importance/                # 新增：重要性分析
    │   ├── __init__.py
    │   └── layer_analyzer.py     # 从 layer_importance.py 移动
    ├── utils/
    │   ├── __init__.py
    │   ├── logger.py
    │   └── get_best_gpu.py
    ├── evaluator/
    │   ├── __init__.py
    │   └── ppl.py
    └── datasets/
        ├── __init__.py
        └── example_samples.py
```

**导入方式**：
```python
# 旧方式
from gqa_aware_pruning import compute_gqa_group_importance
from layer_importance import LayerImportanceAnalyzer

# 新方式
from LLMPruner.methods.gqa_aware import compute_gqa_group_importance
from LLMPruner.importance.layer_analyzer import LayerImportanceAnalyzer
```

### 优化方案2: 重命名 LLMPruner 模块

如果觉得 `LLMPruner` 名字不够准确，可以改为：

**选项A: `pruner` （简洁）**
```python
from pruner.methods.gqa_aware import ...
from pruner.importance.layer_analyzer import ...
```

**选项B: `llama_pruner` （明确针对 Llama）**
```python
from llama_pruner.methods.gqa_aware import ...
from llama_pruner.importance.layer_analyzer import ...
```

**选项C: `pruning_toolkit` （工具箱）**
```python
from pruning_toolkit.methods.gqa_aware import ...
from pruning_toolkit.importance.layer_analyzer import ...
```

### 推荐方案
**保留 `LLMPruner` 名字，重组内部结构**

理由：
- `LLMPruner` 已经在多处使用，改名需要大量修改
- 通过重组内部结构，可以让模块更有组织性
- 符合 Python 模块命名惯例（大写开头）

### 实施步骤
1. 创建 `LLMPruner/methods/` 和 `LLMPruner/importance/` 目录
2. 移动文件并重命名：
   - `gqa_aware_pruning.py` → `LLMPruner/methods/gqa_aware.py`
   - `layer_importance.py` → `LLMPruner/importance/layer_analyzer.py`
3. 更新 `__init__.py` 文件，提供便捷导入
4. 更新主脚本的导入语句

---

## 优先级建议

1. **高优先级**：统一数据加载（问题1）
   - 立即可见的性能提升
   - 代码更简洁

2. **中优先级**：重组代码结构（问题2）
   - 提升可维护性
   - 为未来扩展做准备（如支持其他模型）

---

## 兼容性考虑

为了保持向后兼容，可以在 `LLMPruner/__init__.py` 中提供便捷导入：

```python
# LLMPruner/__init__.py
from .methods.gqa_aware import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

from .importance.layer_analyzer import (
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator
)

# 这样可以简化导入
# from LLMPruner import compute_gqa_group_importance, LayerImportanceAnalyzer
```
