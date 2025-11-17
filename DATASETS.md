# 数据集使用分析

## 项目中使用的数据集

### 1. WikiText-2 (wikitext-2-raw-v1) ⭐ **必需**

**用途**：
- **层重要性分析** (`llama3_unbalanced_pruning_v3_gqa_aware.py:load_evaluation_data()`)
  - 评估每层对模型性能的影响
  - 默认加载 50-100 个样本

- **Taylor Importance 计算** (`LLMPruner/datasets/example_samples.py:get_examples()`)
  - 提供样本数据用于前向和反向传播
  - 计算梯度以确定 GQA 组的重要性
  - 默认 10 个样本，每个长度 64 tokens

- **困惑度 (PPL) 评估** (`LLMPruner/evaluator/ppl.py`)
  - 评估剪枝前后模型质量
  - 使用 test split

**能否删除**: ❌ 不能，这是核心数据集

---

### 2. Penn TreeBank (ptb) 🔷 可选

**用途**：
- 仅用于 PPL 评估的额外对比基准
- 在 `PPLMetric` 中可以作为可选数据集

**能否删除**: ✅ 可以删除
- 不影响核心剪枝流程
- 只是提供额外的评估指标

---

### 3. WikiText-103 (wikitext-103-raw-v1) 🔷 可选

**用途**：
- 仅在 `example_samples.py` 和 `ppl.py` 中支持
- 更大规模的评估数据集
- 当前脚本并未使用

**能否删除**: ✅ 可以删除
- 当前配置下完全未使用
- 可以从代码中移除相关支持

---

### 4. C4 🔷 可选

**用途**：
- 仅在 `example_samples.py` 和 `ppl.py` 中支持
- 大规模网络爬虫数据集
- 当前脚本并未使用

**能否删除**: ✅ 可以删除
- 当前配置下完全未使用
- 下载量大且占用空间
- 可以从代码中移除相关支持

---

## 数据集使用流程图

```
启动剪枝流程
    ↓
1. 加载 WikiText-2 (test split, ~100 samples)
   → 评估层重要性
    ↓
2. 加载 WikiText-2 (train split, ~10 samples)
   → 计算 Taylor Importance 梯度
    ↓
3. 执行 GQA-aware 剪枝
    ↓
4. 加载 WikiText-2 (test split, full)
   → 评估剪枝后的 PPL
    ↓
完成
```

---

## 推荐配置

### 最小化配置（推荐）

**仅保留 WikiText-2**，删除其他数据集支持：

**优点**：
- 减少依赖和代码复杂度
- 加快数据集下载速度
- 降低存储空间需求
- 简化评估流程

**修改建议**：
1. 从 `example_samples.py` 中删除 wikitext103, c4, ptb 的分支
2. 从 `ppl.py` 中删除 wikitext103, c4, ptb 的分支
3. 将 `--test_after_prune` 固定使用 wikitext2

### 完整配置（当前）

保留所有数据集支持，用于：
- 跨数据集对比实验
- 论文写作需要多个评估基准
- 研究不同数据集的影响

---

## 代码修改指南

如果要删除可选数据集，需要修改以下文件：

### 1. `LLMPruner/datasets/example_samples.py`

```python
# 删除这些分支：
elif dataset_name.lower() in ['wikitext103', 'wikitext-103']:
    ...
elif dataset_name.lower() == 'c4':
    ...
elif dataset_name.lower() in ['ptb', 'penn-treebank']:
    ...

# 只保留：
if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    text_field = 'text'
else:
    raise ValueError(f"仅支持 wikitext2 数据集")
```

### 2. `LLMPruner/evaluator/ppl.py`

```python
# 同样删除 wikitext103, c4, ptb 的分支
# 只保留 wikitext2 支持
```

### 3. `llama3_unbalanced_pruning_v3_gqa_aware.py`

```python
# 第 415 行固定使用 wikitext2
ppl = PPLMetric(model, tokenizer, ['wikitext2'],  # 删除 'ptb'
               seq_len=args.max_seq_len, device=args.device)
```

---

## 数据集大小参考

| 数据集 | 下载大小 | 解压后 | 样本数 (test) |
|--------|---------|--------|--------------|
| WikiText-2 | ~4 MB | ~12 MB | ~4K lines |
| WikiText-103 | ~181 MB | ~500 MB | ~60K lines |
| PTB | ~5 MB | ~15 MB | ~3.7K sentences |
| C4 | ~300 GB+ | ~1 TB+ | 数十亿 documents |

---

## 总结

### 必需数据集
- ✅ **WikiText-2**: 核心数据集，不可删除

### 可删除数据集
- ⚠️ **WikiText-103**: 当前未使用，可删除
- ⚠️ **PTB**: 仅用于额外评估，可删除
- ⚠️ **C4**: 当前未使用且体积巨大，强烈建议删除

### 建议操作
对于生产环境和日常使用，建议：
1. **仅保留 WikiText-2**
2. 删除其他数据集的代码支持
3. 简化配置和依赖

对于研究和论文写作，可以：
1. 保留当前完整配置
2. 在需要时启用其他数据集
3. 进行跨数据集对比实验
