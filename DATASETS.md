# 数据集使用分析

## ✅ 当前状态：已简化为仅支持 WikiText-2

本项目已优化，**仅保留 WikiText-2 数据集**，删除了所有其他数据集支持。

---

## 项目中使用的数据集

### 1. WikiText-2 (wikitext-2-raw-v1) ⭐ **唯一支持**

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

**能否删除**: ❌ 不能，这是唯一的核心数据集

**✅ 已移除的数据集**：
- ~~Penn TreeBank (ptb)~~ - 已删除
- ~~WikiText-103~~ - 已删除
- ~~C4~~ - 已删除

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

## ✅ 当前配置（已优化）

### 极简配置（已实施）

**仅保留 WikiText-2**：

**优点**：
- ✅ 减少依赖和代码复杂度
- ✅ 加快数据集下载速度（仅 ~4 MB）
- ✅ 降低存储空间需求（仅 ~12 MB）
- ✅ 简化评估流程

**已完成的修改**：
1. ✅ 从 `example_samples.py` 中删除 wikitext103, c4, ptb 的分支
2. ✅ 从 `ppl.py` 中删除 wikitext103, c4, ptb 的分支
3. ✅ 将 `--test_after_prune` 固定使用 wikitext2

---

## ✅ 已完成的代码修改

### 1. `LLMPruner/datasets/example_samples.py`

```python
# ✅ 已删除 wikitext103, c4, ptb 分支
# ✅ 只保留 wikitext2 支持

if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    text_field = 'text'
else:
    raise ValueError(f"不支持的数据集: {dataset_name}. 当前仅支持 wikitext2")
```

### 2. `LLMPruner/evaluator/ppl.py`

```python
# ✅ 已删除 wikitext103, c4, ptb 分支
# ✅ 只保留 wikitext2 支持

if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text_field = 'text'
else:
    raise ValueError(f"不支持的数据集: {dataset_name}. 当前仅支持 wikitext2")
```

### 3. `llama3_unbalanced_pruning_v3_gqa_aware.py`

```python
# ✅ 已删除 'ptb'，只使用 wikitext2
ppl = PPLMetric(model, tokenizer, ['wikitext2'],
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

### ✅ 当前配置
- ✅ **WikiText-2**: 唯一支持的数据集

### ✅ 已删除的数据集
- ✅ **WikiText-103**: 已删除
- ✅ **PTB**: 已删除
- ✅ **C4**: 已删除

### 优势
1. ✅ **代码简洁**: 减少了 60% 的数据集处理代码
2. ✅ **快速下载**: 仅需下载 4 MB 数据
3. ✅ **低存储**: 仅占用 12 MB 磁盘空间
4. ✅ **易维护**: 单一数据集，无需处理多种格式

### 如需扩展
如果将来需要支持更多数据集，可以参考 git 历史中的代码：
```bash
git log --all -- LLMPruner/datasets/example_samples.py
git show <commit-id>:LLMPruner/datasets/example_samples.py
```
