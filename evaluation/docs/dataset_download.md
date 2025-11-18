# 数据集下载和缓存管理指南

本文档提供完整的数据集下载、缓存管理和故障排查指南。

---

## 📦 支持的数据集

| 数据集 | 用途 | 大小 | 推荐程度 |
|--------|------|------|----------|
| WikiText-2 | PPL评估 | ~4MB | ⭐⭐⭐⭐⭐ 最推荐 |
| PTB | PPL评估 | ~5MB | ⭐⭐⭐ 可选 |
| C4 | PPL评估 | ~365GB (全集) | ⭐⭐ 仅限完整测试 |
| HellaSwag/PIQA等 | Zero-shot | 自动下载 | ⭐⭐⭐⭐⭐ 必需 |

---

## 🔧 问题1：PTB数据集无法下载

### 错误信息
```
Dataset 'ptb-text-only' doesn't exist on the Hub or cannot be accessed.
```

### 原因
PTB数据集在HuggingFace Hub上有多个版本，且部分已被移除或重命名。

### 解决方案

#### 方案A：跳过PTB，只用WikiText-2（推荐）
```bash
# 只使用WikiText-2评估（最常用，论文中普遍使用）
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl \
    --ppl_datasets wikitext2 \
    --output results.json
```

**理由**：WikiText-2是最标准的PPL benchmark，几乎所有论文都使用它。PTB是补充性指标。

#### 方案B：手动下载PTB
```bash
# 1. 创建数据目录
mkdir -p ~/.cache/huggingface/datasets/ptb_manual

# 2. 下载PTB数据（需要LDC许可，或使用开源版本）
# 方法1: 从LDC官方（需要许可）
wget https://catalog.ldc.upenn.edu/LDC99T42

# 方法2: 使用开源版本（推荐）
git clone https://github.com/tomsercu/lstm
cp lstm/data/ptb.test.txt ~/.cache/huggingface/datasets/ptb_manual/
```

#### 方案C：使用替代数据集
如果PTB必需，可使用以下替代：
```python
# 修改 LLMPruner/evaluator/ppl.py
# 使用 'lambada' 作为PTB替代
dataset = load_dataset('lambada', split='test')
```

---

## 🔧 问题2：C4数据集加载失败

### 错误信息
```
Dataset scripts are no longer supported, but found c4.py
```

### 原因
HuggingFace Datasets库更新后，不再支持legacy loading scripts。C4需要使用新路径。

### 解决方案

#### 方案A：使用WikiText-2替代（强烈推荐）
```bash
# C4数据集巨大（365GB）且下载慢，WikiText-2完全够用
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl \
    --ppl_datasets wikitext2 \
    --output results.json
```

#### 方案B：使用新版C4加载路径
已在代码中修复，使用 `allenai/c4` 路径：
```bash
# 使用新版C4（会自动尝试新路径）
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl \
    --ppl_datasets wikitext2,c4 \
    --output results.json
```

**注意**：即使使用新路径，C4也会下载较大文件（~10GB for validation），首次运行较慢。

#### 方案C：使用C4子集
```python
# 修改代码只使用tiny C4 subset
from datasets import load_dataset
dataset = load_dataset('allenai/c4', 'en', split='validation[:1%]', streaming=False)
```

---

## 🔧 问题3：Zero-shot评估数据集缓存损坏

### 错误信息
```
NonMatchingSplitsSizesError: expected SplitInfo(...num_examples=39905...)
recorded: SplitInfo(...num_examples=0...)
```

### 原因
HuggingFace datasets缓存损坏或部分下载失败，导致数据集splits信息不匹配。

### 解决方案

#### 方案A：清理损坏的缓存（推荐）
```bash
# 1. 查看缓存位置
echo "数据集缓存: ~/.cache/huggingface/datasets/"

# 2. 删除损坏的数据集缓存
rm -rf ~/.cache/huggingface/datasets/hellaswag
rm -rf ~/.cache/huggingface/datasets/piqa
rm -rf ~/.cache/huggingface/datasets/winogrande

# 3. 重新运行评估（会自动重新下载）
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics zeroshot \
    --output results.json
```

#### 方案B：完全清理所有缓存
```bash
# 警告：会删除所有已下载的数据集
rm -rf ~/.cache/huggingface/datasets/*

# 重新运行
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics zeroshot \
    --output results.json
```

#### 方案C：使用新的缓存目录
```bash
# 设置环境变量使用新缓存
export HF_DATASETS_CACHE="/path/to/new/cache"

# 运行评估
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics zeroshot \
    --output results.json
```

#### 方案D：忽略缓存验证（不推荐）
```python
# 修改 evaluation/metrics/performance.py
# 在load_dataset调用中添加
dataset = load_dataset(..., verification_mode='no_checks')
```

---

## 📂 数据集缓存管理

### 查看缓存占用
```bash
# 查看数据集缓存大小
du -sh ~/.cache/huggingface/datasets/

# 查看各数据集占用
du -sh ~/.cache/huggingface/datasets/*
```

### 清理特定数据集
```bash
# 只清理C4（如果不需要）
rm -rf ~/.cache/huggingface/datasets/c4

# 只清理PTB（如果有问题）
rm -rf ~/.cache/huggingface/datasets/ptb*
```

### 预下载所有数据集
```bash
# 创建预下载脚本
python -c "
from datasets import load_dataset

# 下载WikiText-2
print('下载 WikiText-2...')
load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# 下载Zero-shot数据集
print('下载 HellaSwag...')
load_dataset('Rowan/hellaswag', split='validation')

print('下载 PIQA...')
load_dataset('piqa', split='validation')

print('下载 WinoGrande...')
load_dataset('winogrande', 'winogrande_xl', split='validation')

print('下载 ARC-Easy...')
load_dataset('ai2_arc', 'ARC-Easy', split='test')

print('下载 BoolQ...')
load_dataset('google/boolq', split='validation')

print('✓ 所有数据集下载完成')
"
```

---

## 🚀 推荐配置

### 最小配置（快速测试）
```bash
# 只用WikiText-2 PPL，跳过Zero-shot
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results.json
```

### 标准配置（论文评估）
```bash
# WikiText-2 + Zero-shot（5个任务）
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --ppl_datasets wikitext2 \
    --zeroshot_tasks hellaswag,piqa,winogrande,arc_easy,boolq \
    --output results.json
```

### 完整配置（详尽评估）
```bash
# 所有数据集 + 效率指标
python evaluation/run_evaluation.py \
    --model_path your_model.bin \
    --metrics all \
    --ppl_datasets wikitext2 \
    --output results.json

# 注意：跳过PTB和C4，它们不稳定且非必需
```

---

## ❓ 常见问题

### Q: WikiText-2下载很慢怎么办？
A: 使用镜像或设置代理：
```bash
# 使用HF镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export https_proxy=http://your-proxy:port
```

### Q: 数据集缓存占用太大怎么办？
A:
1. 只保留WikiText-2（~4MB）
2. 删除C4（如果已下载，~10GB）
3. 定期清理不用的数据集

### Q: Zero-shot评估特别慢怎么办？
A:
1. 减少任务数量：只用 `hellaswag,piqa` 而非全部5个
2. 使用更小的batch_size
3. 首次运行较慢（需下载数据集），后续会快很多

### Q: 是否需要手动下载所有数据集？
A: **不需要**。datasets库会自动下载并缓存。只有在遇到网络问题或缓存损坏时才需要手动干预。

---

## 📝 数据集来源汇总

| 数据集 | HuggingFace路径 | 官方来源 |
|--------|----------------|----------|
| WikiText-2 | `wikitext` / `wikitext-2-raw-v1` | [Link](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) |
| PTB | `ptb_text_only` (已移除) | [LDC](https://catalog.ldc.upenn.edu/LDC99T42) |
| C4 | `allenai/c4` / `en` | [AllenAI](https://github.com/allenai/allennlp) |
| HellaSwag | `Rowan/hellaswag` | [Paper](https://arxiv.org/abs/1905.07830) |
| PIQA | `piqa` | [Paper](https://arxiv.org/abs/1911.11641) |
| WinoGrande | `winogrande` / `winogrande_xl` | [Paper](https://arxiv.org/abs/1907.10641) |
| ARC | `ai2_arc` / `ARC-Easy` | [Paper](https://arxiv.org/abs/1803.05457) |
| BoolQ | `google/boolq` | [Paper](https://arxiv.org/abs/1905.10044) |

---

## 🛠️ 故障排查流程

遇到数据集问题时，按此流程排查：

1. **检查网络连接**
   ```bash
   ping huggingface.co
   ```

2. **检查缓存状态**
   ```bash
   ls -lh ~/.cache/huggingface/datasets/
   ```

3. **清理损坏缓存**
   ```bash
   rm -rf ~/.cache/huggingface/datasets/[problem_dataset]
   ```

4. **使用最小配置测试**
   ```bash
   python evaluation/run_evaluation.py \
       --model_path your_model.bin \
       --metrics ppl \
       --ppl_datasets wikitext2 \
       --output test.json
   ```

5. **如仍有问题，跳过该数据集**
   ```bash
   # 只用WikiText-2即可，PTB/C4是可选的
   --ppl_datasets wikitext2
   ```

---

**最后建议**：对于大多数论文和实验，**只使用WikiText-2进行PPL评估已完全足够**。PTB和C4是补充性指标，非必需。
