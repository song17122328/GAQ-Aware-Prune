# 评估模块快速开始指南

针对您遇到的三个问题的快速解决方案。

---

## ⚡ 问题1：GPU自动选择

### 解决方案
使用 `--auto_select_gpu` 参数自动选择剩余显存最多的GPU：

```bash
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path prune_log/xxx/pytorch_model.bin \
    --metrics ppl,speed,memory \
    --output results/ours.json
```

**说明**：
- 会自动检测所有GPU的剩余显存
- 选择剩余显存最大的GPU
- 覆盖 `--device` 参数

### 手动指定GPU（替代方案）
```bash
# 指定GPU 0
python evaluation/run_evaluation.py \
    --device cuda:0 \
    --model_path ... \
    --output results.json

# 指定GPU 3
python evaluation/run_evaluation.py \
    --device cuda:3 \
    --model_path ... \
    --output results.json
```

---

## ⚡ 问题2：PTB和C4数据集无法下载

### ✅ 推荐方案：只用WikiText-2
对于大多数论文，**WikiText-2已完全足够**，PTB和C4是可选的：

```bash
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path prune_log/xxx/pytorch_model.bin \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results/ours.json
```

**理由**：
- WikiText-2是最标准的PPL benchmark
- 几乎所有LLM论文都使用WikiText-2
- PTB和C4下载不稳定且非必需

### 如果必须使用PTB/C4
参见详细文档：`evaluation/docs/dataset_download.md`

---

## ⚡ 问题3：Zero-shot评估缓存损坏

### 错误信息
```
NonMatchingSplitsSizesError: expected SplitInfo(...num_examples=39905...)
recorded: SplitInfo(...num_examples=0...)
```

### 解决方案：清理损坏的缓存

#### 方法A：使用清理工具（推荐）
```bash
# 清理Zero-shot相关数据集（hellaswag, piqa等）
python evaluation/clean_dataset_cache.py --zeroshot

# 清理PPL相关数据集（wikitext, ptb, c4）
python evaluation/clean_dataset_cache.py --ppl

# 列出所有缓存
python evaluation/clean_dataset_cache.py --list

# 清理特定数据集
python evaluation/clean_dataset_cache.py --dataset hellaswag
```

#### 方法B：手动清理（快速）
```bash
# 删除损坏的数据集缓存
rm -rf ~/.cache/huggingface/datasets/hellaswag
rm -rf ~/.cache/huggingface/datasets/piqa
rm -rf ~/.cache/huggingface/datasets/winogrande
rm -rf ~/.cache/huggingface/datasets/ai2_arc
rm -rf ~/.cache/huggingface/datasets/google___boolq

# 重新运行评估（会自动重新下载）
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path your_model.bin \
    --metrics zeroshot \
    --output results.json
```

#### 方法C：完全清理（最彻底）
```bash
# ⚠️ 警告：删除所有数据集缓存
python evaluation/clean_dataset_cache.py --all
```

---

## 🚀 完整评估流程

### 步骤1：评估剪枝后的模型
```bash
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path prune_log/ppl_search_20251118_005448_ratio_0.7_9.3_freeze_8/pytorch_model.bin \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results/ours.json
```

### 步骤2：评估原始模型
```bash
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results/original.json
```

### 步骤3：添加Zero-shot评估（可选）

**首先清理缓存**（如果之前遇到错误）：
```bash
python evaluation/clean_dataset_cache.py --zeroshot
```

**然后运行Zero-shot**：
```bash
# 剪枝模型（.bin文件直接支持）
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path prune_log/xxx/pytorch_model.bin \
    --metrics zeroshot \
    --zeroshot_tasks hellaswag,piqa,winogrande \
    --output results/ours_zeroshot.json

# 原始模型（HF格式）
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics zeroshot \
    --zeroshot_tasks hellaswag,piqa,winogrande \
    --output results/original_zeroshot.json
```

### 步骤4：生成对比表
```bash
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/original.json,results/ours.json \
    --output results/comparison.md
```

---

## 📊 推荐的最小评估配置

对于论文和实验，以下配置已完全足够：

```bash
# 评估剪枝模型
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path prune_log/xxx/pytorch_model.bin \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results/ours.json

# 评估原始模型
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results/original.json

# 生成对比
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/original.json,results/ours.json \
    --output results/comparison.md
```

**包含的指标**：
- ✅ PPL (WikiText-2) - 语言建模能力
- ✅ 参数量 - 压缩率
- ✅ 推理速度 (tokens/s) - 效率提升
- ✅ GPU显存占用 - 资源节省

**这些指标足以证明您的剪枝方法的有效性！**

---

## 🔍 故障排查

### 问题：GPU自动选择不工作
```bash
# 检查nvidia-smi是否可用
nvidia-smi

# 如果不可用，手动指定GPU
python evaluation/run_evaluation.py --device cuda:0 ...
```

### 问题：数据集下载很慢
```bash
# 使用HuggingFace镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export https_proxy=http://your-proxy:port
```

### 问题：Zero-shot一直失败
```bash
# 1. 完全清理缓存
python evaluation/clean_dataset_cache.py --all

# 2. 只评估PPL（跳过Zero-shot）
python evaluation/run_evaluation.py \
    --auto_select_gpu \
    --model_path your_model.bin \
    --metrics ppl,speed,memory \
    --ppl_datasets wikitext2 \
    --output results.json
```

### 问题：OOM错误
```bash
# 使用force_single_device避免多GPU问题
# 代码已自动处理，但如果仍有问题：

# 1. 减少batch size
--speed_samples 20  # 默认50

# 2. 使用更大的GPU
--device cuda:0  # 选择显存更大的GPU

# 3. 跳过speed测试
--metrics ppl,memory  # 不测试speed
```

---

## 📚 更多文档

- 完整评估指南：`evaluation/README.md`
- 数据集下载详解：`evaluation/docs/dataset_download.md`
- 项目总览：`CLAUDE.md`

---

## 💡 关键要点总结

1. **GPU选择**：使用 `--auto_select_gpu` 自动选择最优GPU
2. **数据集**：只用 `--ppl_datasets wikitext2` 即可，PTB/C4可选
3. **缓存问题**：用 `python evaluation/clean_dataset_cache.py --zeroshot` 清理
4. **.bin文件**：剪枝模型的.bin文件**直接支持所有评估**，无需转换

**现在就可以开始完整评估了！** 🎉
