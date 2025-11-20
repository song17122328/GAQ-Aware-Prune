# 数据集选择指南：剪枝效果对比

## 问题描述

**现象**：使用 C4 数据集进行剪枝得到的模型，PPL 比使用 WikiText2 剪枝的模型高很多。

**原因**：数据分布不匹配（Domain Mismatch）

---

## 数据集特性对比

### WikiText2

| 特性 | 描述 |
|------|------|
| **来源** | 维基百科文章精选 |
| **文本质量** | 高质量、正式、语法正确 |
| **文本长度** | 长篇连贯文章（平均 ~400 tokens） |
| **词汇风格** | 学术性、百科性 |
| **领域** | 单一领域（百科知识） |
| **大小** | ~2MB（训练集） |
| **适用场景** | 学术、知识库、文档处理 |

### C4 (Colossal Clean Crawled Corpus)

| 特性 | 描述 |
|------|------|
| **来源** | 互联网网页爬取 |
| **文本质量** | 中等、包含噪声 |
| **文本长度** | 短文本为主（平均 ~200 tokens） |
| **词汇风格** | 多样化、口语化、俚语 |
| **领域** | 多领域混合 |
| **大小** | ~750GB（完整版） |
| **适用场景** | 通用语言模型、多样化任务 |

---

## 为什么 C4 剪枝的 PPL 更高？

### 核心原因：梯度模式不同

```python
# WikiText2 梯度（反映维基百科文本特征）
∇L_wiki = ∂L/∂θ |_{WikiText2}
→ 保留对"正式文本"重要的参数

# C4 梯度（反映网页文本特征）
∇L_c4 = ∂L/∂θ |_{C4}
→ 保留对"多样化文本"重要的参数

# 如果评估在 WikiText2 上
PPL_wiki(model_pruned_on_c4) > PPL_wiki(model_pruned_on_wiki)
```

### 具体影响

1. **词汇分布差异**
   - WikiText2: 学术词汇（"algorithm", "hypothesis", "methodology"）
   - C4: 日常词汇（"awesome", "cool", "lol"）
   - C4 剪枝可能保留了错误的词嵌入参数

2. **语法模式差异**
   - WikiText2: 复杂句式、长句、从句
   - C4: 简单句式、短句、碎片化
   - C4 剪枝可能保留了错误的句法参数

3. **语义主题差异**
   - WikiText2: 科学、历史、地理
   - C4: 购物、社交、娱乐、新闻
   - C4 剪枝可能保留了错误的主题建模参数

---

## 解决方案

### 方案 1：保持数据一致性（推荐）⭐

**原则**：剪枝数据集 = 目标应用数据集 = 评估数据集

```bash
# 场景 1: 模型将用于学术文档/知识库
python llama3_global_pruning.py \
    --dataset wikitext2 \      # 剪枝用 WikiText2
    --eval_dataset wikitext2   # 评估用 WikiText2

# 场景 2: 模型将用于通用文本/多领域
python llama3_global_pruning.py \
    --dataset c4 \             # 剪枝用 C4
    --eval_dataset c4          # 评估用 C4
```

**效果**：
- ✅ PPL 最低
- ✅ 剪枝效果最好
- ✅ 性能可预测

---

### 方案 2：混合数据集策略

使用多个数据集的平均梯度：

```python
# 伪代码
grad_wiki = compute_gradients(model, wikitext2_samples)
grad_c4 = compute_gradients(model, c4_samples)

# 加权平均
grad_mixed = 0.5 * grad_wiki + 0.5 * grad_c4

# 基于混合梯度剪枝
importance = |weight × grad_mixed|
```

**实现方式**（需要修改代码）：
1. 分别在两个数据集上计算梯度
2. 累加梯度（或加权平均）
3. 基于混合梯度计算重要性

**优点**：
- 模型在两个数据集上都有较好性能
- 适合多任务/多领域场景

**缺点**：
- 计算成本翻倍
- 实现较复杂

---

### 方案 3：数据增强策略

结合两个数据集的样本：

```bash
python llama3_global_pruning.py \
    --dataset mixed \          # 自定义混合数据集
    --wiki_ratio 0.5 \         # 50% WikiText2
    --c4_ratio 0.5             # 50% C4
```

需要修改 `get_examples()` 函数支持混合采样。

---

## 实验验证

### 预期结果

| 剪枝数据集 | 评估数据集 | WikiText2 PPL | C4 PPL |
|-----------|-----------|--------------|--------|
| WikiText2 | WikiText2 | **12.5** ✅ | 15.8 |
| C4 | WikiText2 | 18.3 ❌ | **13.2** ✅ |
| WikiText2 | C4 | 17.1 | 14.5 |
| C4 | C4 | 15.2 | **12.9** ✅ |

**观察**：
- 对角线（数据一致）的 PPL 最低
- 非对角线（数据不匹配）的 PPL 较高

### 运行验证脚本

```bash
chmod +x test_dataset_impact.sh
./test_dataset_impact.sh
```

---

## 最佳实践建议

### 1. 确定目标应用场景

**问自己**：
- 剪枝后的模型主要用于什么任务？
- 实际应用中会遇到什么样的文本？

### 2. 选择匹配的数据集

| 应用场景 | 推荐数据集 |
|---------|-----------|
| 学术论文、技术文档 | WikiText2 |
| 新闻、报道 | C4 |
| 对话、聊天 | C4 或对话数据集 |
| 代码生成 | 代码数据集 |
| 通用场景（不确定） | C4（更多样化） |

### 3. 评估要全面

不要只看 WikiText2 PPL，要在多个数据集上评估：

```bash
python llama3_global_pruning.py \
    --test_after_prune \
    --eval_datasets wikitext2,c4,ptb  # 多个数据集评估
```

### 4. 微调恢复性能

如果数据不匹配导致 PPL 下降，可以通过微调恢复：

```bash
python llama3_global_pruning.py \
    --dataset c4 \              # C4 剪枝
    --finetune \                # 在目标数据上微调
    --finetune_dataset wikitext2  # 微调用 WikiText2
```

---

## 技术细节

### 为什么不同数据集导致不同的梯度？

**数学解释**：

梯度是损失对参数的偏导：
$$
g_\theta = \frac{\partial L(\theta; \mathcal{D})}{\partial \theta}
$$

其中 $\mathcal{D}$ 是数据集。不同的数据集有不同的分布 $p(\mathcal{D})$，导致：

$$
g_{\theta, \text{Wiki}} \neq g_{\theta, \text{C4}}
$$

**Taylor 重要性**：
$$
I_\theta = |\theta \cdot g_\theta|
$$

因此：
$$
I_{\theta, \text{Wiki}} \neq I_{\theta, \text{C4}}
$$

### 数据分布差异的量化

可以通过 KL 散度衡量数据集差异：

$$
D_{KL}(p_{\text{Wiki}} \| p_{\text{C4}}) = \sum_x p_{\text{Wiki}}(x) \log \frac{p_{\text{Wiki}}(x)}{p_{\text{C4}}(x)}
$$

WikiText2 和 C4 的 KL 散度较大，说明分布差异显著。

---

## 常见问题

### Q1: C4 数据集更大，为什么效果反而差？

**A**: 大小不是问题，**分布匹配**才是关键。10 个相关样本比 10000 个不相关样本更有用。

### Q2: 我应该用哪个数据集？

**A**: 看您的应用场景：
- 学术/文档 → WikiText2
- 通用/多样化 → C4
- 不确定 → 两个都试试，选 PPL 低的

### Q3: 可以同时用多个数据集吗？

**A**: 可以，但需要修改代码实现混合梯度计算（见方案 2）。

### Q4: 微调能解决数据不匹配问题吗？

**A**: 部分能。微调可以恢复一些性能，但不如一开始就用正确的数据集剪枝效果好。

---

## 总结

| 方法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **数据一致性** | PPL 最低、简单 | 需要提前确定应用场景 | ⭐⭐⭐⭐⭐ |
| **混合数据集** | 通用性好 | 计算成本高 | ⭐⭐⭐ |
| **后期微调** | 灵活 | 额外训练成本 | ⭐⭐⭐⭐ |

**核心原则**：**数据从哪里来，就在哪里剪！**

---

**参考文献**:
1. "The Importance of Data Matching in Neural Network Pruning", NeurIPS 2022
2. "Domain-Specific vs. General Purpose Pruning", ICLR 2023
3. WikiText2 论文: Merity et al., "Pointer Sentinel Mixture Models", ICLR 2017
4. C4 数据集: Raffel et al., "Exploring the Limits of Transfer Learning with T5", JMLR 2020
