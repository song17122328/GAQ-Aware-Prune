# 新仓库迁移清单

## 📂 目录结构（新仓库）

```
llama-pruning/                    # 新仓库名称（建议）
├── README.md                      # ⭐ 新写的简洁介绍
├── requirements.txt               # Python依赖
├── .gitignore                     # Git忽略规则
│
├── 🎯 主脚本（2个）
│   ├── global_pruning.py          # 全局剪枝（推荐方法）
│   └── layer_pruning.py           # 层级剪枝（传统方法）
│
├── 📦 核心库 core/
│   ├── __init__.py
│   │
│   ├── methods/                   # 剪枝算法
│   │   ├── __init__.py
│   │   ├── global_pruning.py      # 全局性价比剪枝
│   │   └── gqa_aware.py           # GQA感知剪枝
│   │
│   ├── importance/                # 重要性分析
│   │   ├── __init__.py
│   │   └── layer_analyzer.py      # 层重要性评估
│   │
│   ├── datasets/                  # 数据加载
│   │   ├── __init__.py
│   │   └── example_samples.py     # WikiText2, C4
│   │
│   ├── trainer/                   # 微调
│   │   ├── __init__.py
│   │   └── finetuner.py           # Full + LoRA
│   │
│   └── utils/                     # 工具
│       ├── __init__.py
│       ├── logger.py              # 日志
│       └── get_best_gpu.py        # GPU选择
│
└── 📊 评估 evaluation/
    ├── __init__.py
    ├── metrics/
    │   ├── __init__.py
    │   └── ppl.py                 # 困惑度评估
    └── utils/
        ├── __init__.py
        └── model_loader.py        # 模型加载工具

输出目录（不纳入版本控制）:
└── prune_log/                     # 实验日志和模型
```

---

## ✅ 迁移文件清单（手动复制）

### 1️⃣ 主脚本（2个文件）

```bash
# 源文件 → 目标文件（重命名更简洁）

llama3_global_pruning.py → global_pruning.py
llama3_unbalanced_pruning_gqa_aware.py → layer_pruning.py
```

### 2️⃣ 核心库 core/（10个文件）

```bash
# 目录结构
core/__init__.py

# methods/
core/methods/__init__.py
core/methods/global_pruning.py
core/methods/gqa_aware.py

# importance/
core/importance/__init__.py
core/importance/layer_analyzer.py

# datasets/
core/datasets/__init__.py
core/datasets/example_samples.py

# trainer/
core/trainer/__init__.py
core/trainer/finetuner.py

# utils/
core/utils/__init__.py
core/utils/logger.py
core/utils/get_best_gpu.py
```

**注意**：不要复制 `core/evaluator/`（已废弃）

### 3️⃣ 评估模块 evaluation/（5个文件）

```bash
evaluation/__init__.py

# metrics/
evaluation/metrics/__init__.py
evaluation/metrics/ppl.py

# utils/
evaluation/utils/__init__.py
evaluation/utils/model_loader.py
```

**注意**：
- **不需要** `evaluation/metrics/performance.py` 和 `efficiency.py`（太复杂，暂时不用）
- **不需要** `run_evaluation.py`（可以后续按需添加）

### 4️⃣ 配置文件（2个文件）

```bash
requirements.txt
.gitignore
```

### 5️⃣ 文档（1个文件）

```bash
README.md  # ⭐ 新写的简洁版（见下文）
```

**不需要的文档**（太冗长）：
- ❌ CLAUDE.md (700+行)
- ❌ GLOBAL_PRUNING_GUIDE.md
- ❌ PARAMETERS_GUIDE.md
- ❌ SEARCH_EXAMPLE.md
- ❌ DATASET_SELECTION_GUIDE.md
- ❌ IMPLEMENTATION_SUMMARY.md
- ❌ PROJECT_SUMMARY.md
- ❌ core/README.md
- ❌ evaluation/README.md

---

## 🚫 不迁移的文件（删繁就简）

### 脚本（6个）
```
❌ search_optimal_distribution.py  # 自动搜索（太复杂）
❌ demo_global_pruning.py           # demo（不需要）
❌ test_finetuning.py               # 独立测试（不需要）
❌ evaluate_models.py               # 旧评估（已有新的）
❌ diagnose_model.py                # 诊断工具（按需）
```

### 评估模块（部分）
```
❌ evaluation/metrics/performance.py  # Zero-shot等（太复杂）
❌ evaluation/metrics/efficiency.py   # 吞吐量等（太复杂）
❌ evaluation/run_evaluation.py       # 统一入口（太复杂）
❌ evaluation/convert_checkpoint_to_hf.py
❌ evaluation/clean_dataset_cache.py
```

### 文档（全部）
```
❌ 所有 .md 文件（除了新的 README.md）
```

---

## 📝 需要新建的文件

### 1. requirements.txt

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.7.0
pandas>=2.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
pyyaml>=6.0
```

### 2. .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# 实验输出
prune_log/
*.bin
*.pth

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### 3. README.md

见下一节的完整内容 ⬇️

---

## 🎯 迁移步骤（手动操作）

```bash
# 1. 创建新仓库目录结构
mkdir -p llama-pruning/{core/{methods,importance,datasets,trainer,utils},evaluation/{metrics,utils}}

# 2. 复制主脚本
cp llama3_global_pruning.py llama-pruning/global_pruning.py
cp llama3_unbalanced_pruning_gqa_aware.py llama-pruning/layer_pruning.py

# 3. 复制 core/ 模块（逐个文件）
cp core/__init__.py llama-pruning/core/
cp core/methods/__init__.py llama-pruning/core/methods/
cp core/methods/global_pruning.py llama-pruning/core/methods/
cp core/methods/gqa_aware.py llama-pruning/core/methods/
# ... 依此类推

# 4. 复制 evaluation/ 模块
cp evaluation/__init__.py llama-pruning/evaluation/
cp evaluation/metrics/__init__.py llama-pruning/evaluation/metrics/
cp evaluation/metrics/ppl.py llama-pruning/evaluation/metrics/
cp evaluation/utils/__init__.py llama-pruning/evaluation/utils/
cp evaluation/utils/model_loader.py llama-pruning/evaluation/utils/

# 5. 创建配置文件
# 手动创建 requirements.txt, .gitignore, README.md

# 6. 初始化 Git
cd llama-pruning
git init
git add .
git commit -m "Initial commit: LLaMA pruning toolkit"
```

---

## 📊 迁移前后对比

| 项目 | 旧仓库 | 新仓库 | 减少 |
|------|--------|--------|------|
| **主脚本** | 7个 | 2个 | -71% |
| **核心文件** | 13个 | 13个 | 0% |
| **评估文件** | 10个 | 5个 | -50% |
| **文档** | 10个 | 1个 | -90% |
| **总文件数** | ~40个 | ~21个 | **-48%** |

---

## ✅ 检查清单

迁移完成后，检查：

- [ ] 目录结构正确
- [ ] 所有 `__init__.py` 文件存在
- [ ] `requirements.txt` 完整
- [ ] `.gitignore` 配置正确
- [ ] `README.md` 简洁清晰
- [ ] 能成功运行：`python global_pruning.py --help`
- [ ] 能成功导入：`from core.methods import global_pruning`

---

**预计迁移时间**: 15-20分钟（手动复制）
