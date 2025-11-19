# Baseline 剪枝方法

本模块包含多种经典的 LLM 结构化剪枝方法，用于与 GAQ-Aware-Prune 进行对比实验。

## 实现优先级

| 阶段 | 方法 | 状态 | 说明 |
|------|------|------|------|
| **第一阶段** | LLM-Pruner | ⏳ 待实现 | 基于 Taylor 重要性的结构化剪枝 |
| **第一阶段** | Wanda-Structured | ⏳ 待实现 | 基于权重和激活的结构化剪枝 |
| **第一阶段** | Magnitude | ⏳ 待实现 | 基于权重绝对值的剪枝 |
| **第二阶段** | ShortGPT | ⏳ 待实现 | 基于层重要性的深度剪枝 |
| **第三阶段** | SlimGPT | ⏳ 待实现 | 结合稀疏性和结构化剪枝 |
| **第三阶段** | SparseGPT | ⏳ 待实现 | 基于 Hessian 的一次性剪枝 |
| **第四阶段** | FLAP | ⏳ 待实现 | 基于特征的自适应剪枝 |
| **第四阶段** | Random | ⏳ 待实现 | 随机剪枝（下界参考） |

## 文件结构

```
baselines/
├── __init__.py                    # 包导出
├── README.md                      # 本文档
├── run_baseline.py                # 统一运行入口
├── compare_methods.py             # 方法对比脚本
│
├── methods/                       # 剪枝方法实现
│   ├── __init__.py               # 方法注册和获取
│   ├── base_pruner.py            # 基类定义
│   ├── llm_pruner.py             # LLM-Pruner
│   ├── wanda.py                  # Wanda-Structured
│   ├── magnitude.py              # Magnitude
│   ├── shortgpt.py               # ShortGPT
│   ├── slimgpt.py                # SlimGPT
│   ├── sparsegpt.py              # SparseGPT
│   ├── flap.py                   # FLAP
│   └── random_pruner.py          # Random
│
└── utils/                         # 工具函数
    ├── __init__.py
    └── pruning_utils.py          # 通用剪枝工具
```

## 快速开始

### 1. 列出可用方法

```bash
python baselines/run_baseline.py --list_methods
```

### 2. 运行单个方法

```bash
python baselines/run_baseline.py \
    --method llm_pruner \
    --base_model /path/to/model \
    --pruning_ratio 0.25 \
    --save_model \
    --test_after_prune
```

### 3. 对比多个方法

```bash
python baselines/compare_methods.py \
    --methods llm_pruner,wanda,magnitude \
    --base_model /path/to/model \
    --pruning_ratio 0.25 \
    --output_dir comparison_results/
```

## 实现指南

### 基类接口

所有方法需要继承 `BasePruner` 并实现以下方法：

```python
from baselines.methods.base_pruner import BasePruner

class MyPruner(BasePruner):
    def compute_importance(self, calibration_data, **kwargs):
        """计算重要性分数"""
        pass

    def prune(self, pruning_ratio, calibration_data=None, **kwargs):
        """执行剪枝"""
        pass
```

### 实现步骤

1. **阅读骨架文件**：每个方法文件包含详细的实现说明
2. **参考现有代码**：`LLMPruner/methods/gqa_aware.py` 有完整的 GQA 感知实现
3. **保持 GQA 兼容**：确保剪枝后 Q:KV 比例为 4:1
4. **更新状态**：实现完成后在 `methods/__init__.py` 中更新状态

### 更新方法状态

在 `baselines/methods/__init__.py` 中：

```python
AVAILABLE_METHODS = {
    'llm_pruner': {
        'class': 'LLMPruner',
        'module': 'llm_pruner',
        'status': 'implemented',  # pending -> implemented -> tested
        ...
    },
    ...
}
```

## 评估指标

每个方法完成后应报告：

1. **剪枝率**：实际 vs 目标
2. **PPL (Perplexity)**：WikiText-2
3. **Zero-shot 准确率**：7 个标准任务
4. **GQA 比例验证**：是否保持 4:1

## 参考论文

- **LLM-Pruner**: [arXiv:2305.11627](https://arxiv.org/abs/2305.11627)
- **Wanda**: [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
- **ShortGPT**: [arXiv:2403.03853](https://arxiv.org/abs/2403.03853)
- **SparseGPT**: [arXiv:2301.00774](https://arxiv.org/abs/2301.00774)
- **SlimGPT**: [arXiv:2405.14129](https://arxiv.org/abs/2405.14129)
- **FLAP**: [arXiv:2312.11983](https://arxiv.org/abs/2312.11983)

## 注意事项

1. **GQA 兼容性**：所有方法必须保持 Q:KV = 4:1 的比例
2. **校准数据**：大多数方法需要校准数据（128 samples 通常足够）
3. **显存管理**：处理大模型时注意清理缓存
4. **日志记录**：使用 `self.log()` 而非 `print()`
