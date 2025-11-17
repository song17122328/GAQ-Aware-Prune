"""
LLMPruner - 大语言模型剪枝工具包

包含以下模块：
- methods: 剪枝方法（GQA-aware等）
- importance: 重要性分析
- evaluator: 模型评估（PPL等）
- datasets: 数据集加载
- utils: 工具函数
"""

# 剪枝方法
from .methods import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

# 重要性分析
from .importance import (
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator
)

# 评估器
from .evaluator import PPLMetric

# 数据集工具
from .datasets import get_examples, get_examples_from_text

# 工具函数
from .utils.logger import LoggerWithDepth
from .utils.get_best_gpu import get_best_gpu

__version__ = '0.1.0'

__all__ = [
    # Methods
    'compute_gqa_group_importance',
    'select_gqa_groups_to_prune',
    'prune_attention_by_gqa_groups',

    # Importance
    'LayerImportanceAnalyzer',
    'UnbalancedStructuredPruningCalculator',

    # Evaluator
    'PPLMetric',

    # Datasets
    'get_examples',
    'get_examples_from_text',

    # Utils
    'LoggerWithDepth',
    'get_best_gpu',
]
