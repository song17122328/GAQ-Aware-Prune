"""
评估指标模块

包含:
- PPL (Perplexity) 评估
- Zero-shot / Few-shot 评估
- 效率指标 (速度、显存)
"""

# PPL 评估 (核心实现)
from .ppl import PPLMetric, evaluate_perplexity

# 性能评估
from .performance import evaluate_ppl, evaluate_zeroshot, evaluate_fewshot

# 效率评估
from .efficiency import evaluate_efficiency, measure_inference_speed

__all__ = [
    # PPL
    'PPLMetric',
    'evaluate_perplexity',

    # Performance
    'evaluate_ppl',
    'evaluate_zeroshot',
    'evaluate_fewshot',

    # Efficiency
    'evaluate_efficiency',
    'measure_inference_speed'
]
