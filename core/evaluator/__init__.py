"""
Core evaluator module (已弃用)

PPL 评估已移至 evaluation/metrics/ppl.py
请使用: from evaluation.metrics.ppl import PPLMetric
"""

# 为向后兼容，重新导出 PPL
try:
    from evaluation.metrics.ppl import PPLMetric, evaluate_perplexity
    __all__ = ['PPLMetric', 'evaluate_perplexity']
except ImportError:
    __all__ = []
    import warnings
    warnings.warn(
        "PPL 评估已移至 evaluation/metrics/ppl.py，"
        "请使用: from evaluation.metrics.ppl import PPLMetric"
    )
