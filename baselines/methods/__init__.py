#!/usr/bin/env python3
"""
剪枝方法模块

导出所有可用的剪枝方法
"""

from .base_pruner import BasePruner

# 可用方法注册表
AVAILABLE_METHODS = {
    # === 第一阶段：必须实现 ===
    'llm_pruner': {
        'class': 'LLMPruner',
        'module': 'llm_pruner',
        'status': 'pending',  # pending / implemented / tested
        'priority': 1,
        'description': 'LLM-Pruner: 基于Taylor重要性的结构化剪枝'
    },
    'wanda': {
        'class': 'WandaPruner',
        'module': 'wanda',
        'status': 'pending',
        'priority': 1,
        'description': 'Wanda-Structured: 基于权重和激活的结构化剪枝'
    },
    'magnitude': {
        'class': 'MagnitudePruner',
        'module': 'magnitude',
        'status': 'pending',
        'priority': 1,
        'description': 'Magnitude: 基于权重绝对值的剪枝'
    },

    # === 第二阶段：后续实现 ===
    'shortgpt': {
        'class': 'ShortGPTPruner',
        'module': 'shortgpt',
        'status': 'pending',
        'priority': 2,
        'description': 'ShortGPT: 基于层重要性的深度剪枝'
    },

    # === 第三阶段：视难度决定 ===
    'slimgpt': {
        'class': 'SlimGPTPruner',
        'module': 'slimgpt',
        'status': 'pending',
        'priority': 3,
        'description': 'SlimGPT: 结合稀疏性和结构化剪枝'
    },
    'sparsegpt': {
        'class': 'SparseGPTPruner',
        'module': 'sparsegpt',
        'status': 'pending',
        'priority': 3,
        'description': 'SparseGPT: 基于Hessian的一次性剪枝'
    },

    # === 第四阶段：可选 ===
    'flap': {
        'class': 'FLAPPruner',
        'module': 'flap',
        'status': 'pending',
        'priority': 4,
        'description': 'FLAP: 基于特征的自适应剪枝'
    },
    'random': {
        'class': 'RandomPruner',
        'module': 'random_pruner',
        'status': 'pending',
        'priority': 4,
        'description': 'Random: 随机剪枝（作为下界参考）'
    },
}


def get_pruner(method_name: str, **kwargs):
    """
    获取指定的剪枝器实例

    Args:
        method_name: 方法名称 (llm_pruner, wanda, magnitude, etc.)
        **kwargs: 传递给剪枝器的参数

    Returns:
        BasePruner 子类实例

    Raises:
        ValueError: 方法不存在或未实现
    """
    if method_name not in AVAILABLE_METHODS:
        available = ', '.join(AVAILABLE_METHODS.keys())
        raise ValueError(f"未知方法: {method_name}。可用方法: {available}")

    method_info = AVAILABLE_METHODS[method_name]

    if method_info['status'] == 'pending':
        raise NotImplementedError(
            f"方法 '{method_name}' 尚未实现。"
            f"优先级: {method_info['priority']}，描述: {method_info['description']}"
        )

    # 动态导入模块
    import importlib
    module = importlib.import_module(f'.{method_info["module"]}', package='baselines.methods')
    pruner_class = getattr(module, method_info['class'])

    return pruner_class(**kwargs)


def list_methods(show_pending: bool = True) -> None:
    """
    列出所有可用的剪枝方法

    Args:
        show_pending: 是否显示未实现的方法
    """
    print("\n" + "=" * 60)
    print("可用的 Baseline 剪枝方法")
    print("=" * 60)

    # 按优先级分组
    by_priority = {}
    for name, info in AVAILABLE_METHODS.items():
        priority = info['priority']
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append((name, info))

    priority_names = {
        1: "第一阶段（必须实现）",
        2: "第二阶段（后续实现）",
        3: "第三阶段（视难度决定）",
        4: "第四阶段（可选）"
    }

    for priority in sorted(by_priority.keys()):
        print(f"\n{priority_names.get(priority, f'优先级 {priority}')}:")
        print("-" * 40)

        for name, info in by_priority[priority]:
            status_icon = {
                'pending': '⏳',
                'implemented': '✅',
                'tested': '🧪'
            }.get(info['status'], '❓')

            if not show_pending and info['status'] == 'pending':
                continue

            print(f"  {status_icon} {name}: {info['description']}")

    print("\n" + "=" * 60)


__all__ = [
    'BasePruner',
    'AVAILABLE_METHODS',
    'get_pruner',
    'list_methods'
]
