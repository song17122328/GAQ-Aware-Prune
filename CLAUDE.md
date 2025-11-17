# CLAUDE.md - AI Assistant Guide for GAQ-Aware-Prune Repository

This document provides comprehensive guidance for AI assistants working with the GAQ-Aware-Prune codebase. It explains the structure, conventions, workflows, and best practices for this LLM pruning project.

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Codebase Structure](#codebase-structure)
3. [Core Concepts](#core-concepts)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Key Modules](#key-modules)
7. [Running Experiments](#running-experiments)
8. [Testing and Evaluation](#testing-and-evaluation)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)

---

## Repository Overview

**Project Name:** GAQ-Aware-Prune
**Purpose:** GQA (Grouped Query Attention) aware structured pruning for LLaMA-3 models
**Language:** Python 3
**Primary Framework:** PyTorch, Transformers (Hugging Face)

### What This Repository Does

This repository implements a sophisticated neural network pruning technique specifically designed for LLaMA-3 models that use Grouped Query Attention (GQA). The key innovations are:

1. **GQA-Aware Pruning**: Maintains the 4:1 Q:KV head ratio during pruning
2. **Unbalanced Layer-wise Pruning**: More important layers are pruned less aggressively
3. **Taylor Importance**: Uses gradient-based importance metrics for pruning decisions
4. **Post-Pruning Fine-tuning**: Recovers model performance after pruning

### Primary Goals

- Reduce model parameters by 20-30% while maintaining model quality
- Preserve the GQA architecture (4:1 ratio of Query heads to Key-Value heads)
- Minimize perplexity degradation through intelligent layer-wise pruning
- Provide fine-tuning capabilities to recover performance

---

## Codebase Structure

```
GAQ-Aware-Prune/
├── .gitignore                                    # Git ignore file
├── RUN.md                                        # Detailed usage documentation (Chinese)
├── CLAUDE.md                                     # This file - AI assistant guide
├── llama3_unbalanced_pruning_gqa_aware.py       # Main entry point script
│
└── LLMPruner/                                    # Core pruning library
    ├── __init__.py                               # Package exports
    ├── README.md                                 # Module documentation
    │
    ├── methods/                                  # Pruning algorithms
    │   ├── __init__.py
    │   └── gqa_aware.py                          # GQA-aware Taylor importance
    │
    ├── importance/                               # Layer importance analysis
    │   ├── __init__.py
    │   └── layer_analyzer.py                     # Layer importance metrics
    │
    ├── datasets/                                 # Data loading utilities
    │   ├── __init__.py
    │   └── example_samples.py                    # Sample data loaders
    │
    ├── evaluator/                                # Model evaluation
    │   ├── __init__.py
    │   └── ppl.py                                # Perplexity metrics
    │
    ├── trainer/                                  # Fine-tuning module
    │   ├── __init__.py
    │   └── finetuner.py                          # Post-pruning fine-tuning
    │
    └── utils/                                    # Utility functions
        ├── logger.py                             # Logging with timestamps
        └── get_best_gpu.py                       # Auto GPU selection

Output directories (gitignored):
├── prune_log/                                    # Experiment logs and checkpoints
│   └── {experiment_name}/
│       ├── description.txt                       # Config parameters
│       ├── layer_importance_config.json          # Layer importance scores
│       ├── pruning_strategy.png                  # Visualization
│       ├── pytorch_model.bin                     # Pruned model checkpoint
│       ├── pytorch_model_finetuned.bin           # Fine-tuned model
│       └── {timestamp}/
│           ├── training.log                      # Detailed logs
│           └── train.sh                          # Command backup
```

### File Purpose Summary

| File | Purpose | When to Modify |
|------|---------|----------------|
| `llama3_unbalanced_pruning_gqa_aware.py` | Main orchestration script | Add new CLI arguments, modify workflow |
| `LLMPruner/methods/gqa_aware.py` | GQA-aware pruning algorithm | Change importance calculation, modify pruning logic |
| `LLMPruner/importance/layer_analyzer.py` | Layer importance evaluation | Add new importance metrics |
| `LLMPruner/trainer/finetuner.py` | Post-pruning fine-tuning | Modify training loop, add optimizers |
| `LLMPruner/evaluator/ppl.py` | Perplexity evaluation | Add new evaluation metrics |
| `LLMPruner/datasets/example_samples.py` | Data loading | Add new datasets |

---

## Core Concepts

### 1. Grouped Query Attention (GQA)

LLaMA-3 uses GQA with a 4:1 ratio:
- **32 Query (Q) heads** per layer
- **8 Key-Value (KV) heads** per layer
- Each KV head is shared by 4 Q heads

**Critical Constraint**: Pruning must maintain this 4:1 ratio to preserve model architecture.

### 2. GQA Groups

A "GQA group" consists of:
- 1 KV head
- 4 corresponding Q heads
- All associated weight matrices

When pruning, we prune entire GQA groups, not individual heads.

### 3. Taylor Importance

**Formula**: `importance = |weight × gradient|`

Measures the impact of each parameter on the loss. Higher importance means more critical to model performance.

### 4. Layer-wise Unbalanced Pruning

Not all layers are equally important:
- **Important layers** (high perplexity impact when removed): pruned less
- **Less important layers**: pruned more aggressively
- **Strategy**: `inverse` mode prunes important layers less (default)

### 5. Pruning Pipeline

```
1. Layer Importance Analysis
   ↓
2. Calculate Per-Layer Pruning Rates
   ↓
3. For each layer:
   - Compute GQA group importance (Taylor)
   - Select groups to prune (lowest importance)
   - Prune attention and optionally MLP
   ↓
4. Save Pruned Model
   ↓
5. Fine-tune (Optional)
   ↓
6. Evaluate Perplexity
```

---

## Development Workflows

### Typical Workflow for Contributors

1. **Understanding the Codebase**
   - Read `RUN.md` for usage examples
   - Read `LLMPruner/README.md` for module documentation
   - Review the main script: `llama3_unbalanced_pruning_gqa_aware.py`

2. **Making Changes**
   - Modify the appropriate module in `LLMPruner/`
   - Test changes with a small debug run (see "Running Experiments")
   - Update documentation if adding new features

3. **Testing**
   - Run debug mode with limited layers and samples
   - Verify GQA ratio is maintained (check logs for "4:1" confirmation)
   - Check perplexity doesn't explode (PPL should be reasonable, not NaN/Inf)

4. **Committing**
   - Branch naming: Use descriptive names like `feature/new-importance-metric` or `fix/gpu-selection`
   - Commit messages: Clear and descriptive (Chinese or English)
   - Use the designated Claude branch: `claude/claude-md-mi2us1pa3q3g1xh0-01SBkQXujzeqP6vHDvg2rMP3`

### Git Workflow

```bash
# Check current branch
git status

# Make changes
# (edit files)

# Commit changes
git add .
git commit -m "feat: add new importance metric based on attention scores"

# Push to remote (use -u for first push on new branch)
git push -u origin claude/claude-md-mi2us1pa3q3g1xh0-01SBkQXujzeqP6vHDvg2rMP3
```

**Important**: Always push to branches starting with `claude/` when working with Claude Code.

---

## Code Conventions

### Python Style

1. **Language**: Comments and docstrings are in **Chinese** (中文)
2. **Docstring Format**: Google-style docstrings
3. **Imports**: Grouped in standard order (stdlib, third-party, local)
4. **Type Hints**: Used where helpful but not enforced everywhere

### Example Code Style

```python
#!/usr/bin/env python3
"""
模块功能简短描述

详细说明模块的用途和核心思想
"""

import os
import torch
from typing import Dict, List

def compute_importance(layer, method='taylor'):
    """
    计算层的重要性分数

    Args:
        layer: Transformer层
        method: 重要性计算方法，'taylor' 或 'activation'

    Returns:
        importance: float, 重要性分数
    """
    if method == 'taylor':
        # 实现 Taylor importance
        pass
    return importance
```

### Naming Conventions

- **Variables**: `snake_case` (e.g., `num_kv_heads`, `layer_importance`)
- **Functions**: `snake_case` (e.g., `compute_gqa_group_importance`)
- **Classes**: `PascalCase` (e.g., `LayerImportanceAnalyzer`, `FineTuner`)
- **Constants**: `UPPER_SNAKE_CASE` (not heavily used in this codebase)
- **Private**: Prefix with `_` (e.g., `_helper_function`)

### Module Organization

Each module in `LLMPruner/` follows this pattern:

```python
# module_name/__init__.py
from .implementation import PublicClass, public_function

__all__ = ['PublicClass', 'public_function']

# module_name/implementation.py
class PublicClass:
    """Public API"""
    pass

def public_function():
    """Public API"""
    pass
```

### Logging

Always use the logger provided by `LoggerWithDepth`:

```python
from LLMPruner.utils.logger import LoggerWithDepth

logger = LoggerWithDepth(
    env_name='experiment_name',
    config=args.__dict__,
    root_dir='prune_log'
)

logger.log("Starting pruning...")  # Use logger.log(), not print()
logger.log(f"Layer {i}: importance = {importance:.4f}")
```

**Why?**
- Logs are saved to file with timestamps
- Config is automatically saved
- Easy to review experiments later

---

## Key Modules

### 1. `LLMPruner.methods.gqa_aware`

**Purpose**: GQA-aware Taylor importance calculation and pruning

**Key Functions**:

```python
compute_gqa_group_importance(layer, head_dim=128, gqa_ratio=4)
# Returns: Tensor[num_kv_heads] - importance score for each GQA group

select_gqa_groups_to_prune(group_importance, target_num_kv_heads)
# Returns: (keep_indices, prune_indices) - which groups to keep/prune

prune_attention_by_gqa_groups(layer, keep_kv_indices, head_dim=128, gqa_ratio=4)
# Modifies layer in-place, returns (num_q_heads, num_kv_heads)
```

**When to modify**:
- Changing importance calculation (e.g., use attention scores instead of Taylor)
- Modifying pruning granularity
- Adding support for different GQA ratios

### 2. `LLMPruner.importance.layer_analyzer`

**Purpose**: Evaluate layer-wise importance for unbalanced pruning

**Key Classes**:

```python
LayerImportanceAnalyzer(model, tokenizer, device='cuda')
    .measure_layer_importance_by_removal(texts, num_layers)
    # Returns: Dict[int, float] - {layer_idx: importance_score}

    .measure_layer_importance_by_activation(texts)
    # Alternative method using activation statistics

UnbalancedStructuredPruningCalculator(layer_importance, num_layers)
    .compute_layer_pruning_rates(target_overall_rate, strategy, alpha, ...)
    # Returns: Dict[int, float] - {layer_idx: pruning_rate}

    .save_pruning_rates(rates, path)
    .load_pruning_rates(path)
    .visualize_pruning_strategy(rates, save_path)
```

**When to modify**:
- Adding new layer importance metrics
- Changing pruning rate distribution strategy
- Modifying visualization

### 3. `LLMPruner.trainer.finetuner`

**Purpose**: Fine-tune pruned models to recover performance

**Key Class**:

```python
FineTuner(model, tokenizer, device='cuda', logger=None)
    .finetune(dataset_name='wikitext', num_samples=500, lr=1e-5, epochs=1, ...)
    # Returns: Dict with training stats (losses, etc.)

    .save_finetuned_model(save_path, layer_pruning_rates, ...)
    # Saves model + metadata
```

**When to modify**:
- Adding new optimizers (Adam, SGD, etc.)
- Implementing learning rate schedules
- Adding gradient clipping or other training tricks

### 4. `LLMPruner.evaluator.ppl`

**Purpose**: Evaluate model perplexity on standard benchmarks

**Key Function**:

```python
PPLMetric(model, tokenizer, datasets=['wikitext2'], seq_len=128, device='cuda')
# Returns: Dict-like object with perplexity scores
# Usage: ppl['wikitext2 (wikitext-2-raw-v1)'] → float
```

**When to modify**:
- Adding new evaluation datasets
- Implementing other metrics (accuracy, BLEU, etc.)

### 5. `LLMPruner.datasets.example_samples`

**Purpose**: Load samples for gradient computation

**Key Functions**:

```python
get_examples(dataset_name, tokenizer, num_samples=10, seq_len=128, split='train')
# Returns: Tensor[num_samples, seq_len] - tokenized input_ids

get_examples_from_text(texts, tokenizer, seq_len=128)
# Returns: Tensor - tokenized from custom texts
```

**When to modify**:
- Adding support for new datasets (C4, PTB, etc.)
- Changing tokenization strategy

---

## Running Experiments

### Quick Start (Full Pipeline with Fine-tuning)

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned_25pct \
    --pruning_ratio 0.25 \
    --layer_importance_method removal \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_lr 1e-5 \
    --finetune_epochs 1 \
    --finetune_samples 500
```

### Debug Mode (Fast Testing)

```bash
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name debug_test \
    --pruning_ratio 0.25 \
    --layer_importance_samples 10 \
    --channel_importance_samples 5 \
    --layer_start 10 \
    --layer_end 15 \
    --test_after_prune
```

**Note**: Debug mode prunes only layers 10-15 with minimal samples for quick testing.

### Understanding Command-Line Arguments

#### Essential Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--base_model` | str | **required** | Path to base LLaMA-3 model |
| `--save_ckpt_log_name` | str | `llama_gqa_aware_prune` | Experiment name for logs |
| `--pruning_ratio` | float | `0.25` | Target overall pruning rate (0.25 = 25%) |

#### Layer Importance Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer_importance_method` | str | `removal` | `removal` or `activation` |
| `--layer_importance_samples` | int | `50` | Samples for layer importance evaluation |
| `--skip_importance_analysis` | flag | `False` | Skip analysis, load from file |
| `--layer_importance_config` | str | `layer_importance_config.json` | Config file path |

#### Pruning Strategy Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pruning_strategy` | str | `inverse` | `inverse`, `proportional`, or `uniform` |
| `--alpha` | float | `1.0` | Importance weight coefficient (0.5-3.0) |
| `--min_pruning_rate` | float | `0.15` | Minimum per-layer pruning rate |
| `--max_pruning_rate` | float | `0.5` | Maximum per-layer pruning rate |

#### Fine-tuning Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--finetune` | flag | `False` | Enable post-pruning fine-tuning |
| `--finetune_lr` | float | `1e-5` | Fine-tuning learning rate |
| `--finetune_epochs` | int | `1` | Number of fine-tuning epochs |
| `--finetune_samples` | int | `500` | Number of training samples |
| `--finetune_batch_size` | int | `1` | Batch size (limited by VRAM) |
| `--finetune_seq_len` | int | `512` | Sequence length for fine-tuning |

#### Other Flags

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save_model` | flag | `False` | Save pruned model checkpoint |
| `--test_after_prune` | flag | `False` | Evaluate perplexity after pruning |
| `--prune_mlp` | flag | `False` | Also prune MLP layers (not just attention) |

### Interpreting Logs

**Log Location**: `prune_log/{save_ckpt_log_name}/{timestamp}/training.log`

**Key Sections to Monitor**:

1. **Layer Importance Analysis**:
   ```
   层重要性统计:
     平均: 1.234567
     最重要的5层: Layer 15, Layer 16, ...
   ```

2. **Pruning Execution**:
   ```
   处理 Layer 0 (剪枝率: 0.20)
     Attention: 32Q:8KV → 28Q:7KV
   ```
   ✅ **Good**: See gradual reduction, 4:1 ratio maintained
   ❌ **Bad**: Ratio not 4:1, or NaN values

3. **Final Statistics**:
   ```
   参数统计:
     剪枝前: 8,030,261,248
     剪枝后: 6,024,195,936
     实际剪枝率: 25.00%

   GQA比例验证: ✅ 所有层保持4:1
   ```

4. **Perplexity Evaluation**:
   ```
   剪枝后 PPL: {'wikitext2 (wikitext-2-raw-v1)': 12.34}
   微调后 PPL: {'wikitext2 (wikitext-2-raw-v1)': 11.89}
   ```
   ✅ **Good**: PPL < 15, degradation < 5%
   ❌ **Bad**: PPL > 100 or NaN (model broken)

### Output Files

After running, you'll find:

```
prune_log/{save_ckpt_log_name}/
├── description.txt                     # Full config dump
├── layer_importance_config.json        # Reusable importance scores
├── pruning_strategy.png                # Visualization of pruning rates
├── pytorch_model.bin                   # Pruned model (if --save_model)
├── pytorch_model_finetuned.bin         # Fine-tuned model (if --finetune)
└── {timestamp}/
    ├── training.log                    # Detailed execution log
    └── train.sh                        # Command used (for reproducibility)
```

---

## Testing and Evaluation

### Unit Testing Strategy

While this repository doesn't have formal unit tests, you should validate:

1. **GQA Ratio Preservation**:
   ```python
   # After pruning each layer
   assert layer.self_attn.num_heads // layer.self_attn.num_key_value_heads == 4
   ```

2. **Shape Consistency**:
   ```python
   # Ensure weight shapes are correct
   assert layer.self_attn.q_proj.out_features == num_q_heads * head_dim
   assert layer.self_attn.k_proj.out_features == num_kv_heads * head_dim
   ```

3. **Forward Pass**:
   ```python
   # Model should still work after pruning
   with torch.no_grad():
       output = model(input_ids)
   assert not torch.isnan(output.logits).any()
   ```

### Integration Testing

**Test Script Template**:

```bash
# Test 1: Basic pruning (no save, quick)
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name test_basic \
    --pruning_ratio 0.25 \
    --layer_importance_samples 10 \
    --channel_importance_samples 5 \
    --layer_start 10 \
    --layer_end 15

# Test 2: Full pipeline with save
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name test_full \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune

# Test 3: Fine-tuning
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name test_finetune \
    --pruning_ratio 0.25 \
    --save_model \
    --finetune \
    --finetune_samples 100 \
    --test_after_prune
```

### Validation Checklist

After running pruning, verify:

- [ ] Logs show "✅ 所有层保持4:1" (all layers maintain 4:1 ratio)
- [ ] Actual pruning rate matches target (±2%)
- [ ] Perplexity is reasonable (< 20 for 25% pruning)
- [ ] No NaN or Inf values in logs
- [ ] Model checkpoint can be loaded successfully
- [ ] Output files exist in `prune_log/`

---

## Common Tasks

### Task 1: Add a New Importance Metric

**Location**: `LLMPruner/methods/gqa_aware.py`

**Steps**:
1. Add new function:
   ```python
   def compute_gqa_group_importance_attention_based(layer, head_dim=128, gqa_ratio=4):
       """基于注意力分数的重要性计算"""
       # Your implementation
       return group_importance
   ```

2. Update main script to use it:
   ```python
   # In llama3_unbalanced_pruning_gqa_aware.py
   if args.importance_metric == 'taylor':
       group_imp = compute_gqa_group_importance(layer, ...)
   elif args.importance_metric == 'attention':
       group_imp = compute_gqa_group_importance_attention_based(layer, ...)
   ```

3. Add CLI argument:
   ```python
   parser.add_argument('--importance_metric', type=str, default='taylor',
                      choices=['taylor', 'attention'])
   ```

### Task 2: Add Support for New Dataset

**Location**: `LLMPruner/datasets/example_samples.py`

**Steps**:
1. Add dataset loading logic:
   ```python
   def get_examples(dataset_name, tokenizer, ...):
       if dataset_name.lower() == 'c4':
           dataset = load_dataset('c4', 'en', split=split)
           text_field = 'text'
       elif dataset_name.lower() == 'wikitext':
           # existing code
       # ... rest of function
   ```

2. Update `LLMPruner/evaluator/ppl.py` if needed for evaluation

### Task 3: Modify Pruning Strategy

**Location**: `LLMPruner/importance/layer_analyzer.py`

**Steps**:
1. Add new strategy in `UnbalancedStructuredPruningCalculator`:
   ```python
   def compute_layer_pruning_rates(self, ..., strategy='inverse'):
       if strategy == 'custom':
           # Your custom logic
           weights = custom_weight_function(importance_scores)
       # ... rest of function
   ```

2. Update choices in main script:
   ```python
   parser.add_argument('--pruning_strategy', choices=['inverse', 'proportional', 'uniform', 'custom'])
   ```

### Task 4: Save/Load Pruned Models

**Already Implemented**, but to modify:

**Location**: `llama3_unbalanced_pruning_gqa_aware.py` (Step 5 and 6)

**Save**:
```python
save_dict = {
    'model': model,
    'tokenizer': tokenizer,
    'layer_pruning_rates': layer_pruning_rates,
    'layer_importance': layer_importance,
    'pruning_method': 'gqa_aware_taylor',
    'config': args.__dict__
}
torch.save(save_dict, path)
```

**Load**:
```python
checkpoint = torch.load(path, weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']
config = checkpoint['config']
```

### Task 5: Debug CUDA OOM Errors

**Quick Fixes**:
1. Reduce samples:
   ```bash
   --layer_importance_samples 20 \
   --channel_importance_samples 5 \
   --finetune_samples 100
   ```

2. Reduce sequence length:
   ```bash
   --taylor_seq_len 64 \
   --finetune_seq_len 256
   ```

3. Prune fewer layers at once:
   ```bash
   --layer_start 0 --layer_end 16  # First half
   # Then run again with --layer_start 16 --layer_end 32
   ```

---

## Troubleshooting

### Common Issues

#### Issue 1: PPL is NaN or Inf

**Symptoms**: Perplexity shows `nan` or very high values (> 1000)

**Possible Causes**:
- Model weights corrupted during pruning
- Pruning rate too aggressive
- Gradient explosion during fine-tuning

**Solutions**:
1. Check GQA ratio is maintained:
   ```bash
   grep "GQA比例验证" prune_log/{name}/*/training.log
   ```
   Should show: `✅ 所有层保持4:1`

2. Reduce pruning rate:
   ```bash
   --pruning_ratio 0.15  # Instead of 0.25
   ```

3. Add gradient clipping (modify `finetuner.py`):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

#### Issue 2: CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
1. **Immediate**: Use smaller batch/samples (see Task 5 above)

2. **Disable gradient for pruned layers** (already implemented):
   ```python
   for pruned_idx in pruned_layer_indices:
       for param in model.model.layers[pruned_idx].parameters():
           param.requires_grad = False
   ```

3. **Clear cache frequently**:
   ```python
   torch.cuda.empty_cache()
   gc.collect()
   ```

#### Issue 3: Layer Importance Analysis Very Slow

**Symptoms**: "分析层重要性" step takes > 1 hour

**Solutions**:
1. Use cached importance:
   ```bash
   # First run: analyze once
   python script.py --layer_importance_samples 50 --save_ckpt_log_name cache_importance

   # Subsequent runs: reuse
   python script.py --skip_importance_analysis \
       --layer_importance_config prune_log/cache_importance/layer_importance_config.json
   ```

2. Use activation-based method (faster than removal):
   ```bash
   --layer_importance_method activation
   ```

3. Reduce samples:
   ```bash
   --layer_importance_samples 20  # Instead of 50
   ```

#### Issue 4: Model Checkpoint Too Large

**Symptoms**: `pytorch_model.bin` is several GB

**Solutions**:
1. Save only model weights, not optimizer state:
   ```python
   torch.save(model.state_dict(), path)  # Instead of entire model
   ```

2. Use half precision:
   ```python
   model.half()
   torch.save(model.state_dict(), path)
   ```

#### Issue 5: GQA Ratio Not Maintained

**Symptoms**: Logs show different Q:KV ratios per layer

**Diagnosis**:
```bash
grep "Attention:" prune_log/{name}/*/training.log
```

**Solution**: This should NOT happen if using `prune_attention_by_gqa_groups()` correctly. If it does:
1. Check that `gqa_ratio=4` is passed correctly
2. Verify `keep_kv_indices` is computed from `select_gqa_groups_to_prune()`
3. Ensure no manual head pruning outside the GQA-aware functions

---

## Advanced Topics

### Modifying GQA Ratio

If you need to support models with different GQA ratios (e.g., 8:1):

**Update**:
1. Main script:
   ```python
   parser.add_argument('--gqa_ratio', type=int, default=4)
   ```

2. Pass to all pruning functions:
   ```python
   group_imp = compute_gqa_group_importance(layer, head_dim, args.gqa_ratio)
   ```

### Distributed Pruning

For very large models, you might want to distribute across GPUs:

**Not currently implemented**, but suggested approach:
1. Use `torch.nn.parallel.DistributedDataParallel`
2. Prune on rank 0, broadcast pruned model to other ranks
3. Fine-tune with distributed training

### Quantization After Pruning

To further compress the model:

```python
# After pruning
import torch.quantization as quant

model.eval()
quantized_model = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'pruned_quantized.bin')
```

---

## Best Practices for AI Assistants

### When Analyzing This Codebase

1. **Start with the main script**: `llama3_unbalanced_pruning_gqa_aware.py` orchestrates everything
2. **Understand the pipeline**: Read through steps 1-11 in the main script
3. **Check module docs**: `LLMPruner/README.md` and individual `__init__.py` files
4. **Look at RUN.md**: Contains real-world usage examples

### When Modifying Code

1. **Preserve GQA structure**: Always maintain 4:1 ratio
2. **Use the logger**: Don't use `print()`, use `logger.log()`
3. **Test incrementally**: Use debug mode before full runs
4. **Document in Chinese**: Keep consistency with existing comments
5. **Save checkpoints**: Always enable `--save_model` for important runs

### When Debugging

1. **Check logs first**: `prune_log/{name}/{timestamp}/training.log`
2. **Verify GQA ratio**: Look for "✅ 所有层保持4:1" in logs
3. **Check perplexity**: Should be reasonable (< 20 for 25% pruning)
4. **Inspect visualizations**: `pruning_strategy.png` shows per-layer rates

### When Adding Features

1. **Follow existing patterns**: Look at similar functions in the same module
2. **Update __init__.py**: Export new public functions
3. **Add CLI arguments**: Update `llama3_unbalanced_pruning_gqa_aware.py` argparse
4. **Document thoroughly**: Add Chinese docstrings
5. **Test with debug mode**: Use `--layer_start 10 --layer_end 15` for quick testing

---

## Quick Reference

### Essential Commands

```bash
# Full pipeline (pruning + fine-tuning)
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name experiment_name \
    --pruning_ratio 0.25 \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --finetune \
    --finetune_samples 500

# Debug mode (fast)
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --save_ckpt_log_name debug \
    --pruning_ratio 0.25 \
    --layer_importance_samples 10 \
    --channel_importance_samples 5 \
    --layer_start 10 \
    --layer_end 15

# Reuse layer importance
python llama3_unbalanced_pruning_gqa_aware.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --skip_importance_analysis \
    --layer_importance_config prune_log/previous_run/layer_importance_config.json \
    --pruning_ratio 0.30 \
    --save_model
```

### Essential File Locations

| What | Where |
|------|-------|
| Main entry point | `llama3_unbalanced_pruning_gqa_aware.py` |
| GQA pruning logic | `LLMPruner/methods/gqa_aware.py` |
| Layer importance | `LLMPruner/importance/layer_analyzer.py` |
| Fine-tuning | `LLMPruner/trainer/finetuner.py` |
| Evaluation | `LLMPruner/evaluator/ppl.py` |
| Data loading | `LLMPruner/datasets/example_samples.py` |
| Logs | `prune_log/{name}/{timestamp}/training.log` |
| Checkpoints | `prune_log/{name}/pytorch_model.bin` |

### Key Metrics to Monitor

| Metric | Good | Bad | Location in Logs |
|--------|------|-----|------------------|
| GQA Ratio | All layers 4:1 | Mixed ratios | "GQA比例验证" section |
| Pruning Rate | Within ±2% of target | > 5% off target | "实际剪枝率" section |
| Perplexity (25% pruning) | < 15 | > 100 or NaN | "剪枝后 PPL" section |
| PPL after fine-tuning | < 2% degradation | > 10% degradation | "微调后 PPL" section |

---

## Conclusion

This document should provide comprehensive guidance for AI assistants working with the GAQ-Aware-Prune codebase. The key principles to remember:

1. **Preserve GQA structure** (4:1 ratio) at all times
2. **Use layer-wise importance** for intelligent pruning
3. **Test incrementally** with debug mode
4. **Monitor logs closely** for validation
5. **Follow existing conventions** (Chinese comments, snake_case, logger usage)

For more details, refer to:
- `RUN.md` - Comprehensive usage examples
- `LLMPruner/README.md` - Module-level documentation
- Individual module docstrings - Implementation details

**Last Updated**: 2025-11-17
**Version**: 1.0
