# CLAUDE.md - AI Assistant Guide for GAQ-Aware-Prune

> **Project**: GQA-Aware Structured Pruning for LLaMA-3 Models
> **Purpose**: Intelligent neural network pruning that preserves Grouped Query Attention (GQA) structure
> **Last Updated**: 2025-11-17

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Key Concepts](#key-concepts)
4. [Core Components](#core-components)
5. [Development Workflows](#development-workflows)
6. [Coding Conventions](#coding-conventions)
7. [AI Assistant Guidelines](#ai-assistant-guidelines)
8. [Common Tasks](#common-tasks)

---

## Project Overview

### What is This Project?

This repository implements **GQA-aware structured pruning** for LLaMA-3 large language models. The goal is to reduce model size (by ~20-25%) while maintaining model quality (minimal perplexity degradation <5%).

### The Problem It Solves

Traditional pruning methods that naively remove attention heads break the 4:1 Query:Key-Value ratio in LLaMA-3's Grouped Query Attention, causing catastrophic performance degradation (PPL >700k). This project uses Taylor importance to prune entire GQA groups intelligently, maintaining semantic coherence.

### Key Innovation

**GQA Group-Level Pruning**: Instead of pruning individual heads, the system:
1. Groups 4 Q heads + 1 KV head as a single GQA unit
2. Calculates Taylor importance for each group
3. Prunes entire groups based on importance
4. Maintains 4:1 ratio naturally, preserving model structure

### Results Comparison

- **Old Method** (torch_pruning + simple truncation): PPL = 718,107 ❌
- **New Method** (GQA-aware Taylor): PPL degradation <5% ✅

---

## Codebase Structure

```
GAQ-Aware-Prune/
├── 读我.md                                    # Chinese documentation with usage examples
├── run_gqa_aware_pruning.sh                  # Main execution script
├── llama3_unbalanced_pruning_v3_gqa_aware.py # Main pruning pipeline (v3)
├── layer_importance.py                        # Layer importance analysis utilities
├── gqa_aware_pruning.py                      # Core GQA-aware pruning logic
└── LLMPruner/                                # Utility modules
    ├── __init__.py
    ├── README.md                             # Module documentation
    ├── utils/
    │   ├── logger.py                         # Logging with directory management
    │   └── get_best_gpu.py                   # GPU selection utility
    ├── evaluator/
    │   ├── __init__.py
    │   └── ppl.py                            # Perplexity evaluation (NOW IMPLEMENTED)
    └── datasets/
        ├── __init__.py
        └── example_samples.py                # Dataset loading utilities (NOW IMPLEMENTED)
```

### File Purposes

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `gqa_aware_pruning.py` | Core GQA pruning algorithms | `compute_gqa_group_importance()`, `select_gqa_groups_to_prune()`, `prune_attention_by_gqa_groups()` |
| `layer_importance.py` | Layer-wise importance analysis | `LayerImportanceAnalyzer`, `UnbalancedStructuredPruningCalculator` |
| `llama3_unbalanced_pruning_v3_gqa_aware.py` | Main orchestration script | `main()` - full pruning pipeline |
| `run_gqa_aware_pruning.sh` | Convenience wrapper | Shell script with preset hyperparameters |
| `LLMPruner/utils/logger.py` | Experiment logging | `LoggerWithDepth` - hierarchical logging |

---

## Key Concepts

### 1. Grouped Query Attention (GQA)

LLaMA-3 uses GQA architecture where:
- **32 Query (Q) heads** process queries
- **8 Key-Value (KV) heads** share key-value pairs
- **4:1 ratio**: 4 Q heads share 1 KV head

```
Q heads: [0,1,2,3] [4,5,6,7] [8,9,10,11] ... [28,29,30,31]
           ↓           ↓          ↓               ↓
KV heads:  [0]        [1]        [2]      ...    [7]
```

**Why it matters**: Breaking this structure causes misalignment between Q and KV heads, destroying attention semantics.

### 2. Taylor Importance

Measures neuron importance using first-order Taylor expansion:

```
Importance(weight) = |weight × gradient|
```

For GQA groups:
```python
group_importance = sum(Q_heads_importance) + KV_head_importance
                 = sum(q_proj, o_proj) + sum(k_proj, v_proj)
```

### 3. Layer Importance Analysis

Two methods for determining which layers are more critical:

**a) Removal Method** (default):
- Temporarily bypass each layer (identity function)
- Measure perplexity increase
- Higher increase = more important layer

**b) Activation Method**:
- Monitor L2 norm of layer activations
- Higher activation = more important layer

### 4. Unbalanced Pruning Strategy

Different layers get different pruning rates based on importance:

- **Inverse Strategy** (default): Important layers → low pruning rate
- **Proportional**: Important layers → high pruning rate
- **Uniform**: All layers → same pruning rate

---

## Core Components

### Component 1: Layer Importance Analyzer (`layer_importance.py`)

**Class**: `LayerImportanceAnalyzer`

**Key Methods**:
```python
measure_layer_importance_by_removal(texts, num_layers)
# Returns: Dict[layer_idx, importance_score]
# Method: Temporarily remove each layer, measure PPL increase

measure_layer_importance_by_activation(texts)
# Returns: Dict[layer_idx, activation_norm]
# Method: Hook-based activation monitoring
```

**Usage Pattern**:
```python
analyzer = LayerImportanceAnalyzer(model, tokenizer, device='cuda')
importance = analyzer.measure_layer_importance_by_removal(eval_texts, num_layers=32)
```

### Component 2: Pruning Rate Calculator (`layer_importance.py`)

**Class**: `UnbalancedStructuredPruningCalculator`

**Key Methods**:
```python
compute_layer_pruning_rates(
    target_overall_rate=0.25,
    strategy='inverse',
    alpha=1.0,
    min_rate=0.15,
    max_rate=0.5
)
# Returns: Dict[layer_idx, pruning_rate]
```

**Algorithm**:
1. Normalize layer importance scores
2. Apply log transform to handle outliers
3. Invert for 'inverse' strategy (important → low rate)
4. Scale to match target overall rate
5. Clip to [min_rate, max_rate]

### Component 3: GQA-Aware Pruning (`gqa_aware_pruning.py`)

**Core Functions**:

```python
# 1. Calculate importance for each GQA group
compute_gqa_group_importance(layer, head_dim=128, gqa_ratio=4)
# Returns: Tensor[num_kv_heads] - importance per group

# 2. Select which groups to keep
select_gqa_groups_to_prune(group_importance, target_num_kv_heads)
# Returns: (keep_indices, prune_indices)

# 3. Execute pruning
prune_attention_by_gqa_groups(layer, keep_kv_indices, head_dim, gqa_ratio)
# Modifies: layer.self_attn.{q,k,v,o}_proj weights
```

**Critical Implementation Details**:
- Converts head indices to channel indices (head_idx * head_dim)
- Updates both weight tensors AND Linear layer attributes
- Maintains Q:KV ratio by design (4 Q heads per KV head)

### Component 4: Main Pipeline (`llama3_unbalanced_pruning_v3_gqa_aware.py`)

**Pipeline Stages**:

```python
# Stage 1: Load model
model = LlamaForCausalLM.from_pretrained(base_model)

# Stage 2: Evaluate layer importance
analyzer = LayerImportanceAnalyzer(model, tokenizer)
layer_importance = analyzer.measure_layer_importance_by_removal(texts, 32)

# Stage 3: Calculate per-layer pruning rates
calculator = UnbalancedStructuredPruningCalculator(layer_importance, 32)
layer_pruning_rates = calculator.compute_layer_pruning_rates(0.25, 'inverse')

# Stage 4: Prune each layer with GQA-awareness
for layer_idx in pruning_layers:
    # 4a: Compute gradients via forward+backward
    loss = model(examples, labels=examples).loss
    loss.backward()

    # 4b: Calculate GQA group importance
    group_imp = compute_gqa_group_importance(layer, 128, 4)

    # 4c: Select and prune groups
    keep_indices, _ = select_gqa_groups_to_prune(group_imp, target_kv_heads)
    prune_attention_by_gqa_groups(layer, keep_indices, 128, 4)

    # 4d: Optional MLP pruning
    if prune_mlp:
        # Taylor importance for gate/up/down projections
        ...

# Stage 5: Save and evaluate
torch.save(model, checkpoint_path)
ppl = PPLMetric(model, tokenizer, ['wikitext2'])
```

---

## Development Workflows

### Workflow 1: Running Standard Pruning

```bash
# Method 1: Use convenience script
./run_gqa_aware_pruning.sh

# Method 2: Direct Python invocation
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --save_ckpt_log_name llama3_pruned \
    --pruning_ratio 0.25 \
    --importance_method removal \
    --pruning_strategy inverse \
    --prune_mlp \
    --save_model \
    --test_after_prune
```

**Output Locations**:
- Logs: `prune_log/{save_ckpt_log_name}/{timestamp}/training.log`
- Model: `prune_log/{save_ckpt_log_name}/pytorch_model.bin`
- Config: `prune_log/{save_ckpt_log_name}/layer_importance_config.json`
- Visualization: `prune_log/{save_ckpt_log_name}/pruning_strategy.png`

### Workflow 2: Two-Stage Pruning (Reuse Importance Analysis)

```bash
# Stage 1: Analyze importance (slow, ~30min)
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --importance_method removal \
    --importance_samples 100 \
    --save_ckpt_log_name analyze_only

# Stage 2: Prune with saved importance (fast, ~10min)
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --skip_importance_analysis \
    --importance_config prune_log/analyze_only/layer_importance_config.json \
    --pruning_ratio 0.30 \  # Try different rates!
    --save_model
```

### Workflow 3: Analyzing Pruning Results

```bash
# From 读我.md - model size comparison
python check_model_size.py \
    --original_model /path/to/original \
    --pruned_model prune_log/{name}/pytorch_model.bin

# Pruning structure analysis
python analyze_pruning.py \
    --original_model /path/to/original \
    --pruned_model prune_log/{name}/pytorch_model.bin
```

### Workflow 4: Debugging Failed Pruning

**Common Issues**:

1. **Gradient Shape Mismatch**
   - **Cause**: Already-pruned layers still computing gradients
   - **Solution**: Disable gradients for pruned layers (lines 273-275)

2. **PPL Explosion**
   - **Cause**: Too aggressive pruning or wrong GQA ratio
   - **Solution**: Check `min_pruning_rate` (should be ≥12.5% for 8 KV heads)

3. **CUDA OOM**
   - **Cause**: Gradient accumulation during layer-by-layer pruning
   - **Solution**: Clear cache after each layer (line 358)

---

## Coding Conventions

### Python Style

- **Docstrings**: Chinese comments for domain concepts, English for code documentation
- **Type Hints**: Used in function signatures (e.g., `Dict[int, float]`)
- **Logging**: Use `logger.log()` instead of `print()` for all status messages
- **Device Management**: Always move tensors explicitly (avoid implicit device placement)

### Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Functions | `snake_case` + descriptive verb | `compute_gqa_group_importance()` |
| Classes | `PascalCase` + noun | `LayerImportanceAnalyzer` |
| Variables | `snake_case` | `layer_pruning_rates` |
| Constants | `UPPER_SNAKE_CASE` | `MODEL_PATH` (in bash) |
| Temporary | Single letter or `_` prefix | `q_imp`, `_` for unused |

### Module Organization

**Import Order**:
1. Standard library (`os`, `sys`, `json`)
2. Third-party ML (`torch`, `transformers`, `numpy`)
3. Local utilities (`LLMPruner.utils`)
4. Local domain (`layer_importance`, `gqa_aware_pruning`)

**Separation of Concerns**:
- **Pure logic**: `gqa_aware_pruning.py` (no I/O, no logging)
- **Analysis tools**: `layer_importance.py` (has visualization)
- **Orchestration**: `llama3_unbalanced_pruning_v3_gqa_aware.py` (argparse, logging, pipeline)
- **Utilities**: `LLMPruner/` (reusable helpers)

### Error Handling

**Current Pattern** (limited error handling):
- Rely on PyTorch errors for shape mismatches
- Use `assert` for critical invariants
- Manual validation (e.g., `target_num_kv_heads = max(1, target_num_kv_heads)`)

**Best Practices for Extensions**:
```python
# Validate GQA ratio
assert num_q_heads % num_kv_heads == 0, \
    f"Q heads ({num_q_heads}) must be divisible by KV heads ({num_kv_heads})"

# Check pruning feasibility
if target_num_kv_heads < 1:
    logger.log(f"WARNING: Layer {idx} pruning rate too high, keeping 1 group")
    target_num_kv_heads = 1
```

---

## AI Assistant Guidelines

### When Working on This Codebase

#### DO:

✅ **Preserve GQA Structure**
- Always maintain Q:KV ratio (4:1 for LLaMA-3)
- When adding features, ensure they respect GQA groups
- Test that `num_q_heads % num_kv_heads == 0` after any changes

✅ **Understand Gradient Dependencies**
- Pruning requires gradients → model must be in train mode with `requires_grad=True`
- Disable gradients for already-pruned layers to avoid shape mismatches
- Always call `model.zero_grad()` before computing Taylor importance

✅ **Use Existing Utilities**
- Use `LoggerWithDepth` for any new logging needs
- Reuse `get_examples()` for loading sample data
- Leverage `PPLMetric` for evaluation

✅ **Maintain Reproducibility**
- Save all hyperparameters via logger's `config` parameter
- Document pruning rates in JSON format
- Include visualization when changing pruning strategies

✅ **Test Incrementally**
- After modifying pruning logic, test on 1 layer first
- Validate forward pass with `model(example_prompts[:1])` after each layer
- Check parameter count matches expected reduction

#### DON'T:

❌ **Don't Break Head Alignment**
- Never prune Q and KV heads independently
- Don't use generic pruning libraries (torch_pruning) that ignore GQA structure
- Avoid pruning individual channels within a head

❌ **Don't Ignore Device Placement**
- Don't assume all model parts are on same device (may be multi-GPU)
- Always use `first_device = next(model.parameters()).device` pattern
- Don't create tensors without explicit `.to(device)`

❌ **Don't Skip Cleanup**
- Don't accumulate gradients across layers (causes OOM)
- Always delete loss tensors and call `torch.cuda.empty_cache()`
- Don't leave hooks registered after use

❌ **Don't Modify Linear Attributes Without Weights**
- Never change `layer.out_features` without updating `layer.weight.data`
- Both tensor shapes AND module attributes must stay synchronized
- Update in order: 1) weights, 2) biases, 3) Linear attributes

### Understanding Version History

**v1** (not in repo): Basic torch_pruning
**v2** (not in repo): torch_pruning + post-hoc truncation → **FAILED** (PPL 718k)
**v3** (current): GQA-aware group pruning → **SUCCESS** (<5% degradation)

When reading code:
- Comments mentioning "v2" refer to the failed approach
- "GQA-aware" always means the v3 group-level method
- "Taylor importance" is the correct metric (not magnitude)

### Common Modification Scenarios

#### Scenario 1: Add Support for Different Model Architecture

**Example**: Adding Llama-2 support (different Q:KV ratio)

```python
# 1. Detect model's GQA ratio
def detect_gqa_ratio(model):
    first_layer = model.model.layers[0]
    num_q = first_layer.self_attn.num_heads
    num_kv = first_layer.self_attn.num_key_value_heads
    return num_q // num_kv

# 2. Pass ratio dynamically
gqa_ratio = detect_gqa_ratio(model)
group_imp = compute_gqa_group_importance(layer, head_dim=128, gqa_ratio=gqa_ratio)
```

#### Scenario 2: Change Importance Metric

**Example**: Use magnitude instead of Taylor

```python
# In compute_gqa_group_importance()
# OLD:
salience[name] = (sub_layer.weight * sub_layer.weight.grad).abs()

# NEW (magnitude-based):
salience[name] = sub_layer.weight.abs()
# Note: Requires no gradients, faster but less accurate
```

#### Scenario 3: Add Progressive Pruning

**Example**: Prune in multiple iterations

```python
# Multi-stage pruning (25% → 35% → 50%)
for target_rate in [0.25, 0.35, 0.50]:
    # Recompute importance after each stage
    layer_importance = analyzer.measure_layer_importance_by_removal(texts, 32)
    layer_pruning_rates = calculator.compute_layer_pruning_rates(target_rate, 'inverse')

    # Prune incrementally
    for layer_idx in pruning_layers:
        current_kv = layer.self_attn.num_key_value_heads
        target_kv = int(current_kv * (1 - layer_pruning_rates[layer_idx]))
        # ... prune to target_kv

    # Evaluate and decide whether to continue
    ppl = PPLMetric(model, tokenizer, ['wikitext2'])
    if ppl > threshold:
        break
```

### Debugging Guide

**Issue**: Layer forward fails after pruning

**Check**:
1. Print shapes: `logger.log(f"q_proj: {layer.self_attn.q_proj.weight.shape}")`
2. Verify attributes match: `assert layer.q_proj.out_features == layer.q_proj.weight.shape[0]`
3. Check ratio: `assert num_q_heads // num_kv_heads == 4`

**Issue**: Gradients are None during Taylor computation

**Check**:
1. Model mode: `model.train()` (not `model.eval()`)
2. Requires grad: `param.requires_grad` should be `True`
3. Loss backward: `loss.backward()` was called
4. Not detached: Ensure no `.detach()` in forward path

**Issue**: OOM during pruning

**Solutions**:
1. Reduce `--num_examples` (fewer samples for gradient computation)
2. Reduce `--importance_samples` (fewer samples for layer analysis)
3. Use `--skip_importance_analysis` (reuse cached importance scores)
4. Clear cache more frequently: Add `torch.cuda.empty_cache()` after each layer

---

## Common Tasks

### Task 1: Prune with Different Target Ratio

```bash
# Try 30% pruning instead of 25%
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --pruning_ratio 0.30 \  # Changed from 0.25
    --save_ckpt_log_name llama3_pruned_30pct \
    --save_model
```

### Task 2: Prune Only Attention (Skip MLP)

```bash
# Remove --prune_mlp flag
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --pruning_ratio 0.25 \
    # --prune_mlp  # REMOVED
    --save_model
```

### Task 3: Prune Specific Layer Range

```bash
# Only prune layers 10-25 (skip early/late layers)
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --pruning_ratio 0.25 \
    --layer_start 10 \
    --layer_end 25 \
    --save_model
```

### Task 4: Analyze Without Pruning

```bash
# Just compute and save layer importance
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --importance_method removal \
    --importance_samples 100 \
    --save_ckpt_log_name importance_analysis
    # Note: No --save_model flag
```

### Task 5: Change Pruning Strategy

```bash
# Proportional strategy: important layers get pruned MORE
python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model /path/to/model \
    --pruning_ratio 0.25 \
    --pruning_strategy proportional \  # Changed from 'inverse'
    --alpha 2.0 \  # Increase contrast
    --save_model
```

### Task 6: Visualize Layer Importance

```python
# After running analysis, check the visualization
from PIL import Image
img = Image.open('prune_log/{name}/pruning_strategy.png')
img.show()

# Or programmatically load the config
import json
with open('prune_log/{name}/layer_importance_config.json') as f:
    config = json.load(f)
    print(config['layer_importance'])
    print(config['statistics'])
```

---

## Environment & Dependencies

### Required Packages

```python
# Core ML
torch >= 2.0.0
transformers >= 4.30.0
datasets >= 2.0.0

# Numerical
numpy
scipy (implied by transformers)

# Visualization
matplotlib
PIL/Pillow

# Utilities
tqdm
```

### Hardware Requirements

- **GPU**: CUDA-capable GPU with ≥24GB VRAM (for LLaMA-3-8B)
- **RAM**: ≥64GB system memory
- **Storage**: ≥50GB free (model + logs + checkpoints)

### Model Paths

**Expected Structure**:
```
/newdata/LLMs/
└── Llama-3-8B-Instruct/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── pytorch_model.bin (or .safetensors)
```

**Update in Scripts**:
- `run_gqa_aware_pruning.sh`: Line 6 `MODEL_PATH`
- When running Python directly: `--base_model` argument

---

## Testing & Validation

### Validation Checklist After Modifications

- [ ] **Structural Integrity**
  - [ ] All layers maintain Q:KV = 4:1 ratio
  - [ ] Parameter count reduced by expected amount
  - [ ] Forward pass succeeds on test input

- [ ] **Numerical Quality**
  - [ ] PPL on wikitext2 < baseline + 5%
  - [ ] No NaN/Inf in model weights
  - [ ] Loss is finite during gradient computation

- [ ] **Reproducibility**
  - [ ] Config saved to `description.txt`
  - [ ] Layer importance saved to JSON
  - [ ] Visualization generated

- [ ] **Logging**
  - [ ] Training log created in timestamped subdirectory
  - [ ] All hyperparameters recorded
  - [ ] Layer-by-layer pruning progress logged

### Quick Validation Commands

```bash
# 1. Check model loads
python -c "import torch; m = torch.load('prune_log/{name}/pytorch_model.bin'); print(m['model'])"

# 2. Verify GQA ratios
python -c "
import torch
ckpt = torch.load('prune_log/{name}/pytorch_model.bin')
model = ckpt['model']
for i, layer in enumerate(model.model.layers):
    q = layer.self_attn.num_heads
    kv = layer.self_attn.num_key_value_heads
    print(f'Layer {i}: Q={q}, KV={kv}, ratio={q//kv}:1')
"

# 3. Test forward pass
python -c "
import torch
from transformers import AutoTokenizer
ckpt = torch.load('prune_log/{name}/pytorch_model.bin')
model = ckpt['model']
tokenizer = ckpt['tokenizer']
inputs = tokenizer('Hello, world!', return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
print('Forward pass successful!')
"
```

---

## Troubleshooting

### Problem: "RuntimeError: Sizes of tensors must match"

**Cause**: Gradient computation after pruning caused shape mismatch

**Solution**:
```python
# Disable gradients for already-pruned layers
for pruned_idx in pruned_layer_indices:
    for param in model.model.layers[pruned_idx].parameters():
        param.requires_grad = False
```

### Problem: PPL is NaN or extremely high

**Causes**:
1. Pruned too many groups (< 1 KV head remaining)
2. Numerical instability in half precision
3. Broke GQA alignment

**Solutions**:
1. Increase `--min_pruning_rate` to ensure at least 1 group remains
2. Add gradient clipping before backward: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Verify ratio: `assert num_q_heads % num_kv_heads == 0`

### Problem: "KeyError: 'wikitext2 (wikitext-2-raw-v1)'"

**Cause**: PPL evaluation returned different dataset key format

**Solution**:
```python
# In step 7 of main script, use flexible key matching
ppl_results = ppl  # This is a dict
for key, value in ppl_results.items():
    if 'wikitext' in key.lower():
        logger.log(f"Wikitext2 PPL: {value}")
```

### Problem: "CUDA out of memory"

**Solutions** (in order of preference):
1. Reduce batch size: `--num_examples 5` (default 10)
2. Use gradient checkpointing: `model.gradient_checkpointing_enable()`
3. Clear cache more often: Add `torch.cuda.empty_cache()` after each layer
4. Skip importance analysis: `--skip_importance_analysis` and reuse cached results

---

## References & Related Work

### Academic Context

This project implements techniques from:
- **Grouped Query Attention**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- **Taylor Importance**: Molchanov et al., "Pruning Convolutional Neural Networks for Resource Efficient Inference"
- **Structured Pruning**: LLM-Pruner (Ma et al., 2023)

### LLMPruner Module Implementation

**✅ FULLY IMPLEMENTED** - All required modules are now available in `LLMPruner/`:

#### 1. Perplexity Evaluation (`LLMPruner/evaluator/ppl.py`)

```python
from LLMPruner.evaluator.ppl import PPLMetric

# Evaluate on multiple datasets
ppl_metric = PPLMetric(
    model, tokenizer,
    datasets=['wikitext2', 'ptb'],
    seq_len=128,
    device='cuda'
)

# Access results
print(ppl_metric)  # Print all results
wikitext_ppl = ppl_metric['wikitext2 (wikitext-2-raw-v1)']
ptb_ppl = ppl_metric.get('ptb', 'N/A')
```

**Features**:
- Supports multiple datasets: wikitext2, wikitext103, ptb, c4
- Uses sliding window for efficient computation
- Returns dict-like object with dataset names as keys
- Automatic progress tracking with tqdm

#### 2. Sample Data Loading (`LLMPruner/datasets/example_samples.py`)

```python
from LLMPruner.datasets.example_samples import get_examples

# Load samples for gradient computation
examples = get_examples(
    'wikitext',
    tokenizer,
    num_samples=10,
    seq_len=64
).to('cuda')

# Use in pruning
loss = model(examples, labels=examples).loss
loss.backward()
```

**Features**:
- Supports wikitext, wikitext103, c4, ptb datasets
- Returns tokenized tensors ready for model input
- Automatic padding and truncation
- Filters out short/empty texts

#### 3. Utilities

```python
# Logging with hierarchical structure
from LLMPruner.utils.logger import LoggerWithDepth

logger = LoggerWithDepth(
    env_name='my_experiment',
    config={'lr': 0.001},
    root_dir='logs'
)

# Auto-select best GPU
from LLMPruner.utils.get_best_gpu import get_best_gpu
device = f'cuda:{get_best_gpu()}'
```

**For detailed usage**, see `LLMPruner/README.md`

---

## Appendix: Key Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--pruning_ratio` | 0.25 | 0.1-0.5 | Overall target pruning rate |
| `--alpha` | 1.0 | 0.5-3.0 | Importance weighting contrast |
| `--min_pruning_rate` | 0.15 | 0.0-0.3 | Minimum per-layer pruning |
| `--max_pruning_rate` | 0.5 | 0.3-0.8 | Maximum per-layer pruning |
| `--importance_samples` | 50 | 10-200 | Samples for layer importance |
| `--num_examples` | 10 | 5-50 | Samples for Taylor gradients |
| `--head_dim` | 128 | Fixed | LLaMA-3 architecture |
| `--gqa_ratio` | 4 | Fixed | LLaMA-3 architecture |

**Tuning Guide**:
- **Higher `alpha`** → More aggressive unbalanced pruning (wider gap between important/unimportant layers)
- **Higher `importance_samples`** → More accurate layer ranking, slower analysis
- **Higher `num_examples`** → More stable gradients, higher memory usage

---

## Version Notes

**Current Version**: v3 (GQA-Aware)

**Changelog**:
- **v3**: Implemented GQA group-level Taylor importance pruning
  - Fix: Maintains 4:1 Q:KV ratio by design
  - Result: PPL degradation <5%

- **v2** (deprecated): Used torch_pruning + post-processing
  - Issue: Broke GQA alignment
  - Result: PPL explosion (718k)

**Future Enhancements** (suggested):
- [ ] Support for other GQA architectures (Llama-2, Mistral)
- [ ] Progressive pruning with quality gates
- [ ] Sensitivity analysis for hyperparameters
- [ ] Knowledge distillation during pruning
- [ ] Mixed precision pruning (FP16/INT8)

---

## Contact & Contribution

**Repository Status**: Research/experimental code
**Language**: Chinese documentation (`读我.md`), English code comments
**License**: Not specified (check with repository owner)

When contributing:
1. Test on small models first (e.g., LLaMA-2-7B)
2. Validate PPL before committing
3. Update this CLAUDE.md with new patterns
4. Add examples to `读我.md` for end-user workflows

---

**Document Maintained By**: AI Assistant (Claude)
**For**: Future AI assistants and human developers
**Last Validated**: 2025-11-17 with GAQ-Aware-Prune codebase
