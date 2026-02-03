# Using vLLM Recipes for Deployment

The [vllm-project/recipes](https://github.com/vllm-project/recipes) repository provides model-specific deployment guides with tested configurations and best practices. This guide explains how to effectively use these recipes.

## What Are vLLM Recipes?

Recipes are **curated deployment guides** for specific models that include:

- ✅ **Tested configurations** - Known-working parameters
- ✅ **Hardware requirements** - Minimum and recommended GPUs
- ✅ **Performance benchmarks** - Expected throughput/latency
- ✅ **Known issues** - Model-specific quirks and workarounds
- ✅ **Example commands** - Copy-paste ready deployment
- ✅ **Tips and tricks** - Optimization recommendations

## Repository Structure

```
vllm-project/recipes/
├── README.md                  # Overview and index
├── deepseek-r1/              # DeepSeek R1 recipe
│   ├── README.md
│   └── config.yaml
├── deepseek-v3/              # DeepSeek V3 recipe
│   ├── README.md
│   └── config.yaml
├── llama-3.1/                # Llama 3.1 recipe
│   ├── README.md
│   └── config.yaml
├── qwen2.5-vl/               # Qwen2.5-VL recipe
│   ├── README.md
│   └── config.yaml
└── ...
```

---

## How to Use Recipes

### Step 1: Find Your Model's Recipe

**Option 1: Browse the repository**
```
Visit: https://github.com/vllm-project/recipes
Look through the directory listing
```

**Option 2: Use the recipe index (from this skill)**
```python
from vllm_skills.library.deployment import find_recipe

recipe = find_recipe("meta-llama/Llama-3.1-8B-Instruct")
print(recipe['recipe_url'])
# https://github.com/vllm-project/recipes/tree/main/llama-3.1
```

**Option 3: Search GitHub**
```
Search query: "vllm recipe [model-name]"
Example: "vllm recipe llama 3.1"
```

### Step 2: Read the Recipe

Each recipe typically contains:

1. **Model Overview**
   - Model architecture
   - Size variants
   - Supported features

2. **Hardware Requirements**
   - Minimum GPU VRAM
   - Recommended setup
   - Tensor parallel configurations

3. **Installation**
   - vLLM version requirements
   - Special dependencies

4. **Configuration**
   - Recommended parameters
   - Configuration files
   - Command examples

5. **Performance**
   - Benchmark results
   - Throughput/latency numbers
   - Comparison data

6. **Known Issues**
   - Limitations
   - Workarounds
   - Open bugs

### Step 3: Adapt to Your Setup

Most recipes provide **multiple configurations**:

- **Minimal**: Lowest hardware requirements
- **Recommended**: Best balance of performance/cost
- **Maximum**: Highest performance setup

**Example from Llama 3.1 recipe:**

```yaml
# Minimal (1x RTX 4090)
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 1
max_model_len: 4096
gpu_memory_utilization: 0.85

# Recommended (1x A100 80GB)
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 1
max_model_len: 131072  # Full context
gpu_memory_utilization: 0.90
enable_prefix_caching: true

# Maximum (2x A100 80GB for 70B)
model: meta-llama/Llama-3.1-70B-Instruct
tensor_parallel_size: 2
max_model_len: 131072
gpu_memory_utilization: 0.90
dtype: bfloat16
```

---

## Recipe Categories

### 1. Standard Language Models

**Examples**: Llama, Mistral, Qwen

**What they cover**:
- Context length optimization
- Batch size tuning
- Quantization options
- Throughput vs latency trade-offs

**Typical command**:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
```

### 2. Mixture-of-Experts (MoE) Models

**Examples**: Mixtral, DeepSeek-V3, Qwen-MoE

**What they cover**:
- Expert parallelism configuration
- Memory optimization for sparse models
- FP8 quantization requirements
- TP + EP combinations

**Typical command**:
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.80
```

### 3. Vision-Language Models (VLM)

**Examples**: Qwen-VL, LLaVA, Phi-3-Vision

**What they cover**:
- Image input configuration
- Multi-modal batching
- Memory requirements for vision encoders
- Image preprocessing

**Typical command**:
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.85
```

### 4. Quantized Models

**Examples**: AWQ/GPTQ variants

**What they cover**:
- Quantization method specifics
- Quality vs performance trade-offs
- Compatible vLLM versions
- Re-quantization options

**Typical command**:
```bash
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90
```

---

## Common Recipe Patterns

### Pattern 1: Single GPU Deployment

```yaml
# From various recipes
target: Single consumer/professional GPU
models: 7B-13B
configuration:
  tensor_parallel_size: 1
  max_model_len: 4096-8192
  gpu_memory_utilization: 0.80-0.85
  quantization: optional (fp8/awq for larger models)
```

### Pattern 2: Multi-GPU for Large Models

```yaml
# From 70B+ recipes
target: Multi-GPU servers
models: 70B-405B
configuration:
  tensor_parallel_size: 2/4/8
  max_model_len: 8192-32768
  gpu_memory_utilization: 0.85-0.90
  dtype: bfloat16
  enable_prefix_caching: true
```

### Pattern 3: MoE with Expert Parallelism

```yaml
# From MoE recipes (Mixtral, DeepSeek-V3)
target: Large MoE models
configuration:
  tensor_parallel_size: 8
  enable_expert_parallel: true
  quantization: fp8
  kv_cache_dtype: fp8
  gpu_memory_utilization: 0.75-0.80
```

---

## Example: Using DeepSeek-V3 Recipe

### 1. Find the Recipe

**URL**: https://github.com/vllm-project/recipes/tree/main/deepseek-v3

### 2. Check Requirements

From recipe:
- **Model**: DeepSeek-V3 (671B MoE, 37B active)
- **Minimum**: 8x H100 80GB with FP8
- **Recommended**: 8x H100 80GB or 16x A100 80GB

### 3. Follow Configuration

```bash
# From recipe's recommended config
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 \
  --dtype bfloat16
```

### 4. Apply Known Workarounds

From recipe's "Known Issues" section:
- Use vLLM 0.8.0+
- FP8 quantization is required for 8 GPUs
- First inference may take 2-3 minutes (model loading)

---

## Creating Custom Configurations

Recipes are **starting points**, not rigid requirements. Customize based on:

### Your Hardware
```python
# Recipe says: 2x A100 80GB
# You have: 4x A100 40GB

# Adjust TP size:
--tensor-parallel-size 4  # Instead of 2
--gpu-memory-utilization 0.85  # More conservative
```

### Your Use Case
```python
# Recipe optimizes for: Throughput
# You need: Low latency

# Adjust parameters:
--max-num-seqs 64  # Instead of 256
--enable-chunked-prefill  # Add for latency
--max-model-len 8192  # Reduce if not needed
```

### Your Budget
```python
# Recipe recommends: H100s
# You have: A100s

# Trade-offs:
--quantization fp8  # Compensate for less VRAM
--kv-cache-dtype fp8  # Further reduction
--max-model-len 8192  # Reduce context if needed
```

---

## Recipe Best Practices

### ✅ DO:

- **Start with recipe defaults** - They're tested and known to work
- **Read "Known Issues"** - Save time troubleshooting
- **Check vLLM version** - Recipes specify minimum versions
- **Benchmark your setup** - Verify performance matches recipe claims
- **Contribute back** - Share improvements via PR

### ❌ DON'T:

- **Blindly copy** - Understand each parameter
- **Ignore hardware requirements** - Recipes assume specific GPUs
- **Skip version checks** - Older vLLM may not support all features
- **Expect exact numbers** - Performance varies by workload

---

## Contributing to Recipes

### When to Contribute

- Found better configuration for a model
- Discovered workaround for known issue
- Tested model on new hardware
- Created recipe for new model

### How to Contribute

1. **Fork repository**
   ```bash
   git clone https://github.com/vllm-project/recipes
   cd recipes
   ```

2. **Create/update recipe**
   ```
   model-name/
   ├── README.md        # Documentation
   ├── config.yaml      # Configuration
   └── benchmark.md     # Performance data (optional)
   ```

3. **Test thoroughly**
   - Verify configuration works
   - Run benchmarks
   - Document results

4. **Submit PR**
   - Clear description
   - Include hardware used
   - Share benchmark results

---

## Recipe Template

When creating new recipes or configurations:

```markdown
# [Model Name] Deployment Recipe

## Model Overview
- Architecture: [e.g., Llama, Qwen, MoE]
- Parameters: [e.g., 8B, 70B, 671B]
- Context Length: [e.g., 4K, 128K]

## Hardware Requirements

### Minimum
- GPU: [model and count]
- VRAM: [total GB]
- RAM: [GB]

### Recommended
- GPU: [model and count]
- VRAM: [total GB]
- RAM: [GB]

## Installation

```bash
pip install vllm>=0.X.X
```

## Configuration

### Basic Setup
```bash
vllm serve [model] \
  [parameters]
```

### Advanced Setup
```bash
vllm serve [model] \
  [optimized parameters]
```

## Performance

| Setup | Throughput | Latency |
|-------|-----------|---------|
| ... | ... | ... |

## Known Issues
- Issue 1: [description and workaround]
- Issue 2: [description and workaround]

## References
- [Links to docs, papers, etc.]
```

---

## Finding Help

If recipes don't cover your model:

1. **Check supported models**: https://docs.vllm.ai/en/latest/models/supported_models.html
2. **Search similar models**: Use recipe from similar architecture
3. **Ask community**: 
   - Discord: https://discord.gg/vllm
   - GitHub Discussions: https://github.com/vllm-project/vllm/discussions
4. **Create recipe**: Document your configuration and contribute!

---

## Resources

- **Recipe Repository**: https://github.com/vllm-project/recipes
- **Contributing Guide**: https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md
- **vLLM Documentation**: https://docs.vllm.ai/
- **Discord**: https://discord.gg/vllm
