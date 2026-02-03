# vLLM Deployment Assistant - Usage Guide

This guide shows how to use the vLLM Deployment Assistant skill for various deployment scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Hardware Detection](#hardware-detection)
3. [Environment Checking](#environment-checking)
4. [Recipe Lookup](#recipe-lookup)
5. [Configuration Generation](#configuration-generation)
6. [Command Building](#command-building)
7. [Troubleshooting](#troubleshooting)
8. [AI Agent Integration](#ai-agent-integration)

---

## Basic Usage

### Python API

```python
from vllm_skills.library.deployment import DeploymentAssistant

# Create assistant instance
assistant = DeploymentAssistant()

# Use various methods
hardware = assistant.check_hardware()
env = assistant.check_environment()
recipe = assistant.find_recipe("model-name")
config = assistant.suggest_config("model-name", hardware)
command = assistant.generate_command(config)
```

### Convenience Functions

```python
from vllm_skills.library.deployment import (
    check_hardware,
    check_environment,
    find_recipe,
    suggest_config,
    generate_command
)

# Direct function calls
hardware = check_hardware()
env = check_environment()
```

---

## Hardware Detection

### Detect All Hardware

```python
hardware = assistant.check_hardware()

# Output structure:
{
    'gpus': [
        {
            'index': 0,
            'name': 'NVIDIA A100-SXM4-80GB',
            'memory_gb': 80,
            'compute_capability': '8.0',
            'platform': 'nvidia'
        }
    ],
    'gpu_count': 1,
    'total_vram_gb': 80,
    'platform': 'nvidia',
    'cpu': {
        'cores': 64,
        'model': 'AMD EPYC 7763'
    },
    'ram_gb': 512
}
```

### Check Specific Components

```python
from vllm_skills.library.deployment.checks import (
    detect_gpus,
    detect_cpu,
    detect_memory
)

gpus = detect_gpus()
cpu = detect_cpu()
ram_gb = detect_memory()
```

---

## Environment Checking

### Check Software Environment

```python
env = assistant.check_environment()

# Output structure:
{
    'python_version': '3.10.12',
    'pytorch': {
        'version': '2.1.0+cu121',
        'cuda_available': True,
        'cuda_version': '12.1',
        'device_count': 1
    },
    'cuda_driver': '535.86.10',
    'vllm_version': '0.8.2',
    'flash_attn_version': '2.5.0',
    'triton_version': '2.1.0',
    'xformers_version': '0.0.23'
}
```

### Compatibility Check

```python
from vllm_skills.library.deployment.checks import check_compatibility

env = assistant.check_environment()
compat = check_compatibility(env)

# Output includes compatibility matrix
{
    'vllm_version': '0.8.2',
    'vllm_series': '0.8.x',
    'checks': {
        'python': {'compatible': True, 'message': 'Compatible'},
        'pytorch': {'compatible': True, 'message': 'Compatible'},
        'cuda': {'compatible': True, 'message': 'Compatible'},
        'flash_attention': {'compatible': True, 'message': 'Compatible'}
    },
    'overall_compatible': True
}
```

---

## Recipe Lookup

### Find Recipe for Model

```python
# Search by model name
recipe = assistant.find_recipe("meta-llama/Llama-3.1-8B-Instruct")

# Output:
{
    'model': 'Llama-3.1',
    'recipe_url': 'https://github.com/vllm-project/recipes/tree/main/llama-3.1',
    'recipe_path': 'llama-3.1',
    'description': 'Meta Llama 3.1 deployment guide'
}

# Returns None if no recipe found
recipe = assistant.find_recipe("unknown-model")  # None
```

### Check Recipe Index

The recipe index (`models/recipe_index.yaml`) contains mappings for:

- DeepSeek models (R1, V3, etc.)
- Qwen models (Qwen3, Qwen2.5-VL)
- Llama models (3.1, 3.3, 4-Scout)
- Mistral models (Large-3, Mixtral)
- GLM models
- And more...

---

## Configuration Generation

### Basic Configuration

```python
config = assistant.suggest_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    hardware=hardware,
    use_case='balanced'
)

# Output:
{
    'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
    'tensor_parallel_size': 1,
    'gpu_memory_utilization': 0.90,
    'max_model_len': None,
    'max_num_seqs': 256,
    'dtype': 'auto',
    'quantization': None,
    'enable_prefix_caching': False,
    'enable_chunked_prefill': False
}
```

### Use Case Optimizations

```python
# High Throughput
config = assistant.suggest_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_case='throughput'
)
# Sets: max_num_seqs=512, enable_prefix_caching=True

# Low Latency
config = assistant.suggest_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_case='latency'
)
# Sets: max_num_seqs=64, enable_chunked_prefill=True

# Balanced (default)
config = assistant.suggest_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_case='balanced'
)
# Standard settings
```

### MoE Model Configuration

```python
# For Mixture-of-Experts models
config = assistant.suggest_config(
    model_name="deepseek-ai/DeepSeek-V3",
    hardware=hardware
)

# Automatically sets:
# - enable_expert_parallel=True
# - quantization='fp8' (for large MoE)
# - kv_cache_dtype='fp8'
```

---

## Command Building

### Generate Deployment Command

```python
config = {
    'model_name': 'meta-llama/Llama-3.1-70B-Instruct',
    'tensor_parallel_size': 2,
    'gpu_memory_utilization': 0.85,
    'max_model_len': 16384,
    'max_num_seqs': 128,
    'dtype': 'bfloat16',
    'enable_prefix_caching': True
}

command = assistant.generate_command(config)

# Output (formatted for readability):
"""
vllm serve \
  meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --enable-prefix-caching
"""
```

### Full Workflow Example

```python
# Complete deployment workflow
assistant = DeploymentAssistant()

# 1. Check system
hardware = assistant.check_hardware()
env = assistant.check_environment()

# 2. Find recipe
model = "meta-llama/Llama-3.1-8B-Instruct"
recipe = assistant.find_recipe(model)
if recipe:
    print(f"Using recipe: {recipe['recipe_url']}")

# 3. Generate config
config = assistant.suggest_config(model, hardware, use_case='latency')

# 4. Build command
command = assistant.generate_command(config)

# 5. Execute
print(f"Run this command:\n{command}")
```

---

## Troubleshooting

### Access Troubleshooting Guides

```python
# Get path to troubleshooting guide
guide_path = assistant.get_troubleshooting_guide('oom')
# Returns: '/path/to/troubleshooting/memory_issues.md'

# Available guides:
# - 'oom' or 'memory' → memory_issues.md
# - 'cuda' → cuda_errors.md
# - 'flash_attn' → common_issues.md
# - 'moe' → common_issues.md
# - 'kernel' → kernel_issues.md
```

### Read Troubleshooting Content

```python
guide_path = assistant.get_troubleshooting_guide('oom')
if guide_path:
    with open(guide_path) as f:
        guide_content = f.read()
    print(guide_content)
```

### GPU Requirements

```python
from vllm_skills.library.deployment.checks import get_gpu_requirements

# Get requirements for model size
requirements = get_gpu_requirements(70)  # 70B model

# Output:
{
    'min_vram_gb': 140,
    'recommended_vram_gb': 160,
    'example_gpus': ['2x A100 80GB', '2x H100 80GB'],
    'tensor_parallel_recommended': 2
}
```

---

## AI Agent Integration

### For Claude, GPT-4, and other AI Assistants

When helping users with vLLM deployment:

#### 1. Initial Assessment

```python
# Step 1: Detect current setup
assistant = DeploymentAssistant()
hardware = assistant.check_hardware()
env = assistant.check_environment()

# Step 2: Present findings to user
print(f"Found {hardware['gpu_count']} GPUs")
print(f"vLLM installed: {env['vllm_version'] or 'No'}")
```

#### 2. Gather Requirements

```python
# Ask user for model and use case
model_name = input("Which model? ")
use_case = input("Use case (throughput/latency/balanced)? ")

# Find recipe
recipe = assistant.find_recipe(model_name)
if recipe:
    print(f"Found recipe: {recipe['recipe_url']}")
```

#### 3. Generate Configuration

```python
# Create optimal config
config = assistant.suggest_config(
    model_name=model_name,
    hardware=hardware,
    use_case=use_case
)

# Show config to user
print(json.dumps(config, indent=2))
```

#### 4. Build and Explain Command

```python
# Generate command
command = assistant.generate_command(config)

# Explain to user
print(f"\nRun this command:")
print(command)

print("\nExplanation:")
if config['tensor_parallel_size'] > 1:
    print(f"- Using {config['tensor_parallel_size']} GPUs for tensor parallelism")
if config['enable_prefix_caching']:
    print("- Prefix caching enabled for repeated prompts")
# ... more explanations
```

#### 5. Troubleshooting Support

```python
# If user reports error
error_type = 'oom'  # based on user's error
guide_path = assistant.get_troubleshooting_guide(error_type)

# Provide relevant section from guide
with open(guide_path) as f:
    guide = f.read()
    # Extract relevant section
    # Present solution to user
```

### Agent Best Practices

1. **Always detect environment first**
   ```python
   hardware = assistant.check_hardware()
   env = assistant.check_environment()
   ```

2. **Check for recipes**
   ```python
   recipe = assistant.find_recipe(model_name)
   if recipe:
       # Use recipe recommendations
   ```

3. **Explain trade-offs**
   ```python
   # Show different configurations
   for use_case in ['throughput', 'latency', 'balanced']:
       config = assistant.suggest_config(model, hardware, use_case)
       # Explain differences
   ```

4. **Provide complete commands**
   ```python
   command = assistant.generate_command(config)
   # User can copy-paste directly
   ```

5. **Reference official docs**
   - Use `resources/official_sources.yaml` for links
   - Point to vLLM documentation
   - Link to recipes

---

## Advanced Usage

### Custom Configuration

```python
# Start with suggested config
config = assistant.suggest_config(model, hardware)

# Customize
config['max_model_len'] = 8192  # Reduce context
config['quantization'] = 'fp8'  # Add quantization
config['kv_cache_dtype'] = 'fp8'  # FP8 KV cache

# Validate
from vllm_skills.library.deployment.config import validate_configuration
is_valid, errors = validate_configuration(config)

if is_valid:
    command = assistant.generate_command(config)
else:
    print("Configuration errors:", errors)
```

### Parameter Validation

```python
from vllm_skills.library.deployment.config import (
    get_parameter_description,
    validate_configuration
)

# Get parameter info
param_info = get_parameter_description('tensor_parallel_size')
print(param_info['description'])
print(f"Example: {param_info['example']}")

# Validate config
config = {...}
is_valid, errors = validate_configuration(config)
```

### Loading Presets

```python
import yaml

# Load preset
with open('vllm_skills/library/deployment/config/presets/high_throughput.yaml') as f:
    preset = yaml.safe_load(f)

# Apply preset values to config
config.update({
    'gpu_memory_utilization': preset['gpu_memory_utilization'],
    'max_num_seqs': preset['max_num_seqs'],
    'enable_prefix_caching': preset['enable_prefix_caching']
})
```

---

## Examples

See `examples/deployment_example.py` for a complete working example that demonstrates:

- Hardware detection
- Environment checking
- Recipe lookup
- Configuration generation
- Command building
- Troubleshooting guide access

Run it with:
```bash
python examples/deployment_example.py
```

---

## Resources

- **SKILL.md**: Complete skill definition for AI agents
- **Troubleshooting Guides**: `troubleshooting/` directory
- **Resource Index**: `resources/official_sources.yaml`
- **Recipe Index**: `models/recipe_index.yaml`
- **Hardware Matrix**: `models/hardware_matrix.yaml`
- **Configuration Presets**: `config/presets/`
