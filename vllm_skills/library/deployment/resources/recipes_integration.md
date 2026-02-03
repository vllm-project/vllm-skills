# Integration with vllm-project/recipes

## What are vLLM Recipes?

The [vllm-project/recipes](https://github.com/vllm-project/recipes) repository contains:
- Model-specific deployment guides
- Optimized configuration templates
- Hardware recommendations
- Troubleshooting guides
- Performance tuning tips

## How This Skill Uses Recipes

The Deployment Assistant integrates with recipes by:

1. **Model Mapping**: Links popular models to their recipes
2. **Configuration Generation**: Uses recipe templates as starting point
3. **Hardware Matching**: Validates your hardware against recipe requirements
4. **Optimization**: Applies recipe-specific optimizations

## Recipe Structure

Typical recipe includes:
```
recipes/
├── models/
│   └── llama-3.1/
│       ├── README.md          # Deployment guide
│       ├── config.yaml         # Configuration template
│       ├── hardware.yaml       # Hardware requirements
│       └── examples/           # Example commands
```

## Using Recipes with Deployment Assistant

### Step 1: Check if Recipe Exists

```python
from vllm_skills.library.deployment import DeploymentAssistant

assistant = DeploymentAssistant()
recipe = assistant.find_recipe("meta-llama/Llama-3.1-70B-Instruct")

if recipe:
    print(f"Found recipe: {recipe['path']}")
else:
    print("No recipe found, using general configuration")
```

### Step 2: Generate Configuration

```python
# Assistant will use recipe template if available
config = assistant.suggest_config(
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    hardware=hardware_info
)
```

### Step 3: Apply Recipe Optimizations

The assistant automatically:
- Uses recipe's recommended parameters
- Applies model-specific flags
- Suggests compatible hardware
- Provides recipe-specific troubleshooting

## Available Recipe Categories

### By Model Family

- **DeepSeek**: R1, V3, V3.1 deployment guides
- **Qwen**: Qwen3, Qwen3-VL, Qwen2.5-VL configurations
- **Llama**: Llama 3.1, 3.3, 4-Scout recipes
- **Mistral**: Large and Small model setups
- **GLM**: GLM-4 deployment
- **NVIDIA**: Nemotron model configs
- **Kimi**: Moonshot Kimi recipes

### By Use Case

- **High Throughput**: Batch processing optimization
- **Low Latency**: Interactive application setup
- **Multi-GPU**: Tensor/pipeline parallelism guides
- **Quantization**: AWQ/GPTQ deployment
- **Vision**: Vision-language model setup

## Recipe Index

See [models/recipe_index.yaml](../models/recipe_index.yaml) for complete mapping.

## Contributing to Recipes

If you develop an optimal configuration:

1. Test thoroughly on your hardware
2. Document performance metrics
3. Submit PR to vllm-project/recipes
4. Update recipe_index.yaml

## Example: DeepSeek R1 Recipe

```yaml
# From vllm-project/recipes
model: deepseek-ai/DeepSeek-R1

hardware_requirements:
  min_gpu_memory: 80GB
  recommended_gpu: A100-80GB
  tensor_parallel: 2

configuration:
  dtype: bfloat16
  max_model_len: 32768
  gpu_memory_utilization: 0.95
  enable_prefix_caching: true

optimizations:
  - Flash Attention 2
  - Chunked prefill
  - Custom kernels

command: |
  vllm serve deepseek-ai/DeepSeek-R1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching
```

## Manual Recipe Usage

You can also use recipes directly:

```bash
# Clone recipes repository
git clone https://github.com/vllm-project/recipes.git

# Navigate to specific recipe
cd recipes/models/llama-3.1

# Follow the README
cat README.md

# Use provided configuration
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --config config.yaml
```

## When Recipes Aren't Available

For models without recipes:
1. Use general configuration presets
2. Start with conservative settings
3. Iteratively optimize based on monitoring
4. Consider contributing your config as a recipe

## See Also
- [Recipe Repository](https://github.com/vllm-project/recipes)
- [Recipe Index](../models/recipe_index.yaml)
- [Hardware Matrix](../models/hardware_matrix.yaml)
