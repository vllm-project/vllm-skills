# vLLM Deployment Assistant Skill

## Metadata

- **Name**: deployment
- **Version**: 0.1.0
- **Author**: vLLM Project
- **Tags**: vllm, deployment, configuration, troubleshooting
- **Requirements**: psutil, pyyaml

## Purpose

The vLLM Deployment Assistant is an intelligent skill that helps users deploy vLLM models optimally by:

1. **Auto-detecting hardware and environment**
2. **Suggesting optimal configurations**
3. **Integrating with vllm-project/recipes**
4. **Troubleshooting common issues**
5. **Generating deployment commands**

## Capabilities

### Hardware Detection
- GPU detection (name, memory, CUDA version, count)
- CPU core count
- RAM availability
- NVIDIA driver version

### Environment Detection
- Python version
- PyTorch version and CUDA availability
- vLLM installation status
- Flash Attention availability

### Configuration Generation
- Model-specific configurations
- Hardware-optimized parameters
- Preset templates (high throughput, low latency, memory constrained)
- Recipe integration

### Troubleshooting
- OOM (Out of Memory) errors
- CUDA errors and compatibility issues
- Kernel compilation problems
- Performance optimization

## Usage

### As a Skill

```python
from vllm_skills.library.deployment import DeploymentAssistant

# Initialize the skill
assistant = DeploymentAssistant()

# Check hardware
hardware = assistant.check_hardware()
print(f"GPU: {hardware.gpu.name if hardware.gpu else 'None'}")
print(f"GPU Memory: {hardware.gpu.memory_total}MB" if hardware.gpu else "")

# Check environment
environment = assistant.check_environment()
print(f"Python: {environment.python_version}")
print(f"vLLM: {environment.vllm_version}")

# Get configuration for a model
config = assistant.suggest_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    hardware=hardware
)

# Generate deployment command
command = assistant.generate_command(config)
print(f"Command:\n{command}")
```

### Interactive Agent Usage

The skill is designed to work with conversational agents:

**User**: "Help me deploy Llama 3.3 70B on my system"

**Agent** (using skill):
1. Checks hardware and environment
2. Validates compatibility
3. Finds relevant recipe
4. Suggests optimal configuration
5. Generates deployment command
6. Provides troubleshooting tips

## Configuration Presets

### High Throughput
Optimized for batch processing with maximum throughput:
- Large batch sizes (512 sequences)
- High memory utilization (0.95)
- Prefix caching enabled
- Chunked prefill enabled

**Use cases**: Batch inference, offline processing, high-volume API

### Low Latency
Optimized for minimal response time:
- Small batch sizes (64 sequences)
- Conservative memory usage (0.85)
- No extra processing overhead
- Priority scheduling

**Use cases**: Interactive chatbots, real-time applications, demos

### Memory Constrained
Optimized for limited GPU memory:
- Small batch sizes (128 sequences)
- Low memory utilization (0.75)
- Reduced context length
- Quantization recommendations

**Use cases**: Single GPU development, smaller VRAM, testing

## Recipe Integration

The skill integrates with [vllm-project/recipes](https://github.com/vllm-project/recipes):

- **Model Mapping**: Maps models to their deployment recipes
- **Hardware Requirements**: Validates against recipe specifications
- **Configuration Templates**: Uses recipe-recommended settings
- **Optimizations**: Applies recipe-specific flags

### Supported Models with Recipes

- **DeepSeek**: R1, V3, V3.1, V3.2
- **Qwen**: Qwen3, Qwen3-VL, Qwen2.5-VL
- **Llama**: 3.1, 3.3-70B, 4-Scout
- **Mistral**: Large, Small
- **GLM**: GLM-4
- **NVIDIA**: Nemotron
- **Kimi**: Moonshot Kimi

## Hardware Compatibility Matrix

### Small Models (1B-3B)
- **Minimum**: GTX 1060 6GB, T4 16GB
- **Recommended**: RTX 3060 12GB, RTX 4060 8GB
- **Memory**: 2-8GB VRAM

### Medium Models (7B-13B)
- **Minimum**: RTX 3090 24GB, V100 16GB
- **Recommended**: A10 24GB, RTX 4090 24GB
- **Memory**: 14-32GB VRAM

### Large Models (70B+)
- **Minimum**: 2x A100 80GB
- **Recommended**: 2x H100 80GB, 4x A100 40GB
- **Memory**: 140GB+ VRAM (or quantization)

## Troubleshooting Guides

The skill includes comprehensive troubleshooting documentation:

### Common Issues
- Installation failures
- Model loading errors
- Performance issues
- Server crashes
- API timeouts
- Multi-GPU problems

### CUDA Errors
- Out of memory errors
- Version mismatches
- Kernel launch failures
- Initialization errors
- cuDNN issues
- NCCL errors (multi-GPU)

### Memory Issues
- OOM diagnosis
- Memory optimization strategies
- Quantization options
- Batch size tuning
- Context length management

### Kernel Issues
- Flash Attention problems
- Custom kernel compilation
- PagedAttention errors
- Quantization kernels
- MoE routing
- Triton issues

## Agent Behavior Guidelines

When an agent uses this skill:

1. **Always start with detection**: Check hardware and environment first
2. **Validate compatibility**: Ensure model can run on available hardware
3. **Provide context**: Explain why certain configurations are recommended
4. **Offer alternatives**: If optimal config won't work, suggest alternatives
5. **Include troubleshooting**: Anticipate common issues and provide solutions
6. **Reference documentation**: Point to official resources when needed
7. **Be specific**: Generate exact commands users can run

### Example Interaction Flow

1. **User asks for help deploying a model**
2. **Agent checks hardware and environment**
3. **Agent validates model compatibility**
4. **Agent searches for recipe**
5. **Agent generates optimized configuration**
6. **Agent provides deployment command**
7. **Agent includes monitoring tips**
8. **Agent offers troubleshooting resources**

## Advanced Features

### Tensor Parallelism
Automatically suggests tensor parallel size based on:
- Available GPU count
- Model size
- GPU memory

### Quantization
Recommends quantization when:
- GPU memory is insufficient
- User explicitly requests it
- Performance optimization needed

Options: AWQ (4-bit), GPTQ (2-8 bit), FP8 (H100), SqueezeLLM

### Performance Tuning
Suggests optimizations:
- Flash Attention installation
- xFormers for older GPUs
- Prefix caching for repeated prompts
- Chunked prefill for throughput

## Resources

### Official Sources
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Recipes](https://github.com/vllm-project/recipes)
- [Discord Community](https://discord.gg/vllm)

### Related Files
- [Hardware Matrix](models/hardware_matrix.yaml)
- [Recipe Index](models/recipe_index.yaml)
- [Common Issues](troubleshooting/common_issues.md)
- [CUDA Errors](troubleshooting/cuda_errors.md)
- [Memory Issues](troubleshooting/memory_issues.md)
- [Kernel Issues](troubleshooting/kernel_issues.md)
- [Acceleration Libraries](resources/acceleration_libs.md)

## Future Enhancements

- Real-time monitoring integration
- Performance benchmarking
- Cost estimation
- Auto-scaling recommendations
- Cloud deployment templates
- Docker container generation
