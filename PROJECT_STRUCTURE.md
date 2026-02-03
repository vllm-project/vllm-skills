# vLLM Deployment Assistant Skill - Project Structure

```
vllm-skills/
│
├── README.md                           # Main project documentation
├── USAGE_GUIDE.md                      # Comprehensive usage guide
├── .gitignore                          # Git ignore rules
│
├── examples/                           # Usage examples
│   └── deployment_example.py           # Complete working example
│
└── vllm_skills/                        # Main Python package
    ├── __init__.py                     # Package root
    │
    └── library/                        # Skills library
        ├── __init__.py
        │
        └── deployment/                 # 🎯 DEPLOYMENT ASSISTANT SKILL
            │
            ├── SKILL.md                # 📘 Complete skill definition (13.7KB)
            │                           #    - Metadata & requirements
            │                           #    - Environment detection checklist
            │                           #    - Configuration parameters
            │                           #    - Recipe integration
            │                           #    - Troubleshooting flowcharts
            │                           #    - Usage examples
            │                           #    - Agent behavior guidelines
            │
            ├── __init__.py             # 🔧 DeploymentAssistant class (16.3KB)
            │                           #    - check_hardware()
            │                           #    - check_environment()
            │                           #    - find_recipe()
            │                           #    - suggest_config()
            │                           #    - generate_command()
            │
            ├── checks/                 # 🔍 System detection modules
            │   ├── __init__.py
            │   ├── hardware.py         #    - GPU/CPU/RAM detection
            │   ├── environment.py      #    - Python/CUDA/PyTorch/vLLM versions
            │   └── compatibility.py    #    - Version compatibility matrix
            │
            ├── config/                 # ⚙️ Configuration management
            │   ├── __init__.py
            │   ├── parameters.py       #    - Parameter descriptions & validation
            │   │
            │   └── presets/            # 📋 Pre-configured setups
            │       ├── high_throughput.yaml      # Max batch size
            │       ├── low_latency.yaml          # Min response time
            │       └── memory_constrained.yaml   # Limited VRAM
            │
            ├── models/                 # 📊 Model information
            │   ├── recipe_index.yaml   #    - 25+ models → vllm-project/recipes
            │   │                       #    - DeepSeek, Qwen, Llama, Mistral, etc.
            │   │
            │   └── hardware_matrix.yaml#    - GPU requirements by model size
            │                           #    - 1B-3B, 7B-8B, 13B-14B, 70B-72B, 405B+
            │                           #    - MoE models, quantization savings
            │
            ├── troubleshooting/        # 🔧 Problem-solving guides
            │   ├── common_issues.md    #    - Quick reference table (9.1KB)
            │   │                       #    - OOM, Flash Attention, Architecture
            │   │
            │   ├── cuda_errors.md      #    - CUDA diagnostics (9.2KB)
            │   │                       #    - OOM, version mismatch, driver issues
            │   │                       #    - Runtime errors, kernel launches
            │   │
            │   ├── memory_issues.md    #    - Memory optimization (10.3KB)
            │   │                       #    - OOM scenarios & solutions
            │   │                       #    - Quantization strategies
            │   │                       #    - Memory profiling
            │   │
            │   └── kernel_issues.md    #    - Kernel compilation (9.8KB)
            │                           #    - Triton, Punica, Flash Attention
            │                           #    - Compute capability issues
            │
            └── resources/              # 📚 Reference materials
                ├── official_sources.yaml    # Official docs index (9.9KB)
                │                           #    - vLLM documentation links
                │                           #    - Recipes repository
                │                           #    - GitHub resources
                │                           #    - Community links
                │
                ├── acceleration_libs.md    # Acceleration libraries (10.2KB)
                │                           #    - Flash Attention 2
                │                           #    - Triton
                │                           #    - xFormers
                │                           #    - cuBLAS/cuBLASLt
                │                           #    - ROCm support
                │
                └── recipes_integration.md  # Using recipes (10.1KB)
                                            #    - Recipe structure
                                            #    - How to use recipes
                                            #    - Contributing recipes
```

## File Statistics

| Category | Files | Lines | Description |
|----------|-------|-------|-------------|
| **Core Skill** | 1 | 486 | SKILL.md - Agent instructions |
| **Python Implementation** | 7 | 836 | DeploymentAssistant + modules |
| **Configuration** | 4 | 260 | Presets & parameters |
| **Model Data** | 2 | 386 | Recipes & hardware matrix |
| **Troubleshooting** | 4 | 1,352 | Common issues guides |
| **Resources** | 3 | 1,074 | Official docs & libraries |
| **Documentation** | 2 | 519 | README & usage guide |
| **Examples** | 1 | 138 | Working example |
| **Total** | **24** | **5,051** | Complete skill package |

## Key Capabilities

### 1. Hardware Detection ✅
- Automatically detects NVIDIA/AMD GPUs
- Reports VRAM, compute capability
- Detects CPU cores and system RAM
- Platform identification (NVIDIA/AMD/CPU)

### 2. Environment Detection ✅
- Python version checking
- PyTorch + CUDA version detection
- vLLM installation verification
- Flash Attention, Triton, xFormers detection

### 3. Recipe Integration ✅
- Maps 25+ models to deployment guides
- Links to vllm-project/recipes
- Model-specific configurations
- Known issues and workarounds

### 4. Configuration Generation ✅
- Hardware-aware suggestions
- Use case optimization (throughput/latency/balanced)
- MoE model detection and configuration
- Quantization recommendations

### 5. Command Generation ✅
- Builds complete vllm serve commands
- Includes all necessary flags
- Formatted for copy-paste
- Customizable configurations

### 6. Troubleshooting Support ✅
- 4 comprehensive guides (38KB total)
- Systematic debugging flowcharts
- Common issues quick reference
- Solution-oriented documentation

## Usage Patterns

### For AI Agents
```python
# 1. Detect system
hardware = assistant.check_hardware()
env = assistant.check_environment()

# 2. Find recipe
recipe = assistant.find_recipe("model-name")

# 3. Generate config
config = assistant.suggest_config("model-name", hardware, "latency")

# 4. Build command
command = assistant.generate_command(config)
```

### For Users
```bash
# Run example to see capabilities
python examples/deployment_example.py

# Read skill definition
cat vllm_skills/library/deployment/SKILL.md

# Access troubleshooting
cat vllm_skills/library/deployment/troubleshooting/memory_issues.md
```

## Documentation Quality

- ✅ **Comprehensive**: Covers all aspects of vLLM deployment
- ✅ **Practical**: Copy-paste ready commands and solutions
- ✅ **Structured**: Organized by topic and use case
- ✅ **Maintained**: References to official docs and recipes
- ✅ **Tested**: Working Python implementation with examples

## Coverage

### Models (25+ recipes)
- DeepSeek (R1, V3, V3.1, V3.2)
- Qwen (Qwen3, Qwen2.5-VL, Qwen3-VL)
- Llama (3.1, 3.3, 4-Scout)
- Mistral (Large-3, Ministral-3, Mixtral)
- GLM (4, 4.5, 4.6, 4.7)
- NVIDIA Nemotron, Phi-4, Gemma-2, Command-R

### Issues Covered
- CUDA Out of Memory (OOM)
- CUDA version mismatches
- Flash Attention compatibility
- Kernel compilation failures
- Model architecture not supported
- Tensor parallel configuration
- Expert parallel (MoE models)
- Performance optimization

### Configurations
- Hardware: 1B-405B+ models
- GPUs: V100, A100, H100, RTX series, AMD
- Use cases: Throughput, Latency, Memory-constrained
- Features: Quantization, Caching, Chunked prefill

## Next Steps

This skill is ready for:
1. ✅ AI agent integration (Claude, GPT-4, etc.)
2. ✅ User-facing deployment assistance
3. ✅ Community contributions (more recipes, issues)
4. ✅ Extension with additional skills (monitoring, optimization, etc.)
