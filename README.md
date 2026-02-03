# vllm-skills

Agent skills for vLLM - A collection of specialized skills that help AI agents guide users through vLLM deployment, optimization, and troubleshooting.

## Overview

This repository contains reusable skills that AI agents can use to assist users with vLLM-related tasks. Each skill provides:

- **Structured knowledge** - Curated documentation and best practices
- **Automation** - Python implementations for common tasks
- **Guidance** - Step-by-step instructions for AI agents
- **Resources** - Links to official documentation and recipes

## Available Skills

### 1. Deployment Assistant (`vllm_skills/library/deployment/`)

A comprehensive deployment assistant that helps AI agents guide users through vLLM deployment.

**Features:**
- ✅ Auto-detect hardware (GPU/CPU/RAM) and software environment
- ✅ Interactive parameter configuration
- ✅ Recipe integration from [vllm-project/recipes](https://github.com/vllm-project/recipes)
- ✅ Troubleshooting guidance for common issues
- ✅ Resource indexing to official docs

**Key Components:**
- `SKILL.md` - Complete skill definition and agent instructions
- `__init__.py` - Python implementation with DeploymentAssistant class
- `checks/` - Hardware, environment, and compatibility detection
- `config/` - Parameter configuration and presets
- `models/` - Recipe index and hardware requirements matrix
- `troubleshooting/` - Guides for common issues (OOM, CUDA, kernels, etc.)
- `resources/` - Official documentation index and acceleration library guides

**Example Usage:**

```python
from vllm_skills.library.deployment import DeploymentAssistant

assistant = DeploymentAssistant()

# Check hardware
hardware = assistant.check_hardware()
print(f"Found {hardware['gpu_count']} GPUs with {hardware['total_vram_gb']}GB VRAM")

# Check environment
env = assistant.check_environment()
print(f"vLLM {env['vllm_version']}, PyTorch {env['pytorch_version']}")

# Find recipe
recipe = assistant.find_recipe("meta-llama/Llama-3.1-8B-Instruct")
print(f"Recipe: {recipe['recipe_url']}")

# Generate configuration
config = assistant.suggest_config("meta-llama/Llama-3.1-8B-Instruct", hardware)
command = assistant.generate_command(config)
print(command)
```

**Quick Start:**

```bash
# Run the example
python examples/deployment_example.py

# Read the skill definition
cat vllm_skills/library/deployment/SKILL.md
```

## Repository Structure

```
vllm-skills/
├── README.md                           # This file
├── vllm_skills/                        # Main package
│   └── library/                        # Skill library
│       └── deployment/                 # Deployment assistant skill
│           ├── SKILL.md                # Skill definition
│           ├── __init__.py             # Python implementation
│           ├── checks/                 # Detection modules
│           │   ├── hardware.py
│           │   ├── environment.py
│           │   └── compatibility.py
│           ├── config/                 # Configuration
│           │   ├── parameters.py
│           │   └── presets/            # Pre-configured setups
│           ├── models/                 # Model information
│           │   ├── recipe_index.yaml
│           │   └── hardware_matrix.yaml
│           ├── troubleshooting/        # Issue guides
│           │   ├── common_issues.md
│           │   ├── cuda_errors.md
│           │   ├── memory_issues.md
│           │   └── kernel_issues.md
│           └── resources/              # Reference materials
│               ├── official_sources.yaml
│               ├── acceleration_libs.md
│               └── recipes_integration.md
└── examples/                           # Usage examples
    └── deployment_example.py
```

## For AI Agents

This repository is designed to be used by AI agents (like Claude, GPT-4, etc.) to help users with vLLM deployment. Each skill contains:

1. **SKILL.md** - Primary instructions for the AI agent
   - Metadata and description
   - Step-by-step guidance
   - Common patterns and examples
   - Resource links

2. **Python Implementation** - Automation tools
   - Hardware/environment detection
   - Configuration generation
   - Command building
   - Validation

3. **Documentation** - Reference materials
   - Troubleshooting guides
   - Official documentation index
   - Best practices

4. **Configuration** - Pre-built setups
   - Use case presets (throughput, latency, memory-constrained)
   - Hardware requirements
   - Recipe mappings

### Agent Guidelines

When using these skills:

1. **Always read SKILL.md first** - It contains the complete instructions
2. **Use Python tools for automation** - Don't manually replicate logic
3. **Reference official sources** - Link to vLLM docs and recipes
4. **Provide complete solutions** - Give copy-paste ready commands
5. **Explain trade-offs** - Help users understand implications
6. **Troubleshoot systematically** - Use provided flowcharts

## Contributing

We welcome contributions! To add a new skill:

1. Create a new directory in `vllm_skills/library/`
2. Include:
   - `SKILL.md` - Skill definition
   - `__init__.py` - Python implementation
   - Supporting modules and documentation
3. Add examples to `examples/`
4. Update this README

## License

[License information to be added]

## Resources

- **vLLM Documentation**: https://docs.vllm.ai/
- **vLLM Recipes**: https://github.com/vllm-project/recipes
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **vLLM Discord**: https://discord.gg/vllm
