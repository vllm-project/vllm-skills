# vllm-skills 🚀

> Agent skills for vLLM — production-ready tools for LLM agents

## Why vllm-skills?

| Feature | Benefit |
|---------|---------|
| 🔧 Pre-built skills | Deployment, coding, web, data processing |
| 🐳 Sandboxed execution | Safe code execution via Docker |
| ⚡ Optimized for vLLM | Template fixes + FSM warmup for guided decoding |
| 🔌 Easy integration | Mount alongside your vLLM server |
| 📖 Recipe integration | Works with vllm-project/recipes |

## Quickstart

```bash
pip install vllm-skills
```

## Architecture

The vllm-skills framework provides a modular architecture for building and deploying agent skills:

- **core/base.py** - Abstract base class for skills with metadata and execution interface
- **core/registry.py** - Skill discovery and loading system
- **core/sandbox.py** - Safe execution environments (Docker/Local)
- **library/** - Collection of built-in skills
- **vllm_utils/** - vLLM-specific optimizations

## Available Skills

### 🚀 Deployment Assistant

Intelligent vLLM deployment helper that:
- Auto-detects hardware and environment (GPU, CPU, RAM, CUDA)
- Suggests optimal configuration based on your system
- Integrates with [vllm-project/recipes](https://github.com/vllm-project/recipes)
- Troubleshoots common issues (OOM, CUDA errors, kernel issues)
- Provides configuration presets (high throughput, low latency, memory constrained)

### Coming Soon

- **Coding Skills** - Code generation, refactoring, and analysis
- **Web Skills** - Web scraping, API interaction, and data extraction
- **Data Skills** - Data processing, transformation, and analysis

## Integration with vllm-project/recipes

This project works alongside [vllm-project/recipes](https://github.com/vllm-project/recipes) by:

- Mapping popular models (DeepSeek, Qwen, Llama, Mistral, etc.) to their deployment recipes
- Providing hardware compatibility matrices
- Auto-generating optimal launch commands based on your environment
- Troubleshooting using recipe-specific knowledge

## Usage Example

```python
from vllm_skills.client import SkillEnabledClient

# Initialize client with deployment skill
client = SkillEnabledClient(
    base_url="http://localhost:8000/v1",
    skills=["deployment"]
)

# Use the deployment assistant
response = client.chat_with_skills(
    messages=[
        {"role": "user", "content": "Help me deploy Llama 3.3 70B on my system"}
    ]
)
```

## Project Structure

```
vllm-skills/
├── vllm_skills/          # Main package
│   ├── core/             # Core framework
│   ├── library/          # Skill library
│   │   ├── deployment/   # Deployment assistant
│   │   ├── coding/       # Coding skills
│   │   ├── web/          # Web skills
│   │   └── data/         # Data skills
│   ├── vllm_utils/       # vLLM optimizations
│   └── client.py         # Client wrapper
├── examples/             # Usage examples
└── tests/               # Test suite
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Adding new skills
- Skill structure template
- Testing requirements
- PR process

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Recipes](https://github.com/vllm-project/recipes)
- [GitHub Discussions](https://github.com/vllm-project/vllm-skills/discussions)
