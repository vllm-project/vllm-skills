# vLLM Skills

A collection of skills for deploying vLLM as an online service with OpenAI-compatible API. This project follows the [anthropics/skills](https://github.com/anthropics/skills) template format.

## Overview

vLLM Skills provides modular, reusable skills for deploying vLLM servers:

- **vllm-deploy** - Deploy vLLM locally or via Docker with OpenAI-compatible API
- OpenAI-compatible API out of the box - use as a drop-in replacement
- Support for quantization, LoRA adapters, and tensor parallelism

## Project Structure

```
vllm-skills/
├── skills/
│   └── vllm-deploy/          # Local & Docker deployment skill
│       ├── SKILL.md          # Skill documentation (YAML frontmatter + body)
│       ├── scripts/          # Deployment utilities
│       └── references/       # Additional documentation
└── README.md
```

## Skills

### vllm-deploy

Deploy vLLM as an online service with OpenAI-compatible API locally or via Docker.

**Features:**
- Local deployment with `vllm serve`
- Docker container deployment with GPU support
- Flexible configuration for models, quantization, LoRA
- Health check and management utilities

**Quick Start:**

```python
import sys
sys.path.insert(0, "skills/vllm-deploy/scripts")

from vllm_deploy import VLLMConfig, VLLMDeployer

config = VLLMConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    port=8000,
)

deployer = VLLMDeployer(vllm_config=config)
result = deployer.deploy_local(background=True)

print(f"API running at: {result.endpoint_url}")
```

See [skills/vllm-deploy/SKILL.md](skills/vllm-deploy/SKILL.md) for full documentation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hsliuustc0106/vllm-skills.git
   cd vllm-skills
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install vllm
   ```

For Docker deployment, ensure you have Docker and the NVIDIA Container Toolkit installed.

## OpenAI API Compatibility

vLLM implements the OpenAI API protocol. Query any deployed endpoint using the standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # vLLM accepts any key by default
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

**Supported endpoints:**
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models

## Configuration

### VLLMConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"Qwen/Qwen2.5-1.5B-Instruct"` | HuggingFace model ID or local path |
| `host` | str | `"0.0.0.0"` | Host address to bind |
| `port` | int | `8000` | API server port |
| `tensor_parallel_size` | int | `1` | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | float | `0.9` | GPU memory utilization (0.0-1.0) |
| `max_model_len` | int | `None` | Maximum context length |
| `dtype` | str | `"auto"` | Data type for weights |
| `quantization` | str | `None` | Quantization method (awq, gptq, etc.) |
| `api_key` | str | `None` | API key for authentication |
| `enable_lora` | bool | `False` | Enable LoRA adapter support |

## Supported Models

vLLM supports a wide range of models including:

- Llama (2, 3, 3.1, 3.2, 4)
- Qwen (1.5, 2, 2.5)
- Mistral / Mixtral
- Phi (2, 3, 4)
- Gemma / Gemma 2

See [vLLM documentation](https://docs.vllm.ai/en/stable/models/supported_models.html) for the full list.

## Contributing

This project follows the [anthropics/skills](https://github.com/anthropics/skills) template. When adding new skills:

1. Create a new directory under `skills/` (e.g., `skills/your-skill/`)
2. Add a `SKILL.md` file with YAML frontmatter:
   ```yaml
   ---
   name: your-skill
   description: Brief description of what this skill does
   ---
   ```
3. Add optional `scripts/`, `references/`, and `assets/` directories
4. Update this README with your skill documentation

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [anthropics/skills Template](https://github.com/anthropics/skills)
