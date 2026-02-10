---
name: vllm-deploy
description: Deploy vLLM as an online service with OpenAI-compatible API locally or via Docker
---

# vLLM Deploy

Deploy vLLM as an online service with OpenAI-compatible API. This skill helps you quickly spin up vLLM servers locally or using Docker containers.

## What this skill does

This skill provides utilities to deploy vLLM servers with:
- **Local deployment** - Start vLLM directly on your machine
- **Docker deployment** - Run vLLM in containers with GPU support
- **OpenAI-compatible API** - Use as a drop-in replacement for OpenAI API
- **Flexible configuration** - Support for various models, quantization, LoRA adapters

## Quick Start

### Local Deployment

```python
import sys
sys.path.insert(0, "scripts")

from vllm_deploy import VLLMConfig, VLLMDeployer

config = VLLMConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    port=8000,
)

deployer = VLLMDeployer(vllm_config=config)
result = deployer.deploy_local(background=True)

print(f"API running at: {result.endpoint_url}")
```

### Docker Deployment

```python
import sys
sys.path.insert(0, "scripts")

from vllm_deploy import VLLMConfig, VLLMDeployer

config = VLLMConfig(model="Qwen/Qwen2.5-1.5B-Instruct")
deployer = VLLMDeployer(vllm_config=config)

result = deployer.deploy_docker(background=True)
print(f"Container running: {result.endpoint_url}")
```

### Query with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

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

## Prerequisites

For local deployment:
- Python 3.10+
- GPU with CUDA support
- vLLM installed: `pip install vllm`

For Docker deployment:
- Docker installed
- NVIDIA Container Toolkit
- At least one GPU available

## Supported Models

vLLM supports a wide range of models including:
- Llama (2, 3, 3.1, 3.2, 4)
- Qwen (1.5, 2, 2.5)
- Mistral / Mixtral
- Phi (2, 3, 4)
- Gemma / Gemma 2

See [vLLM documentation](https://docs.vllm.ai/en/stable/models/supported_models.html) for the full list.
