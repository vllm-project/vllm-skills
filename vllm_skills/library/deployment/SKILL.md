# vLLM Deployment Assistant

## Metadata

- **Name**: vLLM Deployment Assistant
- **Version**: 1.0.0
- **Description**: Comprehensive deployment assistant that helps AI agents guide users through vLLM deployment, including environment detection, configuration optimization, recipe integration, and troubleshooting.
- **Tags**: deployment, vllm, gpu, configuration, troubleshooting, optimization
- **Requirements**:
  - Python 3.8+
  - vLLM (0.6.x or later)
  - PyTorch 2.0+
  - CUDA 11.8+ or ROCm 5.7+ (for GPU acceleration)

## Overview

This skill enables AI agents to:

1. **Auto-detect system environment** - Detect hardware (GPU/CPU/RAM), software dependencies (Python/CUDA/PyTorch/vLLM versions), and check compatibility
2. **Configure deployment parameters** - Interactively gather user requirements and suggest optimal configurations
3. **Integrate with recipes** - Map models to deployment guides from [vllm-project/recipes](https://github.com/vllm-project/recipes)
4. **Provide troubleshooting guidance** - Help users resolve common deployment issues
5. **Index resources** - Direct users to official documentation and community resources

## Environment Detection Checklist

When helping a user deploy vLLM, perform these checks:

### Hardware Checks

- [ ] **GPU Detection**
  - Detect GPU model(s) using `nvidia-smi` or `rocm-smi`
  - Check total VRAM per GPU
  - Verify GPU compute capability (NVIDIA: 7.0+, AMD: gfx906+)
  - Count available GPUs for tensor parallelism

- [ ] **CPU/RAM Detection**
  - Check CPU core count
  - Verify total system RAM
  - Ensure minimum 16GB RAM for small models

### Software Version Checks

- [ ] **Python Version**
  - Verify Python 3.8, 3.9, 3.10, 3.11, or 3.12
  - Python 3.13 not yet supported

- [ ] **CUDA/ROCm Version**
  - NVIDIA: CUDA 11.8, 12.1, 12.2, 12.3, or 12.4
  - AMD: ROCm 5.7, 6.0, or 6.1

- [ ] **PyTorch Version**
  - Check PyTorch >= 2.0.0
  - Verify CUDA/ROCm compatibility

- [ ] **vLLM Version**
  - Check installed vLLM version
  - Recommend latest stable (0.8.x+)

- [ ] **Flash Attention**
  - Check if Flash Attention 2 is available
  - Verify compatibility with GPU architecture

- [ ] **Triton**
  - Check Triton compiler availability
  - Important for custom kernels

### Compatibility Matrix

| vLLM Version | PyTorch | CUDA | Python | Flash Attention |
|--------------|---------|------|--------|-----------------|
| 0.6.x | 2.0-2.4 | 11.8-12.1 | 3.8-3.11 | 2.3+ |
| 0.7.x | 2.0-2.5 | 11.8-12.4 | 3.8-3.12 | 2.4+ |
| 0.8.x+ | 2.1-2.6 | 11.8-12.6 | 3.8-3.12 | 2.5+ |

## Interactive Configuration Parameters

When configuring a deployment, gather these parameters from the user:

### Required Parameters

1. **model_name** (string)
   - Example: `"meta-llama/Llama-3.1-8B-Instruct"`
   - The HuggingFace model identifier or local path

2. **tensor_parallel_size** (integer)
   - Default: 1
   - Number of GPUs to use for tensor parallelism
   - Should not exceed available GPUs
   - Required for models that don't fit on single GPU

3. **max_model_len** (integer)
   - Default: model's max context length
   - Maximum sequence length
   - Reduce if OOM occurs
   - Examples: 2048, 4096, 8192, 16384, 32768, 131072

4. **gpu_memory_utilization** (float)
   - Default: 0.90
   - Fraction of GPU memory to use (0.0-1.0)
   - Reduce to 0.80-0.85 if OOM occurs
   - Leave headroom for CUDA kernels

5. **max_num_seqs** (integer)
   - Default: 256
   - Maximum number of sequences processed in batch
   - Higher = more throughput but more memory
   - Lower = less memory but lower throughput

6. **quantization** (string or null)
   - Options: `null`, `"awq"`, `"gptq"`, `"squeezellm"`, `"fp8"`, `"bitsandbytes"`
   - Use quantization to reduce memory usage
   - Requires compatible model checkpoint

### Advanced Parameters

7. **dtype** (string)
   - Default: `"auto"` (uses model's default)
   - Options: `"auto"`, `"float16"`, `"bfloat16"`, `"float32"`
   - Use `"bfloat16"` on Ampere+ GPUs for better numerical stability

8. **enforce_eager** (boolean)
   - Default: `false`
   - Set to `true` to disable CUDA graphs
   - Use for debugging or if CUDA graph issues occur

9. **enable_chunked_prefill** (boolean)
   - Default: `false`
   - Improves latency for long prompts
   - May reduce throughput

10. **enable_prefix_caching** (boolean)
    - Default: `false`
    - Caches prompt prefixes for reuse
    - Useful for similar prompts (e.g., system prompts)

11. **enable_expert_parallel** (boolean)
    - Default: `false`
    - For Mixture-of-Experts (MoE) models
    - Distributes experts across GPUs

12. **kv_cache_dtype** (string)
    - Default: `"auto"`
    - Options: `"auto"`, `"fp8"`, `"fp8_e5m2"`
    - FP8 KV cache reduces memory usage

## Recipe Integration

Map common models to their deployment recipes from [vllm-project/recipes](https://github.com/vllm-project/recipes):

### Model-to-Recipe Mapping

| Model Family | Model Name | Recipe Link |
|--------------|------------|-------------|
| DeepSeek | DeepSeek-R1 | [recipes/deepseek-r1](https://github.com/vllm-project/recipes/tree/main/deepseek-r1) |
| DeepSeek | DeepSeek-V3 | [recipes/deepseek-v3](https://github.com/vllm-project/recipes/tree/main/deepseek-v3) |
| Qwen | Qwen3 | [recipes/qwen3](https://github.com/vllm-project/recipes/tree/main/qwen3) |
| Qwen | Qwen2.5-VL | [recipes/qwen2.5-vl](https://github.com/vllm-project/recipes/tree/main/qwen2.5-vl) |
| Llama | Llama-3.1 | [recipes/llama-3.1](https://github.com/vllm-project/recipes/tree/main/llama-3.1) |
| Llama | Llama-3.3-70B | [recipes/llama-3.3-70b](https://github.com/vllm-project/recipes/tree/main/llama-3.3-70b) |
| Mistral | Mistral-Large-3 | [recipes/mistral-large-3](https://github.com/vllm-project/recipes/tree/main/mistral-large-3) |
| GLM | GLM-4 | [recipes/glm-4](https://github.com/vllm-project/recipes/tree/main/glm-4) |

**Note**: Check the [recipes repository](https://github.com/vllm-project/recipes) for the complete list of supported models and their deployment guides.

## Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **CUDA Out of Memory** | `RuntimeError: CUDA out of memory` | Reduce `gpu_memory_utilization` to 0.80-0.85, reduce `max_model_len`, reduce `max_num_seqs`, or use quantization |
| **Flash Attention Not Available** | Warning about Flash Attention | Install `flash-attn` package: `pip install flash-attn` |
| **Model Architecture Not Supported** | `ValueError: Model architecture X not supported` | Check [supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) or request support on GitHub |
| **Tensor Parallel Size Mismatch** | Model doesn't load with TP | Some model architectures require specific TP sizes (e.g., MoE models) |
| **CUDA Version Mismatch** | CUDA errors on startup | Reinstall vLLM with correct CUDA version: `pip install vllm-cu118` or `vllm-cu121` |
| **Kernel Compilation Failed** | Triton kernel errors | Set `VLLM_INSTALL_PUNICA_KERNELS=0` or reinstall from source |
| **Expert Parallel Issues** | MoE model errors | Enable `--enable-expert-parallel` for DeepSeek-V3, Mixtral, etc. |

## Troubleshooting Flowcharts

### Memory Issues Flowchart

```
CUDA OOM Error?
├─ Yes → Try in order:
│   1. Reduce gpu_memory_utilization (0.90 → 0.85 → 0.80)
│   2. Reduce max_model_len (e.g., 32768 → 16384 → 8192)
│   3. Reduce max_num_seqs (256 → 128 → 64)
│   4. Enable quantization (--quantization awq/gptq/fp8)
│   5. Use FP8 KV cache (--kv-cache-dtype fp8)
│   6. Increase tensor_parallel_size (distribute across more GPUs)
│   7. Consider smaller model or more GPUs
└─ No → Continue with deployment
```

### CUDA Issues Flowchart

```
CUDA Error?
├─ Driver Version → Check nvidia-smi, upgrade driver if needed (525.60.13+)
├─ CUDA Version → Reinstall vLLM with matching CUDA:
│   pip install vllm-cu118  # for CUDA 11.8
│   pip install vllm-cu121  # for CUDA 12.1
├─ Kernel Compilation → Set VLLM_INSTALL_PUNICA_KERNELS=0
└─ Flash Attention → pip install flash-attn --no-build-isolation
```

### MoE Model Issues Flowchart

```
MoE Model (DeepSeek-V3, Mixtral, etc.)?
├─ Enable expert parallel: --enable-expert-parallel
├─ FP8 quantization recommended: --quantization fp8
├─ Check TP size compatibility (often requires TP=2,4,8)
└─ Ensure sufficient VRAM (400B+ models need 400GB+)
```

## Resource Links

### Official vLLM Documentation

- **Installation**: https://docs.vllm.ai/en/latest/getting_started/installation.html
- **Supported Models**: https://docs.vllm.ai/en/latest/models/supported_models.html
- **Quantization**: https://docs.vllm.ai/en/latest/quantization/supported_hardware.html
- **Distributed Inference**: https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- **Troubleshooting**: https://docs.vllm.ai/en/latest/getting_started/debugging.html
- **Performance Tuning**: https://docs.vllm.ai/en/latest/performance/performance.html

### vLLM Recipes

- **Main Repository**: https://github.com/vllm-project/recipes
- **How to Use**: Browse model-specific deployment guides with optimal configurations

### GitHub Resources

- **Main Repository**: https://github.com/vllm-project/vllm
- **Issues**: https://github.com/vllm-project/vllm/issues
- **Discussions**: https://github.com/vllm-project/vllm/discussions
- **Releases**: https://github.com/vllm-project/vllm/releases

### Community

- **Discord**: https://discord.gg/vllm
- **Blog**: https://blog.vllm.ai/

## Usage Examples

### Example 1: Full Deployment Flow

**User**: "I want to deploy Llama-3.1-8B-Instruct on my 2x A100 80GB GPUs"

**Assistant using this skill**:

1. **Environment Check**:
   ```
   Detected: 2x NVIDIA A100-SXM4-80GB (160GB total VRAM)
   Python: 3.10.12
   CUDA: 12.1
   PyTorch: 2.1.0+cu121
   vLLM: 0.8.2
   Flash Attention: 2.5.0 ✓
   ```

2. **Configuration Suggestion**:
   ```bash
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.90 \
     --max-model-len 131072 \
     --max-num-seqs 256 \
     --dtype bfloat16 \
     --enable-prefix-caching
   ```

3. **Explanation**:
   - Single GPU is sufficient (8B model ~16GB)
   - Full context length (131K tokens)
   - Prefix caching for repeated system prompts
   - BFloat16 for A100 numerical stability

### Example 2: Troubleshooting OOM

**User**: "I'm getting CUDA OOM with DeepSeek-V3 on 8x H100 GPUs"

**Assistant using this skill**:

1. **Diagnosis**: DeepSeek-V3 is a 671B MoE model requiring significant VRAM
   
2. **Solution**:
   ```bash
   vllm serve deepseek-ai/DeepSeek-V3 \
     --tensor-parallel-size 8 \
     --gpu-memory-utilization 0.85 \
     --max-model-len 16384 \
     --quantization fp8 \
     --enable-expert-parallel \
     --kv-cache-dtype fp8
   ```

3. **Key Changes**:
   - Reduced memory utilization: 0.90 → 0.85
   - Reduced context: 32768 → 16384
   - FP8 quantization for 50% memory reduction
   - Expert parallel enabled
   - FP8 KV cache

4. **Recipe Reference**: See [DeepSeek-V3 recipe](https://github.com/vllm-project/recipes/tree/main/deepseek-v3)

### Example 3: Finding Model Recipe

**User**: "How do I deploy Qwen2.5-VL-72B?"

**Assistant using this skill**:

1. **Recipe Lookup**: Found [Qwen2.5-VL recipe](https://github.com/vllm-project/recipes/tree/main/qwen2.5-vl)

2. **Hardware Check**: 72B vision model requires ~140GB VRAM (TP=2 on A100s)

3. **Recommended Command** (from recipe):
   ```bash
   vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
     --tensor-parallel-size 2 \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.85 \
     --limit-mm-per-prompt image=1
   ```

## Agent Behavior Guidelines

When using this skill, AI agents should:

1. **Always detect environment first** - Run hardware and software checks before suggesting configurations
2. **Ask clarifying questions** - Model name, use case (throughput vs latency), available hardware
3. **Suggest optimal defaults** - Use recipes and hardware matrix to provide good starting points
4. **Explain trade-offs** - Help users understand memory vs throughput vs latency trade-offs
5. **Reference official sources** - Always point to vLLM docs and recipes for authoritative guidance
6. **Provide complete commands** - Give copy-paste ready commands with explanations
7. **Troubleshoot systematically** - Use flowcharts to diagnose issues step-by-step
8. **Stay current** - Remind users to check for latest vLLM releases and recipes

## Advanced Topics

### GPU Memory Requirements by Model Size

| Model Size | Minimum VRAM | Recommended Setup |
|------------|--------------|-------------------|
| 7-8B | 16GB | 1x RTX 4090 / A10G / L4 |
| 13-14B | 28GB | 1x A100 40GB / 2x RTX 4090 |
| 32-34B | 70GB | 1x A100 80GB / 2x A100 40GB |
| 70-72B | 140GB | 2x A100 80GB / 4x A100 40GB |
| 400B+ MoE | 400GB+ | 8x H100 80GB with FP8 quantization |

*Note: Requirements assume FP16/BF16 precision without quantization*

### Acceleration Libraries

- **Flash Attention 2**: Fast attention kernel, 2-4x speedup
- **Triton**: JIT compiler for custom GPU kernels
- **xFormers**: Memory-efficient attention implementations
- **vLLM Custom Kernels**: Optimized PagedAttention kernels
- **cuBLAS**: NVIDIA's optimized linear algebra library

### Configuration Presets

This skill includes preset configurations:

- **high_throughput.yaml**: Maximize batch size and throughput
- **low_latency.yaml**: Minimize time-to-first-token and latency
- **memory_constrained.yaml**: Minimize memory usage for limited VRAM

See `config/presets/` directory for details.

## Version History

- **1.0.0** (2026-02-03): Initial release
  - Environment detection
  - Recipe integration
  - Troubleshooting guides
  - Configuration presets
