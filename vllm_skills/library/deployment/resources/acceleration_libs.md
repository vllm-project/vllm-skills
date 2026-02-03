# Acceleration Libraries for vLLM

## Flash Attention

**What it is**: Optimized attention mechanism that reduces memory usage and increases speed

**Installation**:
```bash
pip install flash-attn --no-build-isolation
```

**Requirements**:
- NVIDIA GPUs: Ampere (A100, RTX 30xx) or newer
- CUDA 11.8+
- Works automatically in vLLM if installed

**Benefits**:
- 2-4x faster attention computation
- Reduced memory usage
- Enables longer context lengths

**Compatibility**:
- ✅ A100, H100, RTX 4090, RTX 4080
- ✅ RTX 3090, RTX 3080
- ❌ V100, T4 (use xFormers instead)

## xFormers

**What it is**: Memory-efficient attention implementation from Meta

**Installation**:
```bash
pip install xformers
```

**Requirements**:
- Broader GPU support than Flash Attention
- CUDA 11.3+

**Benefits**:
- Works on older GPUs (V100, T4)
- Good performance and memory efficiency
- Fallback for Flash Attention

**Use when**:
- GPU doesn't support Flash Attention
- Need broader compatibility

## Triton

**What it is**: GPU programming framework for custom kernels

**Installation**:
```bash
pip install triton
```

**Benefits**:
- Used for custom vLLM kernels
- Automatic tuning for different GPUs
- Good performance on modern GPUs

**Environment variables**:
```bash
# Enable Triton Flash Attention
export VLLM_USE_TRITON_FLASH_ATTN=1

# Disable if causing issues
export VLLM_USE_TRITON_FLASH_ATTN=0
```

## CUTLASS

**What it is**: NVIDIA's CUDA Templates for Linear Algebra Subroutines

**Installation**:
```bash
pip install nvidia-cutlass
```

**Benefits**:
- Optimized GEMM operations
- Used in grouped matrix multiplications
- Better performance on newer GPUs

**Use cases**:
- MoE (Mixture of Experts) models
- Quantized models
- Multi-GPU deployments

## TensorRT-LLM

**What it is**: NVIDIA's optimized LLM inference engine

**Integration**:
- vLLM can use TensorRT-LLM backends
- Best performance on NVIDIA GPUs
- Requires separate installation

**Installation**:
```bash
# Requires NVIDIA NGC account
# See: https://github.com/NVIDIA/TensorRT-LLM
```

**Benefits**:
- Maximum performance on H100
- FP8 quantization support
- Advanced kernel optimizations

## cuBLAS

**What it is**: NVIDIA CUDA Basic Linear Algebra Subprograms

**Installation**:
- Included with CUDA toolkit
- Automatically used by PyTorch

**Optimization**:
```bash
# Enable Tensor Cores
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Performance Comparison

| Library | Speed | Memory | GPU Compatibility | Ease of Use |
|---------|-------|--------|-------------------|-------------|
| Flash Attention | ⚡⚡⚡⚡ | ✅✅✅✅ | Ampere+ | ⭐⭐⭐⭐ |
| xFormers | ⚡⚡⚡ | ✅✅✅ | Most GPUs | ⭐⭐⭐⭐⭐ |
| Triton | ⚡⚡⚡⚡ | ✅✅✅ | Modern GPUs | ⭐⭐⭐ |
| CUTLASS | ⚡⚡⚡⚡ | ✅✅✅ | NVIDIA | ⭐⭐⭐ |
| TensorRT-LLM | ⚡⚡⚡⚡⚡ | ✅✅✅✅ | NVIDIA | ⭐⭐ |

## Recommended Setup

### For A100/H100 (Best Performance)
```bash
pip install flash-attn --no-build-isolation
pip install triton
pip install nvidia-cutlass
```

### For RTX 3090/4090 (Consumer GPU)
```bash
pip install flash-attn --no-build-isolation
pip install xformers
```

### For V100/T4 (Older GPUs)
```bash
pip install xformers
```

### For Production (Maximum Performance)
```bash
pip install flash-attn --no-build-isolation
pip install triton
pip install nvidia-cutlass
# Consider TensorRT-LLM for H100
```

## Troubleshooting

### Flash Attention not working
- Check GPU compatibility (needs Ampere+)
- Verify CUDA version (11.8+)
- Try xFormers as fallback

### Triton compilation errors
```bash
# Clear cache
rm -rf ~/.triton/cache
# Disable if needed
export VLLM_USE_TRITON_FLASH_ATTN=0
```

### Import errors
```bash
# Reinstall with no build isolation
pip uninstall flash-attn xformers triton
pip install flash-attn --no-build-isolation
pip install xformers --no-build-isolation
pip install triton
```

## See Also
- [Official vLLM Docs](https://docs.vllm.ai/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [xFormers GitHub](https://github.com/facebookresearch/xformers)
