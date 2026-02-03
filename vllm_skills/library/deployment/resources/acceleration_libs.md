# Acceleration Libraries for vLLM

Understanding and optimizing acceleration libraries is crucial for peak vLLM performance. This guide covers the key libraries used by vLLM for GPU acceleration.

## Overview

vLLM leverages multiple acceleration libraries:

1. **Flash Attention 2** - Fast attention computation
2. **Triton** - JIT compiler for custom GPU kernels
3. **xFormers** - Memory-efficient attention (fallback)
4. **cuBLAS/cuBLASLt** - Optimized matrix operations
5. **Custom vLLM Kernels** - PagedAttention and specialized operations

---

## Flash Attention 2

### What It Is
- Fast and memory-efficient attention mechanism
- 2-4x speedup over standard attention
- Reduces memory usage during attention computation
- Developed by Tri Dao et al. at Princeton/Stanford

### Why It Matters
- **Primary bottleneck** in LLM inference is attention computation
- Flash Attention 2 significantly reduces this bottleneck
- Essential for competitive performance

### Installation

```bash
# Install Flash Attention 2
pip install flash-attn --no-build-isolation

# Verify installation
python -c "from flash_attn import flash_attn_func; print('Flash Attention available')"
```

### Requirements

| Requirement | Details |
|-------------|---------|
| GPU | NVIDIA with compute capability ≥ 7.5 (Turing or newer) |
| CUDA | 11.4+ |
| PyTorch | 2.0+ |
| Memory | Sufficient for JIT compilation (~1-2GB during install) |

### Supported GPUs

| GPU Series | Compute Cap | Support Status |
|------------|-------------|----------------|
| V100 | 7.0 | Limited (use xFormers instead) |
| T4 | 7.5 | ✅ Full support |
| RTX 20 series | 7.5 | ✅ Full support |
| A100 | 8.0 | ✅ Full support + features |
| RTX 30 series | 8.6 | ✅ Full support |
| H100 | 9.0 | ✅ Full support + FP8 |
| RTX 40 series | 8.9 | ✅ Full support + features |

### Troubleshooting

**Issue: Build fails during installation**
```bash
# Try building with explicit arch
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation

# Or specify CUDA architecture
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
pip install flash-attn --no-build-isolation
```

**Issue: Import error after installation**
```bash
# Check PyTorch CUDA version matches
python -c "import torch; print(torch.version.cuda)"

# Reinstall with matching CUDA
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

**Issue: "Flash Attention not supported on this GPU"**
- GPU compute capability < 7.5
- vLLM will automatically fall back to xFormers
- Performance will be slightly lower but functional

### Performance Impact

| Scenario | vs Standard Attention | vs xFormers |
|----------|----------------------|-------------|
| Throughput (large batch) | 2-4x faster | 1.5-2x faster |
| Memory usage | 50-60% reduction | 20-30% better |
| Latency (small batch) | 1.5-2x faster | 1.2-1.5x faster |

### Configuration

```bash
# Flash Attention is used by default if available
# To disable (for debugging):
export VLLM_USE_TRITON_FLASH_ATTN=0

# Check if it's being used (in vLLM logs):
# "Using Flash Attention backend"
```

---

## Triton

### What It Is
- Python-based JIT compiler for GPU kernels
- Developed by OpenAI
- Enables custom GPU kernels without raw CUDA
- Used by vLLM for PagedAttention and other operations

### Why It Matters
- Allows vLLM to write optimized GPU code efficiently
- Custom kernels for PagedAttention (vLLM's core innovation)
- Easier to maintain and optimize than raw CUDA

### Installation

```bash
# Usually installed automatically with vLLM
pip install triton

# Verify
python -c "import triton; print(f'Triton {triton.__version__}')"
```

### Requirements

| Requirement | Details |
|-------------|---------|
| GPU | NVIDIA with compute capability ≥ 7.0 |
| CUDA | 11.4+ |
| LLVM | 14+ (for compilation) |
| Disk space | ~500MB for kernel cache |

### Kernel Caching

Triton compiles kernels on first use and caches them:

```bash
# Default cache location
~/.triton/cache/

# Set custom cache location
export TRITON_CACHE_DIR=/tmp/triton_cache

# Clear cache if experiencing issues
rm -rf ~/.triton/cache/
```

### Troubleshooting

**Issue: Kernel compilation errors**
```bash
# Update Triton
pip install --upgrade triton

# Clear cache
rm -rf ~/.triton/cache/

# Disable Triton Flash Attention (use native Flash Attn)
export VLLM_USE_TRITON_FLASH_ATTN=0
```

**Issue: Slow first inference**
- Normal: Triton compiles kernels on first use (30-60 seconds)
- Subsequent runs use cached kernels
- Pre-warm in production:
  ```python
  # Send a dummy request after server start
  # to trigger kernel compilation
  ```

### Performance Impact

| Kernel Type | Benefit |
|-------------|---------|
| PagedAttention | Core vLLM functionality |
| Fused operations | 10-20% speedup vs separate ops |
| Custom MoE kernels | Essential for MoE models |

---

## xFormers

### What It Is
- Memory-efficient attention implementations by Meta
- Fallback when Flash Attention unavailable
- Multiple attention variants (memory_efficient, cutlass)

### Why It Matters
- Compatibility layer for older GPUs
- Automatic fallback from Flash Attention
- Still provides memory benefits

### Installation

```bash
# Usually installed with vLLM
pip install xformers

# Or build from source for latest
pip install git+https://github.com/facebookresearch/xformers.git
```

### When It's Used

vLLM uses xFormers when:
- Flash Attention not available
- GPU compute capability < 7.5
- Flash Attention build failed

### Configuration

```bash
# xFormers is used automatically as fallback
# Check logs for: "Using xFormers backend"

# Force xFormers (disable Flash Attention):
export VLLM_USE_TRITON_FLASH_ATTN=0
# Don't install flash-attn
```

### Performance vs Flash Attention

| Metric | Flash Attention 2 | xFormers |
|--------|------------------|----------|
| Throughput | Baseline (best) | 70-80% of FA2 |
| Memory | Baseline (best) | 80-90% of FA2 |
| GPU Support | 7.5+ | 7.0+ |

---

## cuBLAS and cuBLASLt

### What They Are
- NVIDIA's optimized BLAS (Basic Linear Algebra Subprograms) libraries
- Handle matrix multiplications (GEMM operations)
- Core of transformer forward pass

### Why They Matter
- Matrix multiplication is ~80% of inference compute
- Highly optimized by NVIDIA for each GPU generation
- Automatic in PyTorch/vLLM

### Optimization Tips

```bash
# Enable TF32 on Ampere+ GPUs (A100, RTX 30/40, H100)
# Provides ~3x speedup for FP32 operations with minimal quality impact
# Enabled by default in vLLM

# Verify in PyTorch:
import torch
print(torch.backends.cuda.matmul.allow_tf32)  # Should be True
```

### FP8 GEMM (H100 Only)

```bash
# Use FP8 quantization for maximum performance on H100
vllm serve model_name \
  --quantization fp8

# Leverages FP8 Tensor Cores
# ~2x speedup + 50% memory reduction
```

---

## Custom vLLM Kernels

### PagedAttention

**What**: vLLM's core innovation for KV cache management

**How it works**:
- Stores KV cache in non-contiguous memory blocks
- Enables dynamic memory allocation
- Eliminates memory fragmentation

**Performance impact**:
- 2-4x higher throughput vs HuggingFace Transformers
- Near-zero memory waste

### Sampling Kernels

Optimized kernels for:
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- Beam search

### Quantization Kernels

- FP8 quantization/dequantization
- AWQ/GPTQ weight unpacking
- INT4/INT8 matrix multiplication

---

## ROCm Support (AMD GPUs)

vLLM supports AMD GPUs through ROCm:

### Installation

```bash
# Install vLLM for ROCm
pip install vllm  # Auto-detects ROCm

# Or explicitly:
pip install vllm-rocm
```

### Supported GPUs

- MI250X, MI300 series
- RX 7000 series (limited support)

### Limitations

- Flash Attention support varies
- Some kernels may be slower than CUDA equivalent
- Active development area

---

## Performance Comparison

### End-to-End Impact

Configuration comparison on Llama-3.1-8B (A100 80GB):

| Configuration | Throughput (tok/s) | Latency (TTFT) |
|--------------|-------------------|----------------|
| Flash Attention + Triton | 15,000 (baseline) | 25ms (baseline) |
| xFormers + Triton | 11,000 (73%) | 35ms (+40%) |
| No acceleration | 4,000 (27%) | 80ms (+220%) |

### Recommendations by GPU

| GPU | Primary | Fallback | Notes |
|-----|---------|----------|-------|
| H100 | Flash Attn 2 + FP8 | Flash Attn 2 | Use FP8 for max perf |
| A100 | Flash Attn 2 | xFormers | TF32 enabled |
| RTX 4090 | Flash Attn 2 | xFormers | Excellent perf |
| RTX 3090 | Flash Attn 2 | xFormers | Good perf |
| A10G | Flash Attn 2 | xFormers | Cost-effective |
| T4 | Flash Attn 2 | xFormers | Entry-level |
| V100 | xFormers | Standard | Limited Flash Attn |

---

## Debugging Acceleration Issues

### Check What's Being Used

```python
# Check vLLM logs on startup
# Look for messages like:
# "Using Flash Attention backend"
# "Using xFormers backend"
# "Using Triton kernels"

# Or programmatically:
import vllm
# Check log output when initializing LLM
```

### Performance Profiling

```bash
# Use NVIDIA Nsight Systems
nsys profile vllm serve model_name

# Check kernel usage:
# - Flash Attention kernels
# - cuBLAS GEMM
# - Triton kernels
# - Memory transfers
```

### Common Issues

**Slow performance despite Flash Attention installed**
- Verify it's actually being used (check logs)
- Ensure CUDA version compatibility
- Check for fallback to xFormers

**High memory usage**
- Verify Flash Attention is active
- Check kernel compilation is complete
- Monitor with `nvidia-smi`

---

## Keeping Libraries Updated

```bash
# Update all acceleration libraries
pip install --upgrade flash-attn triton xformers torch

# Or update vLLM (pulls compatible versions)
pip install --upgrade vllm

# After update, clear Triton cache
rm -rf ~/.triton/cache/
```

---

## Resources

- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **Triton**: https://github.com/openai/triton
- **xFormers**: https://github.com/facebookresearch/xformers
- **vLLM Kernels**: https://github.com/vllm-project/vllm (kernels/ directory)
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **ROCm**: https://www.amd.com/en/products/software/rocm.html
