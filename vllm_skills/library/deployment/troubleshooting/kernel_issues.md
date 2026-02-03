# Kernel Issues in vLLM

## Flash Attention Kernel Issues

### Issue: Flash Attention not available
**Symptoms**: Warning about Flash Attention not being used

**Solutions**:

1. **Install Flash Attention**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Verify installation**:
   ```python
   import flash_attn
   print(flash_attn.__version__)
   ```

3. **Check GPU compatibility**:
   - Flash Attention requires Ampere (A100, RTX 30xx) or newer
   - Won't work on older GPUs (V100, T4)

### Issue: Flash Attention compilation fails
**Solutions**:

1. **Install with pre-built wheels**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Check CUDA version**:
   ```bash
   nvcc --version  # Should be 11.8+
   ```

3. **Use xFormers as fallback**:
   ```bash
   pip install xformers
   ```

## Custom Kernels Issues

### Issue: Custom kernel compilation fails
**Symptoms**: vLLM cannot compile custom CUDA kernels

**Solutions**:

1. **Ensure CUDA toolkit is installed**:
   ```bash
   nvcc --version
   ```

2. **Install build tools**:
   ```bash
   sudo apt install build-essential
   ```

3. **Set CUDA_HOME**:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   ```

4. **Reinstall vLLM**:
   ```bash
   pip uninstall vllm
   pip install vllm --no-build-isolation
   ```

## PagedAttention Kernel Issues

### Issue: PagedAttention kernel errors
**Symptoms**: Errors during attention computation

**Solutions**:

1. **Update vLLM to latest version**:
   ```bash
   pip install --upgrade vllm
   ```

2. **Check tensor shapes**:
   - Verify input tensors are valid
   - Check max_model_len is reasonable

3. **Disable custom kernels** (fallback):
   ```bash
   # Use PyTorch's native attention
   export VLLM_USE_TRITON_FLASH_ATTN=0
   ```

## Quantization Kernel Issues

### Issue: AWQ/GPTQ kernel errors
**Symptoms**: Errors when loading quantized models

**Solutions**:

1. **Install AutoAWQ**:
   ```bash
   pip install autoawq
   ```

2. **Install AutoGPTQ**:
   ```bash
   pip install auto-gptq
   ```

3. **Verify quantization format**:
   ```python
   # Check if model is properly quantized
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained("model-name")
   print(config.quantization_config)
   ```

4. **Use different quantization method**:
   ```bash
   # If AWQ fails, try GPTQ
   vllm serve model-name --quantization gptq
   ```

## MoE (Mixture of Experts) Kernel Issues

### Issue: MoE routing errors
**Symptoms**: Errors during expert routing in MoE models

**Solutions**:

1. **Update vLLM**:
   ```bash
   pip install --upgrade vllm
   ```

2. **Adjust tensor parallel size**:
   ```bash
   # MoE models may need specific TP sizes
   vllm serve model-name --tensor-parallel-size 2
   ```

3. **Check model compatibility**:
   - Verify vLLM supports this MoE architecture
   - Check GitHub issues for known problems

## Triton Kernel Issues

### Issue: Triton compilation failures
**Symptoms**: Errors related to Triton kernel compilation

**Solutions**:

1. **Update Triton**:
   ```bash
   pip install --upgrade triton
   ```

2. **Clear Triton cache**:
   ```bash
   rm -rf ~/.triton/cache
   ```

3. **Disable Triton** (use CUDA kernels):
   ```bash
   export VLLM_USE_TRITON_FLASH_ATTN=0
   vllm serve model-name
   ```

## CUTLASS Kernel Issues

### Issue: CUTLASS errors in grouped GEMM
**Symptoms**: Errors during matrix multiplications

**Solutions**:

1. **Update NVIDIA CUTLASS**:
   ```bash
   pip install nvidia-cutlass
   ```

2. **Check GPU compatibility**:
   - CUTLASS optimizations require modern GPUs
   - May not work on older architectures

3. **Disable custom GEMM**:
   ```bash
   export VLLM_USE_CUTLASS_GEMM=0
   ```

## Kernel Performance Optimization

### Enabling Kernel Fusion

```bash
# Enable kernel fusion for better performance
export VLLM_ENABLE_KERNEL_FUSION=1
vllm serve model-name
```

### Choosing Attention Backend

```bash
# Use Flash Attention 2 (fastest on A100/H100)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Use xFormers (good compatibility)
export VLLM_ATTENTION_BACKEND=XFORMERS

# Use PyTorch native (fallback)
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
```

### Debugging Kernel Issues

1. **Enable verbose logging**:
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   vllm serve model-name
   ```

2. **Check kernel selection**:
   ```bash
   # vLLM will log which kernels are selected
   # Look for lines like "Using Flash Attention" or "Using xFormers"
   ```

3. **Profile kernel performance**:
   ```bash
   # Use NVIDIA Nsight for profiling
   nsys profile vllm serve model-name
   ```

## See Also
- [Common Issues](common_issues.md)
- [CUDA Errors](cuda_errors.md)
- [Memory Issues](memory_issues.md)
