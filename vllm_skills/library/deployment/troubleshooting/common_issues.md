# Common Issues and Quick Solutions

This guide covers the most common issues encountered when deploying vLLM and their solutions.

## Quick Reference Table

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| `RuntimeError: CUDA out of memory` | Insufficient VRAM | Reduce `gpu_memory_utilization` to 0.80-0.85 |
| `ModuleNotFoundError: No module named 'flash_attn'` | Flash Attention not installed | `pip install flash-attn --no-build-isolation` |
| `ValueError: Model architecture X not supported` | Unsupported model | Check [supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| `ValueError: Tensor parallel size must be divisible by...` | Incorrect TP configuration | Adjust TP to valid value (usually powers of 2) |
| Slow startup time | Normal for large models | Wait 2-5 minutes for model loading |
| `ImportError: cannot import name 'scaled_dot_product_attention'` | Old PyTorch version | Update PyTorch: `pip install --upgrade torch` |
| Model outputs are gibberish | Wrong dtype or quantization | Use `--dtype bfloat16` or check quantization |
| `RuntimeError: No available memory for the cache blocks` | KV cache allocation failed | Reduce `max_model_len` or `max_num_seqs` |

## Detailed Solutions

### 1. CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions (try in order):**

1. **Reduce GPU memory utilization:**
   ```bash
   --gpu-memory-utilization 0.85  # from default 0.90
   --gpu-memory-utilization 0.80  # if still OOM
   ```

2. **Reduce context length:**
   ```bash
   --max-model-len 16384  # from 32768 or default
   --max-model-len 8192   # if still OOM
   ```

3. **Reduce batch size:**
   ```bash
   --max-num-seqs 128  # from default 256
   --max-num-seqs 64   # if still OOM
   ```

4. **Enable quantization:**
   ```bash
   --quantization fp8           # For Hopper GPUs (50% reduction)
   --quantization awq           # For AWQ models (75% reduction)
   --kv-cache-dtype fp8         # FP8 KV cache
   ```

5. **Use tensor parallelism:**
   ```bash
   --tensor-parallel-size 2  # Distribute across 2 GPUs
   --tensor-parallel-size 4  # Distribute across 4 GPUs
   ```

6. **Consider a smaller model** or upgrade GPU

### 2. Flash Attention Issues

**Symptoms:**
```
Warning: Flash Attention is not available
ModuleNotFoundError: No module named 'flash_attn'
```

**Solutions:**

1. **Install Flash Attention 2:**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **If installation fails on older GPUs:**
   - Flash Attention requires compute capability 7.5+ (Turing or newer)
   - For older GPUs (e.g., V100), vLLM will use xFormers instead
   - Performance will be slightly lower but still functional

3. **Alternative installation:**
   ```bash
   # For specific CUDA version
   pip install flash-attn --no-build-isolation --no-cache-dir
   
   # Or build from source
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention
   python setup.py install
   ```

### 3. Model Architecture Not Supported

**Symptoms:**
```
ValueError: Model architecture 'XXX' is not supported by vLLM
NotImplementedError: Architecture XXX is not supported
```

**Solutions:**

1. **Check supported models:**
   - Visit [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
   - Verify your model architecture is listed

2. **Use a supported variant:**
   - Many model families have similar architectures
   - Example: Use `LlamaForCausalLM` instead of custom architecture

3. **Request support:**
   - Open an issue on [vLLM GitHub](https://github.com/vllm-project/vllm/issues)
   - Provide model name and HuggingFace link
   - Check if there's existing support being developed

4. **Convert to supported format:**
   - Some models can be converted to supported architectures
   - Check model documentation for conversion scripts

### 4. Tensor Parallel Configuration Issues

**Symptoms:**
```
ValueError: Tensor parallel size must be divisible by attention head count
AssertionError: TP size must be power of 2
```

**Solutions:**

1. **Use valid TP sizes:**
   - Common valid values: 1, 2, 4, 8, 16
   - Must not exceed number of GPUs available

2. **Check model-specific requirements:**
   - Some models require specific TP sizes
   - MoE models often need TP that divides expert count

3. **Example configurations:**
   ```bash
   # Llama models: any power of 2
   --tensor-parallel-size 2
   
   # DeepSeek-V3: should divide expert count
   --tensor-parallel-size 8
   
   # Mixtral-8x7B: 
   --tensor-parallel-size 2  # or 4, 8
   ```

### 5. Expert Parallel Issues (MoE Models)

**Symptoms:**
```
RuntimeError: Expert parallelism must be enabled for MoE models
ValueError: Number of experts must be divisible by EP size
```

**Solutions:**

1. **Enable expert parallelism:**
   ```bash
   --enable-expert-parallel
   ```

2. **Configure TP and EP together:**
   ```bash
   # For DeepSeek-V3 (256 experts)
   --tensor-parallel-size 8 \
   --enable-expert-parallel
   
   # For Mixtral-8x7B (8 experts)
   --tensor-parallel-size 2 \
   --enable-expert-parallel
   ```

3. **Use recommended quantization:**
   ```bash
   --quantization fp8 \
   --kv-cache-dtype fp8
   ```

### 6. CUDA Version Mismatch

**Symptoms:**
```
ImportError: libcudart.so.11.0: cannot open shared object file
RuntimeError: The detected CUDA version (X.X) mismatches the version that was used to compile PyTorch (Y.Y)
```

**Solutions:**

1. **Reinstall vLLM with correct CUDA version:**
   ```bash
   # For CUDA 11.8
   pip uninstall vllm
   pip install vllm-cu118
   
   # For CUDA 12.1
   pip uninstall vllm
   pip install vllm-cu121
   ```

2. **Check CUDA version:**
   ```bash
   nvidia-smi  # Shows CUDA driver version
   nvcc --version  # Shows CUDA toolkit version
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version
   ```

3. **Update NVIDIA driver if needed:**
   - Minimum driver version: 525.60.13 for CUDA 12.x
   - Download from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

### 7. Kernel Compilation Errors

**Symptoms:**
```
RuntimeError: Failed to compile Triton kernel
Error during kernel compilation
```

**Solutions:**

1. **Disable Punica kernels:**
   ```bash
   export VLLM_INSTALL_PUNICA_KERNELS=0
   pip install --upgrade --force-reinstall vllm
   ```

2. **Update Triton:**
   ```bash
   pip install --upgrade triton
   ```

3. **Use pre-compiled kernels:**
   ```bash
   # Set environment variable before starting vLLM
   export VLLM_USE_PRECOMPILED_KERNELS=1
   ```

### 8. Slow Inference / Poor Performance

**Symptoms:**
- Very slow token generation
- Low tokens per second
- High latency

**Solutions:**

1. **Verify Flash Attention is active:**
   - Check logs for "Using Flash Attention" message
   - Install if missing: `pip install flash-attn`

2. **Enable CUDA graphs (if disabled):**
   ```bash
   # Remove --enforce-eager if present
   # CUDA graphs significantly improve performance
   ```

3. **Optimize configuration:**
   ```bash
   # For throughput
   --gpu-memory-utilization 0.95 \
   --max-num-seqs 512 \
   --enable-prefix-caching
   
   # For latency
   --max-num-seqs 64 \
   --enable-chunked-prefill
   ```

4. **Use optimal dtype:**
   ```bash
   --dtype bfloat16  # For Ampere+ GPUs (A100, H100, RTX 40-series)
   --dtype float16   # For older GPUs
   ```

5. **Check GPU utilization:**
   ```bash
   nvidia-smi dmon  # Monitor GPU usage in real-time
   # Should see high GPU utilization (>80%)
   ```

### 9. Model Download Issues

**Symptoms:**
```
OSError: Can't load tokenizer
HfHubHTTPError: 401 Client Error
```

**Solutions:**

1. **Login to HuggingFace:**
   ```bash
   huggingface-cli login
   # Or set token
   export HF_TOKEN=your_token_here
   ```

2. **Accept model license:**
   - Visit model page on HuggingFace
   - Click "Agree and access repository"
   - Required for Llama, Mistral, and other gated models

3. **Use local model path:**
   ```bash
   # Download first, then serve from disk
   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
   vllm serve /path/to/downloaded/model
   ```

### 10. Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'vllm'
ImportError: cannot import name 'LLM'
```

**Solutions:**

1. **Install vLLM:**
   ```bash
   pip install vllm
   ```

2. **Update to latest version:**
   ```bash
   pip install --upgrade vllm
   ```

3. **Install with specific extras:**
   ```bash
   # For specific features
   pip install vllm[audio]  # Audio models
   pip install vllm[vision]  # Vision models
   ```

## Getting Help

If issues persist:

1. **Check vLLM documentation:** https://docs.vllm.ai/
2. **Search GitHub issues:** https://github.com/vllm-project/vllm/issues
3. **Ask on Discord:** https://discord.gg/vllm
4. **Create GitHub issue** with:
   - vLLM version (`pip show vllm`)
   - PyTorch version
   - CUDA version
   - GPU model
   - Full error traceback
   - Minimal reproduction command
