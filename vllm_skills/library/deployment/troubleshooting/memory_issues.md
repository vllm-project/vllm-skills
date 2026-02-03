# Memory Issues and OOM Solutions

Comprehensive guide for diagnosing and resolving memory-related issues in vLLM deployments.

## Understanding vLLM Memory Usage

vLLM memory consists of three main components:

1. **Model Weights** (~2 bytes/param for FP16, ~1 byte for FP8)
2. **KV Cache** (largest contributor during inference)
3. **Activation Memory** (temporary during forward pass)

### Memory Formula

```
Total Memory ≈ Model Weights + KV Cache + Activation Memory

Model Weights = Parameters × Bytes per parameter
KV Cache = Layers × 2 × Hidden size × Num heads × Max seqs × Seq length × Bytes per element
Activation Memory = Batch size dependent
```

---

## Common OOM Scenarios

### Scenario 1: OOM During Model Loading

**Symptoms:**
```
RuntimeError: CUDA out of memory during model initialization
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause:** Model weights don't fit in VRAM

**Solutions:**

#### Use Tensor Parallelism
```bash
# Distribute model across multiple GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2  # Split across 2 GPUs
```

#### Use Quantization
```bash
# Load quantized checkpoint (requires AWQ/GPTQ model)
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq

# Or use FP8 on H100
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8
```

#### Reduce Model Size
```bash
# Use a smaller variant
vllm serve meta-llama/Llama-3.1-8B-Instruct  # Instead of 70B
```

---

### Scenario 2: OOM During KV Cache Allocation

**Symptoms:**
```
RuntimeError: No available memory for the cache blocks
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`
```

**Cause:** Insufficient memory for KV cache after model loading

**Solutions:**

#### Reduce GPU Memory Utilization
```bash
# Leave more memory for KV cache
vllm serve model_name \
  --gpu-memory-utilization 0.85  # Default is 0.90
```

#### Reduce Maximum Sequence Length
```bash
# Decrease context window
vllm serve model_name \
  --max-model-len 8192  # From 32768 or model default
```

#### Reduce Batch Size
```bash
# Process fewer sequences concurrently
vllm serve model_name \
  --max-num-seqs 64  # From default 256
```

#### Use FP8 KV Cache
```bash
# Reduce KV cache size by ~50%
vllm serve model_name \
  --kv-cache-dtype fp8
```

**Quality Impact:** Minimal for most models, may affect very long contexts

---

### Scenario 3: OOM During Inference

**Symptoms:**
```
RuntimeError: CUDA out of memory (during generation)
OOM after running successfully for some time
```

**Cause:** Memory fragmentation or unexpectedly long sequences

**Solutions:**

#### Set Hard Limits
```bash
vllm serve model_name \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --enforce-eager  # Disable CUDA graphs to save memory
```

#### Enable Memory Monitoring
```python
# Monitor memory in real-time
import subprocess
subprocess.Popen(['watch', '-n', '1', 'nvidia-smi'])
```

#### Restart Service Periodically
```bash
# Add to deployment script for long-running services
# Restart every 24 hours to clear memory fragmentation
```

---

### Scenario 4: OOM with Large Batches

**Symptoms:**
```
OOM when processing many concurrent requests
Memory usage grows with number of requests
```

**Cause:** Batch size exceeds memory capacity

**Solutions:**

#### Limit Concurrent Sequences
```bash
vllm serve model_name \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192
```

#### Use Chunked Prefill
```bash
# Process long prompts in chunks
vllm serve model_name \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096
```

---

## Memory Optimization Strategies

### Strategy 1: Quantization Comparison

| Method | Memory Reduction | Quality Impact | Requirements |
|--------|------------------|----------------|--------------|
| FP8 (W8A8) | ~50% | Minimal | H100, A100 (limited) |
| FP8 KV Cache | ~50% (cache only) | Very minimal | Modern GPUs |
| AWQ (4-bit) | ~75% | Low | AWQ checkpoint |
| GPTQ (4-bit) | ~75% | Low | GPTQ checkpoint |
| bitsandbytes (8-bit) | ~50% | Low | Any GPU |
| bitsandbytes (4-bit) | ~75% | Medium | Any GPU |

**Example Commands:**

```bash
# FP8 quantization (H100)
vllm serve model_name --quantization fp8

# AWQ quantization
vllm serve TheBloke/Model-AWQ --quantization awq

# GPTQ quantization
vllm serve TheBloke/Model-GPTQ --quantization gptq

# FP8 KV cache only
vllm serve model_name --kv-cache-dtype fp8

# bitsandbytes
vllm serve model_name --quantization bitsandbytes --load-format bitsandbytes
```

### Strategy 2: Context Length Optimization

**Recommendation by Use Case:**

| Use Case | Recommended max_model_len | Reasoning |
|----------|---------------------------|-----------|
| Chat/Q&A | 4096-8192 | Most conversations fit |
| Code generation | 8192-16384 | Need space for context + generation |
| Document analysis | 16384-32768 | Long documents |
| Summarization | 32768-131072 | Entire documents/books |

**Example:**
```bash
# Chat application
vllm serve model_name --max-model-len 4096

# Document processing
vllm serve model_name --max-model-len 32768
```

### Strategy 3: Batch Size Tuning

**Find Optimal Batch Size:**

```bash
# Start conservative
vllm serve model_name --max-num-seqs 32

# Monitor GPU memory with nvidia-smi
# If memory usage < 80%, increase batch size
vllm serve model_name --max-num-seqs 64

# Continue until memory usage is 85-90%
vllm serve model_name --max-num-seqs 128
```

**Batch Size Guidelines by Model:**

| Model Size | GPU | Suggested max_num_seqs |
|------------|-----|------------------------|
| 7B | 1x 24GB | 256-512 |
| 7B | 1x 16GB | 128-256 |
| 13B | 1x 40GB | 128-256 |
| 70B | 2x 80GB | 64-128 |
| 70B | 4x 40GB | 32-64 |

### Strategy 4: Multi-GPU Distribution

**Tensor Parallelism:**
```bash
# Split model across GPUs (reduces per-GPU memory)
vllm serve model_name \
  --tensor-parallel-size 4  # Use 4 GPUs

# Example: 70B model on 4x A100-40GB
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85
```

**Pipeline Parallelism (not yet in vLLM):**
- Alternative: Use multiple vLLM instances with load balancing
- Each instance serves a different model or handles different traffic

---

## Memory Profiling

### Check Available Memory

```bash
# Before starting vLLM
nvidia-smi --query-gpu=memory.free,memory.total --format=csv

# During vLLM operation
watch -n 1 nvidia-smi
```

### Estimate Memory Requirements

```python
def estimate_memory(model_size_b, max_len, max_seqs, tp_size=1):
    """
    Estimate vLLM memory requirements.
    
    Args:
        model_size_b: Model size in billions of parameters
        max_len: Maximum sequence length
        max_seqs: Maximum number of concurrent sequences
        tp_size: Tensor parallel size
    """
    # Model weights (FP16)
    model_memory_gb = model_size_b * 2 / tp_size
    
    # KV cache (approximate for Llama-like models)
    # num_layers * 2 * hidden_size * num_heads * max_seqs * max_len * 2 bytes
    # Simplified: ~0.5GB per 1B params per 1K tokens per seq
    kv_cache_gb = (model_size_b * max_len * max_seqs * 0.5) / (1024 * tp_size)
    
    # Activation memory (rough estimate)
    activation_gb = 2
    
    total_gb = model_memory_gb + kv_cache_gb + activation_gb
    
    print(f"Estimated memory per GPU: {total_gb:.1f} GB")
    print(f"  Model weights: {model_memory_gb:.1f} GB")
    print(f"  KV cache: {kv_cache_gb:.1f} GB")
    print(f"  Activations: {activation_gb:.1f} GB")
    
    return total_gb

# Example: 70B model, 8K context, 128 batch, 2 GPUs
estimate_memory(70, 8192, 128, tp_size=2)
```

---

## Advanced Techniques

### Technique 1: CPU Offloading (Swap)

```bash
# Offload some memory to CPU (slower but allows larger models)
vllm serve model_name \
  --swap-space 4  # 4GB swap to CPU
```

**Trade-off:** Slower inference when swapping occurs

### Technique 2: Dynamic Batching

```bash
# Let vLLM automatically adjust batch size
vllm serve model_name \
  --max-num-seqs 256 \
  --scheduler-delay-factor 0.5
```

### Technique 3: Prefix Caching for Repeated Prompts

```bash
# Cache common prompt prefixes (e.g., system prompts)
vllm serve model_name \
  --enable-prefix-caching
```

**Memory Impact:** Increases KV cache usage but reduces recomputation

---

## Troubleshooting Checklist

When encountering OOM:

- [ ] Check actual GPU memory: `nvidia-smi`
- [ ] Verify model size fits: Model params × 2 bytes < VRAM
- [ ] Reduce `gpu_memory_utilization` to 0.80-0.85
- [ ] Reduce `max_model_len` (e.g., 32K → 16K → 8K)
- [ ] Reduce `max_num_seqs` (e.g., 256 → 128 → 64)
- [ ] Enable `--kv-cache-dtype fp8`
- [ ] Try quantization (`--quantization fp8/awq/gptq`)
- [ ] Use tensor parallelism (`--tensor-parallel-size 2/4/8`)
- [ ] Consider smaller model variant
- [ ] Check for memory leaks (restart service)

---

## Memory-Constrained Configurations

### Configuration 1: Single RTX 4090 (24GB) - 7B Model
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8 \
  --dtype float16
```

### Configuration 2: Single A100 40GB - 13B Model
```bash
vllm serve meta-llama/Llama-2-13B-Instruct \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8
```

### Configuration 3: 2x A100 80GB - 70B Model
```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8 \
  --dtype bfloat16
```

### Configuration 4: 8x H100 80GB - DeepSeek-V3 (671B MoE)
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --enable-expert-parallel \
  --dtype bfloat16
```

---

## Resources

- [vLLM Performance Optimization](https://docs.vllm.ai/en/latest/performance/performance.html)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/supported_hardware.html)
- [NVIDIA GPU Memory Guide](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
