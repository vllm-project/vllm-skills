# Memory Issues in vLLM

## Out of Memory (OOM) Errors

### Diagnosing OOM Issues

1. **Check GPU memory usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Monitor vLLM memory**:
   ```bash
   # vLLM logs will show memory allocation
   # Look for lines like "Allocated XX GB for model"
   ```

3. **Calculate memory requirements**:
   - Model weights: `num_parameters * bytes_per_parameter`
   - KV cache: `batch_size * seq_len * num_layers * hidden_size * 2`
   - Activations: Variable based on computation

## Solutions for OOM

### 1. Reduce GPU Memory Utilization

```bash
# Default is 0.9, try lower values
vllm serve model-name --gpu-memory-utilization 0.75
```

**When to use**: First thing to try for OOM errors

### 2. Use Quantization

```bash
# AWQ (4-bit)
vllm serve model-name --quantization awq

# GPTQ (4-bit)
vllm serve model-name --quantization gptq

# FP8 (H100 only)
vllm serve model-name --quantization fp8
```

**Memory savings**:
- FP16 → INT4: ~75% reduction
- FP16 → FP8: ~50% reduction

### 3. Reduce Context Length

```bash
# Limit max sequence length
vllm serve model-name --max-model-len 2048
```

**Impact**: KV cache scales with sequence length

### 4. Reduce Batch Size

```bash
# Smaller batches use less memory
vllm serve model-name --max-num-seqs 32
```

**Tradeoff**: Lower throughput

### 5. Enable Prefix Caching

```bash
vllm serve model-name --enable-prefix-caching
```

**Benefit**: Shares KV cache for common prefixes

### 6. Use Tensor Parallelism

```bash
# Split model across multiple GPUs
vllm serve model-name --tensor-parallel-size 2
```

**Requirement**: Multiple GPUs

## Memory Optimization Strategies

### For Large Models (70B+)

1. **Multi-GPU with TP**:
   ```bash
   vllm serve meta-llama/Llama-3.1-70B-Instruct \
     --tensor-parallel-size 4 \
     --gpu-memory-utilization 0.95
   ```

2. **Quantization + Single GPU**:
   ```bash
   vllm serve meta-llama/Llama-3.1-70B-Instruct \
     --quantization awq \
     --gpu-memory-utilization 0.90
   ```

### For Memory-Constrained GPUs

1. **Small batch sizes**:
   ```bash
   vllm serve model-name \
     --max-num-seqs 16 \
     --gpu-memory-utilization 0.75
   ```

2. **Short context**:
   ```bash
   vllm serve model-name \
     --max-model-len 1024 \
     --gpu-memory-utilization 0.80
   ```

### For Maximum Throughput

1. **High memory utilization**:
   ```bash
   vllm serve model-name \
     --gpu-memory-utilization 0.95 \
     --max-num-seqs 512
   ```

2. **Enable all optimizations**:
   ```bash
   vllm serve model-name \
     --enable-prefix-caching \
     --enable-chunked-prefill \
     --gpu-memory-utilization 0.95
   ```

## Monitoring Memory Usage

### Real-time Monitoring

```bash
# GPU memory
watch -n 0.5 nvidia-smi

# System memory
watch -n 1 free -h

# vLLM metrics (if enabled)
curl http://localhost:8000/metrics
```

### Memory Profiling

```python
import torch

# Check allocated memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Reset peak memory stats
torch.cuda.reset_peak_memory_stats()
```

## Common OOM Patterns

### Pattern 1: OOM During Loading
**Cause**: Model too large for GPU
**Solution**: Use quantization or tensor parallelism

### Pattern 2: OOM During First Request
**Cause**: KV cache allocation fails
**Solution**: Reduce `max_model_len` or `max_num_seqs`

### Pattern 3: OOM During Long Requests
**Cause**: Long sequence exhausts KV cache
**Solution**: Reduce `max_model_len`

### Pattern 4: Gradual Memory Growth
**Cause**: Memory leak or fragmentation
**Solution**: Restart server, update vLLM

## Emergency Recovery

If server is stuck in OOM state:

```bash
# 1. Find vLLM process
ps aux | grep vllm

# 2. Kill it
kill -9 <pid>

# 3. Clear GPU memory
nvidia-smi --gpu-reset

# 4. Restart with lower memory settings
vllm serve model-name --gpu-memory-utilization 0.7
```

## See Also
- [Common Issues](common_issues.md)
- [CUDA Errors](cuda_errors.md)
- [Hardware Matrix](../models/hardware_matrix.yaml)
