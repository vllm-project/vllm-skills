# CUDA Errors in vLLM

## CUDA Out of Memory Errors

### Error: "CUDA out of memory"
**Common causes**:
- Model too large for GPU
- `gpu_memory_utilization` set too high
- Multiple processes using GPU

**Solutions**:

1. **Reduce memory usage**:
   ```bash
   vllm serve model-name --gpu-memory-utilization 0.8
   ```

2. **Use quantization**:
   ```bash
   vllm serve model-name --quantization awq
   ```

3. **Reduce context length**:
   ```bash
   vllm serve model-name --max-model-len 2048
   ```

4. **Use smaller batch sizes**:
   ```bash
   vllm serve model-name --max-num-seqs 64
   ```

5. **Enable CPU offloading** (if supported):
   ```bash
   vllm serve model-name --cpu-offload-gb 10
   ```

## CUDA Version Mismatch

### Error: "CUDA version mismatch"
**Symptoms**: vLLM fails to start, reports CUDA version conflict

**Solutions**:

1. **Check CUDA versions**:
   ```bash
   nvidia-smi  # Driver CUDA version
   nvcc --version  # Toolkit CUDA version
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
   ```

2. **Reinstall PyTorch with correct CUDA**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Reinstall vLLM with matching CUDA**:
   ```bash
   pip uninstall vllm
   pip install vllm --no-build-isolation
   ```

## CUDA Kernel Errors

### Error: "CUDA kernel launch failed"
**Common causes**:
- Invalid tensor operations
- GPU driver issues
- Corrupted CUDA installation

**Solutions**:

1. **Update GPU drivers**:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535  # or latest version
   ```

2. **Verify CUDA installation**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **Reset GPU**:
   ```bash
   sudo nvidia-smi --gpu-reset
   ```

4. **Check for hardware issues**:
   ```bash
   nvidia-smi -q -d MEMORY,ECC
   ```

## CUDA Initialization Errors

### Error: "CUDA initialization failed"
**Symptoms**: Cannot initialize CUDA runtime

**Solutions**:

1. **Check GPU visibility**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Set CUDA device explicitly**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   vllm serve model-name
   ```

3. **Update CUDA drivers**:
   - Download latest drivers from NVIDIA website
   - Reboot after installation

4. **Check for conflicts**:
   ```bash
   # Kill other processes using GPU
   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill
   ```

## CUDNN Errors

### Error: "cuDNN error" or "cuDNN version mismatch"
**Solutions**:

1. **Update cuDNN**:
   ```bash
   pip install nvidia-cudnn-cu12
   ```

2. **Verify cuDNN installation**:
   ```python
   import torch
   print(torch.backends.cudnn.version())
   ```

3. **Disable cuDNN if causing issues**:
   ```python
   torch.backends.cudnn.enabled = False
   ```

## NCCL Errors (Multi-GPU)

### Error: "NCCL error" in tensor parallel mode
**Solutions**:

1. **Update NCCL**:
   ```bash
   pip install nvidia-nccl-cu12
   ```

2. **Enable NCCL debugging**:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   vllm serve model-name --tensor-parallel-size 2
   ```

3. **Use different backend**:
   ```bash
   vllm serve model-name --disable-custom-all-reduce
   ```

4. **Check GPU topology**:
   ```bash
   nvidia-smi topo -m
   ```

## See Also
- [Memory Issues](memory_issues.md)
- [Common Issues](common_issues.md)
- [Kernel Issues](kernel_issues.md)
