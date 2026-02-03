# CUDA Errors and Solutions

Comprehensive guide for diagnosing and resolving CUDA-related errors in vLLM.

## Error Categories

1. [CUDA Out of Memory (OOM)](#cuda-out-of-memory)
2. [CUDA Version Mismatch](#cuda-version-mismatch)
3. [CUDA Driver Issues](#cuda-driver-issues)
4. [CUDA Runtime Errors](#cuda-runtime-errors)
5. [CUDA Kernel Launch Failures](#cuda-kernel-launch-failures)

---

## CUDA Out of Memory

### Error Messages
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB (GPU 0; Y GiB total capacity)
torch.cuda.OutOfMemoryError: CUDA out of memory
RuntimeError: No available memory for the cache blocks
```

### Diagnosis Steps

1. **Check current memory usage:**
   ```bash
   nvidia-smi
   # Look at memory usage per GPU
   ```

2. **Estimate model memory requirements:**
   - Model parameters: ~2 bytes per parameter (FP16)
   - KV cache: Depends on batch size and sequence length
   - Activation memory: Depends on batch size
   - Example: 7B model ≈ 14GB + KV cache + activations

### Solutions

#### Solution 1: Reduce Memory Utilization
```bash
# Start conservative
vllm serve model_name --gpu-memory-utilization 0.80

# Default is 0.90, try reducing by 0.05 increments
--gpu-memory-utilization 0.85
--gpu-memory-utilization 0.80
--gpu-memory-utilization 0.75
```

#### Solution 2: Reduce Context Length
```bash
# Reduce max sequence length
--max-model-len 8192   # from 32768 or default
--max-model-len 4096   # if still having issues
--max-model-len 2048   # minimum for most use cases
```

#### Solution 3: Reduce Batch Size
```bash
# Reduce concurrent sequences
--max-num-seqs 128  # from default 256
--max-num-seqs 64
--max-num-seqs 32
```

#### Solution 4: Enable Memory Optimizations
```bash
# FP8 KV cache (50% memory reduction)
--kv-cache-dtype fp8

# FP8 quantization for Hopper GPUs
--quantization fp8

# Use quantized model checkpoint
--quantization awq   # For AWQ models
--quantization gptq  # For GPTQ models
```

#### Solution 5: Use Tensor Parallelism
```bash
# Distribute model across multiple GPUs
--tensor-parallel-size 2  # Use 2 GPUs
--tensor-parallel-size 4  # Use 4 GPUs
--tensor-parallel-size 8  # Use 8 GPUs
```

#### Solution 6: Enable CPU Swap
```bash
# Swap some tensors to CPU memory (slower but allows larger models)
--swap-space 4  # 4GB CPU swap space
```

### Memory Calculation Example

For Llama-3.1-70B on 2x A100-80GB:
```
Model weights: 70B * 2 bytes = 140GB (requires TP=2)
Per GPU: 140GB / 2 = 70GB
KV cache @ 16K ctx, batch=64: ~8GB per GPU
Activations: ~2GB per GPU
Total per GPU: 70 + 8 + 2 = 80GB ✓ (fits with 0.90 utilization)
```

---

## CUDA Version Mismatch

### Error Messages
```
RuntimeError: The detected CUDA version (12.1) mismatches the version that was used to compile PyTorch (11.8)
ImportError: libcudart.so.11.0: cannot open shared object file
version `GLIBCXX_3.4.30' not found
```

### Diagnosis Steps

1. **Check CUDA versions:**
   ```bash
   # CUDA driver version (from GPU driver)
   nvidia-smi
   
   # CUDA toolkit version
   nvcc --version
   
   # PyTorch CUDA version
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   
   # vLLM CUDA version
   pip show vllm | grep Version
   ```

2. **Understand version requirements:**
   - CUDA driver version ≥ CUDA toolkit version
   - PyTorch CUDA version should match vLLM CUDA version
   - vLLM supports: CUDA 11.8, 12.1, 12.2, 12.3, 12.4, 12.6

### Solutions

#### Solution 1: Reinstall vLLM with Correct CUDA Version
```bash
# Uninstall current vLLM
pip uninstall vllm -y

# Install for CUDA 11.8
pip install vllm-cu118

# Install for CUDA 12.1
pip install vllm-cu121

# Or install default (auto-detects)
pip install vllm
```

#### Solution 2: Reinstall PyTorch with Matching CUDA
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Solution 3: Update CUDA Toolkit
```bash
# Download and install from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Or use conda
conda install -c nvidia cuda-toolkit=12.1
```

---

## CUDA Driver Issues

### Error Messages
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

### Diagnosis Steps

1. **Check driver installation:**
   ```bash
   nvidia-smi
   # Should show driver version and GPU info
   ```

2. **Check minimum driver requirements:**
   - CUDA 11.8: Driver ≥ 450.80.02 (Linux) / 452.39 (Windows)
   - CUDA 12.1: Driver ≥ 525.60.13 (Linux) / 527.41 (Windows)
   - CUDA 12.4: Driver ≥ 550.54.15 (Linux)

### Solutions

#### Solution 1: Update NVIDIA Driver
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Or use NVIDIA installer
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.86.05/NVIDIA-Linux-x86_64-535.86.05.run
sudo sh NVIDIA-Linux-x86_64-535.86.05.run

# Reboot after installation
sudo reboot
```

#### Solution 2: Verify Driver Installation
```bash
# Check loaded modules
lsmod | grep nvidia

# Reinstall if needed
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535
```

---

## CUDA Runtime Errors

### Error Messages
```
RuntimeError: CUDA error: an illegal memory access was encountered
RuntimeError: CUDA error: device-side assert triggered
RuntimeError: CUDA error: invalid configuration argument
```

### Diagnosis Steps

1. **Enable CUDA error checking:**
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   # Re-run to get detailed error location
   ```

2. **Check for GPU hang:**
   ```bash
   nvidia-smi
   # Look for processes using excessive memory
   ```

### Solutions

#### Solution 1: Disable CUDA Graphs (Debugging)
```bash
# Run in eager mode to isolate issue
vllm serve model_name --enforce-eager
```

#### Solution 2: Reset GPU State
```bash
# Kill all GPU processes
nvidia-smi --gpu-reset

# Or reboot if needed
sudo reboot
```

#### Solution 3: Verify Model Checkpoint
```bash
# Try loading model in PyTorch first
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('model_name')"
```

#### Solution 4: Update vLLM
```bash
pip install --upgrade vllm
# Bug fixes often resolve CUDA runtime errors
```

---

## CUDA Kernel Launch Failures

### Error Messages
```
RuntimeError: CUDA error: invalid kernel image
RuntimeError: CUDA error: no kernel image is available for execution on the device
Triton kernel compilation failed
```

### Diagnosis Steps

1. **Check GPU compute capability:**
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   # vLLM requires compute capability ≥ 7.0
   ```

2. **Check Triton installation:**
   ```bash
   python -c "import triton; print(triton.__version__)"
   ```

### Solutions

#### Solution 1: Disable Custom Kernels
```bash
# Disable Punica kernels
export VLLM_INSTALL_PUNICA_KERNELS=0

# Use pre-compiled kernels
export VLLM_USE_PRECOMPILED_KERNELS=1

# Reinstall
pip install --upgrade --force-reinstall vllm
```

#### Solution 2: Update Triton
```bash
pip install --upgrade triton
```

#### Solution 3: Rebuild from Source
```bash
# Clone and build with specific arch
git clone https://github.com/vllm-project/vllm.git
cd vllm
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
pip install -e .
```

---

## CUDA Graph Errors

### Error Messages
```
RuntimeError: CUDA graphs cannot be enabled
RuntimeError: failed to capture CUDA graph
```

### Solutions

#### Solution 1: Reduce Capture Length
```bash
# Capture shorter sequences in CUDA graphs
--max-seq-len-to-capture 2048  # from default 8192
```

#### Solution 2: Disable CUDA Graphs
```bash
# Use eager execution
--enforce-eager
```

---

## Environment Variables for CUDA Debugging

```bash
# Enable synchronous CUDA operations for better error messages
export CUDA_LAUNCH_BLOCKING=1

# Enable CUDA malloc debug
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable CUDA graphs for debugging
export VLLM_DISABLE_CUDA_GRAPH=1

# Enable verbose logging
export VLLM_LOGGING_LEVEL=DEBUG

# Force specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

---

## Diagnostic Commands

```bash
# Full system info
nvidia-smi -L  # List GPUs
nvidia-smi -q  # Detailed GPU info

# Check CUDA installation
nvcc --version
which nvcc

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Test CUDA operations
python -c "import torch; x = torch.tensor([1.0]).cuda(); print(x)"

# Check vLLM installation
python -c "import vllm; print(vllm.__version__)"
```

---

## Getting Help

If CUDA errors persist:

1. Include in bug report:
   - `nvidia-smi` output
   - `nvcc --version` output
   - PyTorch version and CUDA version
   - vLLM version
   - Full error traceback
   - GPU model and compute capability

2. Resources:
   - [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
   - [vLLM Discord](https://discord.gg/vllm)
   - [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
