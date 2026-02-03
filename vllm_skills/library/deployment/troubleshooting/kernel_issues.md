# Kernel Issues and Compilation Errors

Guide for resolving kernel compilation and runtime issues in vLLM.

## Overview

vLLM uses custom CUDA kernels and Triton kernels for optimal performance. This guide covers:

1. [Triton Compilation Errors](#triton-compilation-errors)
2. [Punica Kernel Issues](#punica-kernel-issues)
3. [Flash Attention Kernel Issues](#flash-attention-kernel-issues)
4. [Custom Kernel Failures](#custom-kernel-failures)
5. [Compute Capability Issues](#compute-capability-issues)

---

## Triton Compilation Errors

### Error Messages
```
RuntimeError: Failed to compile Triton kernel
triton.runtime.errors.CompilationError
Error during Triton kernel compilation
```

### Common Causes
- Incompatible Triton version
- Missing compiler dependencies
- GPU compute capability not supported
- Kernel cache corruption

### Solutions

#### Solution 1: Update Triton
```bash
# Update to latest compatible version
pip install --upgrade triton

# Or install specific version
pip install triton==2.1.0  # Check vLLM requirements
```

#### Solution 2: Clear Triton Cache
```bash
# Clear kernel cache
rm -rf ~/.triton/cache/

# Or set new cache location
export TRITON_CACHE_DIR=/tmp/triton_cache
```

#### Solution 3: Disable Triton Kernels
```bash
# Use fallback kernels (slower but more compatible)
export VLLM_USE_TRITON_FLASH_ATTN=0

# Restart vLLM
vllm serve model_name
```

#### Solution 4: Install Compiler Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Verify gcc/g++ installation
gcc --version
g++ --version

# Install LLVM if needed
sudo apt-get install llvm-14
```

---

## Punica Kernel Issues

### Error Messages
```
RuntimeError: Punica kernels failed to compile
ImportError: cannot import punica kernels
Failed to load Punica SGMV kernels
```

### Background
Punica kernels are used for LoRA (Low-Rank Adaptation) support in vLLM.

### Solutions

#### Solution 1: Disable Punica Kernels
```bash
# Set environment variable
export VLLM_INSTALL_PUNICA_KERNELS=0

# Reinstall vLLM
pip install --upgrade --force-reinstall vllm --no-cache-dir
```

#### Solution 2: Build from Source with Punica Disabled
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
export VLLM_INSTALL_PUNICA_KERNELS=0
pip install -e .
```

#### Solution 3: Skip LoRA Features
```bash
# If not using LoRA, Punica kernels aren't needed
# Just ensure VLLM_INSTALL_PUNICA_KERNELS=0 is set
```

---

## Flash Attention Kernel Issues

### Error Messages
```
RuntimeError: FlashAttention is not supported
ImportError: cannot import name 'flash_attn_func'
CUDA error: no kernel image is available for execution
```

### Solutions

#### Solution 1: Install Flash Attention
```bash
# Install Flash Attention 2
pip install flash-attn --no-build-isolation

# If build fails, try with specific CUDA arch
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation
```

#### Solution 2: Check GPU Compatibility
```bash
# Flash Attention requires compute capability 7.5+
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Supported: V100 (7.0 partial), T4 (7.5), A100 (8.0), RTX 30/40 series (8.6, 8.9)
```

#### Solution 3: Use xFormers Fallback
```bash
# If Flash Attention doesn't work, vLLM falls back to xFormers
# Ensure xFormers is installed
pip install xformers

# Check in logs: "Using xFormers for attention"
```

#### Solution 4: Build Flash Attention from Source
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install

# Or for specific GPU architecture
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
python setup.py install
```

---

## Custom Kernel Failures

### Error Messages
```
RuntimeError: CUDA error: invalid kernel image
RuntimeError: no kernel image is available for execution on the device
AssertionError in custom kernel
```

### Solutions

#### Solution 1: Use Pre-compiled Kernels
```bash
# Use pre-built kernels instead of JIT compilation
export VLLM_USE_PRECOMPILED_KERNELS=1

vllm serve model_name
```

#### Solution 2: Rebuild with Correct Architecture
```bash
# Set correct CUDA architectures for your GPU
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# Reinstall vLLM
pip install --upgrade --force-reinstall vllm --no-cache-dir
```

#### Solution 3: Disable Specific Kernels
```bash
# Disable CUDA graphs (uses custom kernels)
vllm serve model_name --enforce-eager

# Disable Flash Attention
export VLLM_USE_TRITON_FLASH_ATTN=0
```

---

## Compute Capability Issues

### Error Messages
```
RuntimeError: Your GPU has compute capability X.X, but vLLM requires 7.0+
CUDA error: no kernel image is available for execution on the device
```

### Check GPU Compute Capability

```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Common GPUs:
# - Pascal (P100): 6.0 (NOT SUPPORTED)
# - Volta (V100): 7.0 (SUPPORTED)
# - Turing (T4, RTX 20): 7.5 (SUPPORTED)
# - Ampere (A100, RTX 30): 8.0, 8.6 (SUPPORTED)
# - Ada (RTX 40): 8.9 (SUPPORTED)
# - Hopper (H100): 9.0 (SUPPORTED)
```

### Solutions

#### Solution 1: Upgrade GPU
- vLLM requires compute capability 7.0+
- Pascal GPUs (P100, GTX 10 series) are not supported

#### Solution 2: Use Older vLLM Version (Not Recommended)
```bash
# Older versions had broader support
pip install vllm==0.4.0
# Note: Missing features and bug fixes
```

#### Solution 3: Use Alternative Framework
- For older GPUs, consider:
  - HuggingFace Transformers
  - Text Generation Inference
  - FastChat

---

## Kernel Debugging Techniques

### Enable Verbose Logging
```bash
# Enable debug logging
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

vllm serve model_name
```

### Check Kernel Compilation
```python
# Test Triton kernel compilation
import triton
print(f"Triton version: {triton.__version__}")

# Test Flash Attention
try:
    from flash_attn import flash_attn_func
    print("Flash Attention is available")
except ImportError:
    print("Flash Attention not available")
```

### Monitor Kernel Launches
```bash
# Use NVIDIA Nsight Systems for profiling
nsys profile --stats=true vllm serve model_name

# Or use nvprof (older)
nvprof vllm serve model_name
```

---

## Environment Variables Reference

```bash
# Triton Configuration
export TRITON_CACHE_DIR=/tmp/triton_cache     # Custom cache location
export VLLM_USE_TRITON_FLASH_ATTN=0           # Disable Triton Flash Attention

# Punica Configuration
export VLLM_INSTALL_PUNICA_KERNELS=0          # Disable Punica kernels

# CUDA Kernel Configuration
export VLLM_USE_PRECOMPILED_KERNELS=1         # Use pre-built kernels
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"    # Target GPU architectures

# Debugging
export CUDA_LAUNCH_BLOCKING=1                 # Synchronous kernel launches
export VLLM_LOGGING_LEVEL=DEBUG               # Verbose logging

# Memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management
```

---

## Build from Source for Custom Kernels

### Prerequisites
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# Install CUDA Toolkit (if not already installed)
# Download from https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
```

### Build Process
```bash
# Clone repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Set build options
export VLLM_INSTALL_PUNICA_KERNELS=0  # Optional: disable Punica
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # Your GPU architectures

# Build and install
pip install -e .

# Or build wheel
pip wheel -e . --no-deps

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

### Troubleshooting Build Issues

#### Issue: Compilation takes too long
```bash
# Use multiple cores
export MAX_JOBS=4
pip install -e .
```

#### Issue: Out of memory during compilation
```bash
# Limit parallel jobs
export MAX_JOBS=1
pip install -e .
```

#### Issue: CUDA version mismatch
```bash
# Ensure consistent CUDA versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version
# These should match
```

---

## Model-Specific Kernel Issues

### MoE Models (Mixtral, DeepSeek-V3)
```bash
# MoE models may need expert parallelism
vllm serve deepseek-ai/DeepSeek-V3 \
  --enable-expert-parallel \
  --tensor-parallel-size 8

# If kernel errors occur, try FP8
vllm serve deepseek-ai/DeepSeek-V3 \
  --quantization fp8 \
  --enable-expert-parallel
```

### Vision Models
```bash
# Vision models may need specific attention kernels
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1
```

---

## Performance Impact of Kernel Fallbacks

| Kernel Type | Performance Impact | When to Use Fallback |
|-------------|-------------------|----------------------|
| Flash Attention → xFormers | -15-25% throughput | GPU compute < 7.5 |
| Triton → Native | -10-20% throughput | Compilation errors |
| CUDA graphs → Eager | -20-40% throughput | Debugging only |
| Custom → Default | -5-15% throughput | Compatibility issues |

---

## Getting Help

For persistent kernel issues:

1. **Gather information:**
   ```bash
   # GPU info
   nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
   
   # Software versions
   pip show vllm torch triton flash-attn
   
   # CUDA version
   nvcc --version
   ```

2. **Check for known issues:**
   - [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
   - Search for your error message

3. **Create detailed bug report:**
   - Include all version information
   - Full error traceback
   - Minimal reproduction command
   - GPU model and compute capability

4. **Community resources:**
   - [vLLM Discord](https://discord.gg/vllm)
   - [GitHub Discussions](https://github.com/vllm-project/vllm/discussions)
