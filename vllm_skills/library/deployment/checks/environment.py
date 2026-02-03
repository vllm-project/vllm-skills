"""Environment detection module for vLLM deployment assistant."""

import subprocess
import sys
from typing import Dict, Optional


def detect_python_version() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def detect_pytorch() -> Dict[str, Optional[str]]:
    """
    Detect PyTorch installation and version.
    
    Returns:
        Dictionary with PyTorch information
    """
    try:
        import torch
        return {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except ImportError:
        return {
            'version': None,
            'cuda_available': False,
            'cuda_version': None,
            'device_count': 0
        }


def detect_cuda() -> Optional[str]:
    """
    Detect CUDA version from nvidia-smi.
    
    Returns:
        CUDA version string or None
    """
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        # Driver version is returned, actual CUDA runtime comes from PyTorch
        return output.strip().split('\n')[0]
    except:
        return None


def detect_vllm() -> Optional[str]:
    """
    Detect vLLM installation and version.
    
    Returns:
        vLLM version string or None
    """
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        return None


def detect_flash_attention() -> Optional[str]:
    """
    Detect Flash Attention installation and version.
    
    Returns:
        Flash Attention version string or None
    """
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return None


def detect_triton() -> Optional[str]:
    """
    Detect Triton installation and version.
    
    Returns:
        Triton version string or None
    """
    try:
        import triton
        return triton.__version__
    except ImportError:
        return None


def detect_xformers() -> Optional[str]:
    """
    Detect xFormers installation and version.
    
    Returns:
        xFormers version string or None
    """
    try:
        import xformers
        return xformers.__version__
    except ImportError:
        return None


def check_environment() -> Dict[str, any]:
    """
    Comprehensive environment detection.
    
    Returns:
        Dictionary containing all environment information
    """
    pytorch_info = detect_pytorch()
    
    return {
        'python_version': detect_python_version(),
        'pytorch': pytorch_info,
        'cuda_driver': detect_cuda(),
        'vllm_version': detect_vllm(),
        'flash_attn_version': detect_flash_attention(),
        'triton_version': detect_triton(),
        'xformers_version': detect_xformers()
    }
