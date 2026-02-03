"""Environment detection utilities"""

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentInfo:
    """Environment information"""

    python_version: str
    pytorch_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    vllm_version: Optional[str]
    flash_attention_version: Optional[str]


def get_python_version() -> str:
    """Get Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_pytorch_version() -> Optional[str]:
    """Get PyTorch version"""
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_version() -> Optional[str]:
    """Get CUDA version"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.version.cuda
        return None
    except ImportError:
        return None


def get_vllm_version() -> Optional[str]:
    """Get vLLM version"""
    try:
        import vllm

        return vllm.__version__
    except ImportError:
        return None


def get_flash_attention_version() -> Optional[str]:
    """Get Flash Attention version"""
    try:
        import flash_attn

        return flash_attn.__version__
    except ImportError:
        return None


def check_environment() -> EnvironmentInfo:
    """
    Check complete environment information

    Returns:
        EnvironmentInfo object with environment details
    """
    return EnvironmentInfo(
        python_version=get_python_version(),
        pytorch_version=get_pytorch_version(),
        cuda_available=check_cuda_available(),
        cuda_version=get_cuda_version(),
        vllm_version=get_vllm_version(),
        flash_attention_version=get_flash_attention_version(),
    )
