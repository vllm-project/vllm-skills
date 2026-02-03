"""Compatibility checking utilities"""

from typing import Optional

from vllm_skills.library.deployment.checks.environment import EnvironmentInfo
from vllm_skills.library.deployment.checks.hardware import HardwareInfo


def check_model_compatibility(
    model_name: str, hardware: HardwareInfo, environment: EnvironmentInfo
) -> dict[str, any]:
    """
    Check if model is compatible with hardware and environment

    Args:
        model_name: Model name or path
        hardware: Hardware information
        environment: Environment information

    Returns:
        Dictionary with compatibility results:
            - compatible: bool
            - issues: list of compatibility issues
            - recommendations: list of recommendations
    """
    issues = []
    recommendations = []

    # Check GPU availability
    if hardware.gpu is None:
        issues.append("No GPU detected - vLLM requires GPU for inference")
        return {
            "compatible": False,
            "issues": issues,
            "recommendations": ["Install NVIDIA GPU with CUDA support"],
        }

    # Check CUDA availability
    if not environment.cuda_available:
        issues.append("CUDA not available")
        recommendations.append("Install PyTorch with CUDA support")

    # Check vLLM installation
    if environment.vllm_version is None:
        issues.append("vLLM not installed")
        recommendations.append("Install vLLM: pip install vllm")

    # Check minimum GPU memory (rough estimate)
    model_lower = model_name.lower()
    required_memory = estimate_model_memory(model_lower)

    if required_memory and hardware.gpu.memory_total < required_memory:
        issues.append(
            f"Insufficient GPU memory: {hardware.gpu.memory_total}MB available, "
            f"~{required_memory}MB required"
        )
        recommendations.append("Consider using quantization or a smaller model")

    compatible = len(issues) == 0

    return {
        "compatible": compatible,
        "issues": issues,
        "recommendations": recommendations,
    }


def estimate_model_memory(model_name: str) -> Optional[int]:
    """
    Estimate GPU memory requirements for a model

    Args:
        model_name: Model name (lowercase)

    Returns:
        Estimated memory in MB or None if unknown
    """
    # Rough estimates based on model size
    # These are conservative estimates for fp16
    if "70b" in model_name or "72b" in model_name:
        return 140000  # ~140GB for 70B models
    elif "13b" in model_name or "14b" in model_name:
        return 26000  # ~26GB for 13B models
    elif "7b" in model_name or "8b" in model_name:
        return 14000  # ~14GB for 7B models
    elif "3b" in model_name:
        return 6000  # ~6GB for 3B models
    elif "1b" in model_name:
        return 2000  # ~2GB for 1B models

    return None  # Unknown model size
