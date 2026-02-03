"""Compatibility checking module for vLLM deployment assistant."""

from typing import Dict, List, Tuple, Optional


# Compatibility matrix
VLLM_COMPATIBILITY = {
    '0.6.x': {
        'pytorch': ['2.0', '2.1', '2.2', '2.3', '2.4'],
        'cuda': ['11.8', '12.1'],
        'python': ['3.8', '3.9', '3.10', '3.11'],
        'flash_attention': '2.3+'
    },
    '0.7.x': {
        'pytorch': ['2.0', '2.1', '2.2', '2.3', '2.4', '2.5'],
        'cuda': ['11.8', '12.1', '12.2', '12.3', '12.4'],
        'python': ['3.8', '3.9', '3.10', '3.11', '3.12'],
        'flash_attention': '2.4+'
    },
    '0.8.x': {
        'pytorch': ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6'],
        'cuda': ['11.8', '12.1', '12.2', '12.3', '12.4', '12.6'],
        'python': ['3.8', '3.9', '3.10', '3.11', '3.12'],
        'flash_attention': '2.5+',
        'rocm': ['5.7', '6.0', '6.1']
    }
}


def parse_version(version: str) -> Tuple[int, ...]:
    """Parse version string into tuple of integers."""
    if not version:
        return (0,)
    # Handle versions like "2.1.0+cu121"
    clean_version = version.split('+')[0]
    return tuple(int(x) for x in clean_version.split('.')[:2])


def check_version_compatibility(
    version: Optional[str],
    compatible_versions: List[str]
) -> Tuple[bool, str]:
    """
    Check if version is compatible with list of compatible versions.
    
    Args:
        version: Version string to check
        compatible_versions: List of compatible version strings
        
    Returns:
        Tuple of (is_compatible, message)
    """
    if not version:
        return False, "Not installed"
    
    version_tuple = parse_version(version)
    
    for compat in compatible_versions:
        if '+' in compat:
            # Handle "2.5+" format
            min_version = parse_version(compat.replace('+', ''))
            if version_tuple >= min_version:
                return True, "Compatible"
        else:
            # Handle exact version match
            compat_tuple = parse_version(compat)
            if version_tuple[:len(compat_tuple)] == compat_tuple:
                return True, "Compatible"
    
    return False, f"Incompatible (found {version}, need {compatible_versions})"


def check_compatibility(env_info: Dict[str, any]) -> Dict[str, any]:
    """
    Check compatibility of installed versions.
    
    Args:
        env_info: Environment information from check_environment()
        
    Returns:
        Dictionary with compatibility results
    """
    vllm_version = env_info.get('vllm_version', '')
    
    # Determine vLLM major version
    vllm_major = '0.8.x'  # Default to latest
    if vllm_version:
        if vllm_version.startswith('0.6.'):
            vllm_major = '0.6.x'
        elif vllm_version.startswith('0.7.'):
            vllm_major = '0.7.x'
    
    compat_matrix = VLLM_COMPATIBILITY.get(vllm_major, VLLM_COMPATIBILITY['0.8.x'])
    
    results = {
        'vllm_version': vllm_version,
        'vllm_series': vllm_major,
        'checks': {}
    }
    
    # Check Python
    python_version = env_info.get('python_version', '')
    is_compat, msg = check_version_compatibility(python_version, compat_matrix['python'])
    results['checks']['python'] = {
        'installed': python_version,
        'compatible': is_compat,
        'message': msg,
        'expected': compat_matrix['python']
    }
    
    # Check PyTorch
    pytorch_version = env_info.get('pytorch', {}).get('version', '')
    is_compat, msg = check_version_compatibility(pytorch_version, compat_matrix['pytorch'])
    results['checks']['pytorch'] = {
        'installed': pytorch_version,
        'compatible': is_compat,
        'message': msg,
        'expected': compat_matrix['pytorch']
    }
    
    # Check CUDA
    cuda_version = env_info.get('pytorch', {}).get('cuda_version', '')
    is_compat, msg = check_version_compatibility(cuda_version, compat_matrix['cuda'])
    results['checks']['cuda'] = {
        'installed': cuda_version,
        'compatible': is_compat,
        'message': msg,
        'expected': compat_matrix['cuda']
    }
    
    # Check Flash Attention
    flash_attn_version = env_info.get('flash_attn_version', '')
    flash_attn_expected = compat_matrix.get('flash_attention', '2.0+')
    if flash_attn_version:
        is_compat, msg = check_version_compatibility(flash_attn_version, [flash_attn_expected])
    else:
        is_compat, msg = False, "Not installed (recommended but optional)"
    
    results['checks']['flash_attention'] = {
        'installed': flash_attn_version or 'Not installed',
        'compatible': is_compat,
        'message': msg,
        'expected': flash_attn_expected
    }
    
    # Overall compatibility
    critical_checks = ['python', 'pytorch', 'cuda']
    results['overall_compatible'] = all(
        results['checks'][check]['compatible'] 
        for check in critical_checks
        if check in results['checks']
    )
    
    return results


def get_gpu_requirements(model_size_b: int) -> Dict[str, any]:
    """
    Get GPU requirements for a model size.
    
    Args:
        model_size_b: Model size in billions of parameters
        
    Returns:
        Dictionary with GPU requirements
    """
    requirements = {
        'min_vram_gb': 0,
        'recommended_vram_gb': 0,
        'example_gpus': [],
        'tensor_parallel_recommended': 1
    }
    
    if model_size_b <= 3:
        requirements.update({
            'min_vram_gb': 8,
            'recommended_vram_gb': 16,
            'example_gpus': ['RTX 3060 12GB', 'T4 16GB']
        })
    elif model_size_b <= 8:
        requirements.update({
            'min_vram_gb': 16,
            'recommended_vram_gb': 24,
            'example_gpus': ['RTX 4090 24GB', 'A10G 24GB', 'L4 24GB']
        })
    elif model_size_b <= 14:
        requirements.update({
            'min_vram_gb': 28,
            'recommended_vram_gb': 40,
            'example_gpus': ['A100 40GB', 'A100 80GB']
        })
    elif model_size_b <= 34:
        requirements.update({
            'min_vram_gb': 70,
            'recommended_vram_gb': 80,
            'example_gpus': ['A100 80GB', 'H100 80GB'],
            'tensor_parallel_recommended': 1
        })
    elif model_size_b <= 72:
        requirements.update({
            'min_vram_gb': 140,
            'recommended_vram_gb': 160,
            'example_gpus': ['2x A100 80GB', '2x H100 80GB'],
            'tensor_parallel_recommended': 2
        })
    else:
        requirements.update({
            'min_vram_gb': model_size_b * 2,
            'recommended_vram_gb': model_size_b * 2.5,
            'example_gpus': ['8x H100 80GB with FP8'],
            'tensor_parallel_recommended': 8
        })
    
    return requirements
