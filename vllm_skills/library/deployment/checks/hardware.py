"""Hardware detection module for vLLM deployment assistant."""

import subprocess
from typing import Dict, List, Optional


def detect_gpus() -> List[Dict[str, any]]:
    """
    Detect available GPUs and their specifications.
    
    Returns:
        List of GPU information dictionaries
    """
    gpus = []
    
    # Try NVIDIA GPUs
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', 
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        
        for line in output.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_gb': int(float(parts[2]) / 1024),
                        'compute_capability': parts[3],
                        'platform': 'nvidia'
                    })
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try AMD GPUs if no NVIDIA found
    if not gpus:
        try:
            output = subprocess.check_output(
                ['rocm-smi', '--showproductname'],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            if 'GPU' in output:
                # Basic AMD detection (would need more parsing for full info)
                gpus.append({
                    'index': 0,
                    'name': 'AMD GPU',
                    'memory_gb': 0,  # Would need additional parsing
                    'compute_capability': 'gfx906+',
                    'platform': 'amd'
                })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return gpus


def detect_cpu() -> Dict[str, any]:
    """
    Detect CPU specifications.
    
    Returns:
        Dictionary with CPU information
    """
    import os
    
    cpu_info = {
        'cores': 0,
        'model': 'Unknown'
    }
    
    # Get core count
    try:
        import psutil
        cpu_info['cores'] = psutil.cpu_count(logical=False) or 0
    except ImportError:
        cpu_info['cores'] = os.cpu_count() or 0
    
    # Try to get CPU model on Linux
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_info['model'] = line.split(':')[1].strip()
                    break
    except:
        pass
    
    return cpu_info


def detect_memory() -> int:
    """
    Detect system RAM in GB.
    
    Returns:
        Total RAM in GB
    """
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        # Fallback for Linux
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return int(kb / (1024**2))
        except:
            pass
    
    return 0


def check_hardware() -> Dict[str, any]:
    """
    Comprehensive hardware detection.
    
    Returns:
        Dictionary containing all hardware information
    """
    gpus = detect_gpus()
    
    return {
        'gpus': gpus,
        'gpu_count': len(gpus),
        'total_vram_gb': sum(gpu.get('memory_gb', 0) for gpu in gpus),
        'platform': gpus[0]['platform'] if gpus else 'cpu',
        'cpu': detect_cpu(),
        'ram_gb': detect_memory()
    }
