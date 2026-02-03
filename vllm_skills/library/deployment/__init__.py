"""
vLLM Deployment Assistant

A comprehensive deployment assistant that helps guide users through vLLM deployment,
including environment detection, configuration optimization, recipe integration,
and troubleshooting.
"""

import os
import json
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class DeploymentAssistant:
    """
    Main deployment assistant class that provides methods for:
    - Hardware detection (GPU/CPU/RAM)
    - Environment detection (Python/CUDA/PyTorch/vLLM versions)
    - Recipe lookup
    - Configuration suggestions
    - Command generation
    """
    
    def __init__(self):
        self.skill_dir = Path(__file__).parent
        self.recipe_index = self._load_recipe_index()
        self.hardware_matrix = self._load_hardware_matrix()
    
    def _load_recipe_index(self) -> Dict[str, Any]:
        """Load the recipe index mapping models to deployment guides."""
        recipe_file = self.skill_dir / "models" / "recipe_index.yaml"
        if recipe_file.exists():
            import yaml
            with open(recipe_file) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_hardware_matrix(self) -> Dict[str, Any]:
        """Load the hardware requirements matrix."""
        matrix_file = self.skill_dir / "models" / "hardware_matrix.yaml"
        if matrix_file.exists():
            import yaml
            with open(matrix_file) as f:
                return yaml.safe_load(f)
        return {}
    
    def check_hardware(self) -> Dict[str, Any]:
        """
        Detect hardware configuration including GPUs, VRAM, CPU, and RAM.
        
        Returns:
            Dictionary containing hardware information:
            {
                'gpus': [{'name': str, 'memory_gb': int, 'index': int}, ...],
                'total_vram_gb': int,
                'gpu_count': int,
                'cpu_cores': int,
                'ram_gb': int,
                'platform': str  # 'nvidia', 'amd', or 'cpu'
            }
        """
        result = {
            'gpus': [],
            'total_vram_gb': 0,
            'gpu_count': 0,
            'cpu_cores': 0,
            'ram_gb': 0,
            'platform': 'cpu'
        }
        
        # Try NVIDIA GPUs first
        try:
            nvidia_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            
            for line in nvidia_output.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_info = {
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_gb': int(float(parts[2]) / 1024)  # Convert MB to GB
                        }
                        result['gpus'].append(gpu_info)
                        result['total_vram_gb'] += gpu_info['memory_gb']
            
            if result['gpus']:
                result['gpu_count'] = len(result['gpus'])
                result['platform'] = 'nvidia'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try AMD ROCm GPUs if no NVIDIA GPUs found
        if not result['gpus']:
            try:
                rocm_output = subprocess.check_output(
                    ['rocm-smi', '--showproductname', '--showmeminfo', 'vram'],
                    stderr=subprocess.DEVNULL,
                    universal_newlines=True
                )
                # Basic ROCm detection (parsing would need refinement for production)
                if 'GPU' in rocm_output:
                    result['platform'] = 'amd'
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # CPU and RAM detection
        try:
            import psutil
            result['cpu_cores'] = psutil.cpu_count(logical=False) or 0
            result['ram_gb'] = int(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            # Fallback if psutil not available
            try:
                result['cpu_cores'] = os.cpu_count() or 0
            except:
                pass
            
            # Try to get RAM on Linux
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            result['ram_gb'] = int(kb / (1024**2))
                            break
            except:
                pass
        
        return result
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Detect software environment including Python, PyTorch, CUDA, vLLM versions.
        
        Returns:
            Dictionary containing environment information:
            {
                'python_version': str,
                'pytorch_version': str or None,
                'pytorch_cuda_version': str or None,
                'cuda_version': str or None,
                'vllm_version': str or None,
                'flash_attn_version': str or None,
                'triton_version': str or None
            }
        """
        result = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'pytorch_version': None,
            'pytorch_cuda_version': None,
            'cuda_version': None,
            'vllm_version': None,
            'flash_attn_version': None,
            'triton_version': None
        }
        
        # Check PyTorch
        try:
            import torch
            result['pytorch_version'] = torch.__version__
            if torch.cuda.is_available():
                result['pytorch_cuda_version'] = torch.version.cuda
        except ImportError:
            pass
        
        # Check CUDA version from nvidia-smi
        try:
            cuda_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            # Get CUDA runtime version
            import torch.version
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                result['cuda_version'] = torch.version.cuda
        except:
            pass
        
        # Check vLLM
        try:
            import vllm
            result['vllm_version'] = vllm.__version__
        except ImportError:
            pass
        
        # Check Flash Attention
        try:
            import flash_attn
            result['flash_attn_version'] = flash_attn.__version__
        except ImportError:
            pass
        
        # Check Triton
        try:
            import triton
            result['triton_version'] = triton.__version__
        except ImportError:
            pass
        
        return result
    
    def find_recipe(self, model_name: str) -> Optional[Dict[str, str]]:
        """
        Find deployment recipe for a given model.
        
        Args:
            model_name: Model name or HuggingFace identifier
            
        Returns:
            Dictionary with recipe information:
            {
                'model': str,
                'recipe_url': str,
                'recipe_path': str,
                'description': str
            }
            or None if no recipe found
        """
        if not self.recipe_index:
            return None
        
        # Normalize model name for matching
        model_lower = model_name.lower()
        
        # Check direct matches first
        for recipe in self.recipe_index.get('recipes', []):
            model_patterns = recipe.get('models', [])
            for pattern in model_patterns:
                if pattern.lower() in model_lower or model_lower in pattern.lower():
                    return {
                        'model': recipe.get('name', ''),
                        'recipe_url': recipe.get('url', ''),
                        'recipe_path': recipe.get('path', ''),
                        'description': recipe.get('description', '')
                    }
        
        return None
    
    def suggest_config(
        self,
        model_name: str,
        hardware: Optional[Dict[str, Any]] = None,
        use_case: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Suggest optimal configuration based on model and hardware.
        
        Args:
            model_name: Model name or HuggingFace identifier
            hardware: Hardware info from check_hardware() (auto-detected if None)
            use_case: 'throughput', 'latency', or 'balanced'
            
        Returns:
            Dictionary with configuration parameters
        """
        if hardware is None:
            hardware = self.check_hardware()
        
        # Estimate model size from name
        model_size = self._estimate_model_size(model_name)
        
        config = {
            'model_name': model_name,
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.90,
            'max_model_len': None,  # Use model default
            'max_num_seqs': 256,
            'dtype': 'auto',
            'quantization': None,
            'enable_prefix_caching': False,
            'enable_chunked_prefill': False,
        }
        
        # Adjust tensor parallel size based on model size and available GPUs
        if model_size and hardware['gpu_count'] > 0:
            vram_per_gpu = hardware['total_vram_gb'] / hardware['gpu_count'] if hardware['gpu_count'] > 0 else 0
            required_vram = self._estimate_vram_requirement(model_size)
            
            if required_vram > vram_per_gpu:
                # Need tensor parallelism
                config['tensor_parallel_size'] = min(
                    hardware['gpu_count'],
                    (required_vram // vram_per_gpu) + 1
                )
        
        # Adjust based on use case
        if use_case == 'throughput':
            config['max_num_seqs'] = 512
            config['enable_prefix_caching'] = True
        elif use_case == 'latency':
            config['max_num_seqs'] = 64
            config['enable_chunked_prefill'] = True
        
        # Check for MoE models
        if self._is_moe_model(model_name):
            config['enable_expert_parallel'] = True
            if model_size and model_size > 100:  # Large MoE models
                config['quantization'] = 'fp8'
                config['kv_cache_dtype'] = 'fp8'
        
        return config
    
    def _estimate_model_size(self, model_name: str) -> Optional[int]:
        """Estimate model size in billions of parameters from name."""
        import re
        
        # Look for patterns like "7B", "70B", "8x7B", etc.
        patterns = [
            r'(\d+)B',  # 7B, 70B
            r'(\d+)x(\d+)B',  # 8x7B (MoE)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_name, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return int(groups[0])
                elif len(groups) == 2:
                    # For MoE, use total size (simplified)
                    return int(groups[0]) * int(groups[1])
        
        return None
    
    def _estimate_vram_requirement(self, model_size_b: int) -> int:
        """Estimate VRAM requirement in GB for model size in billions of parameters."""
        # Rough estimate: 2 bytes per parameter (FP16) + overhead
        return int(model_size_b * 2 * 1.2)  # 20% overhead
    
    def _is_moe_model(self, model_name: str) -> bool:
        """Check if model is a Mixture-of-Experts model."""
        moe_patterns = ['mixtral', 'deepseek-v', 'qwen-moe', 'switch']
        return any(pattern in model_name.lower() for pattern in moe_patterns)
    
    def generate_command(self, config: Dict[str, Any]) -> str:
        """
        Generate vllm serve command from configuration.
        
        Args:
            config: Configuration dictionary from suggest_config()
            
        Returns:
            Complete vllm serve command string
        """
        parts = ['vllm serve', config['model_name']]
        
        # Add tensor parallel
        if config.get('tensor_parallel_size', 1) > 1:
            parts.append(f"--tensor-parallel-size {config['tensor_parallel_size']}")
        
        # Add memory utilization
        if config.get('gpu_memory_utilization'):
            parts.append(f"--gpu-memory-utilization {config['gpu_memory_utilization']}")
        
        # Add max model length
        if config.get('max_model_len'):
            parts.append(f"--max-model-len {config['max_model_len']}")
        
        # Add max num seqs
        if config.get('max_num_seqs'):
            parts.append(f"--max-num-seqs {config['max_num_seqs']}")
        
        # Add dtype
        if config.get('dtype') and config['dtype'] != 'auto':
            parts.append(f"--dtype {config['dtype']}")
        
        # Add quantization
        if config.get('quantization'):
            parts.append(f"--quantization {config['quantization']}")
        
        # Add KV cache dtype
        if config.get('kv_cache_dtype'):
            parts.append(f"--kv-cache-dtype {config['kv_cache_dtype']}")
        
        # Add prefix caching
        if config.get('enable_prefix_caching'):
            parts.append('--enable-prefix-caching')
        
        # Add chunked prefill
        if config.get('enable_chunked_prefill'):
            parts.append('--enable-chunked-prefill')
        
        # Add expert parallel
        if config.get('enable_expert_parallel'):
            parts.append('--enable-expert-parallel')
        
        # Add enforce eager
        if config.get('enforce_eager'):
            parts.append('--enforce-eager')
        
        return ' \\\n  '.join(parts)
    
    def get_troubleshooting_guide(self, error_type: str) -> Optional[str]:
        """
        Get troubleshooting guide for common error types.
        
        Args:
            error_type: Type of error ('oom', 'cuda', 'flash_attn', 'moe', 'kernel')
            
        Returns:
            Path to troubleshooting markdown file or None
        """
        guides = {
            'oom': 'memory_issues.md',
            'memory': 'memory_issues.md',
            'cuda': 'cuda_errors.md',
            'flash_attn': 'common_issues.md',
            'moe': 'common_issues.md',
            'kernel': 'kernel_issues.md',
        }
        
        filename = guides.get(error_type.lower())
        if filename:
            guide_path = self.skill_dir / 'troubleshooting' / filename
            if guide_path.exists():
                return str(guide_path)
        
        return None


# Convenience functions for direct use
def check_hardware() -> Dict[str, Any]:
    """Check hardware configuration."""
    assistant = DeploymentAssistant()
    return assistant.check_hardware()


def check_environment() -> Dict[str, Any]:
    """Check software environment."""
    assistant = DeploymentAssistant()
    return assistant.check_environment()


def find_recipe(model_name: str) -> Optional[Dict[str, str]]:
    """Find deployment recipe for a model."""
    assistant = DeploymentAssistant()
    return assistant.find_recipe(model_name)


def suggest_config(model_name: str, hardware: Optional[Dict[str, Any]] = None, use_case: str = 'balanced') -> Dict[str, Any]:
    """Suggest optimal configuration."""
    assistant = DeploymentAssistant()
    return assistant.suggest_config(model_name, hardware, use_case)


def generate_command(config: Dict[str, Any]) -> str:
    """Generate vllm serve command."""
    assistant = DeploymentAssistant()
    return assistant.generate_command(config)


__all__ = [
    'DeploymentAssistant',
    'check_hardware',
    'check_environment',
    'find_recipe',
    'suggest_config',
    'generate_command',
]
