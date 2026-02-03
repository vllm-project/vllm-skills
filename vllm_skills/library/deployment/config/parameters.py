"""Configuration parameters for vLLM deployment"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeploymentConfig:
    """vLLM deployment configuration"""

    # Model settings
    model: str
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    
    # Performance settings
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    # Serving settings
    host: str = "0.0.0.0"
    port: int = 8000
    max_num_seqs: int = 256
    
    # Advanced settings
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    disable_custom_all_reduce: bool = False
    
    def to_command(self) -> str:
        """
        Generate vLLM server command from configuration
        
        Returns:
            Command string to launch vLLM server
        """
        cmd_parts = [
            "vllm serve",
            self.model,
            f"--dtype {self.dtype}",
            f"--host {self.host}",
            f"--port {self.port}",
            f"--tensor-parallel-size {self.tensor_parallel_size}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--max-num-seqs {self.max_num_seqs}",
        ]
        
        if self.max_model_len:
            cmd_parts.append(f"--max-model-len {self.max_model_len}")
        
        if self.pipeline_parallel_size > 1:
            cmd_parts.append(f"--pipeline-parallel-size {self.pipeline_parallel_size}")
        
        if self.enable_chunked_prefill:
            cmd_parts.append("--enable-chunked-prefill")
        
        if self.enable_prefix_caching:
            cmd_parts.append("--enable-prefix-caching")
        
        if self.disable_custom_all_reduce:
            cmd_parts.append("--disable-custom-all-reduce")
        
        return " \\\n  ".join(cmd_parts)


def validate_config(config: DeploymentConfig) -> list[str]:
    """
    Validate deployment configuration
    
    Args:
        config: Configuration to validate
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not config.model:
        errors.append("Model name is required")
    
    if config.gpu_memory_utilization <= 0 or config.gpu_memory_utilization > 1:
        errors.append("GPU memory utilization must be between 0 and 1")
    
    if config.tensor_parallel_size < 1:
        errors.append("Tensor parallel size must be at least 1")
    
    if config.max_num_seqs < 1:
        errors.append("Max num seqs must be at least 1")
    
    return errors
