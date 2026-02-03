"""Parameter configuration module for vLLM deployment assistant."""

from typing import Dict, List, Optional, Any


def get_parameter_description(param_name: str) -> Dict[str, Any]:
    """
    Get detailed description and validation for a parameter.
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        Dictionary with parameter details
    """
    parameters = {
        'model_name': {
            'type': 'string',
            'required': True,
            'description': 'HuggingFace model ID or local path',
            'example': 'meta-llama/Llama-3.1-8B-Instruct',
            'validation': lambda x: isinstance(x, str) and len(x) > 0
        },
        'tensor_parallel_size': {
            'type': 'integer',
            'required': False,
            'default': 1,
            'description': 'Number of GPUs for tensor parallelism',
            'example': 2,
            'validation': lambda x: isinstance(x, int) and x > 0 and (x & (x - 1)) == 0  # Power of 2
        },
        'gpu_memory_utilization': {
            'type': 'float',
            'required': False,
            'default': 0.90,
            'description': 'Fraction of GPU memory to use (0.0-1.0)',
            'example': 0.85,
            'validation': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0
        },
        'max_model_len': {
            'type': 'integer',
            'required': False,
            'default': None,
            'description': 'Maximum sequence length (context window)',
            'example': 8192,
            'validation': lambda x: x is None or (isinstance(x, int) and x > 0)
        },
        'max_num_seqs': {
            'type': 'integer',
            'required': False,
            'default': 256,
            'description': 'Maximum number of concurrent sequences',
            'example': 128,
            'validation': lambda x: isinstance(x, int) and x > 0
        },
        'quantization': {
            'type': 'string',
            'required': False,
            'default': None,
            'description': 'Quantization method',
            'example': 'fp8',
            'options': ['awq', 'gptq', 'squeezellm', 'fp8', 'bitsandbytes'],
            'validation': lambda x: x is None or x in ['awq', 'gptq', 'squeezellm', 'fp8', 'bitsandbytes']
        },
        'dtype': {
            'type': 'string',
            'required': False,
            'default': 'auto',
            'description': 'Data type for model weights',
            'example': 'bfloat16',
            'options': ['auto', 'float16', 'bfloat16', 'float32'],
            'validation': lambda x: x in ['auto', 'float16', 'bfloat16', 'float32']
        },
        'kv_cache_dtype': {
            'type': 'string',
            'required': False,
            'default': 'auto',
            'description': 'Data type for KV cache',
            'example': 'fp8',
            'options': ['auto', 'fp8', 'fp8_e5m2'],
            'validation': lambda x: x in ['auto', 'fp8', 'fp8_e5m2']
        },
        'enable_prefix_caching': {
            'type': 'boolean',
            'required': False,
            'default': False,
            'description': 'Enable automatic prefix caching',
            'example': True,
            'validation': lambda x: isinstance(x, bool)
        },
        'enable_chunked_prefill': {
            'type': 'boolean',
            'required': False,
            'default': False,
            'description': 'Enable chunked prefill for long prompts',
            'example': True,
            'validation': lambda x: isinstance(x, bool)
        },
        'enable_expert_parallel': {
            'type': 'boolean',
            'required': False,
            'default': False,
            'description': 'Enable expert parallelism for MoE models',
            'example': True,
            'validation': lambda x: isinstance(x, bool)
        },
        'enforce_eager': {
            'type': 'boolean',
            'required': False,
            'default': False,
            'description': 'Disable CUDA graphs (for debugging)',
            'example': False,
            'validation': lambda x: isinstance(x, bool)
        }
    }
    
    return parameters.get(param_name, {})


def get_all_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Get all available parameters with descriptions.
    
    Returns:
        Dictionary of all parameters
    """
    param_names = [
        'model_name', 'tensor_parallel_size', 'gpu_memory_utilization',
        'max_model_len', 'max_num_seqs', 'quantization', 'dtype',
        'kv_cache_dtype', 'enable_prefix_caching', 'enable_chunked_prefill',
        'enable_expert_parallel', 'enforce_eager'
    ]
    
    return {name: get_parameter_description(name) for name in param_names}


def validate_configuration(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required parameters
    if 'model_name' not in config:
        errors.append("Required parameter 'model_name' is missing")
    
    # Validate each parameter
    for param_name, value in config.items():
        param_info = get_parameter_description(param_name)
        
        if not param_info:
            errors.append(f"Unknown parameter: {param_name}")
            continue
        
        # Check type and validation
        validator = param_info.get('validation')
        if validator and not validator(value):
            errors.append(
                f"Invalid value for {param_name}: {value} "
                f"(expected {param_info.get('type')})"
            )
    
    return len(errors) == 0, errors


def suggest_interactive_questions(
    current_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate interactive questions to gather configuration.
    
    Args:
        current_config: Partially filled configuration
        
    Returns:
        List of question dictionaries
    """
    current_config = current_config or {}
    questions = []
    
    # Model name (always first)
    if 'model_name' not in current_config:
        questions.append({
            'parameter': 'model_name',
            'question': 'What model would you like to deploy?',
            'type': 'string',
            'hint': 'e.g., meta-llama/Llama-3.1-8B-Instruct',
            'required': True
        })
    
    # Use case (determines other parameters)
    if 'use_case' not in current_config:
        questions.append({
            'parameter': 'use_case',
            'question': 'What is your primary use case?',
            'type': 'choice',
            'options': [
                'throughput - Maximum requests per second (batch processing)',
                'latency - Minimum response time (interactive chat)',
                'balanced - Balance between throughput and latency'
            ],
            'required': False,
            'default': 'balanced'
        })
    
    # Tensor parallel size
    if 'tensor_parallel_size' not in current_config:
        questions.append({
            'parameter': 'tensor_parallel_size',
            'question': 'How many GPUs do you want to use?',
            'type': 'integer',
            'hint': 'Must be a power of 2 (1, 2, 4, 8, etc.)',
            'required': False,
            'default': 1
        })
    
    # Context length
    if 'max_model_len' not in current_config:
        questions.append({
            'parameter': 'max_model_len',
            'question': 'What maximum context length do you need?',
            'type': 'choice',
            'options': [
                '2048 - Short conversations',
                '4096 - Standard chat',
                '8192 - Long conversations',
                '16384 - Documents',
                '32768 - Long documents',
                'auto - Use model default'
            ],
            'required': False,
            'default': 'auto'
        })
    
    return questions
