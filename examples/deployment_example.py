#!/usr/bin/env python3
"""
Example usage of the vLLM Deployment Assistant Skill

This script demonstrates how to use the deployment assistant to:
1. Check hardware and environment
2. Find recipes for models
3. Generate deployment configurations
4. Create deployment commands
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import the skill
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_skills.library.deployment import DeploymentAssistant


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    """Run example usage of the deployment assistant."""
    
    assistant = DeploymentAssistant()
    
    # 1. Check Hardware
    print_section("Hardware Detection")
    hardware = assistant.check_hardware()
    print(f"Platform: {hardware['platform']}")
    print(f"GPUs Found: {hardware['gpu_count']}")
    for gpu in hardware['gpus']:
        print(f"  - {gpu['name']}: {gpu['memory_gb']}GB")
    print(f"Total VRAM: {hardware['total_vram_gb']}GB")
    print(f"CPU Cores: {hardware['cpu_cores']}")
    print(f"System RAM: {hardware['ram_gb']}GB")
    
    # 2. Check Environment
    print_section("Environment Detection")
    env = assistant.check_environment()
    print(f"Python: {env['python_version']}")
    print(f"PyTorch: {env['pytorch_version']}")
    print(f"CUDA: {env['pytorch_cuda_version'] or env['cuda_version'] or 'Not detected'}")
    print(f"vLLM: {env['vllm_version'] or 'Not installed'}")
    print(f"Flash Attention: {env['flash_attn_version'] or 'Not installed'}")
    print(f"Triton: {env['triton_version'] or 'Not installed'}")
    
    # 3. Find Recipe for Popular Models
    print_section("Recipe Lookup Examples")
    
    test_models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    
    for model_name in test_models:
        recipe = assistant.find_recipe(model_name)
        if recipe:
            print(f"\n✓ {model_name}")
            print(f"  Recipe: {recipe['recipe_url']}")
        else:
            print(f"\n✗ {model_name}")
            print(f"  No recipe found (check supported models)")
    
    # 4. Generate Configuration
    print_section("Configuration Generation")
    
    # Example 1: Small model, balanced
    model = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"\nModel: {model}")
    print("Use case: Balanced")
    config = assistant.suggest_config(model, hardware, use_case='balanced')
    print(json.dumps(config, indent=2))
    
    # Example 2: Generate command
    print_section("Generated Deployment Command")
    command = assistant.generate_command(config)
    print(command)
    
    # 5. Different Use Cases
    print_section("Configuration Presets")
    
    for use_case in ['throughput', 'latency', 'balanced']:
        print(f"\n{use_case.upper()} Configuration:")
        config = assistant.suggest_config(model, hardware, use_case=use_case)
        print(f"  max_num_seqs: {config['max_num_seqs']}")
        print(f"  gpu_memory_utilization: {config['gpu_memory_utilization']}")
        print(f"  enable_prefix_caching: {config.get('enable_prefix_caching', False)}")
        print(f"  enable_chunked_prefill: {config.get('enable_chunked_prefill', False)}")
    
    # 6. Troubleshooting Guide Access
    print_section("Troubleshooting Guides")
    
    error_types = ['oom', 'cuda', 'flash_attn', 'kernel']
    for error_type in error_types:
        guide_path = assistant.get_troubleshooting_guide(error_type)
        if guide_path:
            print(f"✓ {error_type}: {guide_path}")
        else:
            print(f"✗ {error_type}: No guide found")
    
    print_section("Example Complete!")
    print("For more information, see SKILL.md in the deployment directory\n")


if __name__ == "__main__":
    main()
