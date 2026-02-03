"""Example: Deployment Assistant Demo

This example demonstrates the deployment assistant skill capabilities.
"""

import asyncio


async def main():
    """Run deployment assistant demo"""
    from vllm_skills.library.deployment import DeploymentAssistant

    print("=" * 70)
    print("vLLM Deployment Assistant - Demo")
    print("=" * 70)

    # Initialize the assistant
    assistant = DeploymentAssistant()
    print(f"\nSkill: {assistant.metadata.name} v{assistant.metadata.version}")
    print(f"Description: {assistant.metadata.description}")

    # Demo 1: Check hardware and environment
    print("\n" + "-" * 70)
    print("Demo 1: System Check")
    print("-" * 70)

    hardware = assistant.check_hardware()
    environment = assistant.check_environment()

    print("\nHardware:")
    print(f"  CPU Cores: {hardware.cpu_count}")
    print(f"  RAM: {hardware.ram_total} GB total, {hardware.ram_available} GB available")

    if hardware.gpu:
        print(f"  GPU: {hardware.gpu.name}")
        print(f"  GPU Memory: {hardware.gpu.memory_total} MB")
        print(f"  GPU Count: {hardware.gpu.count}")
        print(f"  CUDA Driver: {hardware.gpu.cuda_version}")
    else:
        print("  GPU: Not detected")

    print("\nEnvironment:")
    print(f"  Python: {environment.python_version}")
    print(f"  PyTorch: {environment.pytorch_version or 'Not installed'}")
    print(f"  CUDA Available: {environment.cuda_available}")
    print(f"  vLLM: {environment.vllm_version or 'Not installed'}")
    print(f"  Flash Attention: {environment.flash_attention_version or 'Not installed'}")

    # Demo 2: Find recipe for a model
    print("\n" + "-" * 70)
    print("Demo 2: Recipe Search")
    print("-" * 70)

    models_to_check = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen3-7B",
    ]

    for model in models_to_check:
        recipe = assistant.find_recipe(model)
        if recipe:
            print(f"\n  Model: {model}")
            print(f"  Recipe: {recipe.get('recipes', ['N/A'])[0]}")
            print(f"  Description: {recipe.get('description', 'N/A')}")
        else:
            print(f"\n  Model: {model}")
            print("  Recipe: Not found")

    # Demo 3: Generate configuration
    print("\n" + "-" * 70)
    print("Demo 3: Configuration Generation")
    print("-" * 70)

    test_model = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"\nModel: {test_model}")

    # Try different presets
    presets = ["high_throughput", "low_latency", "memory_constrained"]

    for preset in presets:
        print(f"\n  Preset: {preset}")
        config = assistant.suggest_config(test_model, hardware, preset)
        print(f"    Max Sequences: {config.max_num_seqs}")
        print(f"    GPU Memory Util: {config.gpu_memory_utilization}")
        print(f"    Tensor Parallel: {config.tensor_parallel_size}")
        print(f"    Prefix Caching: {config.enable_prefix_caching}")
        print(f"    Chunked Prefill: {config.enable_chunked_prefill}")

    # Demo 4: Generate deployment command
    print("\n" + "-" * 70)
    print("Demo 4: Deployment Command")
    print("-" * 70)

    config = assistant.suggest_config(test_model, hardware, "high_throughput")
    command = assistant.generate_command(config)

    print(f"\nModel: {test_model}")
    print("Preset: high_throughput")
    print("\nGenerated Command:")
    print("-" * 70)
    print(command)
    print("-" * 70)

    # Demo 5: Complete configuration workflow
    print("\n" + "-" * 70)
    print("Demo 5: Complete Configuration Workflow")
    print("-" * 70)

    result = assistant.configure_model(test_model, "low_latency")

    print(f"\nModel: {result['model']}")
    print(f"GPU: {result['hardware']['gpu']}")
    print(f"GPU Memory: {result['hardware']['gpu_memory']} MB")

    compatibility = result['compatibility']
    print(f"\nCompatibility: {'✓ Compatible' if compatibility['compatible'] else '✗ Incompatible'}")

    if compatibility['issues']:
        print("Issues:")
        for issue in compatibility['issues']:
            print(f"  - {issue}")

    if compatibility['recommendations']:
        print("Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"  - {rec}")

    print(f"\nRecipe: {result['recipe'] or 'Not found'}")

    print("\nConfiguration:")
    for key, value in result['configuration'].items():
        print(f"  {key}: {value}")

    # Demo 6: Troubleshooting resources
    print("\n" + "-" * 70)
    print("Demo 6: Troubleshooting Resources")
    print("-" * 70)

    troubleshooting = assistant.get_troubleshooting_info()

    print("\nAvailable Guides:")
    for name in troubleshooting['guides']:
        print(f"  - {name}")

    print("\nOfficial Resources:")
    for name, url in troubleshooting['resources'].items():
        print(f"  - {name}: {url}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
