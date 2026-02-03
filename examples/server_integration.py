"""Example: Server integration with vLLM

This example shows how to mount vllm-skills alongside a vLLM server.
"""

import asyncio

from vllm_skills.core.registry import SkillRegistry


async def main():
    """Run example server integration"""
    print("=" * 60)
    print("vLLM Skills - Server Integration Example")
    print("=" * 60)

    # Initialize skill registry
    print("\n1. Initializing skill registry...")
    registry = SkillRegistry()

    # Load deployment skill
    print("2. Loading deployment skill...")
    try:
        registry.load_skills("deployment")
        print("   ✓ Deployment skill loaded")
    except Exception as e:
        print(f"   ✗ Error loading skill: {e}")
        return

    # Get the skill
    deployment_skill = registry.get("deployment")
    if deployment_skill:
        print(f"   Skill: {deployment_skill.metadata.name}")
        print(f"   Version: {deployment_skill.metadata.version}")
        print(f"   Description: {deployment_skill.metadata.description}")

    # Execute skill to check environment
    print("\n3. Checking environment with skill...")
    result = await deployment_skill.execute(context={}, action="check")

    print("\n4. Environment Information:")
    print("-" * 60)

    # Display hardware info
    if "hardware" in result:
        hw = result["hardware"]
        print(f"CPU Cores: {hw.get('cpu_count', 'N/A')}")
        print(f"RAM: {hw.get('ram_total_gb', 'N/A')} GB total, "
              f"{hw.get('ram_available_gb', 'N/A')} GB available")

        if hw.get("gpu"):
            gpu = hw["gpu"]
            print(f"GPU: {gpu.get('name', 'N/A')}")
            print(f"GPU Memory: {gpu.get('memory_total_mb', 'N/A')} MB total, "
                  f"{gpu.get('memory_free_mb', 'N/A')} MB free")
            print(f"GPU Count: {gpu.get('count', 0)}")
            print(f"CUDA Version: {gpu.get('cuda_version', 'N/A')}")
        else:
            print("GPU: Not detected")

    # Display environment info
    if "environment" in result:
        env = result["environment"]
        print(f"\nPython: {env.get('python_version', 'N/A')}")
        print(f"PyTorch: {env.get('pytorch_version', 'N/A')}")
        print(f"CUDA Available: {env.get('cuda_available', False)}")
        print(f"vLLM: {env.get('vllm_version', 'Not installed')}")
        print(f"Flash Attention: {env.get('flash_attention_version', 'Not installed')}")

    print("\n" + "=" * 60)
    print("Server integration example complete!")
    print("=" * 60)

    # In a real server integration, you would:
    # 1. Start vLLM server
    # 2. Register skills with the server
    # 3. Handle skill execution in request handlers
    # 4. Return skill results to clients


if __name__ == "__main__":
    asyncio.run(main())
