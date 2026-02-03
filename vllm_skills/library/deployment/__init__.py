"""vLLM Deployment Assistant Skill"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from vllm_skills.core.base import BaseSkill, SkillMetadata
from vllm_skills.library.deployment.checks.compatibility import (
    check_model_compatibility,
)
from vllm_skills.library.deployment.checks.environment import (
    EnvironmentInfo,
    check_environment,
)
from vllm_skills.library.deployment.checks.hardware import (
    HardwareInfo,
    check_hardware,
)
from vllm_skills.library.deployment.config.parameters import (
    DeploymentConfig,
    validate_config,
)


class DeploymentAssistant(BaseSkill):
    """Intelligent vLLM deployment assistant skill"""

    def __init__(self):
        self._recipe_index = None
        self._hardware_matrix = None
        self._load_resources()

    def _load_resources(self):
        """Load recipe index and hardware matrix"""
        resources_dir = Path(__file__).parent / "models"

        # Load recipe index
        recipe_file = resources_dir / "recipe_index.yaml"
        if recipe_file.exists():
            with open(recipe_file) as f:
                self._recipe_index = yaml.safe_load(f)

        # Load hardware matrix
        hardware_file = resources_dir / "hardware_matrix.yaml"
        if hardware_file.exists():
            with open(hardware_file) as f:
                self._hardware_matrix = yaml.safe_load(f)

    @property
    def metadata(self) -> SkillMetadata:
        """Return skill metadata"""
        return SkillMetadata(
            name="deployment",
            description="Intelligent vLLM deployment assistant",
            version="0.1.0",
            author="vLLM Project",
            tags=["vllm", "deployment", "configuration", "troubleshooting"],
            requires=["psutil", "pyyaml"],
        )

    async def execute(self, context: dict, **kwargs: Any) -> Any:
        """
        Execute deployment assistant

        Args:
            context: Execution context
            **kwargs: Additional parameters
                - action: Action to perform (check, configure, troubleshoot)
                - model_name: Model to deploy
                - preset: Configuration preset name

        Returns:
            Result of the action
        """
        action = kwargs.get("action", "check")

        if action == "check":
            return self.check_all()
        elif action == "configure":
            model_name = kwargs.get("model_name")
            if not model_name:
                return {"error": "model_name is required for configure action"}
            return self.configure_model(model_name, kwargs.get("preset"))
        elif action == "troubleshoot":
            return self.get_troubleshooting_info()
        else:
            return {"error": f"Unknown action: {action}"}

    def check_hardware(self) -> HardwareInfo:
        """
        Check hardware information

        Returns:
            HardwareInfo object
        """
        return check_hardware()

    def check_environment(self) -> EnvironmentInfo:
        """
        Check environment information

        Returns:
            EnvironmentInfo object
        """
        return check_environment()

    def check_all(self) -> dict[str, Any]:
        """
        Check both hardware and environment

        Returns:
            Dictionary with hardware and environment information
        """
        hardware = self.check_hardware()
        environment = self.check_environment()

        return {
            "hardware": {
                "cpu_count": hardware.cpu_count,
                "ram_total_gb": hardware.ram_total,
                "ram_available_gb": hardware.ram_available,
                "gpu": {
                    "name": hardware.gpu.name if hardware.gpu else None,
                    "memory_total_mb": (
                        hardware.gpu.memory_total if hardware.gpu else None
                    ),
                    "memory_free_mb": (
                        hardware.gpu.memory_free if hardware.gpu else None
                    ),
                    "cuda_version": hardware.gpu.cuda_version if hardware.gpu else None,
                    "count": hardware.gpu.count if hardware.gpu else 0,
                }
                if hardware.gpu
                else None,
            },
            "environment": {
                "python_version": environment.python_version,
                "pytorch_version": environment.pytorch_version,
                "cuda_available": environment.cuda_available,
                "cuda_version": environment.cuda_version,
                "vllm_version": environment.vllm_version,
                "flash_attention_version": environment.flash_attention_version,
            },
        }

    def find_recipe(self, model_name: str) -> Optional[dict[str, Any]]:
        """
        Find recipe for a model

        Args:
            model_name: Model name or path

        Returns:
            Recipe information or None
        """
        if not self._recipe_index:
            return None

        models = self._recipe_index.get("models", [])

        # Normalize model name for comparison
        model_lower = model_name.lower()

        for model_info in models:
            repo = model_info.get("repository", "").lower()
            name = model_info.get("name", "").lower()

            # Check if model name matches
            if name in model_lower or any(
                part in model_lower for part in repo.split("/")
            ):
                return model_info

        return None

    def suggest_config(
        self, model_name: str, hardware: Optional[HardwareInfo] = None, preset: Optional[str] = None
    ) -> DeploymentConfig:
        """
        Suggest optimal configuration for a model

        Args:
            model_name: Model name or path
            hardware: Hardware information (auto-detected if not provided)
            preset: Configuration preset (high_throughput, low_latency, memory_constrained)

        Returns:
            DeploymentConfig object
        """
        if hardware is None:
            hardware = self.check_hardware()

        # Start with base configuration
        config = DeploymentConfig(model=model_name)

        # Load preset if specified
        if preset:
            config = self._apply_preset(config, preset)

        # Adjust based on hardware
        if hardware.gpu:
            # Suggest tensor parallelism for multi-GPU
            if hardware.gpu.count > 1:
                # Use all GPUs for large models
                if "70b" in model_name.lower() or "72b" in model_name.lower():
                    config.tensor_parallel_size = min(hardware.gpu.count, 8)
                elif "13b" in model_name.lower():
                    config.tensor_parallel_size = min(hardware.gpu.count, 2)

            # Adjust memory utilization based on available memory
            if hardware.gpu.memory_total < 16000:  # Less than 16GB
                config.gpu_memory_utilization = 0.75
            elif hardware.gpu.memory_total < 24000:  # Less than 24GB
                config.gpu_memory_utilization = 0.85

        return config

    def _apply_preset(
        self, config: DeploymentConfig, preset: str
    ) -> DeploymentConfig:
        """Apply configuration preset"""
        preset_dir = Path(__file__).parent / "config" / "presets"
        preset_file = preset_dir / f"{preset}.yaml"

        if not preset_file.exists():
            return config

        with open(preset_file) as f:
            preset_data = yaml.safe_load(f)

        settings = preset_data.get("settings", {})

        # Apply preset settings
        if "max_num_seqs" in settings:
            config.max_num_seqs = settings["max_num_seqs"]
        if "gpu_memory_utilization" in settings:
            config.gpu_memory_utilization = settings["gpu_memory_utilization"]
        if "enable_chunked_prefill" in settings:
            config.enable_chunked_prefill = settings["enable_chunked_prefill"]
        if "enable_prefix_caching" in settings:
            config.enable_prefix_caching = settings["enable_prefix_caching"]
        if "max_model_len" in settings:
            config.max_model_len = settings["max_model_len"]

        return config

    def generate_command(self, config: DeploymentConfig) -> str:
        """
        Generate vLLM deployment command

        Args:
            config: Deployment configuration

        Returns:
            Command string
        """
        return config.to_command()

    def configure_model(
        self, model_name: str, preset: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Complete configuration workflow for a model

        Args:
            model_name: Model name
            preset: Optional preset name

        Returns:
            Complete configuration information
        """
        # Check hardware and environment
        hardware = self.check_hardware()
        environment = self.check_environment()

        # Check compatibility
        compatibility = check_model_compatibility(model_name, hardware, environment)

        # Find recipe
        recipe = self.find_recipe(model_name)

        # Generate configuration
        config = self.suggest_config(model_name, hardware, preset)

        # Validate configuration
        config_errors = validate_config(config)

        # Generate command
        command = self.generate_command(config)

        return {
            "model": model_name,
            "hardware": {
                "gpu": hardware.gpu.name if hardware.gpu else "No GPU",
                "gpu_memory": hardware.gpu.memory_total if hardware.gpu else 0,
                "gpu_count": hardware.gpu.count if hardware.gpu else 0,
            },
            "compatibility": compatibility,
            "recipe": recipe.get("recipes")[0] if recipe else None,
            "configuration": {
                "tensor_parallel_size": config.tensor_parallel_size,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_num_seqs": config.max_num_seqs,
                "max_model_len": config.max_model_len,
            },
            "command": command,
            "errors": config_errors,
        }

    def get_troubleshooting_info(self) -> dict[str, Any]:
        """
        Get troubleshooting information and common issues

        Returns:
            Dictionary with troubleshooting resources
        """
        troubleshooting_dir = Path(__file__).parent / "troubleshooting"

        guides = {
            "common_issues": troubleshooting_dir / "common_issues.md",
            "cuda_errors": troubleshooting_dir / "cuda_errors.md",
            "memory_issues": troubleshooting_dir / "memory_issues.md",
            "kernel_issues": troubleshooting_dir / "kernel_issues.md",
        }

        available_guides = {
            name: str(path) for name, path in guides.items() if path.exists()
        }

        return {
            "guides": available_guides,
            "resources": {
                "documentation": "https://docs.vllm.ai/",
                "github": "https://github.com/vllm-project/vllm",
                "discord": "https://discord.gg/vllm",
            },
        }
