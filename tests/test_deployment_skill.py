"""Tests for deployment skill"""

import pytest

from vllm_skills.library.deployment import DeploymentAssistant
from vllm_skills.library.deployment.checks.compatibility import (
    check_model_compatibility,
    estimate_model_memory,
)
from vllm_skills.library.deployment.checks.environment import check_environment
from vllm_skills.library.deployment.checks.hardware import check_hardware
from vllm_skills.library.deployment.config.parameters import (
    DeploymentConfig,
    validate_config,
)


class TestDeploymentAssistant:
    """Tests for DeploymentAssistant skill"""

    def test_skill_metadata(self):
        """Test skill metadata"""
        assistant = DeploymentAssistant()
        metadata = assistant.metadata

        assert metadata.name == "deployment"
        assert metadata.version == "0.1.0"
        assert "vllm" in metadata.tags
        assert "deployment" in metadata.tags

    @pytest.mark.asyncio
    async def test_execute_check(self):
        """Test execute with check action"""
        assistant = DeploymentAssistant()
        result = await assistant.execute(context={}, action="check")

        assert "hardware" in result
        assert "environment" in result

    @pytest.mark.asyncio
    async def test_execute_configure(self):
        """Test execute with configure action"""
        assistant = DeploymentAssistant()
        result = await assistant.execute(
            context={},
            action="configure",
            model_name="meta-llama/Llama-3.1-8B-Instruct"
        )

        assert "model" in result
        assert "configuration" in result
        assert "command" in result

    @pytest.mark.asyncio
    async def test_execute_troubleshoot(self):
        """Test execute with troubleshoot action"""
        assistant = DeploymentAssistant()
        result = await assistant.execute(context={}, action="troubleshoot")

        assert "guides" in result
        assert "resources" in result

    def test_check_hardware(self):
        """Test hardware check"""
        assistant = DeploymentAssistant()
        hardware = assistant.check_hardware()

        assert hardware is not None
        assert hardware.cpu_count > 0
        assert hardware.ram_total >= 0

    def test_check_environment(self):
        """Test environment check"""
        assistant = DeploymentAssistant()
        environment = assistant.check_environment()

        assert environment is not None
        assert environment.python_version is not None
        assert "." in environment.python_version  # Has version format

    def test_find_recipe_llama(self):
        """Test finding recipe for Llama model"""
        assistant = DeploymentAssistant()
        recipe = assistant.find_recipe("meta-llama/Llama-3.1-8B-Instruct")

        # May or may not find depending on recipe index
        if recipe:
            assert "recipes" in recipe
            assert recipe["name"] is not None

    def test_find_recipe_nonexistent(self):
        """Test finding recipe for nonexistent model"""
        assistant = DeploymentAssistant()
        recipe = assistant.find_recipe("nonexistent/model")

        assert recipe is None

    def test_suggest_config(self):
        """Test configuration suggestion"""
        assistant = DeploymentAssistant()
        config = assistant.suggest_config("meta-llama/Llama-3.1-8B-Instruct")

        assert config is not None
        assert config.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.tensor_parallel_size >= 1
        assert 0 < config.gpu_memory_utilization <= 1

    def test_suggest_config_with_preset(self):
        """Test configuration with preset"""
        assistant = DeploymentAssistant()
        config = assistant.suggest_config(
            "meta-llama/Llama-3.1-8B-Instruct",
            preset="high_throughput"
        )

        assert config is not None
        # High throughput should have larger batch size
        assert config.max_num_seqs >= 256

    def test_generate_command(self):
        """Test command generation"""
        assistant = DeploymentAssistant()
        config = DeploymentConfig(model="test-model")
        command = assistant.generate_command(config)

        assert "vllm serve" in command
        assert "test-model" in command

    def test_configure_model(self):
        """Test complete model configuration"""
        assistant = DeploymentAssistant()
        result = assistant.configure_model("meta-llama/Llama-3.1-8B-Instruct")

        assert "model" in result
        assert "hardware" in result
        assert "compatibility" in result
        assert "configuration" in result
        assert "command" in result


class TestHardwareDetection:
    """Tests for hardware detection"""

    def test_check_hardware(self):
        """Test hardware check"""
        hardware = check_hardware()

        assert hardware is not None
        assert hardware.cpu_count > 0
        assert hardware.ram_total >= 0
        # GPU may or may not be present


class TestEnvironmentDetection:
    """Tests for environment detection"""

    def test_check_environment(self):
        """Test environment check"""
        environment = check_environment()

        assert environment is not None
        assert environment.python_version is not None


class TestCompatibility:
    """Tests for compatibility checking"""

    def test_estimate_model_memory(self):
        """Test memory estimation"""
        # 7B model
        memory_7b = estimate_model_memory("llama-7b")
        assert memory_7b == 14000

        # 70B model
        memory_70b = estimate_model_memory("llama-70b")
        assert memory_70b == 140000

        # Unknown model
        memory_unknown = estimate_model_memory("unknown-model")
        assert memory_unknown is None

    def test_check_model_compatibility(self):
        """Test model compatibility check"""
        hardware = check_hardware()
        environment = check_environment()

        result = check_model_compatibility(
            "meta-llama/Llama-3.1-8B-Instruct",
            hardware,
            environment
        )

        assert "compatible" in result
        assert "issues" in result
        assert "recommendations" in result
        assert isinstance(result["compatible"], bool)


class TestDeploymentConfig:
    """Tests for deployment configuration"""

    def test_config_creation(self):
        """Test config creation"""
        config = DeploymentConfig(model="test-model")

        assert config.model == "test-model"
        assert config.dtype == "auto"
        assert config.tensor_parallel_size == 1

    def test_config_to_command(self):
        """Test command generation"""
        config = DeploymentConfig(
            model="test-model",
            tensor_parallel_size=2,
            max_num_seqs=512
        )

        command = config.to_command()

        assert "vllm serve" in command
        assert "test-model" in command
        assert "--tensor-parallel-size 2" in command
        assert "--max-num-seqs 512" in command

    def test_validate_config_valid(self):
        """Test validating valid config"""
        config = DeploymentConfig(model="test-model")
        errors = validate_config(config)

        assert len(errors) == 0

    def test_validate_config_invalid(self):
        """Test validating invalid config"""
        config = DeploymentConfig(
            model="",  # Empty model name
            gpu_memory_utilization=1.5  # Invalid value
        )
        errors = validate_config(config)

        assert len(errors) > 0
