"""Tests for core framework"""

import pytest

from vllm_skills.core.base import BaseSkill, SkillMetadata
from vllm_skills.core.registry import SkillRegistry
from vllm_skills.core.sandbox import LocalSandbox


class TestSkill(BaseSkill):
    """Test skill implementation"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="test_skill",
            description="A test skill",
            version="0.1.0",
            author="Test Author",
            tags=["test"],
            requires=[],
        )

    async def execute(self, context: dict, **kwargs):
        return {"result": "test_executed"}


class TestBaseSkill:
    """Tests for BaseSkill"""

    def test_skill_metadata(self):
        """Test skill metadata"""
        skill = TestSkill()
        metadata = skill.metadata

        assert metadata.name == "test_skill"
        assert metadata.description == "A test skill"
        assert metadata.version == "0.1.0"
        assert metadata.author == "Test Author"
        assert "test" in metadata.tags

    @pytest.mark.asyncio
    async def test_skill_execute(self):
        """Test skill execution"""
        skill = TestSkill()
        result = await skill.execute(context={})

        assert result is not None
        assert result["result"] == "test_executed"

    def test_skill_validate_requirements(self):
        """Test requirements validation"""
        skill = TestSkill()
        assert skill.validate_requirements() is True


class TestSkillRegistry:
    """Tests for SkillRegistry"""

    def test_registry_creation(self):
        """Test registry creation"""
        registry = SkillRegistry()
        assert registry is not None
        assert registry.list_skills() == []

    def test_register_skill(self):
        """Test skill registration"""
        registry = SkillRegistry()
        skill = TestSkill()

        registry.register(skill)

        assert "test_skill" in registry.list_skills()
        assert registry.get("test_skill") == skill

    def test_register_duplicate_skill(self):
        """Test duplicate skill registration fails"""
        registry = SkillRegistry()
        skill = TestSkill()

        registry.register(skill)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(skill)

    def test_get_nonexistent_skill(self):
        """Test getting nonexistent skill"""
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_load_deployment_skill(self):
        """Test loading deployment skill"""
        registry = SkillRegistry()

        try:
            registry.load_skills("deployment")
            assert "deployment" in registry.list_skills()

            skill = registry.get("deployment")
            assert skill is not None
            assert skill.metadata.name == "deployment"
        except ImportError:
            # Skill module may not be importable in test environment
            pytest.skip("Deployment skill not available")


class TestLocalSandbox:
    """Tests for LocalSandbox"""

    @pytest.mark.asyncio
    async def test_simple_command(self):
        """Test simple command execution"""
        sandbox = LocalSandbox()
        result = await sandbox.execute("echo 'Hello, World!'", timeout=5)

        assert result["return_code"] == 0
        assert "Hello, World!" in result["stdout"]
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """Test command timeout"""
        sandbox = LocalSandbox()
        result = await sandbox.execute("sleep 10", timeout=1)

        assert result["return_code"] == -1
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_command_error(self):
        """Test command with error"""
        sandbox = LocalSandbox()
        result = await sandbox.execute("exit 1", timeout=5)

        assert result["return_code"] == 1
