"""Base classes for vllm-skills framework"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SkillMetadata:
    """Metadata for a skill"""

    name: str
    description: str
    version: str
    author: str
    tags: list[str]
    requires: list[str]


class BaseSkill(ABC):
    """Abstract base class for all skills"""

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """Return skill metadata"""
        pass

    @abstractmethod
    async def execute(self, context: dict, **kwargs) -> Any:
        """
        Execute the skill

        Args:
            context: Execution context with runtime information
            **kwargs: Additional skill-specific parameters

        Returns:
            Skill execution result
        """
        pass

    def validate_requirements(self) -> bool:
        """
        Validate that all requirements are met

        Returns:
            True if requirements are satisfied, False otherwise
        """
        # Default implementation - can be overridden
        return True
