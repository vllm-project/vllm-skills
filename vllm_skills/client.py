"""OpenAI client wrapper with skill support"""

from typing import Any, Optional

from openai import OpenAI

from vllm_skills.core.registry import SkillRegistry


class SkillEnabledClient:
    """OpenAI-compatible client with skill execution support"""

    def __init__(
        self,
        base_url: str,
        skills: Optional[list[str]] = None,
        api_key: str = "EMPTY",
    ):
        """
        Initialize skill-enabled client

        Args:
            base_url: Base URL for vLLM server
            skills: List of skill names to load
            api_key: API key (default: "EMPTY" for vLLM)
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.registry = SkillRegistry()

        if skills:
            self.registry.load_skills(*skills)

    async def chat_with_skills(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        """
        Chat completion with skill execution support

        Args:
            messages: Chat messages
            **kwargs: Additional arguments for chat completion

        Returns:
            Chat completion response
        """
        # Basic implementation - would be enhanced with:
        # 1. Tool call detection
        # 2. Skill execution
        # 3. Response integration

        response = self.client.chat.completions.create(
            messages=messages, **kwargs
        )

        return response

    def list_available_skills(self) -> list[str]:
        """
        List all available skills

        Returns:
            List of skill names
        """
        return self.registry.list_skills()
