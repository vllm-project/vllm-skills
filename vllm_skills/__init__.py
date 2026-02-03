"""vllm-skills: Agent skills for vLLM"""

__version__ = "0.1.0"

from vllm_skills.core.base import BaseSkill, SkillMetadata
from vllm_skills.core.registry import SkillRegistry

__all__ = ["BaseSkill", "SkillMetadata", "SkillRegistry"]
