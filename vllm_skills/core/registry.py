"""Skill registry for loading and managing skills"""

import importlib
from typing import Optional

from vllm_skills.core.base import BaseSkill


class SkillRegistry:
    """Registry for discovering and loading skills"""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """
        Register a skill instance

        Args:
            skill: Skill instance to register
        """
        skill_name = skill.metadata.name
        if skill_name in self._skills:
            raise ValueError(f"Skill '{skill_name}' is already registered")
        self._skills[skill_name] = skill

    def load_skills(self, *skill_names: str) -> None:
        """
        Load skills by name from the library

        Args:
            *skill_names: Names of skills to load
        """
        for skill_name in skill_names:
            if skill_name in self._skills:
                continue  # Already loaded

            try:
                # Try to import from library
                module = importlib.import_module(
                    f"vllm_skills.library.{skill_name}"
                )

                # Get the skill class (convention: capitalize first letter of each word)
                class_name = "".join(
                    word.capitalize() for word in skill_name.split("_")
                )
                if not hasattr(module, class_name):
                    # Fallback to searching for BaseSkill subclasses
                    skill_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, BaseSkill)
                            and attr is not BaseSkill
                        ):
                            skill_class = attr
                            break
                    if skill_class is None:
                        raise AttributeError(
                            f"No BaseSkill subclass found in {module.__name__}"
                        )
                else:
                    skill_class = getattr(module, class_name)

                # Instantiate and register
                skill_instance = skill_class()
                self.register(skill_instance)

            except ImportError as e:
                raise ImportError(
                    f"Failed to import skill '{skill_name}': {e}"
                ) from e

    def get(self, name: str) -> Optional[BaseSkill]:
        """
        Get a skill by name

        Args:
            name: Skill name

        Returns:
            Skill instance or None if not found
        """
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """
        List all registered skills

        Returns:
            List of skill names
        """
        return list(self._skills.keys())
