"""Example: ReAct agent loop with skills

This example demonstrates a ReAct (Reasoning + Acting) loop using vllm-skills.
"""

import asyncio
from typing import Any


class SimpleReActAgent:
    """Simple ReAct agent for demonstration"""

    def __init__(self, skills: list):
        self.skills = {skill.metadata.name: skill for skill in skills}

    async def run(self, task: str, max_iterations: int = 5):
        """
        Run ReAct loop

        Args:
            task: Task description
            max_iterations: Maximum number of iterations
        """
        print("=" * 60)
        print("ReAct Agent Loop")
        print("=" * 60)
        print(f"\nTask: {task}\n")

        for i in range(max_iterations):
            print(f"--- Iteration {i + 1} ---")

            # Thought: Reason about what to do next
            thought = self._think(task, i)
            print(f"Thought: {thought}")

            # Action: Decide which skill to use
            action, action_input = self._decide_action(task, i)
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")

            # Execute action
            if action == "FINISH":
                print("\nTask complete!")
                break

            if action in self.skills:
                observation = await self._execute_skill(action, action_input)
                print(f"Observation: {observation[:200]}...")
            else:
                observation = f"Unknown skill: {action}"
                print(f"Observation: {observation}")

            print()

    def _think(self, task: str, iteration: int) -> str:
        """Generate thought (simplified for demo)"""
        if iteration == 0:
            return "I should first check the hardware and environment"
        elif iteration == 1:
            return "Now I should configure the model deployment"
        else:
            return "I have enough information to complete the task"

    def _decide_action(self, task: str, iteration: int) -> tuple[str, dict]:
        """Decide which action to take (simplified for demo)"""
        if iteration == 0:
            return "deployment", {"action": "check"}
        elif iteration == 1:
            # Extract model name from task (simplified)
            model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Default
            if "llama" in task.lower():
                model_name = "meta-llama/Llama-3.1-8B-Instruct"
            return "deployment", {"action": "configure", "model_name": model_name}
        else:
            return "FINISH", {}

    async def _execute_skill(self, skill_name: str, inputs: dict) -> Any:
        """Execute a skill and return observation"""
        skill = self.skills.get(skill_name)
        if skill:
            result = await skill.execute(context={}, **inputs)
            return str(result)
        return "Skill not found"


async def main():
    """Run example ReAct loop"""
    from vllm_skills.core.registry import SkillRegistry

    # Load skills
    print("Loading skills...")
    registry = SkillRegistry()
    registry.load_skills("deployment")

    # Get loaded skills
    skills = [registry.get(name) for name in registry.list_skills()]

    # Create agent
    agent = SimpleReActAgent(skills)

    # Run task
    task = "Help me deploy Llama 3.1 8B on my system"
    await agent.run(task)

    print("\n" + "=" * 60)
    print("ReAct loop example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
