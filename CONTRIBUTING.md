# Contributing to vllm-skills

Thank you for your interest in contributing to vllm-skills! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Provide clear reproduction steps
- Include system information (OS, Python version, vLLM version)

### Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Adding New Skills

Skills are self-contained modules that provide specific functionality to LLM agents. Here's how to add a new skill:

### Skill Structure Template

```
vllm_skills/library/your_skill/
├── __init__.py           # Skill class implementation
├── SKILL.md              # Skill documentation
├── [module1.py]          # Additional modules
├── [module2.py]
└── resources/            # Configuration files, data
    ├── config.yaml
    └── data.json
```

### Skill Implementation

Your skill must inherit from `BaseSkill`:

```python
from vllm_skills.core.base import BaseSkill, SkillMetadata

class YourSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="your_skill",
            description="What your skill does",
            version="0.1.0",
            author="Your Name",
            tags=["tag1", "tag2"],
            requires=["package1", "package2"]
        )
    
    async def execute(self, context: dict, **kwargs):
        """Execute the skill logic"""
        # Your implementation
        pass
    
    def validate_requirements(self) -> bool:
        """Check if requirements are met"""
        # Your validation logic
        return True
```

### SKILL.md Documentation

Each skill must include a SKILL.md file with:

- **Metadata**: Name, description, version, tags, requirements
- **Purpose**: What problem does this skill solve?
- **Usage**: How to use the skill
- **Examples**: Code examples and use cases
- **Configuration**: Available parameters and options
- **Agent Behavior**: Guidelines for how agents should use this skill
- **Troubleshooting**: Common issues and solutions

### Testing Requirements

1. Unit tests for core functionality
2. Integration tests for skill execution
3. Test coverage should be at least 80%
4. All tests must pass before PR is merged

Example test structure:

```python
import pytest
from vllm_skills.library.your_skill import YourSkill

@pytest.mark.asyncio
async def test_skill_execute():
    skill = YourSkill()
    result = await skill.execute(context={})
    assert result is not None

def test_skill_metadata():
    skill = YourSkill()
    metadata = skill.metadata
    assert metadata.name == "your_skill"
    assert metadata.version is not None
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)
- Use Ruff for linting: `ruff check .`
- Format code with: `ruff format .`

## Development Setup

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm-skills.git
cd vllm-skills

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .
```

## PR Process

1. **Before submitting**:
   - Ensure all tests pass
   - Add tests for new functionality
   - Update documentation
   - Run linter and fix issues

2. **PR description should include**:
   - What changes were made
   - Why the changes were needed
   - Any breaking changes
   - Related issues

3. **Review process**:
   - Maintainers will review your PR
   - Address feedback and questions
   - Once approved, PR will be merged

## Skill Categories

We organize skills into categories:

- **deployment** - vLLM deployment and configuration
- **coding** - Code generation, refactoring, analysis
- **web** - Web scraping, API interaction
- **data** - Data processing and transformation

Choose the appropriate category for your skill or propose a new one.

## Questions?

- GitHub Discussions: https://github.com/vllm-project/vllm-skills/discussions
- Issue Tracker: https://github.com/vllm-project/vllm-skills/issues

Thank you for contributing to vllm-skills!
