# vLLM Skills

A collection of skills for deploying and invoking vLLM. This project follows the [anthropics/skills](https://github.com/anthropics/skills) template format.

## Overview

This repository provides modular, reusable agent skills required to operate and invoke vLLM, following the Anthropics `SKILL.md` specification. Each skill is a self-contained directory implementing automation, scripts, and metadata for a specific operational task.

All skills adhere to the Anthropics skills template and can be copied into a Claude Code skills directory for use.

## Project Structure

```
vllm-skills/
├── skills/
│   └── vllm-deploy-simple/   # Local deployment skill
│       ├── SKILL.md          # Skill documentation (YAML frontmatter + body)
│       └── scripts/          # Deployment utilities
└── README.md
```

## Skills Example

### vllm-deploy-simple

Deploy vLLM as an online service with OpenAI-compatible API locally.

**Features:**
- Auto detect hardware type and install vllm
- Local deployment with `vllm serve`
- Test and management utilities

**Quick Start for Claude Code:**

1. Clone the repository

   ```bash
   git clone https://github.com/vllm-project/vllm-skills.git
   cd vllm-skills
   ```

2. Copy skills needed to your Claude Code skills directory

   Copy the vllm-deploy-simple skill to global skill folder:
   
   ```bash
   cp -r skills/vllm-deploy-simple ~/.claude/skills/
   ```

   Or copy to the project skill folder:
   
   ```bash
   cp -r skills/vllm-deploy-simple .claude/skills/
   ```

3. Use the skills (with sample user prompts):

   Once installed, you can use the skill in Claude Code like:

   ```
   /vllm-deploy-simple
   ```

   Or with natural language:

   ```
   Deploy vLLM with Qwen2.5-1.5B-Instruct on port 8000
   ```

   ```
   Install and start a vLLM server using the vllm-deploy-simple skill
   ```

   ```
   Set up vLLM in a virtual environment at current folder with Qwen2.5-1.5B-Instruct
   ```

## Supported Models

See [vLLM documentation](https://docs.vllm.ai/en/stable/models/supported_models.html) for the full list.

## Contributing

This project follows the [anthropics/skills](https://github.com/anthropics/skills) template. When adding new skills:

1. Create a new directory under `skills/` (e.g., `skills/your-skill/`)
2. Add a `SKILL.md` file with YAML frontmatter:
   ```yaml
   ---
   name: your-skill
   description: Brief description of what this skill does
   ---
   ```
3. Add optional `scripts/`, `references/`, and `assets/` directories
4. Update this README with your skill documentation

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [anthropics/skills Template](https://github.com/anthropics/skills)
