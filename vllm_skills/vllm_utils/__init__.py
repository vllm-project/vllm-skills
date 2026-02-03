"""vLLM-specific utilities and optimizations"""

from vllm_skills.vllm_utils.templates import (
    fix_chat_template,
    get_tool_use_template,
)
from vllm_skills.vllm_utils.warmup import warmup_guided_decoding

__all__ = ["fix_chat_template", "get_tool_use_template", "warmup_guided_decoding"]
