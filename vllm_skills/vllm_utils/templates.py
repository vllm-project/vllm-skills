"""Chat template fixes for tool use support"""

from typing import Any, Optional

# Known models with broken or suboptimal chat templates for tool use
TEMPLATE_FIXES = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "issue": "Missing tool use template",
        "fix": "Add tool call formatting",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "issue": "Tool formatting inconsistent",
        "fix": "Standardize tool call format",
    },
}


def fix_chat_template(model_name: str, tokenizer: Any) -> Optional[str]:
    """
    Fix broken chat templates for tool use

    Args:
        model_name: Model identifier
        tokenizer: Tokenizer instance

    Returns:
        Fixed template string or None if no fix needed
    """
    if model_name not in TEMPLATE_FIXES:
        return None

    # Get current template
    current_template = getattr(tokenizer, "chat_template", None)
    if current_template is None:
        return None

    # Apply fixes based on model
    # This is a placeholder - actual fixes would be model-specific
    fix_info = TEMPLATE_FIXES[model_name]

    # Return modified template (implementation would go here)
    return current_template


def get_tool_use_template(model_name: str) -> str:
    """
    Get optimized template for tool calling

    Args:
        model_name: Model identifier

    Returns:
        Tool use template string
    """
    # Basic tool use template
    template = """{% for message in messages %}
{% if message['role'] == 'system' %}{{ message['content'] }}{% endif %}
{% if message['role'] == 'user' %}User: {{ message['content'] }}{% endif %}
{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}
{% if message.get('tool_calls') %}
Tool calls: {{ message['tool_calls'] }}
{% endif %}
{% endfor %}
Assistant:"""

    # Model-specific optimizations would go here
    return template
