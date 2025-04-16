"""
Centralized cost calculation utility for LLM usage.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Define cost per 1K tokens (approximate) - Consider moving to a config file
# or a more dynamic pricing provider in the future.
MODEL_COSTS = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    # Add other models as needed
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated cost based on token usage and model name.

    Args:
        model_name: The name of the language model used.
        prompt_tokens: The number of tokens in the prompt.
        completion_tokens: The number of tokens in the completion.

    Returns:
        The estimated cost in USD.
    """
    model_costs = MODEL_COSTS.get(model_name)

    if model_costs is None:
        logger.warning(
            f"Cost calculation not available for model: {model_name}. Returning cost 0.0"
        )
        return 0.0

    prompt_cost = (prompt_tokens / 1000) * model_costs.get("prompt", 0)
    completion_cost = (completion_tokens / 1000) * model_costs.get("completion", 0)

    return prompt_cost + completion_cost

# Removed track_token_usage context manager
# Removed track_usage decorator