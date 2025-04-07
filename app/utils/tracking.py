"""
Token tracking and cost calculation utilities
"""

import time
from contextlib import contextmanager
from typing import Dict

@contextmanager
def track_token_usage():
    """Context manager for tracking OpenAI token usage"""
    class TokenTracker:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_cost = 0.0
            self.start_time = time.time()
            
        def update(self, response: Dict):
            """Update metrics from OpenAI response"""
            usage = response.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self._calculate_cost()
            
        def _calculate_cost(self):
            """Estimate costs based on GPT-4 pricing"""
            self.total_cost += (self.prompt_tokens/1000)*0.03 + \
                             (self.completion_tokens/1000)*0.06
    
    tracker = TokenTracker()
    try:
        yield tracker
    finally:
        duration = time.time() - tracker.start_time
        print(f"Execution took {duration:.2f}s")
        print(f"Total cost: ${tracker.total_cost:.4f}")