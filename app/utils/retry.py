"""
Async retry logic for API calls
"""

import asyncio
from typing import Callable, Optional, Any

class AsyncRetry:
    """Configurable retry mechanism for async operations"""
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0
    ):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.delay * (self.backoff ** attempt))
        raise RuntimeError("Retry logic failed unexpectedly")