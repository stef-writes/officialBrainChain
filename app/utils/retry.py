"""
Retry functionality for asynchronous operations
"""

import asyncio
import random
from typing import Type, Tuple, Callable, Any, Optional
from app.utils.logging import logger

class AsyncRetry:
    """Asynchronous retry mechanism for API calls"""
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        max_delay: float = 60.0,
        jitter: bool = True,
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        """Initialize retry mechanism.
        
        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay after each retry
            exceptions: Tuple of exceptions to catch and retry
            max_delay: Maximum delay between retries in seconds
            jitter: Whether to add random jitter to delays
            on_retry: Optional callback function called on retry
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
        self.max_delay = max_delay
        self.jitter = jitter
        self.on_retry = on_retry
        self._retry_count = 0
        
        logger.debug(
            f"Initialized AsyncRetry with max_retries={max_retries}, "
            f"delay={delay}, backoff={backoff}"
        )
        
    async def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from function execution
            
        Raises:
            Exception: If all retry attempts fail
        """
        while True:
            try:
                return await func(*args, **kwargs)
                
            except self.exceptions as e:
                self._retry_count += 1
                
                if self._retry_count > self.max_retries:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded. "
                        f"Last error: {str(e)}"
                    )
                    raise
                    
                # Calculate delay with exponential backoff
                delay = min(
                    self.delay * (self.backoff ** (self._retry_count - 1)),
                    self.max_delay
                )
                
                # Add jitter if enabled
                if self.jitter:
                    delay *= random.uniform(0.5, 1.5)
                    
                logger.warning(
                    f"Retry {self._retry_count}/{self.max_retries} after "
                    f"{delay:.2f}s. Error: {str(e)}"
                )
                
                # Call retry callback if provided
                if self.on_retry:
                    self.on_retry(e, self._retry_count)
                    
                await asyncio.sleep(delay)
                
    def reset(self) -> None:
        """Reset retry counter"""
        self._retry_count = 0