"""
Asynchronous retry mechanism for resilient API calls
"""

import asyncio
import logging
import time
from typing import Callable, Any, Dict, Type, Tuple, Optional, Union, List
import traceback

logger = logging.getLogger(__name__)

class AsyncRetry:
    """Configurable asynchronous retry mechanism for resilient API calls
    
    This class provides a flexible retry mechanism for asynchronous functions,
    with configurable retry counts, delays, backoff strategies, and exception handling.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        max_delay: float = 60.0,
        jitter: bool = True,
        on_retry: Optional[Callable[[int, Exception, Dict[str, Any]], None]] = None
    ):
        """Initialize the retry mechanism.
        
        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay after each retry
            exceptions: Exception(s) to catch and retry on
            max_delay: Maximum delay between retries in seconds
            jitter: Whether to add random jitter to delays
            on_retry: Optional callback function called on each retry
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions if isinstance(exceptions, tuple) else (exceptions,)
        self.max_delay = max_delay
        self.jitter = jitter
        self.on_retry = on_retry
        self.current_retry = 0
        
        logger.info(f"Initialized AsyncRetry with max_retries={max_retries}, delay={delay}, backoff={backoff}")
    
    async def execute(
        self,
        func: Callable,
        *args,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute a function with retry logic.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            metadata: Optional metadata for retry callbacks
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function execution
            
        Raises:
            The last exception encountered after all retries are exhausted
        """
        metadata = metadata or {}
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Reset retry counter on successful execution
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} for {func.__name__}")
                
                self.current_retry = attempt
                return await func(*args, **kwargs)
                
            except self.exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(self.delay * (self.backoff ** attempt), self.max_delay)
                    
                    # Add jitter if enabled
                    if self.jitter:
                        delay = delay * (0.5 + 0.5 * asyncio.get_event_loop().time() % 1)
                    
                    # Call retry callback if provided
                    if self.on_retry:
                        try:
                            self.on_retry(attempt, e, metadata)
                        except Exception as callback_error:
                            logger.error(f"Error in retry callback: {str(callback_error)}")
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                    )
                    logger.debug(traceback.format_exc())
        
        # If we've exhausted all retries, raise the last exception
        raise last_exception
    
    def reset(self) -> None:
        """Reset the retry counter."""
        self.current_retry = 0