"""
Logging configuration for the application
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str = "gaffer", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger

# Create default logger instance
logger = setup_logger() 