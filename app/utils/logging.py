"""
Logging configuration for the application
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create file handler for all logs
    log_file = logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create file handler for errors
    error_log_file = logs_dir / f"error_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

# Create logger instance
logger = setup_logging() 