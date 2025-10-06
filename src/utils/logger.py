"""
Logging utilities for the MMS Finance ML project
"""

import logging
import os
from typing import Dict, Any
from pathlib import Path


def setup_logger(config: Dict[str, Any] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logger
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/mms_finance.log'
        }
    
    # Get logging level
    level = getattr(logging, config.get('level', 'INFO').upper())
    
    # Get log format
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get log file path
    log_file = config.get('file', 'logs/mms_finance.log')
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create logger
    logger = logging.getLogger('mms_finance')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'mms_finance.{name}')


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)
