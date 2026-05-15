"""
Logging configuration for Mind Games
"""

import logging
import os
from datetime import datetime
from config import LOGS_DIR, DEBUG_MODE

def setup_logger(name: str, log_file: str = None, level: int = None) -> logging.Logger:
    """
    Setup a logger with console and optional file handler.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Filename inside LOGS_DIR (None = no file logging)
        level: Logging level (default: DEBUG if DEBUG_MODE else INFO)
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (if requested)
    if log_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_path = os.path.join(LOGS_DIR, log_file)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default application logger
app_logger = setup_logger('MindGames', 'game.log')