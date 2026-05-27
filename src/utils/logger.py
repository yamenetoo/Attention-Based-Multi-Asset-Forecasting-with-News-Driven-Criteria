"""
Logging configuration utilities.
"""

import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(
    log_file: str = "logs/training.log",
    level: str = "INFO",
    console: bool = True,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Configure application logger with console and file handlers.
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Whether to output to console
        rotation: Log file rotation size
        retention: How long to keep old log files
    """
    # Remove default handler
    _logger.remove()
    
    # Console handler
    if console:
        _logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        _logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )


# Create module-level logger instance
logger = _logger

__all__ = ["setup_logger", "logger"]
