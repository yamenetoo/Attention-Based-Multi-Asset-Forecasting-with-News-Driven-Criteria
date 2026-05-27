"""
Attention-Based Multi-Asset Forecasting with News-Driven Criteria Extraction
A Case Study on Gold and Crude Oil

Package initialization for the commodity forecasting framework.
"""

__version__ = "0.1.0"
__author__ = "Mohamad Yamen Al-Mohamad"
__email__ = "author@email.com"

from .utils.seed import set_seed
from .utils.logger import setup_logger

__all__ = ["set_seed", "setup_logger"]
