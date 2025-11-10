"""Utility functions for the agent system."""

import logging
import sys

from .error_handling import (
    CircuitBreaker,
    ErrorCategory,
    ErrorType,
    classify_error,
    retry_with_backoff,
)
from .json_utils import extract_json_from_text, safe_json_loads


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/log.txt', mode='w'),
        ],
    )
    return logging.getLogger('agent_system')


__all__ = [
    'extract_json_from_text',
    'safe_json_loads',
    'classify_error',
    'retry_with_backoff',
    'CircuitBreaker',
    'ErrorType',
    'ErrorCategory',
    'setup_logging',
]

