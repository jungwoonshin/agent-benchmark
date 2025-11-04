"""Utility functions for the agent system."""

import logging
import sys


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/log.txt', mode='a'),
        ],
    )
    return logging.getLogger('agent_system')
