"""Structured logging system for TrendPulse-AI.

Provides uniform logging formatters, console handlers, and contextual
logging capabilities compliant with enterprise security standards.
"""

from __future__ import annotations

import logging
import sys


def setup_logger(
    name: str = "TrendPulse",
    level: str = "INFO",
) -> logging.Logger:
    """Configure and return a structured logger instance.

    Args:
        name: Name of the logger or module.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.propagate = False

    return logger


logger = setup_logger()
