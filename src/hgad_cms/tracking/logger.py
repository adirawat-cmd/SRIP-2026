"""Structured logging setup for hgad_cms."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    logger_name: str = "hgad_cms",
) -> logging.Logger:
    """
    Configure root project logger with console and optional file handlers.

    Parameters
    ----------
    log_level:
        Logging level name (DEBUG, INFO, WARNING, ERROR).
    log_file:
        Optional path to append log output.
    logger_name:
        Logger namespace.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
