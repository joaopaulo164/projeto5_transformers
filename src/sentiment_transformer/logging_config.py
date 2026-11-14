"""Logging configuration for sentiment transformer pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def close_logging_handlers() -> None:
    """Close and remove all handlers attached to root and package loggers."""
    for logger_name in ["", "sentiment_transformer"]:
        current_logger = logging.getLogger(logger_name)
        for handler in list(current_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            current_logger.removeHandler(handler)


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """Set up and configure the root logger for the application.

    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write log records to disk.
        log_format: Optional logging format string.

    Returns:
        The configured root or package logger.
    """
    close_logging_handlers()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = (
        log_format
        if log_format is not None
        else "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger = logging.getLogger("sentiment_transformer")
    return logger
