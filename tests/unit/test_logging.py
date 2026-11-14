"""Unit tests for logging configuration."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.logging_config import setup_logging


def test_setup_logging(temp_dir: Path) -> None:
    log_file = temp_dir / "test.log"
    logger = setup_logging(level="DEBUG", log_file=log_file)
    logger.info("Test log message")
    logger.debug("Debug log message")

    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message" in content
    assert "Debug log message" in content
