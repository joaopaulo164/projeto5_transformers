"""Sentiment Transformer package for binary sentiment classification in Portuguese."""

__version__ = "0.1.0"
__author__ = "Joao Paulo dos Santos Mendes"

from sentiment_transformer.config import AppConfig, load_config
from sentiment_transformer.exceptions import (
    ConfigError,
    DataAuditError,
    DataDownloadError,
    DataValidationError,
    ModelingError,
    SentimentTransformerError,
)

__all__ = [
    "AppConfig",
    "load_config",
    "SentimentTransformerError",
    "ConfigError",
    "DataDownloadError",
    "DataValidationError",
    "DataAuditError",
    "ModelingError",
]
