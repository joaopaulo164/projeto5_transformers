"""Custom domain exceptions for the sentiment transformer pipeline."""


class SentimentTransformerError(Exception):
    """Base exception for all sentiment transformer errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigError(SentimentTransformerError):
    """Raised when configuration is missing, invalid or malformed."""


class DataDownloadError(SentimentTransformerError):
    """Raised when dataset acquisition or download fails."""


class DataValidationError(SentimentTransformerError):
    """Raised when dataset fails schema, content or invariant validation."""


class DataAuditError(SentimentTransformerError):
    """Raised when dataset auditing fails or finds fatal inconsistencies."""


class PreprocessingError(SentimentTransformerError):
    """Raised when text preprocessing or token composition encounters an error."""


class ModelingError(SentimentTransformerError):
    """Raised when model instantiation, training or checkpoint saving fails."""


class InferenceError(SentimentTransformerError):
    """Raised when batch inference or evaluation fails."""


class EvidenceError(SentimentTransformerError):
    """Raised when generating experiment artifacts, plots or manifests fails."""


class PhaseExecutionError(SentimentTransformerError):
    """Raised when a specific pipeline phase fails to execute."""
