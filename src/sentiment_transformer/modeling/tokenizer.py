"""Hugging Face tokenizer loader and text tokenization helpers."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sentiment_transformer.constants import DEFAULT_TRANSFORMER_CHECKPOINT
from sentiment_transformer.exceptions import ModelingError

logger = logging.getLogger(__name__)


def load_tokenizer(
    checkpoint: str = DEFAULT_TRANSFORMER_CHECKPOINT,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """Load a pretrained Hugging Face tokenizer with error handling and fallback messages.

    Args:
        checkpoint: Hugging Face hub model identifier or local directory path.
        **kwargs: Additional keyword arguments forwarded to AutoTokenizer.from_pretrained.

    Returns:
        PreTrainedTokenizerBase instance.

    Raises:
        ModelingError: If tokenizer loading fails.
    """
    logger.info("Loading tokenizer for checkpoint: %s", checkpoint)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
        logger.info(
            "Tokenizer loaded. Vocab size: %d, sep_token: '%s', cls_token: '%s'",
            tokenizer.vocab_size,
            tokenizer.sep_token,
            tokenizer.cls_token,
        )
        return tokenizer
    except Exception as e:
        raise ModelingError(
            f"Failed to load tokenizer for checkpoint '{checkpoint}': {e}\n"
            f"Ensure internet connectivity is available for initial download, or specify a valid local directory path."
        ) from e


def get_sep_token(tokenizer: PreTrainedTokenizerBase | None = None) -> str:
    """Return the separator token from tokenizer or default '[SEP]'."""
    if tokenizer is not None and getattr(tokenizer, "sep_token", None):
        return str(tokenizer.sep_token)
    return "[SEP]"
