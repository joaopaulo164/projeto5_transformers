"""Batch inference utilities for fine-tuned Transformer models."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sentiment_transformer.exceptions import InferenceError
from sentiment_transformer.modeling.dataset import ReviewSentimentDataset

logger = logging.getLogger(__name__)


def predict_transformer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str] | np.ndarray,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run batch inference on a list of texts using the fine-tuned Transformer model.

    Args:
        model: Loaded Hugging Face sequence classification model.
        tokenizer: Loaded Hugging Face tokenizer.
        texts: List or array of text strings.
        device: PyTorch device (CPU or CUDA).
        batch_size: Batch size for DataLoader.
        max_length: Max sequence length for tokenizer.

    Returns:
        Tuple of (probabilities array shape [N, 2], predicted class labels shape [N], inference_duration_seconds).

    Raises:
        InferenceError: If inference fails.
    """
    if len(texts) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int), 0.0

    logger.info(
        "Running Transformer inference on %d samples (batch_size=%d, max_length=%d)...",
        len(texts),
        batch_size,
        max_length,
    )

    dataset = ReviewSentimentDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)  # type: ignore[arg-type]

    all_probabilities: list[np.ndarray] = []
    start_time = time.time()

    try:
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")

                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                outputs = model(**kwargs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probabilities.append(probs)

        duration = time.time() - start_time
        probabilities = np.vstack(all_probabilities)
        predictions = np.argmax(probabilities, axis=1)

        logger.info(
            "Inference completed in %.2f seconds (%.2f ms/sample).",
            duration,
            (duration / len(texts)) * 1000 if len(texts) > 0 else 0.0,
        )

        return probabilities, predictions, duration

    except Exception as e:
        raise InferenceError(
            f"Transformer inference encountered an error: {e}"
        ) from e
