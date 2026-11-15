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

    dataset = ReviewSentimentDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(data_loader)
    inf_log_interval = max(1, min(25, total_batches // 4))

    logger.info(
        "Running Transformer inference on %d samples (%d batches, batch_size=%d, max_length=%d)...",
        len(texts),
        total_batches,
        batch_size,
        max_length,
    )

    model.eval()
    model.to(device)  # type: ignore[arg-type]

    all_probabilities: list[np.ndarray] = []
    start_time = time.time()
    last_log_time = start_time
    samples_done = 0

    try:
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
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

                batch_len = len(input_ids)
                samples_done += batch_len
                now = time.time()
                is_last_step = (step + 1) == total_batches

                if (
                    ((step + 1) % inf_log_interval == 0)
                    or (now - last_log_time >= 20.0)
                    or is_last_step
                ):
                    elapsed = now - start_time
                    speed = samples_done / max(0.001, elapsed)
                    logger.info(
                        "  [Inference] Batch %d/%d (%.1f%%) | %d/%d samples | Speed: %.1f samples/s | Elapsed: %.1fs",
                        step + 1,
                        total_batches,
                        ((step + 1) / total_batches) * 100,
                        samples_done,
                        len(texts),
                        speed,
                        elapsed,
                    )
                    last_log_time = now

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
