"""PyTorch Dataset implementation for review sentiment classification."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ReviewSentimentDataset(Dataset):
    """PyTorch Dataset that tokenizes review text for sequence classification models."""

    def __init__(
        self,
        texts: list[str] | np.ndarray,
        labels: list[int] | np.ndarray | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 128,
    ) -> None:
        """Initialize Dataset.

        Args:
            texts: Sequence of review strings.
            labels: Optional sequence of integer labels (0 or 1).
            tokenizer: Hugging Face tokenizer instance.
            max_length: Maximum sequence length for padding/truncation.
        """
        self.texts = [str(t) for t in texts]
        self.labels = [int(lbl) for lbl in labels] if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch and tokenize sample at index idx.

        Args:
            idx: Integer sample index.

        Returns:
            Dictionary containing input_ids, attention_mask, and optionally labels as PyTorch tensors.
        """
        text = self.texts[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item: dict[str, torch.Tensor] = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            # Fallback dummy tensor representation for tests without tokenizer
            item = {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.ones(self.max_length, dtype=torch.long),
            }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
