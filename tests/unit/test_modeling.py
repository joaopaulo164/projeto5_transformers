"""Unit tests for modeling dataset, tokenizer helpers, and inference batching."""

from __future__ import annotations

from sentiment_transformer.modeling.dataset import ReviewSentimentDataset
from sentiment_transformer.modeling.tokenizer import get_sep_token


def test_review_sentiment_dataset_without_tokenizer() -> None:
    texts = ["Texto um", "Texto dois"]
    labels = [0, 1]
    dataset = ReviewSentimentDataset(texts, labels, max_length=16)

    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 0


def test_get_sep_token_default() -> None:
    assert get_sep_token(None) == "[SEP]"
