"""Unit tests for TransformerTrainer using lightweight mock architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sentiment_transformer.config import AppConfig
from sentiment_transformer.modeling.dataset import ReviewSentimentDataset
from sentiment_transformer.modeling.trainer import TransformerTrainer


class DummyClassificationOutput:
    def __init__(self, loss: torch.Tensor, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class DummyTransformerModel(nn.Module):
    """Minimal PyTorch module matching Hugging Face sequence classification interface for fast testing."""

    def __init__(self, num_labels: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(16, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DummyClassificationOutput:
        # Dummy forward pass projecting tensor float
        x = input_ids.float()
        logits = self.linear(x)
        loss = self.loss_fn(logits, labels) if labels is not None else torch.tensor(0.0)
        return DummyClassificationOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}", encoding="utf-8")


def test_transformer_trainer_evaluate(temp_dir: Path) -> None:
    cfg = AppConfig()
    model = DummyTransformerModel(num_labels=2)
    device = torch.device("cpu")

    trainer = TransformerTrainer(config=cfg, model=model, device=device)

    # Fake dataset
    dataset = ReviewSentimentDataset(
        texts=["sample 1", "sample 2", "sample 3", "sample 4"],
        labels=[0, 1, 0, 1],
        max_length=16,
    )
    val_loader = DataLoader(dataset, batch_size=2)

    metrics = trainer.evaluate(val_loader)
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    assert "val_f1_macro" in metrics


def test_transformer_trainer_train(temp_dir: Path) -> None:
    cfg = AppConfig()
    cfg.transformer.epochs = 1
    cfg.transformer.batch_size = 2
    cfg.transformer.eval_batch_size = 2
    cfg.transformer.max_length = 16

    model = DummyTransformerModel(num_labels=2)
    device = torch.device("cpu")
    trainer = TransformerTrainer(config=cfg, model=model, device=device)

    train_texts = ["texto 1", "texto 2", "texto 3", "texto 4"]
    train_labels = [0, 1, 0, 1]
    val_texts = ["val 1", "val 2"]
    val_labels = [0, 1]

    res = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=temp_dir,
    )

    assert "best_epoch" in res
    assert (temp_dir / "training_history.csv").is_file()
    assert (temp_dir / "best_checkpoint").is_dir()
