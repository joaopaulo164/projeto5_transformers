"""Unit tests for confusion matrix and training curves plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_transformer.evaluation.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)


def test_plot_confusion_matrix(temp_dir: Path) -> None:
    cm = np.array([[10, 2], [3, 15]])
    out_p = temp_dir / "cm.png"
    saved = plot_confusion_matrix(cm, "Modelo Teste", out_p)
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_plot_training_curves(temp_dir: Path) -> None:
    history_df = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [0.6, 0.4, 0.3],
            "val_loss": [0.55, 0.42, 0.35],
            "val_f1_macro": [0.70, 0.82, 0.85],
            "val_accuracy": [0.72, 0.83, 0.86],
        }
    )
    out_p = temp_dir / "curves.png"
    saved = plot_training_curves(history_df, out_p)
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_plot_training_curves_empty(temp_dir: Path) -> None:
    history_df = pd.DataFrame()
    out_p = temp_dir / "curves_empty.png"
    saved = plot_training_curves(history_df, out_p)
    assert saved.is_file()
