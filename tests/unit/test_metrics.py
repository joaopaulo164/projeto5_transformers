"""Unit tests for metrics calculation, classification reporting, and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sentiment_transformer.evaluation.metrics import (
    compute_classification_metrics,
    save_metrics_artifacts,
)


def test_compute_classification_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = compute_classification_metrics(y_true, y_pred, model_name="TestModel")

    assert metrics["model_name"] == "TestModel"
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.75
    assert "f1_macro" in metrics
    assert metrics["confusion_matrix"]["true_negative"] == 1
    assert metrics["confusion_matrix"]["false_positive"] == 1
    assert metrics["confusion_matrix"]["true_positive"] == 2
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_save_metrics_artifacts(temp_dir: Path) -> None:
    y_true = [0, 1]
    y_pred = [0, 1]
    metrics = compute_classification_metrics(y_true, y_pred, model_name="Dummy")

    json_p, rep_csv, cm_csv = save_metrics_artifacts(metrics, "dummy_model", temp_dir)

    assert json_p.is_file()
    assert rep_csv.is_file()
    assert cm_csv.is_file()
