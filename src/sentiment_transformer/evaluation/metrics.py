"""Scientific metrics calculation and reporting for binary sentiment classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    model_name: str = "model",
    training_time: float | None = None,
    inference_time: float | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for binary evaluation.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        model_name: Identifier of the model.
        training_time: Optional training duration in seconds.
        inference_time: Optional inference duration in seconds.

    Returns:
        Structured metrics dictionary.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    f1_mac = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_wgt = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_t, y_p, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    report_dict = classification_report(
        y_t,
        y_p,
        labels=[0, 1],
        target_names=["Negativa (0)", "Positiva (1)"],
        output_dict=True,
        zero_division=0,
    )

    metrics_result: dict[str, Any] = {
        "model_name": model_name,
        "sample_count": len(y_t),
        "accuracy": round(acc, 5),
        "f1_macro": round(f1_mac, 5),
        "f1_weighted": round(f1_wgt, 5),
        "class_0_negative": {
            "precision": round(float(precision[0]), 5),
            "recall": round(float(recall[0]), 5),
            "f1_score": round(float(f1[0]), 5),
            "support": int(support[0]),
        },
        "class_1_positive": {
            "precision": round(float(precision[1]), 5),
            "recall": round(float(recall[1]), 5),
            "f1_score": round(float(f1[1]), 5),
            "support": int(support[1]),
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "training_time_seconds": (
            round(training_time, 3) if training_time is not None else None
        ),
        "inference_time_seconds": (
            round(inference_time, 3) if inference_time is not None else None
        ),
        "classification_report": report_dict,
    }

    logger.info(
        "Metrics for %s -> Accuracy: %.4f | F1 Macro: %.4f | F1 Neg: %.4f | F1 Pos: %.4f",
        model_name,
        acc,
        f1_mac,
        float(f1[0]),
        float(f1[1]),
    )

    return metrics_result


def save_metrics_artifacts(
    metrics: dict[str, Any],
    output_prefix: str,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Save metrics JSON, classification report CSV, and confusion matrix CSV.

    Args:
        metrics: Computed metrics dictionary.
        output_prefix: Filename prefix (e.g. 'baseline_majority', 'transformer').
        output_dir: Directory where files should be written.

    Returns:
        Tuple of (json_path, report_csv_path, cm_csv_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics JSON
    json_path = out_dir / f"{output_prefix}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 2. Classification report CSV
    report_dict = metrics.get("classification_report", {})
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = (
        out_dir / f"classification_report_{output_prefix.replace('baseline_', '')}.csv"
    )
    report_df.to_csv(report_csv_path, index=True, encoding="utf-8")

    # 3. Confusion Matrix CSV
    cm = np.array(metrics["confusion_matrix"]["matrix"])
    cm_df = pd.DataFrame(
        cm,
        index=["Real_Negativa (0)", "Real_Positiva (1)"],
        columns=["Prevista_Negativa (0)", "Prevista_Positiva (1)"],
    )
    cm_csv_path = (
        out_dir / f"confusion_matrix_{output_prefix.replace('baseline_', '')}.csv"
    )
    cm_df.to_csv(cm_csv_path, index=True, encoding="utf-8")

    return json_path, report_csv_path, cm_csv_path
