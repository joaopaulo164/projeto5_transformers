"""Evaluation subsystem: metrics calculation, plotting, predictions generation, and error analysis."""

from sentiment_transformer.evaluation.error_analysis import build_error_analysis
from sentiment_transformer.evaluation.metrics import compute_classification_metrics
from sentiment_transformer.evaluation.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)
from sentiment_transformer.evaluation.predictions import generate_predictions_dataframe

__all__ = [
    "compute_classification_metrics",
    "generate_predictions_dataframe",
    "plot_confusion_matrix",
    "plot_training_curves",
    "build_error_analysis",
]
