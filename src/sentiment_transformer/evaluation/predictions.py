"""Test set predictions formatting and saving."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_transformer.constants import (
    COL_COMBINED_TEXT,
    COL_LABEL,
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
)

logger = logging.getLogger(__name__)


def generate_predictions_dataframe(
    test_df: pd.DataFrame,
    y_pred: np.ndarray | list[int],
    y_proba: np.ndarray,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Construct structured predictions DataFrame with probabilities and text for auditing.

    Args:
        test_df: Original test partition DataFrame.
        y_pred: Array of predicted labels (0/1).
        y_proba: Array of shape (N, 2) with negative (col 0) and positive (col 1) probabilities.
        output_csv_path: Optional path to save CSV.

    Returns:
        Structured predictions DataFrame.
    """
    preds_arr = np.asarray(y_pred, dtype=int)
    proba_arr = np.asarray(y_proba, dtype=float)
    true_labels = np.asarray(test_df[COL_LABEL], dtype=int)
    is_correct_arr = (true_labels == preds_arr).astype(int)

    preds_df = pd.DataFrame(
        {
            COL_REVIEW_ID: test_df[COL_REVIEW_ID].values,
            COL_OVERALL_RATING: test_df[COL_OVERALL_RATING].values,
            "true_label": true_labels,
            "predicted_label": preds_arr,
            "negative_probability": np.round(proba_arr[:, 0], 5),
            "positive_probability": np.round(proba_arr[:, 1], 5),
            "is_correct": is_correct_arr,
            COL_REVIEW_TITLE: test_df[COL_REVIEW_TITLE].values,
            COL_REVIEW_TEXT: test_df[COL_REVIEW_TEXT].values,
            COL_COMBINED_TEXT: test_df[COL_COMBINED_TEXT].values,
        }
    )

    if output_csv_path is not None:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(out_p, index=False, encoding="utf-8")
        logger.info(
            "Saved predictions DataFrame with %d rows to: %s",
            len(preds_df),
            out_p,
        )

    return preds_df
