"""Majority class dummy baseline classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from sentiment_transformer.exceptions import ModelingError

logger = logging.getLogger(__name__)


class MajorityBaseline:
    """Baseline classifier predicting the most frequent class in training data."""

    def __init__(self, strategy: str = "most_frequent") -> None:
        self.strategy = strategy
        self.model = DummyClassifier(strategy=self.strategy)
        self.is_fitted = False

    def fit(self, X: Any, y: np.ndarray | list[int]) -> MajorityBaseline:
        """Fit the dummy classifier on training labels.

        Args:
            X: Input features or texts (ignored by DummyClassifier).
            y: Target binary labels (0/1).

        Returns:
            Fitted MajorityBaseline instance.
        """
        logger.info("Fitting Majority Baseline (strategy='%s')...", self.strategy)
        y_arr = np.asarray(y)
        self.model.fit(X, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels for input samples.

        Args:
            X: Input samples.

        Returns:
            Array of predicted class labels.
        """
        if not self.is_fitted:
            raise ModelingError("MajorityBaseline model is not fitted yet.")
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for input samples.

        Args:
            X: Input samples.

        Returns:
            Array of predicted probabilities shape (N, 2).
        """
        if not self.is_fitted:
            raise ModelingError("MajorityBaseline model is not fitted yet.")
        proba = np.asarray(self.model.predict_proba(X))
        # Ensure 2-column probability output even if training set had 1 class
        if proba.shape[1] == 1:
            classes = list(self.model.classes_)
            if 0 in classes:
                # only negative
                return np.column_stack([proba[:, 0], np.zeros(len(proba))])
            else:
                # only positive
                return np.column_stack([np.zeros(len(proba)), proba[:, 0]])
        return proba

    def save(self, path: str | Path) -> None:
        """Save model to disk using joblib.

        Args:
            path: Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved MajorityBaseline to: %s", p)

    @classmethod
    def load(cls, path: str | Path) -> MajorityBaseline:
        """Load model from disk.

        Args:
            path: Path to serialized model.

        Returns:
            Loaded MajorityBaseline instance.
        """
        p = Path(path)
        if not p.is_file():
            raise ModelingError(f"Model file not found at: {p}")
        obj = joblib.load(p)
        if not isinstance(obj, MajorityBaseline):
            raise ModelingError(
                f"Loaded object is not a MajorityBaseline instance: {type(obj)}"
            )
        return obj
