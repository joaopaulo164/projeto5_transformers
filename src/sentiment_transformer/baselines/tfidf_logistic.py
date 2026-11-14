"""TF-IDF with Logistic Regression baseline classifier pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_transformer.config import BaselineTfidfConfig
from sentiment_transformer.exceptions import ModelingError

logger = logging.getLogger(__name__)


class TfidfLogisticBaseline:
    """TF-IDF Vectorizer + Logistic Regression classification pipeline."""

    def __init__(
        self,
        ngram_range: tuple[int, int] | list[int] | tuple[int, ...] = (1, 2),
        max_features: int = 10000,
        min_df: int | float = 2,
        max_df: float = 0.95,
        c_regularization: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.ngram_range = (
            (ngram_range[0], ngram_range[1])
            if len(ngram_range) >= 2
            else (1, 2)
        )
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.c_regularization = c_regularization
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            C=self.c_regularization,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.pipeline = Pipeline(
            [
                ("tfidf", self.vectorizer),
                ("clf", self.classifier),
            ]
        )
        self.is_fitted = False

    @classmethod
    def from_config(cls, config: BaselineTfidfConfig) -> TfidfLogisticBaseline:
        """Instantiate pipeline from BaselineTfidfConfig."""
        ng_tuple = (
            (config.ngram_range[0], config.ngram_range[1])
            if len(config.ngram_range) >= 2
            else (1, 2)
        )
        return cls(
            ngram_range=ng_tuple,
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            c_regularization=config.c_regularization,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )

    def fit(
        self, texts: list[str] | np.ndarray, y: np.ndarray | list[int]
    ) -> TfidfLogisticBaseline:
        """Fit the TF-IDF vectorizer and logistic regression classifier on training data.

        Args:
            texts: List or array of text strings.
            y: Target binary labels (0/1).

        Returns:
            Fitted TfidfLogisticBaseline instance.
        """
        logger.info(
            "Fitting TF-IDF Logistic Regression (ngram_range=%s, max_features=%d, C=%.2f)...",
            self.ngram_range,
            self.max_features,
            self.c_regularization,
        )
        y_arr = np.asarray(y)
        self.pipeline.fit(texts, y_arr)
        self.is_fitted = True
        logger.info("TF-IDF vocabulary size: %d", len(self.vectorizer.vocabulary_))
        return self

    def predict(self, texts: list[str] | np.ndarray) -> np.ndarray:
        """Predict labels for input texts.

        Args:
            texts: List or array of text strings.

        Returns:
            Array of predicted binary labels (0/1).
        """
        if not self.is_fitted:
            raise ModelingError("TfidfLogisticBaseline model is not fitted yet.")
        return np.asarray(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str] | np.ndarray) -> np.ndarray:
        """Predict class probabilities for input texts.

        Args:
            texts: List or array of text strings.

        Returns:
            Array of shape (N, 2) containing class probabilities.
        """
        if not self.is_fitted:
            raise ModelingError("TfidfLogisticBaseline model is not fitted yet.")
        return np.asarray(self.pipeline.predict_proba(texts))

    def save(self, path: str | Path) -> None:
        """Save model pipeline to disk using joblib.

        Args:
            path: Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved TfidfLogisticBaseline to: %s", p)

    @classmethod
    def load(cls, path: str | Path) -> TfidfLogisticBaseline:
        """Load model pipeline from disk.

        Args:
            path: Path to serialized model file.

        Returns:
            Loaded TfidfLogisticBaseline instance.
        """
        p = Path(path)
        if not p.is_file():
            raise ModelingError(f"Model file not found at: {p}")
        obj = joblib.load(p)
        if not isinstance(obj, TfidfLogisticBaseline):
            raise ModelingError(
                f"Loaded object is not a TfidfLogisticBaseline instance: {type(obj)}"
            )
        return obj
