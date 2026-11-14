"""Baseline models: Majority class classifier and TF-IDF Logistic Regression."""

from sentiment_transformer.baselines.majority import MajorityBaseline
from sentiment_transformer.baselines.tfidf_logistic import TfidfLogisticBaseline

__all__ = [
    "MajorityBaseline",
    "TfidfLogisticBaseline",
]
