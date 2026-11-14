"""Unit tests for Majority and TF-IDF baseline models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sentiment_transformer.baselines.majority import MajorityBaseline
from sentiment_transformer.baselines.tfidf_logistic import TfidfLogisticBaseline
from sentiment_transformer.config import BaselineTfidfConfig


def test_majority_baseline(temp_dir: Path) -> None:
    X_train = ["pessimo", "ruim", "horrivel", "otimo"]
    y_train = np.array([0, 0, 0, 1])

    clf = MajorityBaseline(strategy="most_frequent")
    clf.fit(X_train, y_train)

    X_test = ["novo texto", "outro teste"]
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    assert len(preds) == 2
    assert (preds == 0).all()
    assert proba.shape == (2, 2)
    assert proba[0, 0] >= 0.5

    # Test serialization
    save_p = temp_dir / "maj.joblib"
    clf.save(save_p)
    loaded = MajorityBaseline.load(save_p)
    loaded_preds = loaded.predict(X_test)
    assert (loaded_preds == preds).all()


def test_tfidf_logistic_baseline(temp_dir: Path) -> None:
    X_train = [
        "produto pessimo e defeituoso",
        "nao gostei muito ruim quebrado",
        "horrivel pessima compra",
        "excelente produto muito bom",
        "otima qualidade adorei recomendo",
        "perfeito maravilhoso entrega rapida",
    ]
    y_train = np.array([0, 0, 0, 1, 1, 1])

    cfg = BaselineTfidfConfig(ngram_range=[1, 2], max_features=100, min_df=1)
    clf = TfidfLogisticBaseline.from_config(cfg)
    clf.fit(X_train, y_train)

    X_test = ["produto excelente maravilhoso", "pessimo quebrado defeito"]
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    assert preds[0] == 1  # positive
    assert preds[1] == 0  # negative
    assert proba.shape == (2, 2)
    assert proba[0, 1] > 0.5
    assert proba[1, 0] > 0.5

    # Test serialization
    save_p = temp_dir / "tfidf.joblib"
    clf.save(save_p)
    loaded = TfidfLogisticBaseline.load(save_p)
    loaded_preds = loaded.predict(X_test)
    assert (loaded_preds == preds).all()
