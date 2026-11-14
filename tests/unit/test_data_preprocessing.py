"""Unit tests for binary labeling, rating 3 exclusion, and textual composition."""

from __future__ import annotations

import pandas as pd

from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
)
from sentiment_transformer.data.preprocessing import (
    compose_input_text,
    map_rating_to_label,
    normalize_spaces,
    preprocess_dataset,
)


def test_normalize_spaces() -> None:
    raw = "  Olá,   mundo!\n\tTeste  com espaços   múltiplos.  "
    norm = normalize_spaces(raw)
    assert norm == "Olá, mundo! Teste com espaços múltiplos."


def test_compose_input_text_with_title_and_body() -> None:
    title = "Excelente compra"
    body = "Adorei a qualidade e rapidez na entrega."
    res = compose_input_text(title, body, sep_token="[SEP]")
    assert res == "Excelente compra [SEP] Adorei a qualidade e rapidez na entrega."


def test_compose_input_text_without_title() -> None:
    title = ""
    body = "Apenas o corpo da avaliacao."
    res = compose_input_text(title, body, sep_token="[SEP]")
    assert res == "Apenas o corpo da avaliacao."


def test_compose_input_text_empty_body() -> None:
    title = "Apenas titulo"
    body = "   "
    res = compose_input_text(title, body, sep_token="[SEP]")
    assert res == ""


def test_map_rating_to_label() -> None:
    cfg = AppConfig()
    # Negatives (1, 2)
    assert map_rating_to_label(1, cfg) == LABEL_NEGATIVE
    assert map_rating_to_label(2, cfg) == LABEL_NEGATIVE
    assert map_rating_to_label("1.0", cfg) == LABEL_NEGATIVE

    # Excluded (3)
    assert map_rating_to_label(3, cfg) is None
    assert map_rating_to_label(3.0, cfg) is None

    # Positives (4, 5)
    assert map_rating_to_label(4, cfg) == LABEL_POSITIVE
    assert map_rating_to_label(5, cfg) == LABEL_POSITIVE

    # Invalid / None
    assert map_rating_to_label(None, cfg) is None
    assert map_rating_to_label("invalid", cfg) is None


def test_preprocess_dataset_pipeline() -> None:
    df = pd.DataFrame(
        [
            {
                COL_REVIEW_ID: "1",
                COL_REVIEW_TITLE: "Otimo",
                COL_REVIEW_TEXT: "Muito bom",
                COL_OVERALL_RATING: 5,
            },
            {
                COL_REVIEW_ID: "2",
                COL_REVIEW_TITLE: "Ruim",
                COL_REVIEW_TEXT: "Pessimo produto",
                COL_OVERALL_RATING: 1,
            },
            {
                COL_REVIEW_ID: "3",
                COL_REVIEW_TITLE: "Medio",
                COL_REVIEW_TEXT: "Mais ou menos",
                COL_OVERALL_RATING: 3,
            },  # Excluded
            {
                COL_REVIEW_ID: "4",
                COL_REVIEW_TITLE: "",
                COL_REVIEW_TEXT: "Sem titulo mas positivo",
                COL_OVERALL_RATING: 4,
            },
            {
                COL_REVIEW_ID: "5",
                COL_REVIEW_TITLE: "Titulo",
                COL_REVIEW_TEXT: "   ",
                COL_OVERALL_RATING: 2,
            },  # Empty body excluded
        ]
    )

    cfg = AppConfig()
    prep_df, stats = preprocess_dataset(df, cfg, sep_token="[SEP]")

    assert len(prep_df) == 3
    assert stats["rating_3_excluded"] == 1
    assert stats["empty_text_excluded"] == 1
    assert stats["label_distribution"]["negative_0"] == 1
    assert stats["label_distribution"]["positive_1"] == 2
    assert stats["with_title_count"] == 2
    assert stats["without_title_count"] == 1
