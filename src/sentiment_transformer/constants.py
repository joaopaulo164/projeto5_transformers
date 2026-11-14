"""Centralized constants for the sentiment transformer project."""

from typing import Final

# Label definitions
LABEL_NEGATIVE: Final[int] = 0
LABEL_POSITIVE: Final[int] = 1
LABEL_NAMES: Final[dict[int, str]] = {
    LABEL_NEGATIVE: "negative",
    LABEL_POSITIVE: "positive",
}

# Rating classifications
DEFAULT_NEGATIVE_RATINGS: Final[list[int]] = [1, 2]
DEFAULT_EXCLUDED_RATINGS: Final[list[int]] = [3]
DEFAULT_POSITIVE_RATINGS: Final[list[int]] = [4, 5]

# Standard canonical column names
COL_REVIEW_ID: Final[str] = "review_id"
COL_REVIEW_TITLE: Final[str] = "review_title"
COL_REVIEW_TEXT: Final[str] = "review_text"
COL_COMBINED_TEXT: Final[str] = "combined_text"
COL_NORMALIZED_KEY: Final[str] = "normalized_duplicate_key"
COL_OVERALL_RATING: Final[str] = "overall_rating"
COL_LABEL: Final[str] = "label"
COL_SPLIT: Final[str] = "split"

REQUIRED_PROCESSED_COLUMNS: Final[list[str]] = [
    COL_REVIEW_ID,
    COL_REVIEW_TITLE,
    COL_REVIEW_TEXT,
    COL_COMBINED_TEXT,
    COL_NORMALIZED_KEY,
    COL_OVERALL_RATING,
    COL_LABEL,
    COL_SPLIT,
]

# Dataset candidate column names
CANDIDATE_ID_COLUMNS: Final[list[str]] = [
    "review_id",
    "id",
    "submission_date",
    "reviewer_id",
]
CANDIDATE_TITLE_COLUMNS: Final[list[str]] = [
    "review_title",
    "title",
    "titulo",
    "review_header",
    "product_name",
]
CANDIDATE_TEXT_COLUMNS: Final[list[str]] = [
    "review_text",
    "text",
    "texto",
    "review_body",
    "review_comment_message",
]
CANDIDATE_RATING_COLUMNS: Final[list[str]] = [
    "overall_rating",
    "rating",
    "nota",
    "review_score",
    "stars",
    "pontuacao",
]

# Partition names
SPLIT_TRAIN: Final[str] = "train"
SPLIT_VAL: Final[str] = "val"
SPLIT_TEST: Final[str] = "test"
ALL_SPLITS: Final[list[str]] = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]

# Default model identifiers
DEFAULT_TRANSFORMER_CHECKPOINT: Final[str] = "neuralmind/bert-base-portuguese-cased"
DEFAULT_MAX_SEQ_LENGTH: Final[int] = 128

# Negation heuristics for error analysis
PORTUGUESE_NEGATION_TERMS: Final[list[str]] = [
    "nao",
    "não",
    "nunca",
    "jamais",
    "nem",
    "tampouco",
    "nada",
    "nenhum",
    "nenhuma",
    "sem",
    "decepcionou",
    "pessimo",
    "péssimo",
    "ruim",
    "horrivel",
    "horrível",
    "estragado",
    "quebrado",
    "defeituoso",
]

# Error analysis error types
ERROR_FALSE_POSITIVE: Final[str] = "false_positive"
ERROR_FALSE_NEGATIVE: Final[str] = "false_negative"
CORRECT_PREDICTION: Final[str] = "correct"

# Methodological justification for rating 3 exclusion
RATING_3_EXCLUSION_JUSTIFICATION: Final[str] = (
    "A polaridade foi definida a partir das avaliações numéricas, considerando notas 1 e 2 "
    "como negativas e notas 4 e 5 como positivas. Avaliações com nota 3 foram excluídas "
    "porque a posição intermediária na escala não garante neutralidade textual. Esses registros "
    "podem expressar opiniões mistas, polaridade moderada ou avaliações divergentes entre produto, "
    "entrega e experiência de compra. Como as estrelas funcionam como rótulos fracos do conteúdo "
    "textual, a exclusão da classe intermediária reduz ruído de rotulagem e permite avaliar, "
    "neste primeiro experimento, casos de polaridade mais claramente definida."
)
