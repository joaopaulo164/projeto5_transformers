"""Validation of raw dataset files, headers, encodings, and column resolutions."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from sentiment_transformer.config import AppConfig
from sentiment_transformer.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def detect_encoding_and_separator(file_path: str | Path) -> tuple[str, str]:
    """Detect encoding and delimiter for a CSV file.

    Tries standard encodings (utf-8, utf-8-sig, latin-1) and inspects the first
    few lines to determine the delimiter (comma, semicolon, or tab).

    Args:
        file_path: Path to the CSV file.

    Returns:
        Tuple of (encoding, delimiter).

    Raises:
        DataValidationError: If file cannot be read or delimiter cannot be deduced.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DataValidationError(f"File does not exist: {path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]
    detected_encoding = None
    sample_text = None

    for enc in encodings_to_try:
        try:
            with open(path, encoding=enc, errors="strict") as f:
                sample_text = f.read(16384)
            detected_encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if detected_encoding is None or sample_text is None:
        raise DataValidationError(
            f"Unable to decode file {path} using supported encodings ({encodings_to_try})."
        )

    # Detect delimiter
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        # Fallback heuristic: count occurrences in first line
        first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
        comma_count = first_line.count(",")
        semi_count = first_line.count(";")
        tab_count = first_line.count("\t")
        if semi_count > comma_count and semi_count > tab_count:
            delimiter = ";"
        elif tab_count > comma_count and tab_count > semi_count:
            delimiter = "\t"
        else:
            delimiter = ","

    logger.info(
        "Detected file %s encoding='%s', delimiter='%s'",
        path.name,
        detected_encoding,
        delimiter,
    )
    return detected_encoding, delimiter


def find_candidate_column(
    available_columns: list[str], candidate_names: list[str]
) -> str | None:
    """Find the best matching column name from a list of candidate names.

    Matching is case-insensitive, ignores leading/trailing whitespace, and handles underscores.

    Args:
        available_columns: Columns present in the dataset.
        candidate_names: Preferred candidate column names.

    Returns:
        Matched column name from available_columns, or None.
    """
    col_map = {col.strip().lower().replace(" ", "_"): col for col in available_columns}
    for cand in candidate_names:
        cleaned_cand = cand.strip().lower().replace(" ", "_")
        if cleaned_cand in col_map:
            return col_map[cleaned_cand]
    return None


def validate_raw_file(file_path: str | Path, config: AppConfig) -> dict[str, Any]:
    """Validate a raw CSV file and resolve required columns against configuration.

    Args:
        file_path: Path to the raw CSV file.
        config: Application configuration.

    Returns:
        Dictionary containing encoding, separator, resolved column mappings, and column list.

    Raises:
        DataValidationError: If validation fails or mandatory columns cannot be determined.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DataValidationError(
            f"Raw dataset file not found at: {path}\n"
            f"Please run the download step or copy the dataset to this path manually."
        )

    size = path.stat().st_size
    if size == 0:
        raise DataValidationError(f"Raw dataset file at {path} is empty (0 bytes).")

    encoding, delimiter = detect_encoding_and_separator(path)

    # Read first line / header
    try:
        with open(path, encoding=encoding, errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
    except Exception as e:
        raise DataValidationError(f"Failed to read header from {path}: {e}") from e

    if not header:
        raise DataValidationError(f"Could not read header row from {path}.")

    clean_header = [col.strip() for col in header]

    # Resolve required columns
    candidates = config.dataset.column_candidates
    id_col = find_candidate_column(clean_header, candidates.get("id", []))
    title_col = find_candidate_column(clean_header, candidates.get("title", []))
    text_col = find_candidate_column(clean_header, candidates.get("text", []))
    rating_col = find_candidate_column(clean_header, candidates.get("rating", []))

    missing = []
    if not text_col:
        missing.append(f"Text column (candidates: {candidates.get('text', [])})")
    if not rating_col:
        missing.append(f"Rating column (candidates: {candidates.get('rating', [])})")

    if missing:
        raise DataValidationError(
            f"Required columns could not be determined in {path}.\n"
            f"Available columns: {clean_header}\n"
            f"Unresolved mandatory columns: {', '.join(missing)}\n"
            f"Please specify correct column names in configuration."
        )

    resolved_mapping = {
        "id_col": id_col,
        "title_col": title_col,
        "text_col": text_col,
        "rating_col": rating_col,
    }

    logger.info(
        "Successfully validated raw file %s. Resolved mapping: ID='%s', Title='%s', Text='%s', Rating='%s'",
        path.name,
        id_col,
        title_col,
        text_col,
        rating_col,
    )

    return {
        "file_path": str(path),
        "size_bytes": size,
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": clean_header,
        "resolved_mapping": resolved_mapping,
    }
