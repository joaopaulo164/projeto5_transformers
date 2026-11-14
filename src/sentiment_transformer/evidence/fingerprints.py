"""Cryptographic data fingerprinting for experiment lineage and auditability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sentiment_transformer.paths import resolve_path
from sentiment_transformer.reproducibility import calculate_sha256

logger = logging.getLogger(__name__)


def compute_dataset_fingerprints(
    raw_file_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate SHA-256 fingerprints for raw dataset and processed partition files.

    Args:
        raw_file_path: Path to raw dataset CSV.
        processed_dir: Path to directory containing processed train/val/test CSVs.
        output_json_path: Optional path to save fingerprints JSON.

    Returns:
        Dictionary of dataset fingerprints.
    """
    fingerprints: dict[str, Any] = {
        "raw_file": None,
        "partitions": {},
    }

    if raw_file_path is not None:
        raw_p = resolve_path(raw_file_path)
        if raw_p.is_file():
            fingerprints["raw_file"] = {
                "file_name": raw_p.name,
                "file_path": str(raw_p),
                "size_bytes": raw_p.stat().st_size,
                "sha256": calculate_sha256(raw_p),
            }

    if processed_dir is not None:
        proc_p = resolve_path(processed_dir)
        if proc_p.is_dir():
            for split_name in ["train.csv", "val.csv", "test.csv", "all_processed.csv"]:
                split_file = proc_p / split_name
                if split_file.is_file():
                    fingerprints["partitions"][split_name] = {
                        "file_path": str(split_file),
                        "size_bytes": split_file.stat().st_size,
                        "sha256": calculate_sha256(split_file),
                    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(fingerprints, f, indent=2, ensure_ascii=False)
        logger.info("Saved dataset fingerprints to: %s", out_p)

    return fingerprints
