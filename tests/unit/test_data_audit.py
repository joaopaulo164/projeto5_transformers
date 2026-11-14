"""Unit tests for dataset auditing."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.config import AppConfig
from sentiment_transformer.data.audit import audit_dataset


def test_audit_dataset(sample_csv_path: Path, temp_dir: Path) -> None:
    cfg = AppConfig()
    out_json = temp_dir / "audit.json"
    audit_res = audit_dataset(sample_csv_path, cfg, output_json_path=out_json)

    assert audit_res["total_rows"] == 20
    assert "sha256" in audit_res
    assert audit_res["rating_distribution"]["5"] > 0
    assert audit_res["rating_distribution"]["1"] > 0
    assert audit_res["rating_3_count"] == 2
    assert out_json.is_file()
