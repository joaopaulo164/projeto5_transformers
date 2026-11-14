"""Integration tests for the end-to-end data preparation phase."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.config import AppConfig
from sentiment_transformer.pipeline.phases import run_data_phase


def test_run_data_phase_integration(test_config: AppConfig, temp_dir: Path) -> None:
    run_dir = temp_dir / "run_data_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    res = run_data_phase(config=test_config, run_dir=run_dir, force_download=False)

    assert res["status"] == "success"
    assert "audit_stats" in res
    assert "prep_stats" in res
    assert "dedup_stats" in res
    assert "split_stats" in res

    proc_dir = Path(test_config.dataset.processed_dir)
    assert (proc_dir / "train.csv").is_file()
    assert (proc_dir / "val.csv").is_file()
    assert (proc_dir / "test.csv").is_file()
    assert (proc_dir / "all_processed.csv").is_file()
