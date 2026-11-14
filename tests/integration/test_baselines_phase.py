"""Integration tests for baseline models training phase."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.config import AppConfig
from sentiment_transformer.pipeline.phases import run_baselines_phase, run_data_phase


def test_run_baselines_phase_integration(
    test_config: AppConfig, temp_dir: Path
) -> None:
    run_dir = temp_dir / "run_baselines_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run data phase first
    run_data_phase(config=test_config, run_dir=run_dir)

    # 2. Run baselines phase
    res = run_baselines_phase(config=test_config, run_dir=run_dir)

    assert res["status"] == "success"
    assert (run_dir / "baseline_majority.joblib").is_file()
    assert (run_dir / "baseline_tfidf.joblib").is_file()
