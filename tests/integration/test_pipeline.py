"""Integration tests for experiment orchestration."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.config import AppConfig
from sentiment_transformer.pipeline.phases import (
    run_baselines_phase,
    run_data_phase,
    run_evaluation_phase,
    run_evidence_phase,
)


def test_orchestration_flow_modular(test_config: AppConfig, temp_dir: Path) -> None:
    run_dir = temp_dir / "run_flow_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data phase
    data_res = run_data_phase(test_config, run_dir=run_dir)
    assert data_res["status"] == "success"

    # 2. Baselines phase
    base_res = run_baselines_phase(test_config, run_dir=run_dir)
    assert base_res["status"] == "success"

    # 3. Evaluation phase for baselines
    eval_res = run_evaluation_phase(
        test_config,
        run_dir=run_dir,
        majority_model=base_res["majority_model"],
        tfidf_model=base_res["tfidf_model"],
    )
    assert eval_res["status"] == "success"
    assert (run_dir / "baseline_majority_metrics.json").is_file()
    assert (run_dir / "baseline_tfidf_metrics.json").is_file()

    # 4. Evidence phase
    ev_res = run_evidence_phase(
        test_config,
        run_dir=run_dir,
        eval_results=eval_res,
        data_results=data_res,
        phase_durations={"data": 1.0, "baselines": 1.0, "eval": 1.0},
    )
    assert ev_res["status"] == "success"
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "model_comparison.md").is_file()
