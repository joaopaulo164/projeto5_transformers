"""Unit tests for ExperimentOrchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from sentiment_transformer.config import AppConfig
from sentiment_transformer.pipeline.orchestrator import ExperimentOrchestrator


def test_orchestrator_initialization(temp_dir: Path) -> None:
    cfg = AppConfig()
    cfg.artifacts.output_dir = str(temp_dir / "runs")

    orch = ExperimentOrchestrator(config=cfg, run_id="custom_run_01")
    assert orch.run_id == "custom_run_01"
    assert orch.run_dir.exists()
    assert (orch.run_dir / "execution.log").exists()


@patch("sentiment_transformer.pipeline.orchestrator.run_data_phase")
@patch("sentiment_transformer.pipeline.orchestrator.run_baselines_phase")
@patch("sentiment_transformer.pipeline.orchestrator.run_transformer_phase")
@patch("sentiment_transformer.pipeline.orchestrator.run_evaluation_phase")
@patch("sentiment_transformer.pipeline.orchestrator.run_evidence_phase")
def test_orchestrator_run_mocked(
    mock_ev: MagicMock,
    mock_eval: MagicMock,
    mock_tf: MagicMock,
    mock_base: MagicMock,
    mock_data: MagicMock,
    temp_dir: Path,
) -> None:
    mock_data.return_value = {"duration_seconds": 1.0, "status": "success"}
    mock_base.return_value = {"duration_seconds": 1.0, "status": "success"}
    mock_tf.return_value = {"duration_seconds": 1.0, "status": "success"}
    mock_eval.return_value = {"duration_seconds": 1.0, "status": "success"}
    mock_ev.return_value = {"duration_seconds": 1.0, "status": "success"}

    cfg = AppConfig()
    cfg.artifacts.output_dir = str(temp_dir / "runs")

    orch = ExperimentOrchestrator(config=cfg, run_id="test_mock_run")
    result = orch.run()

    assert result["status"] == "success"
    assert (orch.run_dir / "config_resolved.yaml").is_file()
