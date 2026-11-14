"""Unit tests for evidence, fingerprints, environment, reports, and run manifest."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.config import AppConfig
from sentiment_transformer.evidence.environment import capture_environment_metadata
from sentiment_transformer.evidence.fingerprints import compute_dataset_fingerprints
from sentiment_transformer.evidence.manifest import create_run_manifest
from sentiment_transformer.evidence.report import generate_comparison_reports


def test_capture_environment_metadata(temp_dir: Path) -> None:
    out_json = temp_dir / "env.json"
    env_data = capture_environment_metadata(out_json)
    assert "hardware_and_os" in env_data
    assert "packages" in env_data
    assert out_json.is_file()


def test_compute_dataset_fingerprints(sample_csv_path: Path, temp_dir: Path) -> None:
    out_json = temp_dir / "fingerprints.json"
    fp = compute_dataset_fingerprints(
        raw_file_path=sample_csv_path, output_json_path=out_json
    )
    assert fp["raw_file"] is not None
    assert "sha256" in fp["raw_file"]
    assert out_json.is_file()


def test_generate_comparison_reports(temp_dir: Path) -> None:
    metrics_list = [
        {
            "model_name": "Majority",
            "accuracy": 0.5,
            "f1_macro": 0.33,
            "f1_weighted": 0.33,
            "class_0_negative": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0},
            "class_1_positive": {"f1_score": 0.66, "precision": 0.5, "recall": 1.0},
        },
        {
            "model_name": "TF-IDF",
            "accuracy": 0.85,
            "f1_macro": 0.84,
            "f1_weighted": 0.85,
            "class_0_negative": {"f1_score": 0.84, "precision": 0.85, "recall": 0.83},
            "class_1_positive": {"f1_score": 0.86, "precision": 0.85, "recall": 0.87},
        },
    ]
    csv_p, md_p = generate_comparison_reports(metrics_list, temp_dir)
    assert csv_p.is_file()
    assert md_p.is_file()
    md_content = md_p.read_text(encoding="utf-8")
    assert "TF-IDF" in md_content
    assert "Majority" in md_content


def test_create_run_manifest(temp_dir: Path) -> None:
    cfg = AppConfig()
    out_json = temp_dir / "manifest.json"
    manifest = create_run_manifest(
        run_id="test_run_001",
        start_time="2026-11-13T10:00:00Z",
        end_time="2026-11-13T10:05:00Z",
        total_duration_seconds=300.0,
        config=cfg,
        environment_metadata={},
        dataset_fingerprints={},
        dataset_audit_stats={},
        preprocessing_stats={},
        deduplication_stats={},
        split_stats={},
        phase_durations={"data": 10.0},
        phase_statuses={"data": "completed"},
        artifacts_created=["config_resolved.yaml"],
        output_json_path=out_json,
    )
    assert manifest["run_id"] == "test_run_001"
    assert out_json.is_file()
