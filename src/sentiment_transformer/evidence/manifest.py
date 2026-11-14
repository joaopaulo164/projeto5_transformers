"""Run manifest generation consolidating complete scientific provenance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sentiment_transformer.config import AppConfig

logger = logging.getLogger(__name__)


def create_run_manifest(
    run_id: str,
    start_time: str,
    end_time: str,
    total_duration_seconds: float,
    config: AppConfig,
    environment_metadata: dict[str, Any],
    dataset_fingerprints: dict[str, Any],
    dataset_audit_stats: dict[str, Any] | None,
    preprocessing_stats: dict[str, Any] | None,
    deduplication_stats: dict[str, Any] | None,
    split_stats: dict[str, Any] | None,
    phase_durations: dict[str, float],
    phase_statuses: dict[str, str],
    artifacts_created: list[str],
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate comprehensive run manifest for scientific lineage.

    Args:
        run_id: Unique string identifier for the execution run.
        start_time: ISO-8601 formatted start timestamp.
        end_time: ISO-8601 formatted end timestamp.
        total_duration_seconds: Total wall clock duration in seconds.
        config: Application configuration.
        environment_metadata: Environment audit dictionary.
        dataset_fingerprints: Cryptographic hashes of datasets and partitions.
        dataset_audit_stats: Audit results on raw data.
        preprocessing_stats: Filtering and labeling statistics.
        deduplication_stats: Duplicate and conflict resolution statistics.
        split_stats: Partition statistics (train/val/test).
        phase_durations: Dictionary of execution durations per phase.
        phase_statuses: Dictionary of execution statuses per phase.
        artifacts_created: List of all generated files in run directory.
        output_json_path: Optional path to save run_manifest.json.

    Returns:
        Structured manifest dictionary.
    """
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamps": {
            "start": start_time,
            "end": end_time,
            "total_duration_seconds": round(total_duration_seconds, 2),
            "phase_durations_seconds": {
                k: round(v, 2) for k, v in phase_durations.items()
            },
        },
        "phase_statuses": phase_statuses,
        "config_resolved": config.to_dict(),
        "environment": environment_metadata,
        "data_provenance": {
            "fingerprints": dataset_fingerprints,
            "raw_audit": dataset_audit_stats,
            "preprocessing": preprocessing_stats,
            "deduplication": deduplication_stats,
            "splits": split_stats,
        },
        "artifacts": sorted(artifacts_created),
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Saved run manifest to: %s", out_p)

    return manifest
