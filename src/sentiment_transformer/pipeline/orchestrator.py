"""Experiment orchestrator managing end-to-end pipeline execution and run artifact lifecycle."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from sentiment_transformer.config import AppConfig
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.pipeline.phases import (
    run_baselines_phase,
    run_data_phase,
    run_evaluation_phase,
    run_evidence_phase,
    run_transformer_phase,
)
from sentiment_transformer.reproducibility import set_seed

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Orchestrates end-to-end scientific experiment pipeline across all five phases."""

    def __init__(
        self,
        config: AppConfig,
        run_id: str | None = None,
        force_download: bool = False,
    ) -> None:
        """Initialize orchestrator.

        Args:
            config: Application configuration.
            run_id: Optional custom run identifier. If None, generated from timestamp and config name.
            force_download: Whether to force fresh dataset download.
        """
        self.config = config
        self.force_download = force_download

        # Generate unique, collision-free run_id
        if run_id is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_name = config.project.name.replace(" ", "_").lower()
            self.run_id = f"{timestamp_str}_{sanitized_name}"
        else:
            self.run_id = run_id

        # Setup run directory
        base_output_dir = resolve_path(config.artifacts.output_dir)
        self.run_dir = ensure_dir(base_output_dir / self.run_id)

        # Configure run logging to file and stdout
        self.log_file = self.run_dir / "execution.log"
        setup_logging(
            level=config.logging.level,
            log_file=self.log_file,
            log_format=config.logging.format,
        )

        logger.info("Initialized ExperimentOrchestrator for run '%s'", self.run_id)
        logger.info("Run directory: %s", self.run_dir)

    def run(self) -> dict[str, Any]:
        """Execute all experiment phases in sequence.

        Returns:
            Dictionary with overall experiment execution summary.
        """
        start_wall_time = time.time()
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(
            "================================================================================"
        )
        logger.info("STARTING EXPERIMENT RUN: %s", self.run_id)
        logger.info(
            "================================================================================"
        )

        # 1. Enforce reproducibility & seeds
        seed_info = set_seed(self.config.project.seed)
        logger.info("Global seed initialized: %s", seed_info)

        # 2. Save resolved config
        resolved_config_path = self.run_dir / "config_resolved.yaml"
        self.config.save_yaml(resolved_config_path)
        logger.info("Saved resolved configuration to: %s", resolved_config_path)

        phase_durations: dict[str, float] = {}
        phase_statuses: dict[str, str] = {}

        data_results: dict[str, Any] = {}
        baseline_results: dict[str, Any] = {}
        transformer_results: dict[str, Any] = {}
        eval_results: dict[str, Any] = {}
        evidence_results: dict[str, Any] = {}

        try:
            # Phase 1: Data
            phase_statuses["data"] = "running"
            data_results = run_data_phase(
                config=self.config,
                run_dir=self.run_dir,
                force_download=self.force_download,
            )
            phase_durations["data"] = data_results["duration_seconds"]
            phase_statuses["data"] = "completed"

            # Phase 2: Baselines
            phase_statuses["baselines"] = "running"
            baseline_results = run_baselines_phase(
                config=self.config,
                run_dir=self.run_dir,
            )
            phase_durations["baselines"] = baseline_results["duration_seconds"]
            phase_statuses["baselines"] = "completed"

            # Phase 3: Transformer
            phase_statuses["transformer"] = "running"
            transformer_results = run_transformer_phase(
                config=self.config,
                run_dir=self.run_dir,
            )
            phase_durations["transformer"] = transformer_results["duration_seconds"]
            phase_statuses["transformer"] = "completed"

            # Phase 4: Evaluation
            phase_statuses["evaluation"] = "running"
            eval_results = run_evaluation_phase(
                config=self.config,
                run_dir=self.run_dir,
                majority_model=baseline_results.get("majority_model"),
                tfidf_model=baseline_results.get("tfidf_model"),
                transformer_model=getattr(
                    transformer_results.get("trainer"), "model", None
                ),
                tokenizer=getattr(
                    transformer_results.get("trainer"), "tokenizer", None
                ),
            )
            phase_durations["evaluation"] = eval_results["duration_seconds"]
            phase_statuses["evaluation"] = "completed"

            # Phase 5: Evidence
            phase_statuses["evidence"] = "running"
            evidence_results = run_evidence_phase(
                config=self.config,
                run_dir=self.run_dir,
                eval_results=eval_results,
                data_results=data_results,
                phase_durations=phase_durations,
                phase_statuses=phase_statuses,
                start_iso=start_iso,
            )
            phase_durations["evidence"] = evidence_results["duration_seconds"]
            phase_statuses["evidence"] = "completed"

        except Exception as e:
            logger.exception("Experiment run failed during phase execution: %s", e)
            for phase, status in list(phase_statuses.items()):
                if status == "running":
                    phase_statuses[phase] = f"failed: {e}"
            raise

        total_duration = time.time() - start_wall_time
        logger.info(
            "================================================================================"
        )
        logger.info(
            "EXPERIMENT RUN %s COMPLETED SUCCESSFULLY in %.2fs",
            self.run_id,
            total_duration,
        )
        logger.info("All artifacts saved to: %s", self.run_dir)
        logger.info(
            "================================================================================"
        )

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": "success",
            "total_duration_seconds": round(total_duration, 2),
            "phase_durations": phase_durations,
            "phase_statuses": phase_statuses,
            "data_results": data_results,
            "eval_results": eval_results,
            "evidence_results": evidence_results,
        }
