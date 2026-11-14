"""Command-Line Interface (CLI) entrypoint for the Sentiment Transformer pipeline."""

from __future__ import annotations

import argparse
import sys

from sentiment_transformer.config import load_config
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.pipeline.orchestrator import ExperimentOrchestrator
from sentiment_transformer.pipeline.phases import (
    run_baselines_phase,
    run_data_phase,
    run_evaluation_phase,
    run_evidence_phase,
    run_transformer_phase,
)


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="sentiment-transformer",
        description=(
            "Binary Sentiment Classification pipeline using BERTimbau, Baselines, "
            "and PyTorch on Portuguese product reviews."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Execution mode")

    # Command: run (Pipeline Completo - Nível 3)
    run_parser = subparsers.add_parser(
        "run",
        help="Execute end-to-end experiment pipeline (Data -> Baselines -> Transformer -> Evaluation -> Evidence)",
    )
    run_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional custom run ID name (defaults to timestamp-based ID)",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload of dataset even if local file exists",
    )

    # Command: phase (Execução por Fase - Nível 2)
    phase_parser = subparsers.add_parser(
        "phase",
        help="Execute a specific pipeline phase in isolation",
    )
    phase_parser.add_argument(
        "phase_name",
        choices=["data", "baselines", "transformer", "evaluation", "evidence"],
        help="Name of the phase to execute",
    )
    phase_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    phase_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Target run directory for phase artifacts (defaults to artifacts/runs/latest_phase)",
    )
    phase_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload during data phase",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """CLI execution entrypoint.

    Args:
        args: Optional list of CLI arguments (defaults to sys.argv[1:]).

    Returns:
        Integer exit code (0 for success, non-zero for error).
    """
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    try:
        config = load_config(parsed_args.config)
        setup_logging(level=config.logging.level, log_format=config.logging.format)

        if parsed_args.command == "run":
            orchestrator = ExperimentOrchestrator(
                config=config,
                run_id=parsed_args.run_id,
                force_download=parsed_args.force,
            )
            result = orchestrator.run()
            print(f"\n[SUCCESS] Run completed: {result['run_id']}")
            print(f"[ARTIFACTS] Directory: {result['run_dir']}\n")
            return 0

        elif parsed_args.command == "phase":
            phase_name = parsed_args.phase_name
            run_dir_path = (
                resolve_path(parsed_args.run_dir)
                if parsed_args.run_dir
                else ensure_dir(
                    resolve_path(config.artifacts.output_dir) / f"phase_{phase_name}"
                )
            )

            if phase_name == "data":
                run_data_phase(
                    config, run_dir=run_dir_path, force_download=parsed_args.force
                )
            elif phase_name == "baselines":
                run_baselines_phase(config, run_dir=run_dir_path)
            elif phase_name == "transformer":
                run_transformer_phase(config, run_dir=run_dir_path)
            elif phase_name == "evaluation":
                run_evaluation_phase(config, run_dir=run_dir_path)
            elif phase_name == "evidence":
                # For isolated evidence phase, run evaluation first if needed
                eval_res = run_evaluation_phase(config, run_dir=run_dir_path)
                run_evidence_phase(config, run_dir=run_dir_path, eval_results=eval_res)

            print(f"\n[SUCCESS] Phase '{phase_name}' completed.")
            print(f"[ARTIFACTS] Saved in: {run_dir_path}\n")
            return 0

        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
