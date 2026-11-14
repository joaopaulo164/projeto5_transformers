#!/usr/bin/env python
"""Wrapper script to execute end-to-end scientific experiment pipeline."""

import argparse
import sys

from sentiment_transformer.config import load_config
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.pipeline.orchestrator import ExperimentOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end sentiment classification experiment"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Custom run identifier"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload of dataset"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)

    try:
        orchestrator = ExperimentOrchestrator(
            config=config,
            run_id=args.run_id,
            force_download=args.force,
        )
        res = orchestrator.run()
        print(f"\n[SUCCESS] Experiment finished: {res['run_id']}")
        print(f"Artifacts saved to: {res['run_dir']}\n")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Experiment execution failed: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
