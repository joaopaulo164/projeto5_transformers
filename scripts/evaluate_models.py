#!/usr/bin/env python
"""Wrapper script to evaluate models and compute metrics."""

import argparse
import sys

from sentiment_transformer.config import load_config
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.pipeline.phases import run_evaluation_phase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and Transformer models on test set"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory with models and outputs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)
    run_dir = ensure_dir(resolve_path(args.run_dir))

    try:
        res = run_evaluation_phase(config, run_dir=run_dir)
        print(
            f"[SUCCESS] Evaluation completed in {res['duration_seconds']:.2f}s. Artifacts in {run_dir}"
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
