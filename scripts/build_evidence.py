#!/usr/bin/env python
"""Wrapper script to consolidate evidence package and manifest."""

import argparse
from pathlib import Path
import sys

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from sentiment_transformer.config import load_config
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.pipeline.phases import (
    run_evaluation_phase,
    run_evidence_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate evidence, error analysis, and manifest"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--run-dir", type=str, default="artifacts/evidence", help="Run directory"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)
    run_dir = ensure_dir(resolve_path(args.run_dir))

    try:
        eval_res = run_evaluation_phase(config, run_dir=run_dir)
        run_evidence_phase(config, run_dir=run_dir, eval_results=eval_res)
        print(f"[SUCCESS] Evidence consolidated in {run_dir}")
        return 0
    except Exception as e:
        print(f"[ERROR] Evidence building failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
