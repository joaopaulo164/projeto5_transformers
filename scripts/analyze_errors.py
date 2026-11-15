#!/usr/bin/env python
"""Wrapper script to execute error analysis on Transformer predictions."""

import argparse
from pathlib import Path
import sys

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pandas as pd

from sentiment_transformer.config import load_config
from sentiment_transformer.evaluation.error_analysis import build_error_analysis
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run error analysis on model predictions"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        default="artifacts/evaluation/predictions_test_transformer.csv",
        help="Predictions CSV",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="artifacts/evaluation/error_analysis.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)

    pred_path = resolve_path(args.predictions)
    if not pred_path.is_file():
        print(f"[ERROR] Predictions file not found at: {pred_path}", file=sys.stderr)
        return 1

    try:
        preds_df = pd.read_csv(pred_path, encoding="utf-8")
        out_p = resolve_path(args.output)
        error_df, stats = build_error_analysis(preds_df, output_csv_path=out_p)
        print(
            f"[SUCCESS] Error analysis completed ({len(error_df)} cases selected). Saved to {out_p}"
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Error analysis failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
