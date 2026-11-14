#!/usr/bin/env python
"""Wrapper script to execute data preparation phase."""

import argparse
import sys

from sentiment_transformer.config import load_config
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.pipeline.phases import run_data_phase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare dataset: preprocess, deduplicate, and split"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)

    try:
        res = run_data_phase(config, force_download=args.force)
        print(
            f"[SUCCESS] Data preparation completed in {res['duration_seconds']:.2f}s."
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
