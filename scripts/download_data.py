#!/usr/bin/env python
"""Wrapper script to download dataset from official source."""

import argparse
import sys

from sentiment_transformer.config import load_config
from sentiment_transformer.data.download import download_dataset
from sentiment_transformer.logging_config import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Download B2W-Reviews01 dataset")
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
        res = download_dataset(config, force=args.force)
        print(
            f"[SUCCESS] Dataset ready at: {res['file_path']} (SHA-256: {res['sha256']})"
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
