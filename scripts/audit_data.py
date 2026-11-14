#!/usr/bin/env python
"""Wrapper script to audit raw dataset."""

import argparse
import sys

from sentiment_transformer.config import load_config
from sentiment_transformer.data.audit import audit_dataset
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit B2W-Reviews01 raw dataset")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="artifacts/dataset_audit.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)

    raw_path = resolve_path(config.dataset.raw_dir) / config.dataset.raw_file_name
    if not raw_path.is_file():
        print(f"[ERROR] Raw dataset not found at: {raw_path}", file=sys.stderr)
        print("Please run scripts/download_data.py first.", file=sys.stderr)
        return 1

    try:
        res = audit_dataset(
            raw_path, config, output_json_path=resolve_path(args.output)
        )
        print(
            f"[SUCCESS] Audit completed for {res['total_rows']} rows. JSON saved to {args.output}"
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Audit failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
