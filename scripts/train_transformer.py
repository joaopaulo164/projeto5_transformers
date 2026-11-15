#!/usr/bin/env python
"""Wrapper script to fine-tune BERTimbau Transformer."""

import argparse
from pathlib import Path
import sys

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from sentiment_transformer.config import load_config
from sentiment_transformer.logging_config import setup_logging
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.pipeline.phases import run_transformer_phase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTimbau Transformer model"
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
        default="artifacts/transformer",
        help="Directory to save checkpoint",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level)
    run_dir = ensure_dir(resolve_path(args.run_dir))

    try:
        res = run_transformer_phase(config, run_dir=run_dir)
        print(
            f"[SUCCESS] Transformer fine-tuning completed in {res['duration_seconds']:.2f}s."
        )
        print(f"Best checkpoint at: {res['checkpoint_dir']}")
        return 0
    except Exception as e:
        print(f"[ERROR] Transformer training failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
