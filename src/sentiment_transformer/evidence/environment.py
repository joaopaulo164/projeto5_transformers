"""Execution environment metadata and dependency auditing."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

from sentiment_transformer.modeling.device import get_device_info
from sentiment_transformer.paths import get_project_root

logger = logging.getLogger(__name__)


def get_git_info() -> dict[str, Any]:
    """Retrieve Git commit hash, branch, and dirty status if available."""
    git_info: dict[str, Any] = {
        "available": False,
        "commit_hash": None,
        "branch": None,
        "is_dirty": None,
    }
    try:
        root = get_project_root()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        git_info["available"] = True
        git_info["commit_hash"] = commit
        git_info["branch"] = branch
        git_info["is_dirty"] = bool(len(status) > 0)
    except Exception:
        pass

    return git_info


def capture_environment_metadata(
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Capture full environment specification including hardware, OS, and critical packages.

    Args:
        output_json_path: Optional path to save environment.json.

    Returns:
        Structured dictionary of environment metadata.
    """
    device_info = get_device_info()
    git_info = get_git_info()

    env_data: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "hardware_and_os": device_info,
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "git": git_info,
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(env_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved environment metadata to: %s", out_p)

    return env_data
