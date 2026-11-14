"""Filesystem path helpers using pathlib for cross-platform Windows compatibility."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Attempts detection based on this file's location (src/sentiment_transformer/paths.py)
    or falls back to current working directory if pyproject.toml is found there.
    """
    # 3 levels up: src/sentiment_transformer/paths.py -> src/sentiment_transformer -> src -> root
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "pyproject.toml").exists() or (candidate / "configs").exists():
        return candidate

    cwd = Path(os.getcwd()).resolve()
    if (cwd / "pyproject.toml").exists() or (cwd / "configs").exists():
        return cwd

    return candidate


def resolve_path(path_str: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path string or Path object relative to project root if not absolute.

    Args:
        path_str: The path string or Path object to resolve.
        base_dir: Optional base directory to resolve against. Defaults to project root.

    Returns:
        Absolute Path object.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    base = base_dir if base_dir is not None else get_project_root()
    return (base / p).resolve()


def ensure_dir(dir_path: str | Path) -> Path:
    """Ensure a directory exists, creating parents if necessary.

    Args:
        dir_path: Path to directory.

    Returns:
        The ensured Path object.
    """
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p
