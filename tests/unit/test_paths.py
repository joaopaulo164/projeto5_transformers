"""Unit tests for path management and filesystem resolution."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.paths import ensure_dir, get_project_root, resolve_path


def test_get_project_root() -> None:
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.is_dir()
    assert (root / "configs").exists() or (root / "src").exists()


def test_resolve_path_relative() -> None:
    p = resolve_path("configs/smoke.yaml")
    assert p.is_absolute()
    assert p.name == "smoke.yaml"


def test_resolve_path_absolute(temp_dir: Path) -> None:
    abs_p = temp_dir / "file.txt"
    resolved = resolve_path(abs_p)
    assert resolved == abs_p


def test_ensure_dir(temp_dir: Path) -> None:
    target = temp_dir / "nested" / "subdir"
    assert not target.exists()
    ensured = ensure_dir(target)
    assert target.is_dir()
    assert ensured == target
