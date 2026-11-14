"""Shared pytest fixtures and test configuration."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest

from sentiment_transformer.config import (
    AppConfig,
    DatasetConfig,
    SamplingConfig,
    SplitConfig,
)
from sentiment_transformer.logging_config import close_logging_handlers
from sentiment_transformer.paths import get_project_root


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return get_project_root()


@pytest.fixture
def sample_csv_path(project_root: Path) -> Path:
    """Return path to fixture sample reviews CSV."""
    return project_root / "tests" / "fixtures" / "sample_reviews.csv"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for isolated file I/O tests."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            yield Path(tmp)
        finally:
            close_logging_handlers()


@pytest.fixture
def test_config(temp_dir: Path, sample_csv_path: Path) -> AppConfig:
    """Create test configuration pointing to temporary directories."""
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(sample_csv_path, raw_dir / "B2W-Reviews01.csv")

    cfg = AppConfig(
        dataset=DatasetConfig(
            raw_dir=str(raw_dir),
            raw_file_name="B2W-Reviews01.csv",
            interim_dir=str(temp_dir / "interim"),
            processed_dir=str(temp_dir / "processed"),
            min_file_size_bytes=10,
        ),
        sampling=SamplingConfig(
            enabled=False,
            total_samples=100,
            samples_per_class=50,
            seed=42,
        ),
        split=SplitConfig(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        ),
    )
    return cfg


@pytest.fixture
def sample_processed_df() -> pd.DataFrame:
    """Create a sample processed DataFrame for modeling/evaluation testing."""
    records = []
    for i in range(40):
        label = 0 if i < 20 else 1
        rating = 1 if label == 0 else 5
        text = (
            f"Produto pessimo nao recomendo defeito numero {i}"
            if label == 0
            else f"Excelente produto adorei muito bom qualidade {i}"
        )
        records.append(
            {
                "review_id": f"rev_{i:03d}",
                "review_title": f"Titulo {i}",
                "review_text": text,
                "combined_text": f"Titulo {i} [SEP] {text}",
                "normalized_duplicate_key": f"titulo {i} || {text}",
                "overall_rating": rating,
                "label": label,
                "split": "train" if i < 28 else ("val" if i < 34 else "test"),
            }
        )
    return pd.DataFrame(records)
