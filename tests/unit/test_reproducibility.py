"""Unit tests for random seed setting, determinism flags, and hashing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentiment_transformer.reproducibility import (
    calculate_dataframe_hash,
    calculate_sha256,
    set_seed,
)


def test_set_seed_deterministic() -> None:
    set_seed(42)
    py_val1 = random.random()
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()

    set_seed(42)
    py_val2 = random.random()
    np_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()

    assert py_val1 == py_val2
    assert np_val1 == np_val2
    assert torch_val1 == torch_val2


def test_calculate_sha256(temp_dir: Path) -> None:
    test_file = temp_dir / "hash_test.txt"
    test_file.write_text("Hello scientific world!", encoding="utf-8")

    digest1 = calculate_sha256(test_file)
    digest2 = calculate_sha256(test_file)
    assert len(digest1) == 64
    assert digest1 == digest2


def test_calculate_dataframe_hash() -> None:
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df3 = pd.DataFrame({"a": [1, 3], "b": ["x", "y"]})

    h1 = calculate_dataframe_hash(df1)
    h2 = calculate_dataframe_hash(df2)
    h3 = calculate_dataframe_hash(df3)

    assert h1 == h2
    assert h1 != h3
