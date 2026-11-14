"""Reproducibility utilities: seed setting, deterministic flags, and cryptographic hashing."""

from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> dict[str, Any]:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible runs.

    Note on GPU determinism:
    While setting seeds and PyTorch flags ensures high reproducibility, some CUDA
    operations (like certain atomic additions or specific cuDNN convolution kernels)
    may still exhibit minor numerical variations across different GPU architectures
    or driver versions.

    Args:
        seed: Integer seed value.
        deterministic_torch: Whether to enforce PyTorch deterministic algorithms.

    Returns:
        Dictionary summarizing the configured seed settings.
    """
    logger.info(
        "Setting global random seed to %d (deterministic_torch=%s)",
        seed,
        deterministic_torch,
    )

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic operations if available in PyTorch version
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as e:
                logger.warning(
                    "Could not enable strict deterministic algorithms: %s", e
                )

    return {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_available": cuda_available,
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", True)),
    }


def calculate_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file on disk.

    Args:
        file_path: Path to the file.
        chunk_size: Byte size per chunk during streaming read.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot calculate SHA-256: file not found at {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash for a pandas DataFrame content.

    Args:
        df: Pandas DataFrame.

    Returns:
        Hexadecimal SHA-256 digest string of CSV representation.
    """
    csv_bytes = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()
