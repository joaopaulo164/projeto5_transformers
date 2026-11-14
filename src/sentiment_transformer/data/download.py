"""Automatic download and manual fallback handling for the B2W-Reviews01 dataset."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import requests

from sentiment_transformer.config import AppConfig
from sentiment_transformer.exceptions import DataDownloadError
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.reproducibility import calculate_sha256

logger = logging.getLogger(__name__)


def download_dataset(
    config: AppConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Download the dataset from the configured official source with streaming and validation.

    Args:
        config: Application configuration.
        force: If True, redownload even if target file already exists and is valid.

    Returns:
        Dictionary with download metadata (file_path, size_bytes, sha256, downloaded).

    Raises:
        DataDownloadError: If download fails, returns non-200, HTML, or corrupted file.
    """
    raw_dir = ensure_dir(resolve_path(config.dataset.raw_dir))
    target_path = raw_dir / config.dataset.raw_file_name
    url = config.dataset.source_url
    timeout = config.dataset.timeout_seconds
    min_size = config.dataset.min_file_size_bytes

    if target_path.exists() and not force:
        file_size = target_path.stat().st_size
        if file_size >= min_size:
            logger.info(
                "Dataset already exists at %s (%d bytes). Skipping download. Use --force to redownload.",
                target_path,
                file_size,
            )
            sha256 = calculate_sha256(target_path)
            return {
                "file_path": str(target_path),
                "size_bytes": file_size,
                "sha256": sha256,
                "downloaded": False,
            }
        else:
            logger.warning(
                "Existing file %s is smaller than minimum required (%d < %d bytes). Re-downloading.",
                target_path,
                file_size,
                min_size,
            )

    logger.info("Initiating download from: %s", url)
    logger.info("Destination path: %s", target_path)

    temp_fd = None
    temp_path = None
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code != 200:
            raise DataDownloadError(
                f"Download failed with HTTP status {response.status_code} for URL: {url}\n"
                f"Manual fallback: Please download '{config.dataset.raw_file_name}' manually and "
                f"place it at '{target_path}'.",
                details={"status_code": response.status_code, "url": url},
            )

        with tempfile.NamedTemporaryFile(
            delete=False, dir=raw_dir, suffix=".tmp"
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    tmp_file.write(chunk)

        # Validate downloaded temp file
        downloaded_size = temp_path.stat().st_size
        if downloaded_size < min_size:
            # Check if content is HTML error page
            with open(temp_path, "rb") as f:
                head = f.read(1024).lower()
                if b"<html" in head or b"<!doctype html" in head:
                    raise DataDownloadError(
                        f"Downloaded content from {url} is HTML rather than CSV data.\n"
                        f"Manual fallback: Download '{config.dataset.raw_file_name}' manually and "
                        f"place it at '{target_path}'.",
                        details={"content_type": "html", "size_bytes": downloaded_size},
                    )

            raise DataDownloadError(
                f"Downloaded file is smaller than required minimum: {downloaded_size} < {min_size} bytes.\n"
                f"Manual fallback: Download '{config.dataset.raw_file_name}' manually and "
                f"place it at '{target_path}'.",
                details={"size_bytes": downloaded_size, "min_size": min_size},
            )

        # Atomic move/replace
        shutil.move(str(temp_path), str(target_path))
        temp_path = None

        sha256 = calculate_sha256(target_path)
        final_size = target_path.stat().st_size
        logger.info(
            "Download completed successfully. Path: %s | Size: %d bytes | SHA-256: %s",
            target_path,
            final_size,
            sha256,
        )

        return {
            "file_path": str(target_path),
            "size_bytes": final_size,
            "sha256": sha256,
            "downloaded": True,
        }

    except requests.RequestException as e:
        raise DataDownloadError(
            f"Network error while downloading dataset from {url}: {e}\n"
            f"Manual fallback: Please download '{config.dataset.raw_file_name}' manually and "
            f"place it at '{target_path}'.",
            details={"error": str(e), "url": url},
        ) from e
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
