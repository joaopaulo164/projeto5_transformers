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
    urls_to_try = [config.dataset.source_url]
    known_mirrors = [
        "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/master/B2W-Reviews01.csv",
        "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv",
        "https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv",
    ]
    for mirror in known_mirrors:
        if mirror not in urls_to_try:
            urls_to_try.append(mirror)

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

    logger.info("Destination path: %s", target_path)

    last_error: Exception | None = None
    temp_path: Path | None = None

    for url in urls_to_try:
        logger.info("Attempting download from: %s", url)
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            if response.status_code != 200:
                logger.warning("HTTP %d received from %s", response.status_code, url)
                last_error = DataDownloadError(
                    f"Download failed with HTTP status {response.status_code} for URL: {url}",
                    details={"status_code": response.status_code, "url": url},
                )
                continue

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
                is_html = False
                with open(temp_path, "rb") as f:
                    head = f.read(1024).lower()
                    if b"<html" in head or b"<!doctype html" in head:
                        is_html = True

                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                temp_path = None

                if is_html:
                    logger.warning("Downloaded content from %s is HTML.", url)
                    last_error = DataDownloadError(
                        f"Downloaded content from {url} is HTML rather than CSV data.",
                        details={"content_type": "html", "size_bytes": downloaded_size},
                    )
                else:
                    last_error = DataDownloadError(
                        f"Downloaded file from {url} is smaller than required minimum: {downloaded_size} < {min_size} bytes.",
                        details={"size_bytes": downloaded_size, "min_size": min_size},
                    )
                continue

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
            logger.warning("Network error attempting %s: %s", url, e)
            last_error = e
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
                temp_path = None

    raise DataDownloadError(
        f"All download attempts failed for dataset '{config.dataset.raw_file_name}'. Last error: {last_error}\n"
        f"Manual fallback: Please download '{config.dataset.raw_file_name}' manually and "
        f"place it at '{target_path}'.",
        details={"last_error": str(last_error), "urls_attempted": urls_to_try},
    )
