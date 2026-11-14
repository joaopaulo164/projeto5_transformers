"""Unit tests for dataset download, streaming, and error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentiment_transformer.config import AppConfig
from sentiment_transformer.data.download import download_dataset
from sentiment_transformer.exceptions import DataDownloadError


def test_download_dataset_already_exists(sample_csv_path: Path, temp_dir: Path) -> None:
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    target_f = raw_dir / "B2W-Reviews01.csv"
    shutil.copy(sample_csv_path, target_f)

    cfg = AppConfig()
    cfg.dataset.raw_dir = str(raw_dir)
    cfg.dataset.raw_file_name = "B2W-Reviews01.csv"
    cfg.dataset.min_file_size_bytes = 10

    res = download_dataset(cfg, force=False)
    assert res["downloaded"] is False
    assert res["size_bytes"] == target_f.stat().st_size


@patch("sentiment_transformer.data.download.requests.get")
def test_download_dataset_success(mock_get: MagicMock, temp_dir: Path) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"col1,col2\n", b"val1,val2\n"]
    mock_get.return_value = mock_resp

    cfg = AppConfig()
    cfg.dataset.raw_dir = str(temp_dir / "raw")
    cfg.dataset.min_file_size_bytes = 5

    res = download_dataset(cfg, force=True)
    assert res["downloaded"] is True
    assert Path(res["file_path"]).is_file()


@patch("sentiment_transformer.data.download.requests.get")
def test_download_dataset_http_error(mock_get: MagicMock, temp_dir: Path) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    cfg = AppConfig()
    cfg.dataset.raw_dir = str(temp_dir / "raw")

    with pytest.raises(DataDownloadError, match="HTTP status 404"):
        download_dataset(cfg, force=True)


@patch("sentiment_transformer.data.download.requests.get")
def test_download_dataset_html_error(mock_get: MagicMock, temp_dir: Path) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [
        b"<!DOCTYPE html><html><body>Error</body></html>"
    ]
    mock_get.return_value = mock_resp

    cfg = AppConfig()
    cfg.dataset.raw_dir = str(temp_dir / "raw")
    cfg.dataset.min_file_size_bytes = 1000

    with pytest.raises(DataDownloadError, match="HTML rather than CSV"):
        download_dataset(cfg, force=True)
