"""Unit tests for configuration loading, validation, and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentiment_transformer.config import AppConfig, load_config
from sentiment_transformer.exceptions import ConfigError


def test_default_config_valid() -> None:
    cfg = AppConfig()
    cfg.validate()
    assert cfg.project.name == "sentiment_transformer"
    assert cfg.transformer.num_labels == 2
    assert cfg.split.train_ratio == 0.70


def test_config_invalid_split_sum() -> None:
    cfg = AppConfig()
    cfg.split.train_ratio = 0.50
    cfg.split.val_ratio = 0.20
    cfg.split.test_ratio = 0.20  # sum = 0.90 != 1.0
    with pytest.raises(ConfigError, match="Split ratios must sum to 1.0"):
        cfg.validate()


def test_config_invalid_epochs() -> None:
    cfg = AppConfig()
    cfg.transformer.epochs = 0
    with pytest.raises(ConfigError, match="epochs must be >= 1"):
        cfg.validate()


def test_config_save_and_load(temp_dir: Path) -> None:
    cfg = AppConfig()
    cfg.project.name = "test_custom_run"
    cfg.transformer.batch_size = 12

    yaml_path = temp_dir / "custom_config.yaml"
    cfg.save_yaml(yaml_path)
    assert yaml_path.is_file()

    loaded = load_config(yaml_path)
    assert loaded.project.name == "test_custom_run"
    assert loaded.transformer.batch_size == 12


def test_load_config_missing_file() -> None:
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config("non_existent_file.yaml")
