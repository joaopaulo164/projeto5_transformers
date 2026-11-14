"""Configuration loading, validation, and schema definitions."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sentiment_transformer.constants import (
    CANDIDATE_ID_COLUMNS,
    CANDIDATE_RATING_COLUMNS,
    CANDIDATE_TEXT_COLUMNS,
    CANDIDATE_TITLE_COLUMNS,
    DEFAULT_EXCLUDED_RATINGS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_NEGATIVE_RATINGS,
    DEFAULT_POSITIVE_RATINGS,
    DEFAULT_TRANSFORMER_CHECKPOINT,
)
from sentiment_transformer.exceptions import ConfigError
from sentiment_transformer.paths import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Project-level metadata and global seed."""

    name: str = "sentiment_transformer"
    seed: int = 42


@dataclass
class DatasetConfig:
    """Dataset source, local paths, and column candidate specifications."""

    source_url: str = (
        "https://raw.githubusercontent.com/b2wads/b2w-reviews01/master/B2W-Reviews01.csv"
    )
    raw_file_name: str = "B2W-Reviews01.csv"
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    timeout_seconds: int = 60
    min_file_size_bytes: int = 1048576  # 1 MB
    column_candidates: dict[str, list[str]] = field(
        default_factory=lambda: {
            "id": list(CANDIDATE_ID_COLUMNS),
            "title": list(CANDIDATE_TITLE_COLUMNS),
            "text": list(CANDIDATE_TEXT_COLUMNS),
            "rating": list(CANDIDATE_RATING_COLUMNS),
        }
    )
    label_mapping: dict[str, list[int]] = field(
        default_factory=lambda: {
            "negative_ratings": list(DEFAULT_NEGATIVE_RATINGS),
            "excluded_ratings": list(DEFAULT_EXCLUDED_RATINGS),
            "positive_ratings": list(DEFAULT_POSITIVE_RATINGS),
        }
    )


@dataclass
class SamplingConfig:
    """Sampling and class balancing configuration."""

    enabled: bool = True
    total_samples: int = 10000
    samples_per_class: int = 5000
    balance_classes: bool = True
    seed: int = 42


@dataclass
class SplitConfig:
    """Data split proportions and options."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    save_parquet: bool = False


@dataclass
class BaselineMajorityConfig:
    """Majority class dummy baseline configuration."""

    strategy: str = "most_frequent"


@dataclass
class BaselineTfidfConfig:
    """TF-IDF Logistic Regression baseline configuration."""

    ngram_range: list[int] = field(default_factory=lambda: [1, 2])
    max_features: int = 10000
    min_df: int = 2
    max_df: float = 0.95
    c_regularization: float = 1.0
    max_iter: int = 1000
    random_state: int = 42


@dataclass
class TransformerConfig:
    """Transformer fine-tuning configuration."""

    model_checkpoint: str = DEFAULT_TRANSFORMER_CHECKPOINT
    max_length: int = DEFAULT_MAX_SEQ_LENGTH
    num_labels: int = 2
    epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    seed: int = 42


@dataclass
class DeviceConfig:
    """Hardware device selection configuration."""

    use_cuda_if_available: bool = True
    device_id: int = 0


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class ArtifactsConfig:
    """Artifacts output directory configuration."""

    output_dir: str = "artifacts/runs"


@dataclass
class AppConfig:
    """Root configuration object combining all sub-configurations."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    baseline_majority: BaselineMajorityConfig = field(
        default_factory=BaselineMajorityConfig
    )
    baseline_tfidf: BaselineTfidfConfig = field(default_factory=BaselineTfidfConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def validate(self) -> None:
        """Validate configuration values and consistency.

        Raises:
            ConfigError: If configuration values are invalid.
        """
        # Validate split proportions
        total_ratio = (
            self.split.train_ratio + self.split.val_ratio + self.split.test_ratio
        )
        if abs(total_ratio - 1.0) > 1e-4:
            raise ConfigError(
                f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
                f"(train={self.split.train_ratio}, val={self.split.val_ratio}, test={self.split.test_ratio})"
            )

        if (
            self.split.train_ratio <= 0
            or self.split.val_ratio <= 0
            or self.split.test_ratio <= 0
        ):
            raise ConfigError(
                "All split ratios (train, val, test) must be strictly positive."
            )

        # Validate sampling
        if self.sampling.enabled:
            if self.sampling.total_samples <= 0:
                raise ConfigError(
                    f"total_samples must be positive, got {self.sampling.total_samples}"
                )
            if self.sampling.balance_classes and self.sampling.samples_per_class <= 0:
                raise ConfigError(
                    f"samples_per_class must be positive when balance_classes is True, "
                    f"got {self.sampling.samples_per_class}"
                )

        # Validate transformer
        if self.transformer.epochs <= 0:
            raise ConfigError(f"epochs must be >= 1, got {self.transformer.epochs}")
        if self.transformer.batch_size <= 0:
            raise ConfigError(
                f"batch_size must be >= 1, got {self.transformer.batch_size}"
            )
        if self.transformer.max_length <= 0:
            raise ConfigError(
                f"max_length must be >= 1, got {self.transformer.max_length}"
            )
        if self.transformer.learning_rate <= 0:
            raise ConfigError(
                f"learning_rate must be > 0, got {self.transformer.learning_rate}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration dataclass hierarchy to plain dictionary."""
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save resolved configuration as YAML file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _deep_update(
    mapping: dict[str, Any], updating_dict: dict[str, Any]
) -> dict[str, Any]:
    """Recursively update a dictionary."""
    for k, v in updating_dict.items():
        if isinstance(v, dict) and k in mapping and isinstance(mapping[k], dict):
            mapping[k] = _deep_update(mapping[k], v)
        else:
            mapping[k] = v
    return mapping


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file or return default configuration.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If configuration file is missing or cannot be parsed.
    """
    if config_path is None:
        cfg = AppConfig()
        cfg.validate()
        return cfg

    path = resolve_path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found at: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML file at {path}: {e}") from e

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration YAML root must be a dictionary, got {type(raw_data).__name__}"
        )

    # Build dataclass instance merging defaults with raw_data
    base = AppConfig()
    base_dict = base.to_dict()
    merged = _deep_update(base_dict, raw_data)

    try:
        cfg = AppConfig(
            project=ProjectConfig(**merged.get("project", {})),
            dataset=DatasetConfig(**merged.get("dataset", {})),
            sampling=SamplingConfig(**merged.get("sampling", {})),
            split=SplitConfig(**merged.get("split", {})),
            baseline_majority=BaselineMajorityConfig(
                **merged.get("baseline_majority", {})
            ),
            baseline_tfidf=BaselineTfidfConfig(**merged.get("baseline_tfidf", {})),
            transformer=TransformerConfig(**merged.get("transformer", {})),
            device=DeviceConfig(**merged.get("device", {})),
            logging=LoggingConfig(**merged.get("logging", {})),
            artifacts=ArtifactsConfig(**merged.get("artifacts", {})),
        )
        cfg.validate()
        return cfg
    except TypeError as e:
        raise ConfigError(f"Configuration schema mismatch: {e}") from e
