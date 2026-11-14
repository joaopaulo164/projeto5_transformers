"""Evidence subsystem: environment metadata, dataset fingerprints, run manifest, and comparison reports."""

from sentiment_transformer.evidence.environment import capture_environment_metadata
from sentiment_transformer.evidence.fingerprints import compute_dataset_fingerprints
from sentiment_transformer.evidence.manifest import create_run_manifest
from sentiment_transformer.evidence.report import generate_comparison_reports

__all__ = [
    "capture_environment_metadata",
    "compute_dataset_fingerprints",
    "create_run_manifest",
    "generate_comparison_reports",
]
