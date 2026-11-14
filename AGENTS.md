# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Commands
- Single test: `python -m pytest tests/unit/test_metrics.py -k test_compute_classification_metrics -v`
- Full test suite with coverage: `python -m pytest tests/` (enforces >=80% coverage on `src/sentiment_transformer`)
- Lint and format check: `python -m ruff check src tests scripts; python -m black --check src tests scripts configs`
- Type checking: `python -m mypy src/sentiment_transformer`
- Execute single phase: `python -m sentiment_transformer.cli phase <data|baselines|transformer|evaluation|evidence> --config configs/smoke.yaml`
- End-to-end pipeline: `python -m sentiment_transformer.cli run --config configs/smoke.yaml`

## Critical Project-Specific Conventions
- **Path Resolution**: Never use raw string concatenation or OS-specific separators. Always use `resolve_path(path)` or `get_project_root()` from [`src/sentiment_transformer/paths.py`](src/sentiment_transformer/paths.py:1).
- **Label Mapping**: Binary classification only. Ratings 1 and 2 = class `0` (negative); ratings 4 and 5 = class `1` (positive). Rating 3 **MUST be excluded** from all partitions (never mapped to neutral).
- **Text Composition**: Combine review title and body with tokenizer-native separator: `f"{title} {tokenizer.sep_token} {body}"`. Never hardcode `"[SEP]"` string directly; query `tokenizer.sep_token` via [`src/sentiment_transformer/modeling/tokenizer.py`](src/sentiment_transformer/modeling/tokenizer.py:1). Empty body reviews are excluded.
- **Deduplication & Conflicts**: Generate deduplication keys via `normalize_duplicate_key()` in [`src/sentiment_transformer/data/deduplication.py`](src/sentiment_transformer/data/deduplication.py:1). When duplicate normalized texts have conflicting labels, **the entire conflicting group must be dropped**, never arbitrated.
- **Data Partition Isolation**: Splits (70% train, 15% validation, 15% test) must have zero overlap in IDs or normalized duplicate keys. Test set must **never** be used for checkpoint selection, early stopping, or threshold tuning.
- **Reproducibility & Manifest**: All runs output to immutable `artifacts/runs/<run_id>/`. Never overwrite previous runs. Always generate `run_manifest.json` and compute SHA-256 dataset fingerprints via [`src/sentiment_transformer/evidence/`](src/sentiment_transformer/evidence/).
