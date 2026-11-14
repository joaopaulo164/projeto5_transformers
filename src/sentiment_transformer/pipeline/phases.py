"""Executable phase functions for modular pipeline execution."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification

from sentiment_transformer.baselines.majority import MajorityBaseline
from sentiment_transformer.baselines.tfidf_logistic import TfidfLogisticBaseline
from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_COMBINED_TEXT,
    COL_LABEL,
)
from sentiment_transformer.data.audit import audit_dataset
from sentiment_transformer.data.deduplication import deduplicate_dataset
from sentiment_transformer.data.download import download_dataset
from sentiment_transformer.data.loader import load_raw_dataset
from sentiment_transformer.data.preprocessing import preprocess_dataset
from sentiment_transformer.data.splitting import split_and_save_dataset
from sentiment_transformer.evaluation.error_analysis import build_error_analysis
from sentiment_transformer.evaluation.metrics import (
    compute_classification_metrics,
    save_metrics_artifacts,
)
from sentiment_transformer.evaluation.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)
from sentiment_transformer.evaluation.predictions import generate_predictions_dataframe
from sentiment_transformer.evidence.environment import capture_environment_metadata
from sentiment_transformer.evidence.fingerprints import compute_dataset_fingerprints
from sentiment_transformer.evidence.manifest import create_run_manifest
from sentiment_transformer.evidence.report import generate_comparison_reports
from sentiment_transformer.exceptions import PhaseExecutionError
from sentiment_transformer.modeling.device import get_device
from sentiment_transformer.modeling.inference import predict_transformer
from sentiment_transformer.modeling.tokenizer import get_sep_token, load_tokenizer
from sentiment_transformer.modeling.trainer import TransformerTrainer
from sentiment_transformer.paths import resolve_path

logger = logging.getLogger(__name__)


def run_data_phase(
    config: AppConfig,
    run_dir: Path | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    """Execute complete data preparation phase.

    Steps:
    1. Download dataset (or validate existing file).
    2. Audit raw dataset.
    3. Preprocess texts and assign binary labels (excluding rating 3).
    4. Deduplicate and remove conflicting label records.
    5. Stratified train/val/test splitting and saving to data/processed.

    Args:
        config: Application configuration.
        run_dir: Optional run directory to save run-specific copies of stats.
        force_download: If True, redownload raw dataset.

    Returns:
        Dictionary of phase statistics and artifacts.
    """
    logger.info("=== STARTING DATA PHASE ===")
    start_time = time.time()

    try:
        # Step 1: Download
        dl_info = download_dataset(config, force=force_download)
        raw_path = Path(dl_info["file_path"])

        # Step 2: Audit
        audit_json = run_dir / "dataset_audit.json" if run_dir else None
        audit_stats = audit_dataset(raw_path, config, output_json_path=audit_json)

        # Step 3: Load raw
        raw_df = load_raw_dataset(raw_path, config)

        # Step 4: Preprocess
        # Attempt loading tokenizer for actual sep_token, fallback if offline/mock
        try:
            tokenizer = load_tokenizer(config.transformer.model_checkpoint)
            sep_token = get_sep_token(tokenizer)
        except Exception:
            sep_token = "[SEP]"

        prep_df, prep_stats = preprocess_dataset(raw_df, config, sep_token=sep_token)

        # Step 5: Deduplicate
        dedup_df, dedup_stats = deduplicate_dataset(prep_df)

        # Step 6: Split & Save
        splits_dict, split_stats = split_and_save_dataset(dedup_df, config)

        if run_dir:
            # Save stats copy in run_dir as well
            with open(run_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
                json.dump(prep_stats, f, indent=2, ensure_ascii=False)
            with open(run_dir / "split_statistics.json", "w", encoding="utf-8") as f:
                json.dump(split_stats, f, indent=2, ensure_ascii=False)

        duration = time.time() - start_time
        logger.info("=== DATA PHASE COMPLETED in %.2fs ===", duration)

        return {
            "status": "success",
            "duration_seconds": duration,
            "raw_path": str(raw_path),
            "audit_stats": audit_stats,
            "prep_stats": prep_stats,
            "dedup_stats": dedup_stats,
            "split_stats": split_stats,
            "splits": splits_dict,
        }

    except Exception as e:
        logger.exception("Data phase failed: %s", e)
        raise PhaseExecutionError(f"Data phase failed: {e}") from e


def run_baselines_phase(
    config: AppConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """Train and save Majority and TF-IDF Logistic Regression baselines.

    Args:
        config: Application configuration.
        run_dir: Run directory for model persistence.

    Returns:
        Dictionary with trained baseline model instances and training durations.
    """
    logger.info("=== STARTING BASELINES PHASE ===")
    start_time = time.time()

    proc_dir = resolve_path(config.dataset.processed_dir)
    train_csv = proc_dir / "train.csv"

    if not train_csv.is_file():
        raise PhaseExecutionError(
            f"Train partition not found at {train_csv}. Please run data phase first."
        )

    train_df = pd.read_csv(train_csv, encoding="utf-8")
    X_train = [str(t) for t in train_df[COL_COMBINED_TEXT]]
    y_train = np.asarray(train_df[COL_LABEL], dtype=int)

    # 1. Majority Baseline
    t0 = time.time()
    majority_model = MajorityBaseline(strategy=config.baseline_majority.strategy)
    majority_model.fit(X_train, y_train)
    majority_train_time = time.time() - t0
    majority_path = run_dir / "baseline_majority.joblib"
    majority_model.save(majority_path)

    # 2. TF-IDF Logistic Baseline
    t1 = time.time()
    tfidf_model = TfidfLogisticBaseline.from_config(config.baseline_tfidf)
    tfidf_model.fit(X_train, y_train)
    tfidf_train_time = time.time() - t1
    tfidf_path = run_dir / "baseline_tfidf.joblib"
    tfidf_model.save(tfidf_path)

    duration = time.time() - start_time
    logger.info("=== BASELINES PHASE COMPLETED in %.2fs ===", duration)

    return {
        "status": "success",
        "duration_seconds": duration,
        "majority_model": majority_model,
        "majority_train_time": majority_train_time,
        "majority_path": str(majority_path),
        "tfidf_model": tfidf_model,
        "tfidf_train_time": tfidf_train_time,
        "tfidf_path": str(tfidf_path),
    }


def run_transformer_phase(
    config: AppConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """Fine-tune BERTimbau sequence classification model with early stopping.

    Args:
        config: Application configuration.
        run_dir: Run directory for checkpoint persistence.

    Returns:
        Dictionary with trainer summary and model checkpoint path.
    """
    logger.info("=== STARTING TRANSFORMER PHASE ===")
    start_time = time.time()

    proc_dir = resolve_path(config.dataset.processed_dir)
    train_csv = proc_dir / "train.csv"
    val_csv = proc_dir / "val.csv"

    if not train_csv.is_file() or not val_csv.is_file():
        raise PhaseExecutionError(
            "Train or Val partition CSV not found. Please run data phase first."
        )

    train_df = pd.read_csv(train_csv, encoding="utf-8")
    val_df = pd.read_csv(val_csv, encoding="utf-8")

    train_texts = [str(t) for t in train_df[COL_COMBINED_TEXT]]
    train_labels = np.asarray(train_df[COL_LABEL], dtype=int)
    val_texts = [str(t) for t in val_df[COL_COMBINED_TEXT]]
    val_labels = np.asarray(val_df[COL_LABEL], dtype=int)

    device = get_device(config)
    tokenizer = load_tokenizer(config.transformer.model_checkpoint)

    trainer = TransformerTrainer(
        config=config,
        tokenizer=tokenizer,
        device=device,
    )

    trainer_result = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=run_dir,
    )

    # Plot training curves
    plot_training_curves(trainer_result["history_df"], run_dir / "training_curves.png")

    duration = time.time() - start_time
    logger.info("=== TRANSFORMER PHASE COMPLETED in %.2fs ===", duration)

    return {
        "status": "success",
        "duration_seconds": duration,
        "best_epoch": trainer_result["best_epoch"],
        "best_val_metric": trainer_result["best_metric_value"],
        "checkpoint_dir": trainer_result["best_checkpoint_dir"],
        "training_time": trainer_result["total_training_time_seconds"],
        "trainer": trainer,
    }


def run_evaluation_phase(
    config: AppConfig,
    run_dir: Path,
    majority_model: MajorityBaseline | None = None,
    tfidf_model: TfidfLogisticBaseline | None = None,
    transformer_model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Evaluate all three models on test partition and generate evaluation artifacts.

    Args:
        config: Application configuration.
        run_dir: Run directory where artifacts will be saved.
        majority_model: Optional pre-loaded Majority baseline.
        tfidf_model: Optional pre-loaded TF-IDF baseline.
        transformer_model: Optional pre-loaded Transformer model.
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        Dictionary of computed metrics for all models.
    """
    logger.info("=== STARTING EVALUATION PHASE ===")
    start_time = time.time()

    proc_dir = resolve_path(config.dataset.processed_dir)
    test_csv = proc_dir / "test.csv"
    if not test_csv.is_file():
        raise PhaseExecutionError(f"Test partition not found at {test_csv}.")

    test_df = pd.read_csv(test_csv, encoding="utf-8")
    X_test = [str(t) for t in test_df[COL_COMBINED_TEXT]]
    y_test = np.asarray(test_df[COL_LABEL], dtype=int)

    device = get_device(config)

    # 1. Load / Prepare Majority Baseline
    if majority_model is None:
        maj_path = run_dir / "baseline_majority.joblib"
        majority_model = MajorityBaseline.load(maj_path)

    t0 = time.time()
    maj_preds = majority_model.predict(X_test)
    maj_proba = majority_model.predict_proba(X_test)
    maj_inf_time = time.time() - t0

    maj_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=maj_preds,
        model_name="Majority Class Baseline",
        inference_time=maj_inf_time,
    )
    save_metrics_artifacts(maj_metrics, "baseline_majority", run_dir)
    plot_confusion_matrix(
        maj_metrics["confusion_matrix"]["matrix"],
        "Baseline Majoritário",
        run_dir / "confusion_matrix_majority.png",
    )
    generate_predictions_dataframe(
        test_df,
        maj_preds,
        maj_proba,
        run_dir / "predictions_test_majority.csv",
    )

    # 2. Load / Prepare TF-IDF Baseline
    if tfidf_model is None:
        tfidf_path = run_dir / "baseline_tfidf.joblib"
        tfidf_model = TfidfLogisticBaseline.load(tfidf_path)

    t1 = time.time()
    tfidf_preds = tfidf_model.predict(X_test)
    tfidf_proba = tfidf_model.predict_proba(X_test)
    tfidf_inf_time = time.time() - t1

    tfidf_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=tfidf_preds,
        model_name="TF-IDF + Logistic Regression",
        inference_time=tfidf_inf_time,
    )
    save_metrics_artifacts(tfidf_metrics, "baseline_tfidf", run_dir)
    plot_confusion_matrix(
        tfidf_metrics["confusion_matrix"]["matrix"],
        "Baseline TF-IDF + Regressão Logística",
        run_dir / "confusion_matrix_tfidf.png",
    )
    generate_predictions_dataframe(
        test_df,
        tfidf_preds,
        tfidf_proba,
        run_dir / "predictions_test_tfidf.csv",
    )

    # 3. Load / Prepare Transformer Model
    if tokenizer is None:
        tokenizer = load_tokenizer(config.transformer.model_checkpoint)

    if transformer_model is None:
        ckpt_dir = run_dir / "best_checkpoint"
        if (ckpt_dir / "config.json").exists():
            transformer_model = AutoModelForSequenceClassification.from_pretrained(
                ckpt_dir
            )
        else:
            transformer_model = AutoModelForSequenceClassification.from_pretrained(
                config.transformer.model_checkpoint,
                num_labels=config.transformer.num_labels,
            )

    t_proba, t_preds, t_inf_time = predict_transformer(
        model=transformer_model,
        tokenizer=tokenizer,
        texts=X_test,
        device=device,
        batch_size=config.transformer.eval_batch_size,
        max_length=config.transformer.max_length,
    )

    transformer_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=t_preds,
        model_name=f"BERTimbau ({config.transformer.model_checkpoint})",
        inference_time=t_inf_time,
    )
    save_metrics_artifacts(transformer_metrics, "transformer", run_dir)
    plot_confusion_matrix(
        transformer_metrics["confusion_matrix"]["matrix"],
        "BERTimbau Fine-Tuning",
        run_dir / "confusion_matrix_transformer.png",
    )
    t_preds_df = generate_predictions_dataframe(
        test_df,
        t_preds,
        t_proba,
        run_dir / "predictions_test_transformer.csv",
    )

    duration = time.time() - start_time
    logger.info("=== EVALUATION PHASE COMPLETED in %.2fs ===", duration)

    return {
        "status": "success",
        "duration_seconds": duration,
        "majority_metrics": maj_metrics,
        "tfidf_metrics": tfidf_metrics,
        "transformer_metrics": transformer_metrics,
        "transformer_predictions_df": t_preds_df,
    }


def run_evidence_phase(
    config: AppConfig,
    run_dir: Path,
    eval_results: dict[str, Any],
    data_results: dict[str, Any] | None = None,
    phase_durations: dict[str, float] | None = None,
    phase_statuses: dict[str, str] | None = None,
    start_iso: str | None = None,
) -> dict[str, Any]:
    """Generate final evidence package: error analysis, comparison reports, and run manifest.

    Args:
        config: Application configuration.
        run_dir: Run directory.
        eval_results: Output dictionary from evaluation phase.
        data_results: Optional output dictionary from data phase.
        phase_durations: Optional phase durations dictionary.
        phase_statuses: Optional phase execution statuses.
        start_iso: Optional ISO start timestamp string.

    Returns:
        Dictionary of evidence phase outputs.
    """
    logger.info("=== STARTING EVIDENCE PHASE ===")
    start_time = time.time()

    # 1. Error analysis on Transformer predictions
    preds_df = eval_results.get("transformer_predictions_df")
    if preds_df is None:
        preds_path = run_dir / "predictions_test_transformer.csv"
        if preds_path.is_file():
            preds_df = pd.read_csv(preds_path, encoding="utf-8")

    if preds_df is not None:
        error_df, error_stats = build_error_analysis(
            predictions_df=preds_df,
            output_csv_path=run_dir / "error_analysis.csv",
        )
    else:
        error_stats = None

    # 2. Model comparison reports (CSV & MD)
    all_metrics = [
        eval_results["majority_metrics"],
        eval_results["tfidf_metrics"],
        eval_results["transformer_metrics"],
    ]
    generate_comparison_reports(all_metrics, run_dir)

    # 3. Environment metadata
    env_metadata = capture_environment_metadata(run_dir / "environment.json")

    # 4. Fingerprints
    raw_path = (
        resolve_path(config.dataset.raw_dir) / config.dataset.raw_file_name
    )
    proc_dir = resolve_path(config.dataset.processed_dir)
    fingerprints = compute_dataset_fingerprints(
        raw_file_path=raw_path,
        processed_dir=proc_dir,
        output_json_path=run_dir / "data_fingerprints.json",
    )

    # 5. Manifest
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_ts = start_iso if start_iso is not None else end_iso

    all_artifacts = [p.name for p in run_dir.iterdir() if p.is_file()]

    manifest = create_run_manifest(
        run_id=run_dir.name,
        start_time=start_ts,
        end_time=end_iso,
        total_duration_seconds=sum((phase_durations or {}).values()),
        config=config,
        environment_metadata=env_metadata,
        dataset_fingerprints=fingerprints,
        dataset_audit_stats=(data_results or {}).get("audit_stats"),
        preprocessing_stats=(data_results or {}).get("prep_stats"),
        deduplication_stats=(data_results or {}).get("dedup_stats"),
        split_stats=(data_results or {}).get("split_stats"),
        phase_durations=phase_durations or {},
        phase_statuses=phase_statuses or {},
        artifacts_created=all_artifacts,
        output_json_path=run_dir / "run_manifest.json",
    )

    duration = time.time() - start_time
    logger.info("=== EVIDENCE PHASE COMPLETED in %.2fs ===", duration)

    return {
        "status": "success",
        "duration_seconds": duration,
        "manifest": manifest,
        "error_stats": error_stats,
    }
