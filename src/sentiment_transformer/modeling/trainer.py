"""Fine-tuning trainer for BERTimbau / Transformer sequence classification models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from sentiment_transformer.config import AppConfig
from sentiment_transformer.exceptions import ModelingError
from sentiment_transformer.modeling.dataset import ReviewSentimentDataset
from sentiment_transformer.modeling.device import get_device

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds into a human-readable string (e.g. '1h 05m 20s', '4m 12s', '35s')."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


class TransformerTrainer:
    """Trainer handling fine-tuning, validation, early stopping, and checkpointing for Transformers."""

    def __init__(
        self,
        config: AppConfig,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Application configuration.
            model: Optional pre-instantiated model.
            tokenizer: Optional pre-instantiated tokenizer.
            device: Optional torch.device.
        """
        self.config = config
        self.device = device if device is not None else get_device(config)
        self.tokenizer = tokenizer
        self.model = model
        self.training_history: list[dict[str, Any]] = []
        self.best_metric_value: float = (
            -float("inf") if config.transformer.greater_is_better else float("inf")
        )
        self.best_epoch: int = -1

    def _init_model(self) -> None:
        """Instantiate model if not already provided."""
        if self.model is None:
            checkpoint = self.config.transformer.model_checkpoint
            num_labels = self.config.transformer.num_labels
            logger.info(
                "Instantiating AutoModelForSequenceClassification from checkpoint: %s",
                checkpoint,
            )
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint,
                    num_labels=num_labels,
                )
            except Exception as e:
                raise ModelingError(
                    f"Failed to load model from checkpoint '{checkpoint}': {e}\n"
                    f"Ensure internet connectivity is available or supply a local checkpoint directory."
                ) from e

        self.model.to(self.device)  # type: ignore[arg-type]

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate current model weights on validation DataLoader.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Dictionary with loss, accuracy, precision, recall, and f1_macro.
        """
        if self.model is None:
            raise ModelingError("Model is not initialized.")

        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        total_val_batches = len(val_loader)
        eval_log_interval = max(1, min(25, total_val_batches // 4))

        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                outputs = self.model(**kwargs)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                if (val_step + 1) % eval_log_interval == 0 or (val_step + 1) == total_val_batches:
                    logger.info(
                        "  [Validation] Batch %d/%d (%.1f%%)",
                        val_step + 1,
                        total_val_batches,
                        ((val_step + 1) / total_val_batches) * 100,
                    )

        num_samples = len(all_labels)
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

        acc = float(accuracy_score(all_labels, all_preds)) if num_samples > 0 else 0.0
        prec = (
            float(precision_score(all_labels, all_preds, zero_division=0))
            if num_samples > 0
            else 0.0
        )
        rec = (
            float(recall_score(all_labels, all_preds, zero_division=0))
            if num_samples > 0
            else 0.0
        )
        f1_mac = (
            float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
            if num_samples > 0
            else 0.0
        )

        return {
            "val_loss": avg_loss,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1_macro": f1_mac,
        }

    def train(
        self,
        train_texts: list[str] | np.ndarray,
        train_labels: list[int] | np.ndarray,
        val_texts: list[str] | np.ndarray,
        val_labels: list[int] | np.ndarray,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Execute fine-tuning with validation, early stopping, and checkpoint saving.

        Args:
            train_texts: Training text samples.
            train_labels: Training labels (0/1).
            val_texts: Validation text samples.
            val_labels: Validation labels (0/1).
            output_dir: Directory where best checkpoint and training history will be saved.

        Returns:
            Dictionary with training summary, paths, and history DataFrame.
        """
        self._init_model()
        assert self.model is not None

        checkpoint_dir = Path(output_dir) / "best_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        t_cfg = self.config.transformer
        batch_size = t_cfg.batch_size
        eval_batch_size = t_cfg.eval_batch_size
        max_length = t_cfg.max_length
        epochs = t_cfg.epochs
        lr = t_cfg.learning_rate
        weight_decay = t_cfg.weight_decay
        grad_accum = max(1, t_cfg.gradient_accumulation_steps)
        patience = t_cfg.early_stopping_patience
        use_fp16 = t_cfg.fp16 and (self.device.type == "cuda")

        logger.info(
            "Setting up datasets. Train: %d samples, Val: %d samples | Batch size: %d, Max length: %d",
            len(train_texts),
            len(val_texts),
            batch_size,
            max_length,
        )

        train_dataset = ReviewSentimentDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        val_dataset = ReviewSentimentDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

        # Optimizer and Scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        total_steps = (len(train_loader) // grad_accum) * epochs
        warmup_steps = int(total_steps * t_cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )

        # Optional GradScaler for fp16
        scaler = None
        if use_fp16:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except Exception:
                scaler = None

        logger.info(
            "Starting fine-tuning for %d epochs on device '%s' (fp16=%s)...",
            epochs,
            self.device,
            use_fp16,
        )

        epochs_without_improvement = 0
        start_time = time.time()

        total_batches = len(train_loader)
        log_interval = max(1, min(50, total_batches // 10))

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            last_log_time = epoch_start
            self.model.train()
            train_loss_total = 0.0
            samples_processed = 0
            optimizer.zero_grad()

            logger.info(
                "--- Starting Epoch %d/%d (%d training batches, %d samples) ---",
                epoch,
                epochs,
                total_batches,
                len(train_dataset),
            )

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                batch_samples = len(labels)
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                if use_fp16 and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**kwargs)
                        raw_loss = outputs.loss
                        loss = raw_loss / grad_accum
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum == 0 or (step + 1) == total_batches:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = self.model(**kwargs)
                    raw_loss = outputs.loss
                    loss = raw_loss / grad_accum
                    loss.backward()
                    if (step + 1) % grad_accum == 0 or (step + 1) == total_batches:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                loss_val = raw_loss.item()
                train_loss_total += loss_val * batch_samples
                samples_processed += batch_samples

                now = time.time()
                is_last_step = (step + 1) == total_batches
                should_log = (
                    ((step + 1) % log_interval == 0)
                    or (now - last_log_time >= 30.0)
                    or is_last_step
                )

                if should_log:
                    elapsed = now - epoch_start
                    speed = samples_processed / max(0.001, elapsed)
                    avg_running_loss = train_loss_total / max(1, samples_processed)
                    batches_remaining = total_batches - (step + 1)
                    avg_time_per_batch = elapsed / (step + 1)
                    eta_epoch = batches_remaining * avg_time_per_batch
                    current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr

                    logger.info(
                        "  [Epoch %d/%d] Step %d/%d (%.1f%%) | Loss: %.4f (batch: %.4f) | LR: %.2e | Speed: %.1f samples/s | Elapsed: %s | ETA: %s",
                        epoch,
                        epochs,
                        step + 1,
                        total_batches,
                        ((step + 1) / total_batches) * 100,
                        avg_running_loss,
                        loss_val,
                        current_lr,
                        speed,
                        _format_duration(elapsed),
                        _format_duration(eta_epoch),
                    )
                    last_log_time = now

            avg_train_loss = (
                train_loss_total / len(train_dataset) if len(train_dataset) > 0 else 0.0
            )

            # Validation step
            logger.info(
                "--- Starting Validation for Epoch %d/%d (%d validation samples) ---",
                epoch,
                epochs,
                len(val_dataset),
            )
            val_metrics = self.evaluate(val_loader)
            epoch_duration = time.time() - epoch_start

            current_metric = val_metrics[f"val_{t_cfg.metric_for_best_model}"]
            is_better = (
                (current_metric > self.best_metric_value)
                if t_cfg.greater_is_better
                else (current_metric < self.best_metric_value)
            )

            history_entry = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 5),
                "val_loss": round(val_metrics["val_loss"], 5),
                "val_accuracy": round(val_metrics["val_accuracy"], 4),
                "val_precision": round(val_metrics["val_precision"], 4),
                "val_recall": round(val_metrics["val_recall"], 4),
                "val_f1_macro": round(val_metrics["val_f1_macro"], 4),
                "epoch_duration_seconds": round(epoch_duration, 2),
                "is_best": is_better,
            }
            self.training_history.append(history_entry)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val F1 Macro: %.4f | Val Acc: %.4f (%.1fs)",
                epoch,
                epochs,
                avg_train_loss,
                val_metrics["val_loss"],
                val_metrics["val_f1_macro"],
                val_metrics["val_accuracy"],
                epoch_duration,
            )

            if is_better:
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                epochs_without_improvement = 0
                logger.info(
                    "--> New best model found at epoch %d (%s=%.4f). Saving checkpoint...",
                    epoch,
                    t_cfg.metric_for_best_model,
                    current_metric,
                )
                self.model.save_pretrained(checkpoint_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(checkpoint_dir)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No improvement for %d consecutive epoch(s) (patience=%d).",
                    epochs_without_improvement,
                    patience,
                )
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        total_training_time = time.time() - start_time
        logger.info(
            "Fine-tuning finished in %.2f seconds. Best epoch: %d",
            total_training_time,
            self.best_epoch,
        )

        history_df = pd.DataFrame(self.training_history)
        history_path = Path(output_dir) / "training_history.csv"
        history_df.to_csv(history_path, index=False, encoding="utf-8")
        logger.info("Saved training history to: %s", history_path)

        # Reload best model if saved
        if (checkpoint_dir / "config.json").exists():
            try:
                logger.info("Reloading best weights from %s...", checkpoint_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_dir
                )
                self.model.to(self.device)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning("Could not reload best checkpoint via AutoModel: %s", e)

        return {
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "best_checkpoint_dir": str(checkpoint_dir),
            "training_history_path": str(history_path),
            "total_training_time_seconds": round(total_training_time, 2),
            "history_df": history_df,
        }
