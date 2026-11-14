"""Scientific plotting utilities with Portuguese labels."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    model_title: str,
    output_path: str | Path,
) -> Path:
    """Generate and save a confusion matrix heatmap plot with Portuguese labels.

    Args:
        cm: 2x2 confusion matrix array.
        model_title: Display name for the model in the plot title.
        output_path: Destination PNG image file path.

    Returns:
        Path to saved PNG image.
    """
    matrix = np.asarray(cm)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Class names in Portuguese
    labels = ["Negativa (0)", "Positiva (1)"]

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_title(
        f"Matriz de Confusão\n{model_title}", fontsize=13, pad=12, weight="bold"
    )
    ax.set_xlabel("Classe Prevista", fontsize=11, labelpad=8)
    ax.set_ylabel("Classe Real", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close(fig)

    logger.info("Saved confusion matrix plot to: %s", out_p)
    return out_p


def plot_training_curves(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Generate and save loss and validation metric curves across training epochs.

    Args:
        history_df: DataFrame containing epoch, train_loss, val_loss, and val_f1_macro columns.
        output_path: Destination PNG image file path.

    Returns:
        Path to saved PNG image.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if history_df.empty or "epoch" not in history_df.columns:
        logger.warning(
            "Empty history DataFrame. Generating placeholder training curves plot."
        )
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.text(
            0.5, 0.5, "Sem dados de histórico de treinamento", ha="center", va="center"
        )
        plt.savefig(out_p, dpi=300)
        plt.close(fig)
        return out_p

    epochs = history_df["epoch"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Loss curves
    ax1.plot(
        epochs,
        history_df["train_loss"],
        marker="o",
        color="#1f77b4",
        label="Perda de Treino (Train Loss)",
    )
    ax1.plot(
        epochs,
        history_df["val_loss"],
        marker="s",
        color="#ff7f0e",
        label="Perda de Validação (Val Loss)",
    )
    ax1.set_title("Evolução da Perda (Loss)", fontsize=12, weight="bold")
    ax1.set_xlabel("Época", fontsize=10)
    ax1.set_ylabel("Perda (Cross-Entropy)", fontsize=10)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # Subplot 2: F1 Macro metric curve
    if "val_f1_macro" in history_df.columns:
        ax2.plot(
            epochs,
            history_df["val_f1_macro"],
            marker="^",
            color="#2ca02c",
            label="F1 Macro (Validação)",
        )
    if "val_accuracy" in history_df.columns:
        ax2.plot(
            epochs,
            history_df["val_accuracy"],
            marker="d",
            color="#9467bd",
            label="Acurácia (Validação)",
        )

    ax2.set_title("Métricas de Validação por Época", fontsize=12, weight="bold")
    ax2.set_xlabel("Época", fontsize=10)
    ax2.set_ylabel("Pontuação (Score)", fontsize=10)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close(fig)

    logger.info("Saved training curves plot to: %s", out_p)
    return out_p
