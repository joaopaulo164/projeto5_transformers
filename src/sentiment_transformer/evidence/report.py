"""Model comparison reporting in CSV and Markdown formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_comparison_reports(
    models_metrics: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Generate model comparison tables in CSV and Markdown formats.

    Args:
        models_metrics: List of metrics dictionaries computed for each model.
        output_dir: Directory where report files will be saved.

    Returns:
        Tuple of (comparison_csv_path, comparison_md_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for m in models_metrics:
        name = m.get("model_name", "Unknown")
        acc = m.get("accuracy", 0.0)
        f1_mac = m.get("f1_macro", 0.0)
        f1_wgt = m.get("f1_weighted", 0.0)
        c0 = m.get("class_0_negative", {})
        c1 = m.get("class_1_positive", {})
        train_t = m.get("training_time_seconds")
        inf_t = m.get("inference_time_seconds")

        prec_mac = round((c0.get("precision", 0.0) + c1.get("precision", 0.0)) / 2, 4)
        rec_mac = round((c0.get("recall", 0.0) + c1.get("recall", 0.0)) / 2, 4)

        rows.append(
            {
                "Modelo": name,
                "Acurácia": acc,
                "F1_Macro": f1_mac,
                "F1_Weighted": f1_wgt,
                "F1_Negativa_0": c0.get("f1_score", 0.0),
                "F1_Positiva_1": c1.get("f1_score", 0.0),
                "Precisao_Macro": prec_mac,
                "Recall_Macro": rec_mac,
                "Tempo_Treino_s": train_t if train_t is not None else 0.0,
                "Tempo_Inferencia_s": inf_t if inf_t is not None else 0.0,
            }
        )

    df_comp = pd.DataFrame(rows)

    # 1. Save CSV
    csv_path = out_dir / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False, encoding="utf-8")

    # 2. Build Markdown Report
    md_path = out_dir / "model_comparison.md"

    md_lines = [
        "# Relatório Comparativo de Modelos",
        "",
        "## 1. Tabela Comparativa de Desempenho",
        "",
        "| Modelo | Acurácia | F1 Macro (Principal) | F1 Negativa (0) | F1 Positiva (1) | Precisão Macro | Recall Macro | Tempo Treino (s) | Tempo Inferência (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df_comp.iterrows():
        md_lines.append(
            f"| **{r['Modelo']}** | {r['Acurácia']:.4f} | **{r['F1_Macro']:.4f}** | "
            f"{r['F1_Negativa_0']:.4f} | {r['F1_Positiva_1']:.4f} | "
            f"{r['Precisao_Macro']:.4f} | {r['Recall_Macro']:.4f} | "
            f"{r['Tempo_Treino_s']:.2f} | {r['Tempo_Inferencia_s']:.2f} |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Análise Metodológica",
            "",
            "- **Métrica Principal**: F1 Macro é adotado como métrica primária para avaliar o equilíbrio "
            "entre classes sem favorecer desbalanceamentos residuais.",
            "- **Hipótese de Pesquisa**: Avalia-se o ganho do fine-tuning do BERTimbau "
            "(`neuralmind/bert-base-portuguese-cased`) sobre o baseline majoritário e o baseline TF-IDF + Logistic Regression.",
            "- **Conclusão Experimental**: Os artefatos e métricas completos desta execução estão "
            "registrados e versionados nesta pasta de run.",
            "",
        ]
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Saved comparison reports to %s and %s", csv_path, md_path)
    return csv_path, md_path
