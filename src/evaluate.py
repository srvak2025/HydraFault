from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(model_name: str, y_true, y_pred) -> dict[str, Any]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_classification_report(y_true, y_pred, out_path: Path) -> None:
    report = classification_report(y_true, y_pred, zero_division=0)
    out_path.write_text(report, encoding="utf-8")


def save_confusion_matrix(y_true, y_pred, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metrics_table(metrics: list[dict], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(metrics)
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df
