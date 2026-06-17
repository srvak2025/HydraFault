from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


def get_builtin_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """
    Returns feature importances for tree-based models when available.
    Handles sklearn Pipelines by looking at the final estimator.
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    if not hasattr(estimator, "feature_importances_"):
        return None

    values = estimator.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_permutation_importance(
    model,
    X,
    y,
    n_repeats: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Model-agnostic explanation. Slower than built-in tree importances but more honest.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_importance_plot(importance_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(top))} Important Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def humanize_feature_name(feature: str) -> str:
    return feature.replace("_", " ")


def build_explanation_text(importance_df: pd.DataFrame, top_n: int = 5) -> str:
    top_features = importance_df.head(top_n)["feature"].tolist()
    bullets = "\n".join(f"- {humanize_feature_name(f)}" for f in top_features)

    return (
        "The model's prediction is driven most strongly by these engineered sensor features:\n\n"
        f"{bullets}\n\n"
        "Interpretation: features involving late-cycle drift, standard deviation, range, or slope "
        "suggest that the model is detecting instability and degradation patterns over time rather "
        "than only comparing raw average sensor readings."
    )
