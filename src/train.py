from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_metrics_table,
)
from src.explain import (
    build_explanation_text,
    get_builtin_feature_importance,
    get_permutation_importance,
    save_importance_plot,
)
from src.features import build_feature_table, make_xy
from src.load_data import load_dataset, TARGET_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VISUALS_DIR = PROJECT_ROOT / "visuals"


def make_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def choose_stratify(y: pd.Series):
    """
    Stratified split fails if any class has only one sample.
    This keeps the script robust.
    """
    counts = y.value_counts()
    if counts.min() < 2:
        return None
    return y


def train(target: str, test_size: float = 0.25, random_state: int = 42) -> None:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}'. Choose from: {TARGET_COLUMNS}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    sensors, profile = load_dataset()

    print("Building feature table...")
    features = build_feature_table(sensors)
    X, y = make_xy(features, profile, target)

    feature_path = PROCESSED_DIR / f"features_{target}.csv"
    label_path = PROCESSED_DIR / f"labels_{target}.csv"
    X.to_csv(feature_path, index=False)
    y.to_frame(name=target).to_csv(label_path, index=False)

    print(f"Saved features: {feature_path}")
    print(f"Saved labels: {label_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=choose_stratify(y),
    )

    models = make_models(random_state=random_state)
    metrics = []
    predictions_for_best = None
    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_metrics = compute_metrics(name, y_test, y_pred)
        metrics.append(model_metrics)

        report_path = ARTIFACTS_DIR / f"classification_report_{target}_{name}.txt"
        cm_path = VISUALS_DIR / f"confusion_matrix_{target}_{name}.png"

        save_classification_report(y_test, y_pred, report_path)
        save_confusion_matrix(y_test, y_pred, cm_path, title=f"{name} - {target}")

        score = model_metrics["macro_f1"]
        print(f"{name} macro F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
            predictions_for_best = y_pred

    assert best_model is not None
    assert best_name is not None
    assert predictions_for_best is not None

    metrics_df = save_metrics_table(metrics, ARTIFACTS_DIR / f"model_comparison_{target}.csv")
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))

    model_path = MODELS_DIR / f"best_model_{target}.joblib"
    joblib.dump(
        {
            "model": best_model,
            "target": target,
            "feature_names": list(X.columns),
            "best_model_name": best_name,
            "classes": sorted(y.unique().tolist()),
        },
        model_path,
    )
    print(f"\nSaved best model: {model_path}")

    prediction_df = pd.DataFrame(
        {
            "cycle_id": X_test.index,
            "actual": y_test.values,
            "predicted": predictions_for_best,
        }
    )
    prediction_df.to_csv(ARTIFACTS_DIR / f"test_predictions_{target}.csv", index=False)

    print("\nComputing feature importance...")
    importance_df = get_builtin_feature_importance(best_model, list(X.columns))
    if importance_df is None:
        importance_df = get_permutation_importance(best_model, X_test, y_test)

    importance_path = ARTIFACTS_DIR / f"feature_importance_{target}.csv"
    importance_df.to_csv(importance_path, index=False)
    save_importance_plot(
        importance_df,
        VISUALS_DIR / f"feature_importance_{target}.png",
        top_n=15,
    )

    explanation_text = build_explanation_text(importance_df, top_n=5)
    (ARTIFACTS_DIR / f"explanation_{target}.txt").write_text(explanation_text, encoding="utf-8")

    print(f"Saved feature importance: {importance_path}")
    print("\nDONE. Now run:")
    print("    streamlit run app.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="accumulator",
        choices=TARGET_COLUMNS,
        help="Which hydraulic condition to classify.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(target=args.target, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
