from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import build_feature_table
from src.load_data import load_dataset, summarize_dataset, TARGET_COLUMNS
from src.recommend import describe_label, make_recommendation

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VISUALS_DIR = PROJECT_ROOT / "visuals"

st.set_page_config(
    page_title="HydraFault",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ HydraFault")
st.subheader("Explainable Time-Series Health Monitoring for Hydraulic Systems")

st.markdown(
    """
HydraFault takes hydraulic sensor time-series data, extracts engineering features,
classifies component health, explains the prediction, and turns it into a maintenance recommendation.
"""
)


@st.cache_data(show_spinner="Loading HydraFault data...")
def cached_dataset():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    has_profile = (raw_dir / "profile.txt").exists() or any(raw_dir.rglob("profile.txt"))

    if not has_profile:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "download_data.py")],
            check=True,
        )

    sensors, profile = load_dataset()
    summary = summarize_dataset()

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_file = processed_dir / "features_accumulator.csv"

    if feature_file.exists():
        features = pd.read_csv(feature_file)
    else:
        features = build_feature_table(sensors)
        features.to_csv(feature_file, index=False)

    return sensors, profile, summary, features


def load_model_bundle(target: str):
    path = MODELS_DIR / f"best_model_{target}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def get_prediction_confidence(model, X_row: pd.DataFrame, predicted_label: str) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(X_row)[0]
    classes = [str(c) for c in model.classes_]

    if str(predicted_label) not in classes:
        return None

    idx = classes.index(str(predicted_label))
    return float(probs[idx])


def plot_sensor_cycle(sensor_matrix: pd.DataFrame, cycle_id: int, sensor_name: str):
    values = sensor_matrix.iloc[cycle_id].to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines",
            name=sensor_name,
        )
    )
    fig.update_layout(
        title=f"{sensor_name} Time-Series for Cycle {cycle_id}",
        xaxis_title="Sample index within cycle",
        yaxis_title="Sensor value",
        height=380,
    )
    return fig


try:
    sensors, profile, summary, features = cached_dataset()
except Exception as exc:
    st.error("Dataset not found or could not be loaded.")
    st.code("python scripts/download_data.py")
    st.info("If the download fails, manually unzip the UCI Hydraulic Systems dataset into data/raw/.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Controls")

    target = st.selectbox(
        "Prediction target",
        TARGET_COLUMNS,
        index=TARGET_COLUMNS.index("accumulator") if "accumulator" in TARGET_COLUMNS else 0,
    )

    cycle_id = st.number_input(
        "Test cycle ID",
        min_value=0,
        max_value=len(profile) - 1,
        value=min(100, len(profile) - 1),
        step=1,
    )

    sensor_name = st.selectbox("Sensor to visualize", sorted(sensors.keys()))

    st.markdown("---")
    st.caption("Train first if model is missing:")
    st.code(f"python -m src.train --target {target}", language="bash")


tab_overview, tab_predict, tab_models, tab_explain = st.tabs(
    ["Dataset", "Prediction", "Model Comparison", "Explainability"]
)

with tab_overview:
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Sensor files loaded:")
        st.dataframe(summary, use_container_width=True)

    with col2:
        st.write("Condition labels:")
        st.dataframe(profile.head(12), use_container_width=True)

    st.plotly_chart(plot_sensor_cycle(sensors[sensor_name], int(cycle_id), sensor_name), use_container_width=True)

with tab_predict:
    st.header("Health Classification")

    bundle = load_model_bundle(target)

    if bundle is None:
        st.warning(f"No trained model found for target: {target}")
        st.code(f"python -m src.train --target {target}", language="bash")
    else:
        model = bundle["model"]
        model_name = bundle["best_model_name"]

        X_row = features.iloc[[int(cycle_id)]]
        predicted = str(model.predict(X_row)[0])
        confidence = get_prediction_confidence(model, X_row, predicted)

        true_label = str(profile.iloc[int(cycle_id)][target])
        top_features = []

        importance_path = ARTIFACTS_DIR / f"feature_importance_{target}.csv"
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            top_features = importance_df["feature"].head(5).tolist()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best model", model_name)

        with col2:
            st.metric("Predicted condition", describe_label(target, predicted))

        with col3:
            if confidence is not None:
                st.metric("Confidence", f"{confidence * 100:.1f}%")
            else:
                st.metric("Confidence", "N/A")

        st.markdown("### Ground truth")
        st.write(f"Actual label for this cycle: **{describe_label(target, true_label)}**")

        st.markdown("### Maintenance recommendation")
        st.success(make_recommendation(target, predicted, top_features))

        st.markdown("### Top evidence features")
        if top_features:
            for i, feature in enumerate(top_features, start=1):
                st.write(f"{i}. `{feature}`")
        else:
            st.info("Train the model to generate feature importance.")

with tab_models:
    st.header("Model Comparison")

    comparison_path = ARTIFACTS_DIR / f"model_comparison_{target}.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True)

        st.markdown(
            """
**Model-selection logic:** Logistic regression is a baseline. Tree-based models are strong here because the project converts
sensor cycles into nonlinear tabular features like slopes, ranges, RMS values, and late-cycle instability shifts.
"""
        )
    else:
        st.warning("No model comparison file found.")
        st.code(f"python -m src.train --target {target}", language="bash")

with tab_explain:
    st.header("Explainability")

    importance_path = ARTIFACTS_DIR / f"feature_importance_{target}.csv"
    explanation_path = ARTIFACTS_DIR / f"explanation_{target}.txt"
    importance_img = VISUALS_DIR / f"feature_importance_{target}.png"

    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        st.dataframe(importance.head(20), use_container_width=True)

        if importance_img.exists():
            st.image(str(importance_img), caption="Top feature importance")

        if explanation_path.exists():
            st.markdown("### Interpretation")
            st.write(explanation_path.read_text(encoding="utf-8"))
    else:
        st.warning("No explainability artifacts found.")
        st.code(f"python -m src.train --target {target}", language="bash")
