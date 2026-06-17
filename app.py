from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.recommend import describe_label, make_recommendation

PROJECT_ROOT = Path(__file__).resolve().parent

TARGET = "accumulator"

MODEL_PATH = PROJECT_ROOT / "models" / "best_model_accumulator.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_accumulator.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels_accumulator.csv"
COMPARISON_PATH = PROJECT_ROOT / "artifacts" / "model_comparison_accumulator.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "artifacts" / "feature_importance_accumulator.csv"
EXPLANATION_PATH = PROJECT_ROOT / "artifacts" / "explanation_accumulator.txt"

st.set_page_config(
    page_title="HydraFault",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ HydraFault")
st.subheader("Explainable Time-Series Health Monitoring for Hydraulic Systems")

st.markdown(
    """
HydraFault is a prototype ML dashboard for hydraulic system condition monitoring.
It uses engineered time-series features, compares classification models, explains
important sensor signals, and generates a maintenance recommendation.
"""
)


@st.cache_data(show_spinner="Loading precomputed HydraFault artifacts...")
def load_artifacts():
    missing = [
        str(path)
        for path in [MODEL_PATH, FEATURES_PATH, LABELS_PATH, COMPARISON_PATH, IMPORTANCE_PATH]
        if not path.exists()
    ]

    if missing:
        return None, missing

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    explanation = ""
    if EXPLANATION_PATH.exists():
        explanation = EXPLANATION_PATH.read_text(encoding="utf-8")

    return {
        "features": features,
        "labels": labels,
        "comparison": comparison,
        "importance": importance,
        "explanation": explanation,
    }, []


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return joblib.load(MODEL_PATH)


data, missing = load_artifacts()

if missing:
    st.error("Missing required deployment files.")
    st.write("These files were not found:")
    for item in missing:
        st.code(item)
    st.stop()

model_bundle = load_model()
model = model_bundle["model"]
model_name = model_bundle.get("best_model_name", "trained model")

features = data["features"]
labels = data["labels"]
comparison = data["comparison"]
importance = data["importance"]
explanation = data["explanation"]

if TARGET not in labels.columns:
    st.error(f"Expected label column '{TARGET}' in labels file.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    cycle_id = st.number_input(
        "Cycle ID",
        min_value=0,
        max_value=len(features) - 1,
        value=min(100, len(features) - 1),
        step=1,
    )

    st.markdown("---")
    st.caption("Deployment mode")
    st.success("Using precomputed features and trained model")


tab_dataset, tab_prediction, tab_models, tab_explain = st.tabs(
    ["Dataset", "Prediction", "Model Comparison", "Explainability"]
)

with tab_dataset:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Operating cycles", len(features))

    with col2:
        st.metric("Engineered features", features.shape[1])

    with col3:
        st.metric("Target", TARGET)

    st.markdown("### Accumulator condition distribution")
    label_counts = labels[TARGET].astype(str).value_counts().reset_index()
    label_counts.columns = ["condition", "count"]
    label_counts["meaning"] = label_counts["condition"].apply(lambda x: describe_label(TARGET, x))
    st.dataframe(label_counts, use_container_width=True)

    fig = px.bar(
        label_counts,
        x="meaning",
        y="count",
        title="Accumulator Condition Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sample engineered feature table")
    st.dataframe(features.head(20), use_container_width=True)

with tab_prediction:
    st.header("Health Classification")

    X_row = features.iloc[[int(cycle_id)]]
    predicted = str(model.predict(X_row)[0])
    actual = str(labels.iloc[int(cycle_id)][TARGET])

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_row)[0]
        classes = [str(c) for c in model.classes_]
        if predicted in classes:
            confidence = float(probs[classes.index(predicted)])

    top_features = importance["feature"].head(5).tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", model_name)

    with col2:
        st.metric("Predicted condition", describe_label(TARGET, predicted))

    with col3:
        if confidence is not None:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        else:
            st.metric("Confidence", "N/A")

    st.markdown("### Ground truth")
    st.write(f"Actual condition for cycle `{cycle_id}`: **{describe_label(TARGET, actual)}**")

    st.markdown("### Maintenance recommendation")
    st.success(make_recommendation(TARGET, predicted, top_features))

    st.markdown("### Top evidence features")
    for i, feature in enumerate(top_features, start=1):
        st.write(f"{i}. `{feature}`")

    st.markdown("### Selected cycle feature values")
    selected_values = X_row[top_features].T.reset_index()
    selected_values.columns = ["feature", "value"]
    st.dataframe(selected_values, use_container_width=True)

with tab_models:
    st.header("Model Comparison")
    st.dataframe(comparison, use_container_width=True)

    st.markdown(
        """
**Model-selection logic:** Logistic regression provides a baseline. Tree-based models
are effective here because the system converts raw time-series sensor cycles into
nonlinear tabular features such as RMS, slope, range, late-cycle drift, and instability shifts.
"""
    )

with tab_explain:
    st.header("Explainability")

    top = importance.head(20).copy()
    st.dataframe(top, use_container_width=True)

    fig = px.bar(
        top.iloc[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Importances",
    )
    st.plotly_chart(fig, use_container_width=True)

    if explanation:
        st.markdown("### Interpretation")
        st.write(explanation)
