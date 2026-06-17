# HydraFault

**Explainable Time-Series Health Monitoring for Hydraulic Systems**

HydraFault is a fast, professor-ready ML project that takes raw hydraulic sensor time-series data, extracts engineering features, trains classification models, explains the prediction, and displays the result in a Streamlit dashboard.

## What it demonstrates

- Multivariate time-series data handling
- Sensor feature extraction
- Classification
- Model comparison
- Model-selection reasoning
- Explainability through permutation importance
- Practical engineering recommendation system
- Clean deployable dashboard

## Dataset

This project is designed for the UCI **Condition Monitoring of Hydraulic Systems** dataset.

The dataset contains repeated hydraulic test cycles, where each sensor file stores one time-series run per row. The `profile.txt` file stores condition labels.

Target columns used by this project:

| Target name | Meaning |
|---|---|
| `cooler` | Cooler condition |
| `valve` | Valve condition |
| `pump` | Internal pump leakage |
| `accumulator` | Hydraulic accumulator pressure |
| `stable` | Stable flag |

## Quickstart

```bash
# 1. Create environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 2. Install packages
pip install -r requirements.txt

# 3. Download and unpack the dataset
python scripts/download_data.py

# 4. Train all models on one target
python -m src.train --target accumulator

# 5. Run the dashboard
streamlit run app.py
```

If the download script fails, manually download the UCI dataset ZIP, unzip it, and put all `.txt` files directly inside:

```text
data/raw/
```

You should see files like:

```text
PS1.txt
PS2.txt
TS1.txt
VS1.txt
profile.txt
```

## Recommended first target

Use:

```bash
python -m src.train --target accumulator
```

Why: accumulator degradation is easy to explain as an engineering maintenance problem.

## Project structure

```text
hydrafault/
├── app.py
├── requirements.txt
├── scripts/
│   └── download_data.py
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── features.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── recommend.py
│   └── train.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── artifacts/
├── visuals/
└── report/
```

## Model choice explanation

HydraFault tests simple and practical models:

- **Logistic Regression** as an interpretable baseline.
- **Random Forest** because engineered sensor statistics are nonlinear and tabular.
- **Extra Trees** because it is fast and robust for noisy tabular features.
- **Gradient Boosting** because boosted trees often perform well on structured datasets.

Deep learning is intentionally not required for the first version because this project is focused on reliable feature engineering, evaluation, and explainable engineering decisions.

## Limitations

This is a prototype system built for applied machine learning experimentation. The model currently uses engineered statistical and windowed features rather than deep sequence models. This makes the system easier to explain, but it may miss more complex temporal patterns that could be captured by recurrent or convolutional time-series models.

The dashboard is intended as a demonstration tool, not a production monitoring system.

## Future Work

* Add support for multiple prediction targets in one unified dashboard.
* Compare feature-based models against LSTM, GRU, or 1D-CNN sequence models.
* Add SHAP explanations for individual predictions.
* Improve dashboard startup speed by loading precomputed feature artifacts.
* Add upload support for new hydraulic sensor runs.
* Package the app for cloud deployment.

