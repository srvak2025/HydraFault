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

## What to show a professor

1. The dashboard.
2. The model comparison table.
3. The top feature importance plot.
4. A selected test run with predicted fault condition.
5. The README section explaining why each model was used.

## Example pitch

> I built HydraFault to demonstrate hands-on ML implementation on engineering time-series data. It converts raw hydraulic sensor cycles into statistical and windowed features, compares multiple classification models, explains which sensor signals drive predictions, and turns the result into a maintenance recommendation.
