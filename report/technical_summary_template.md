# HydraFault Technical Summary

## Problem

Hydraulic systems generate multivariate sensor time-series data during operating cycles. The goal of this project is to classify the health condition of system components from raw sensor signals and explain which time-series patterns drive the prediction.

## Dataset

The project uses the UCI Condition Monitoring of Hydraulic Systems dataset. Each row in a sensor file represents one operating cycle, while each column represents a time sample within that cycle. The `profile.txt` file contains component condition labels.

## Feature Engineering

For each sensor cycle, HydraFault extracts:

- Mean
- Standard deviation
- Minimum
- Maximum
- Range
- RMS
- First value
- Last value
- Delta
- Linear slope
- Coefficient of variation
- Windowed first-25%, mid-50%, last-25%, and last-10% features
- Late-cycle mean shift
- Late-cycle instability shift

These features capture both overall sensor magnitude and time-dependent degradation behavior.

## Models

HydraFault compares:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Tree-based models are appropriate because the engineered features form a nonlinear tabular dataset. Logistic regression is kept as a simple baseline.

## Evaluation

The project reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Confusion matrix

Macro F1 is emphasized because component condition classes may not be equally represented.

## Explainability

The project uses feature importance or permutation importance to identify which engineered sensor features most affect the model's prediction.

## Deliverable

The final Streamlit dashboard allows a user to select a cycle, view a sensor time-series plot, see the predicted health condition, compare models, inspect feature importance, and receive a maintenance recommendation.
