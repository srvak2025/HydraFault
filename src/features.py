from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _slope_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Vectorized slope for every row.
    arr shape: [n_cycles, n_timesteps]
    """
    arr = arr.astype(float)
    n = arr.shape[1]

    if n < 2:
        return np.zeros(arr.shape[0])

    x = np.linspace(0.0, 1.0, n)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)

    y_centered = arr - arr.mean(axis=1, keepdims=True)
    return y_centered @ x_centered / denom


def _window(arr: np.ndarray, start_frac: float, end_frac: float) -> np.ndarray:
    n = arr.shape[1]
    start = int(start_frac * n)
    end = max(start + 1, int(end_frac * n))
    return arr[:, start:end]


def _add_basic_features(out: dict, arr: np.ndarray, prefix: str) -> None:
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    min_v = arr.min(axis=1)
    max_v = arr.max(axis=1)

    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = std
    out[f"{prefix}_min"] = min_v
    out[f"{prefix}_max"] = max_v
    out[f"{prefix}_range"] = max_v - min_v
    out[f"{prefix}_rms"] = np.sqrt(np.mean(arr ** 2, axis=1))
    out[f"{prefix}_first"] = arr[:, 0]
    out[f"{prefix}_last"] = arr[:, -1]
    out[f"{prefix}_delta"] = arr[:, -1] - arr[:, 0]
    out[f"{prefix}_slope"] = _slope_matrix(arr)
    out[f"{prefix}_cv"] = std / (np.abs(mean) + 1e-9)


def _add_window_features(out: dict, arr: np.ndarray, sensor_name: str) -> None:
    windows = {
        "first25": _window(arr, 0.00, 0.25),
        "mid50": _window(arr, 0.25, 0.75),
        "last25": _window(arr, 0.75, 1.00),
        "last10": _window(arr, 0.90, 1.00),
    }

    means = {}
    stds = {}

    for window_name, w in windows.items():
        prefix = f"{sensor_name}_{window_name}"

        mean = w.mean(axis=1)
        std = w.std(axis=1)
        min_v = w.min(axis=1)
        max_v = w.max(axis=1)

        means[window_name] = mean
        stds[window_name] = std

        out[f"{prefix}_mean"] = mean
        out[f"{prefix}_std"] = std
        out[f"{prefix}_min"] = min_v
        out[f"{prefix}_max"] = max_v
        out[f"{prefix}_range"] = max_v - min_v
        out[f"{prefix}_slope"] = _slope_matrix(w)

    out[f"{sensor_name}_late_mean_shift"] = means["last25"] - means["first25"]
    out[f"{sensor_name}_late_instability_shift"] = stds["last25"] - stds["first25"]


def extract_features_for_sensor(matrix: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    arr = matrix.to_numpy(dtype=float)
    out = {}

    _add_basic_features(out, arr, sensor_name)
    _add_window_features(out, arr, sensor_name)

    return pd.DataFrame(out)


def build_feature_table(
    sensors: Dict[str, pd.DataFrame],
    selected_sensors: Iterable[str] | None = None,
) -> pd.DataFrame:
    if selected_sensors is None:
        selected_sensor_names = list(sensors.keys())
    else:
        selected_sensor_names = [s for s in selected_sensors if s in sensors]

    if not selected_sensor_names:
        raise ValueError("No valid sensors selected.")

    feature_frames = []

    for sensor_name in selected_sensor_names:
        print(f"Extracting features from {sensor_name}...")
        feature_frames.append(extract_features_for_sensor(sensors[sensor_name], sensor_name))

    features = pd.concat(feature_frames, axis=1)
    features.index.name = "cycle_id"

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))

    return features


def make_xy(features: pd.DataFrame, profile: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in profile.columns:
        raise ValueError(f"Unknown target '{target}'. Choose from: {list(profile.columns)}")

    X = features.copy()
    y = profile[target].copy().astype(str)

    return X, y
