from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TARGET_COLUMNS = ["cooler", "valve", "pump", "accumulator", "stable"]

# Real sensor files in the UCI hydraulic systems dataset.
VALID_SENSOR_PREFIXES = (
    "PS",   # pressure sensors
    "TS",   # temperature sensors
    "FS",   # volume flow sensors
    "VS",   # vibration sensor
    "CE",   # cooling efficiency
    "CP",   # cooling power
    "SE",   # efficiency factor
    "EPS",  # motor power
)


def find_raw_dir(raw_dir: Path = RAW_DIR) -> Path:
    if (raw_dir / "profile.txt").exists():
        return raw_dir

    matches = list(raw_dir.rglob("profile.txt"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find profile.txt under {raw_dir}. "
            "Run scripts/download_data.py or manually unzip the dataset into data/raw/."
        )

    return matches[0].parent


def is_sensor_file(path: Path) -> bool:
    name = path.stem.upper()

    if path.name.lower() == "profile.txt":
        return False

    return name.startswith(VALID_SENSOR_PREFIXES)


def read_txt_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")


def load_profiles(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    data_dir = find_raw_dir(raw_dir)
    profile_path = data_dir / "profile.txt"

    profile = read_txt_matrix(profile_path)

    if profile.shape[1] < 5:
        raise ValueError(
            f"profile.txt should have at least 5 columns, found {profile.shape[1]}"
        )

    profile = profile.iloc[:, :5].copy()
    profile.columns = TARGET_COLUMNS
    profile.index.name = "cycle_id"
    return profile


def load_sensor_matrices(raw_dir: Path = RAW_DIR) -> Dict[str, pd.DataFrame]:
    data_dir = find_raw_dir(raw_dir)
    profile = load_profiles(raw_dir)
    n_cycles = len(profile)

    sensors: Dict[str, pd.DataFrame] = {}

    for path in sorted(data_dir.glob("*.txt")):
        if not is_sensor_file(path):
            print(f"Skipping non-sensor file: {path.name}")
            continue

        name = path.stem
        matrix = read_txt_matrix(path)

        if matrix.shape[0] != n_cycles:
            print(
                f"Skipping {path.name}: expected {n_cycles} rows, found {matrix.shape[0]}"
            )
            continue

        sensors[name] = matrix

    if not sensors:
        raise FileNotFoundError(
            f"No valid sensor TXT files found in {data_dir}. "
            "Expected files like PS1.txt, TS1.txt, VS1.txt, etc."
        )

    return sensors


def load_dataset(raw_dir: Path = RAW_DIR) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    sensors = load_sensor_matrices(raw_dir)
    profile = load_profiles(raw_dir)
    return sensors, profile


def summarize_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    sensors, profile = load_dataset(raw_dir)
    rows = []

    for sensor_name, matrix in sensors.items():
        rows.append(
            {
                "sensor": sensor_name,
                "cycles": matrix.shape[0],
                "samples_per_cycle": matrix.shape[1],
            }
        )

    return pd.DataFrame(rows).sort_values("sensor").reset_index(drop=True)
