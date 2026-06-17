"""
Download and unpack the UCI Condition Monitoring of Hydraulic Systems dataset.

Run:
    python scripts/download_data.py

If this fails because the UCI URL changes, manually download the dataset ZIP
and unzip all TXT files into data/raw/.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "hydraulic_systems.zip"

URLS = [
    # UCI static public dataset path. If UCI changes this, use manual download.
    "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip",
]


def download_file(url: str, dest: Path) -> bool:
    print(f"Trying: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print(f"Downloaded {len(response.content):,} bytes to {dest}")
        return True
    except Exception as exc:
        print(f"Failed: {exc}")
        return False


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted dataset to {out_dir}")


def flatten_txt_files(raw_dir: Path) -> None:
    """
    UCI zips sometimes contain a nested folder. This moves all TXT files into data/raw/.
    """
    for txt in raw_dir.rglob("*.txt"):
        if txt.parent == raw_dir:
            continue
        target = raw_dir / txt.name
        if not target.exists():
            target.write_bytes(txt.read_bytes())
            print(f"Moved {txt.name} to data/raw/")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / "profile.txt").exists():
        print("profile.txt already exists. Dataset appears ready.")
        return

    success = False
    for url in URLS:
        if download_file(url, ZIP_PATH):
            success = True
            break

    if not success:
        raise SystemExit(
            "Automatic download failed. Manually download the UCI Hydraulic Systems dataset, "
            "unzip it, and place all .txt files inside data/raw/."
        )

    extract_zip(ZIP_PATH, RAW_DIR)
    flatten_txt_files(RAW_DIR)

    if not (RAW_DIR / "profile.txt").exists():
        raise SystemExit(
            "Download/extract completed, but profile.txt was not found. "
            "Check data/raw/ and move the TXT files there manually."
        )

    print("Dataset ready.")


if __name__ == "__main__":
    main()
