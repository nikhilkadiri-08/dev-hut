"""Derive the ML training domain from the supplied synthetic dataset.

The optimiser must warn when a user's inputs fall outside the range the kNN residual
model was actually fitted on. Rather than hardcoding those ranges, this script reads
them straight out of the dataset and writes backend/data/ml_domain.json.

Run from the repository root:

    python scripts/extract_ml_domain.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "synthetic_dataset.csv"
OUTPUT = ROOT / "backend" / "data" / "ml_domain.json"

# Columns that a user can influence through the input form. Front/tail resistance and
# stage count are included because they bound what the residual model has ever seen,
# even though the optimiser (not the user) will eventually choose them.
INPUT_COLUMNS = [
    "Test_kV",
    "Load_C_pF",
    "Divider_C_pF",
    "Stray_C_pF",
    "L_uH",
    "Efficiency",
    "Stages",
    "Charge_kV_Stage",
    "Front_R_Stage",
    "Tail_R_Stage",
]

TARGET_COLUMNS = [
    "Physics_FrontPeak_us",
    "Physics_Tail_us",
    "Observed_FrontPeak_us",
    "Observed_Tail_us",
    "Observed_Crest_kV",
    "Residual_FrontPeak",
    "Residual_Tail",
    "Residual_Crest",
]


def _ranges(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    return {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
            "mean": float(frame[column].mean()),
        }
        for column in columns
    }


def build_domain(frame: pd.DataFrame) -> dict:
    training = frame[frame["Split"] == "Train"]

    per_impulse = {}
    for impulse_type, group in training.groupby("Impulse_Type"):
        total_load = group["Load_C_pF"] + group["Divider_C_pF"] + group["Stray_C_pF"]
        per_impulse[str(impulse_type).lower()] = {
            "training_records": int(len(group)),
            "inputs": _ranges(group, INPUT_COLUMNS),
            "derived": {
                "Total_Load_C_pF": {
                    "min": float(total_load.min()),
                    "max": float(total_load.max()),
                    "mean": float(total_load.mean()),
                }
            },
            "targets": _ranges(group, TARGET_COLUMNS),
        }

    return {
        "source": "data/synthetic_dataset.csv (exported from the supplied workbook)",
        "note": (
            "Ranges are computed from the Train split only. Validation and hidden-test "
            "rows are excluded so that this file cannot leak evaluation data into the "
            "in-domain check."
        ),
        "split_counts": {
            str(split): int(count)
            for split, count in frame["Split"].value_counts().items()
        },
        "impulse_types": per_impulse,
    }


def main() -> None:
    frame = pd.read_csv(DATASET)
    domain = build_domain(frame)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(domain, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for impulse_type, block in domain["impulse_types"].items():
        print(f"  {impulse_type}: {block['training_records']} training records")


if __name__ == "__main__":
    main()
