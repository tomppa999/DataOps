#!/usr/bin/env python3
"""
Gold transformation layer.

Reads Silver data, applies feature selection from config/selected_features.yaml,
and produces an ML-ready Gold dataset for next-day meantemp forecasting.
No scaling or train/test splitting is applied — those are deferred to the
ModelOps stage.
"""

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SILVER_CSV_PATH = Path("data/silver/silver.csv")
FEATURES_CONFIG_PATH = Path("config/selected_features.yaml")
GOLD_CSV_PATH = Path("data/gold/gold.csv")
GOLD_REPORT_PATH = Path("data/gold/gold_report.json")

TARGET_SOURCE_COL = "meantemp"
TARGET_COL = "target"


def load_features_config(path: Path) -> list[str]:
    """Parse selected features list from a simple YAML config."""
    text = path.read_text()
    features: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s+-\s+(.+)$", line)
        if m:
            value = m.group(1).strip()
            if not value.startswith("#"):
                features.append(value)
    if not features:
        raise ValueError(f"No features found in {path}")
    return features


def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Keep only the columns listed in the config. Fail on missing columns."""
    available = set(df.columns)
    missing = [f for f in features if f not in available]
    if missing:
        raise ValueError(
            f"Features in config but missing from data: {missing}. "
            f"Available columns: {sorted(available)}"
        )
    extra = sorted(available - set(features))
    if extra:
        logger.info("Dropping columns not in config: %s", extra)
    return df[features]


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Shift meantemp by -1 to create a next-day prediction target.

    The last row is dropped because it has no next-day value.
    """
    df[TARGET_COL] = df[TARGET_SOURCE_COL].shift(-1)
    rows_before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    logger.info(
        "Created target column (dropped %d trailing row(s) with no next-day value)",
        rows_before - len(df),
    )
    return df


def validate_gold(df: pd.DataFrame) -> None:
    """Sanity-check the Gold dataset before writing."""
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        logger.error("Gold data contains NaN values:\n%s", nan_counts[nan_counts > 0])
        sys.exit(1)

    if TARGET_COL not in df.columns:
        logger.error("Target column '%s' missing from Gold data", TARGET_COL)
        sys.exit(1)

    if len(df) == 0:
        logger.error("Gold dataset is empty")
        sys.exit(1)

    logger.info("Gold validation passed (%d rows, %d columns)", len(df), len(df.columns))


def build_report(df: pd.DataFrame) -> dict:
    """Assemble metadata report for the Gold dataset."""
    return {
        "prediction_task": f"next-day {TARGET_SOURCE_COL}",
        "rows": len(df),
        "columns": list(df.columns),
        "feature_columns": [c for c in df.columns if c not in ("date", TARGET_COL)],
        "target_column": TARGET_COL,
        "date_range": [
            str(df["date"].min()),
            str(df["date"].max()),
        ],
        "target_stats": {
            "mean": round(float(df[TARGET_COL].mean()), 4),
            "std": round(float(df[TARGET_COL].std()), 4),
            "min": round(float(df[TARGET_COL].min()), 4),
            "max": round(float(df[TARGET_COL].max()), 4),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    silver_path = project_root / SILVER_CSV_PATH
    if not silver_path.exists():
        logger.error("Silver file not found: %s", silver_path)
        logger.error("Run validate.py first.")
        sys.exit(1)

    config_path = project_root / FEATURES_CONFIG_PATH
    if not config_path.exists():
        logger.error("Feature config not found: %s", config_path)
        sys.exit(1)

    logger.info("Loading Silver data from %s", silver_path)
    df = pd.read_csv(silver_path)
    rows_in = len(df)
    logger.info("Loaded %d rows, %d columns", rows_in, len(df.columns))

    features = load_features_config(config_path)
    logger.info("Applying feature selection: keeping %s", features)
    df = select_features(df, features)

    df = df.sort_values("date").reset_index(drop=True)
    df = create_target(df)

    validate_gold(df)

    report = build_report(df)

    gold_path = project_root / GOLD_CSV_PATH
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(gold_path, index=False)
    logger.info("Wrote Gold dataset to %s (%d rows)", gold_path, len(df))

    report_path = project_root / GOLD_REPORT_PATH
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote Gold report to %s", report_path)


if __name__ == "__main__":
    main()
