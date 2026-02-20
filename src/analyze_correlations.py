#!/usr/bin/env python3
"""
One-off correlation analysis for feature selection.

Reads Silver data, computes feature-feature and feature-target correlations,
and writes a report. The results inform which features to keep in
config/selected_features.yaml.

This script is NOT part of the automated pipeline; run it manually once after
all batches have been ingested and validated.

Usage:
    python src/analyze_correlations.py [--target meantemp]
"""

import argparse
import json
import logging
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
REPORT_PATH = Path("data/silver/correlation_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation analysis for feature selection")
    parser.add_argument(
        "--target", default="meantemp",
        help="Target variable for feature-target correlation (default: meantemp)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    silver_path = project_root / SILVER_CSV_PATH
    if not silver_path.exists():
        logger.error("Silver file not found: %s", silver_path)
        logger.error("Run validate.py first.")
        sys.exit(1)

    df = pd.read_csv(silver_path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), silver_path)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        logger.error("No numeric columns found.")
        sys.exit(1)

    logger.info("Numeric columns: %s", list(numeric_df.columns))

    corr_matrix = numeric_df.corr(method="pearson")
    logger.info("Feature-feature correlation matrix:\n%s", corr_matrix.to_string())

    # Identify highly correlated pairs (|r| > 0.9, excluding self-correlations)
    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if abs(r) > 0.9:
                high_corr_pairs.append({"feature_1": c1, "feature_2": c2, "r": round(r, 4)})

    if high_corr_pairs:
        logger.warning("Highly correlated pairs (|r| > 0.9):")
        for p in high_corr_pairs:
            logger.warning("  %s <-> %s : r=%.4f", p["feature_1"], p["feature_2"], p["r"])
    else:
        logger.info("No feature pairs with |r| > 0.9 found.")

    # Feature-target correlations
    target = args.target
    target_corr = {}
    if target in numeric_df.columns:
        for col in numeric_df.columns:
            if col != target:
                r = numeric_df[target].corr(numeric_df[col])
                target_corr[col] = round(r, 4)
        logger.info("Feature-target correlations (target=%s):", target)
        for col, r in sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            logger.info("  %s : r=%.4f", col, r)
    else:
        logger.warning("Target '%s' not in numeric columns; skipping feature-target analysis.", target)

    report = {
        "target": target,
        "numeric_columns": list(numeric_df.columns),
        "correlation_matrix": {
            c1: {c2: round(corr_matrix.loc[c1, c2], 4) for c2 in cols}
            for c1 in cols
        },
        "high_correlation_pairs": high_corr_pairs,
        "feature_target_correlations": target_corr,
    }

    report_path = project_root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote correlation report to %s", report_path)

    logger.info(
        "Review the report and update config/selected_features.yaml "
        "to drop redundant or low-relevance features."
    )


if __name__ == "__main__":
    main()
