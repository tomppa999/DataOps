#!/usr/bin/env python3
"""
Silver validation, cleaning, and enrichment pipeline.

Reads bronze data, validates, cleans, adds derived features, and outputs
silver data with a validation report.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging for deterministic, readable output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
BRONZE_PATH = Path("data/bronze/bronze.csv")
SILVER_CSV_PATH = Path("data/silver/silver.csv")
SILVER_REPORT_PATH = Path("data/silver/validation_report.json")

# Value range constraints: (min, max, hard_fail)
VALUE_RANGES = {
    "meantemp": (-20, 55, True),
    "humidity": (0, 100, True),
    "wind_speed": (0, 50, False),
    "meanpressure": (900, 1100, False),
}


def detect_date_column(df: pd.DataFrame) -> str:
    """Detect date column: 'date', first column, or any column containing 'date'."""
    cols = df.columns.tolist()
    # Exact match
    if "date" in cols:
        return "date"
    # First column
    if len(cols) > 0:
        first = cols[0].lower()
        if "date" in first:
            return cols[0]
    # Any column containing 'date'
    for c in cols:
        if "date" in c.lower():
            return c
    raise ValueError(f"No date column found. Columns: {cols}")


def load_bronze(path: Path) -> pd.DataFrame:
    """Load bronze CSV and parse date column."""
    logger.info("Loading bronze data from %s", path)
    df = pd.read_csv(path)
    date_col = detect_date_column(df)
    logger.info("Detected date column: %s", date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates > 0:
        logger.error("Found %d invalid date values", invalid_dates)
        sys.exit(1)
    return df, date_col


def run_checks(df: pd.DataFrame, date_col: str, report: dict) -> bool:
    """Run validation checks. Returns False if hard-fail occurred."""
    all_ok = True

    # 1) Duplicated dates
    dup_mask = df.duplicated(subset=[date_col], keep=False)
    n_dups = dup_mask.sum()
    if n_dups > 0:
        dup_dates = df.loc[dup_mask, date_col].unique().tolist()
        logger.warning("Found %d rows with duplicated dates: %s", n_dups, dup_dates[:10])
        if len(dup_dates) > 10:
            logger.warning("... and %d more", len(dup_dates) - 10)
        report["duplicates_found"] = int(n_dups)
    else:
        report["duplicates_found"] = 0

    # 2) Time index continuity
    df_sorted = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="first")
    min_date = df_sorted[date_col].min()
    max_date = df_sorted[date_col].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq="D")
    existing_dates = set(df_sorted[date_col].dt.normalize())
    missing_dates = [d for d in full_range if d.normalize() not in existing_dates]
    n_missing = len(missing_dates)
    if n_missing > 0:
        logger.warning("Found %d missing dates between %s and %s", n_missing, min_date, max_date)
        if n_missing <= 20:
            logger.warning("Missing dates: %s", missing_dates)
        else:
            logger.warning("First 10: %s ... last 10: %s", missing_dates[:10], missing_dates[-10:])
        report["missing_dates"] = n_missing
        report["missing_date_list"] = [d.strftime("%Y-%m-%d") for d in missing_dates]
    else:
        report["missing_dates"] = 0
        report["missing_date_list"] = []

    # 3) Value ranges
    numeric_cols = [c for c in VALUE_RANGES if c in df.columns]
    report["value_range_violations"] = {}
    report["value_range_warnings"] = {}

    for col in numeric_cols:
        min_val, max_val, hard_fail = VALUE_RANGES[col]
        series = pd.to_numeric(df[col], errors="coerce")
        low = (series < min_val) & series.notna()
        high = (series > max_val) & series.notna()
        n_low = low.sum()
        n_high = high.sum()
        if n_low > 0 or n_high > 0:
            if hard_fail:
                logger.error(
                    "%s: %d values below %s, %d above %s (HARD FAIL)",
                    col, n_low, min_val, n_high, max_val,
                )
                report["value_range_violations"][col] = {
                    "below_min": int(n_low),
                    "above_max": int(n_high),
                    "min_allowed": min_val,
                    "max_allowed": max_val,
                }
                all_ok = False
            else:
                logger.warning(
                    "%s: %d values below %s, %d above %s (WARN)",
                    col, n_low, min_val, n_high, max_val,
                )
                report["value_range_warnings"][col] = {
                    "below_min": int(n_low),
                    "above_max": int(n_high),
                    "min_allowed": min_val,
                    "max_allowed": max_val,
                }

    return all_ok


def clean_data(df: pd.DataFrame, date_col: str, report: dict) -> pd.DataFrame:
    """Apply cleaning: drop duplicates, fill missing dates, impute missing values."""
    rows_in = len(df)
    report["rows_in"] = rows_in

    # Drop duplicate dates (keep first)
    df = df.sort_values(date_col)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[date_col], keep="first")
    n_dups_removed = before_dedup - len(df)
    report["duplicates_removed"] = int(n_dups_removed)
    if n_dups_removed > 0:
        logger.info("Dropped %d duplicate date rows (kept first)", n_dups_removed)

    # Reindex to full date range for continuity
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq="D")
    n_rows_before_reindex = len(df)
    df = df.set_index(date_col)
    df = df.reindex(full_range)
    df.index.name = date_col
    df = df.reset_index()

    n_missing_dates_added = len(full_range) - n_rows_before_reindex
    report["missing_dates_filled"] = int(n_missing_dates_added)
    if n_missing_dates_added > 0:
        logger.info("Added %d rows for missing dates", n_missing_dates_added)

    # Replace soft-fail out-of-range values with NaN (to be imputed below)
    numeric_cols = [c for c in df.columns if c != date_col]
    report["soft_range_replacements"] = {}
    for col in numeric_cols:
        if col in VALUE_RANGES:
            min_val, max_val, hard_fail = VALUE_RANGES[col]
            if not hard_fail:
                mask = (df[col] < min_val) | (df[col] > max_val)
                n_replaced = mask.sum()
                if n_replaced > 0:
                    df.loc[mask, col] = float("nan")
                    report["soft_range_replacements"][col] = int(n_replaced)
                    logger.info("Replaced %d out-of-range values in %s with NaN", n_replaced, col)

    # Impute missing numeric values: forward-fill then backward-fill
    missing_before = {}
    for col in numeric_cols:
        n = df[col].isna().sum()
        if n > 0:
            missing_before[col] = int(n)

    if missing_before:
        report["missing_values_before_impute"] = missing_before
        total_missing = sum(missing_before.values())
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        report["missing_values_imputed"] = int(total_missing)
        logger.info("Imputed %d missing numeric values (ffill then bfill)", total_missing)
    else:
        report["missing_values_before_impute"] = {}
        report["missing_values_imputed"] = 0

    rows_out = len(df)
    report["rows_out"] = rows_out

    # Round numeric columns to 2 decimals
    df[numeric_cols] = df[numeric_cols].round(2)

    # Min/max per variable for report (after rounding)
    report["min_max"] = {}
    for col in numeric_cols:
        if col in df.columns:
            report["min_max"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

    return df


def add_derived_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create weather-derived features from the cleaned Silver data."""
    df = df.sort_values(date_col).reset_index(drop=True)

    df["meantemp_rolling_7d"] = df["meantemp"].rolling(window=7, min_periods=1).mean()
    df["meantemp_lag_1"] = df["meantemp"].shift(1)
    df["meantemp_lag_7"] = df["meantemp"].shift(7)
    df["day_of_year"] = pd.to_datetime(df[date_col]).dt.dayofyear

    df["meantemp_lag_1"] = df["meantemp_lag_1"].bfill()
    df["meantemp_lag_7"] = df["meantemp_lag_7"].bfill()

    return df


def main() -> None:
    """Run validation, cleaning, and enrichment pipeline."""
    bronze_path = Path(__file__).resolve().parent.parent / BRONZE_PATH
    if not bronze_path.exists():
        logger.error("Bronze file not found: %s", bronze_path)
        sys.exit(1)

    silver_csv = Path(__file__).resolve().parent.parent / SILVER_CSV_PATH
    silver_report = Path(__file__).resolve().parent.parent / SILVER_REPORT_PATH
    silver_csv.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {}

    df, date_col = load_bronze(bronze_path)
    report["date_column"] = date_col

    if not run_checks(df, date_col, report):
        logger.error("Validation failed (hard-fail). Exiting.")
        report["validation_passed"] = False
        silver_report.parent.mkdir(parents=True, exist_ok=True)
        with open(silver_report, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    df_clean = clean_data(df, date_col, report)
    report["validation_passed"] = True

    logger.info("Adding derived features")
    df_clean = add_derived_features(df_clean, date_col)
    derived_cols = ["meantemp_rolling_7d", "meantemp_lag_1", "meantemp_lag_7", "day_of_year"]
    report["enrichment"] = {"derived_columns": derived_cols}
    logger.info("Added derived columns: %s", derived_cols)

    df_clean = df_clean.sort_values(date_col).reset_index(drop=True)

    df_clean.to_csv(silver_csv, index=False)
    logger.info("Wrote silver data to %s (%d rows)", silver_csv, len(df_clean))

    with open(silver_report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote validation report to %s", silver_report)


if __name__ == "__main__":
    main()
