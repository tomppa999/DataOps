"""
Data quality tests for Bronze, Silver, and Gold layers.

Run after each pipeline execution (dvc repro) to verify data integrity:
    pytest tests/test_validation.py -v
"""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRONZE_CSV = PROJECT_ROOT / "data" / "bronze" / "bronze.csv"
INGESTED_LOG = PROJECT_ROOT / "data" / "bronze" / "ingested_batches.json"
SILVER_CSV = PROJECT_ROOT / "data" / "silver" / "silver.csv"
SILVER_REPORT = PROJECT_ROOT / "data" / "silver" / "validation_report.json"
GOLD_CSV = PROJECT_ROOT / "data" / "gold" / "gold.csv"
GOLD_REPORT = PROJECT_ROOT / "data" / "gold" / "gold_report.json"

VALUE_RANGES = {
    "meantemp": (-20, 55),
    "humidity": (0, 100),
    "wind_speed": (0, 50),
    "meanpressure": (900, 1100),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bronze():
    pytest.importorskip("pandas")
    if not BRONZE_CSV.exists():
        pytest.skip("Bronze data not yet produced")
    return pd.read_csv(BRONZE_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def silver():
    pytest.importorskip("pandas")
    if not SILVER_CSV.exists():
        pytest.skip("Silver data not yet produced")
    return pd.read_csv(SILVER_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def gold():
    pytest.importorskip("pandas")
    if not GOLD_CSV.exists():
        pytest.skip("Gold data not yet produced")
    return pd.read_csv(GOLD_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def silver_report():
    if not SILVER_REPORT.exists():
        pytest.skip("Silver validation report not yet produced")
    with open(SILVER_REPORT) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gold_report():
    if not GOLD_REPORT.exists():
        pytest.skip("Gold report not yet produced")
    with open(GOLD_REPORT) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bronze layer tests
# ---------------------------------------------------------------------------
class TestBronze:
    def test_file_exists(self):
        assert BRONZE_CSV.exists(), "bronze.csv must exist after ingestion"

    def test_expected_columns(self, bronze):
        required = {"date", "meantemp", "humidity", "wind_speed", "meanpressure"}
        assert required.issubset(set(bronze.columns))

    def test_not_empty(self, bronze):
        assert len(bronze) > 0

    def test_ingestion_log_consistent(self, bronze):
        """Ingested batches log exists and lists at least one batch."""
        assert INGESTED_LOG.exists(), "ingested_batches.json must exist"
        with open(INGESTED_LOG) as f:
            ingested = json.load(f)
        assert len(ingested) > 0, "At least one batch should be recorded"
        assert all(1 <= b <= 5 for b in ingested), "Batch IDs must be in 1..5"


# ---------------------------------------------------------------------------
# Silver layer tests
# ---------------------------------------------------------------------------
class TestSilver:
    def test_no_duplicate_dates(self, silver):
        dups = silver["date"].duplicated()
        assert dups.sum() == 0, f"Silver has {dups.sum()} duplicate dates"

    def test_time_index_continuity(self, silver):
        """Every calendar day between min and max date must be present."""
        dates = silver["date"].sort_values()
        full_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
        missing = set(full_range) - set(dates)
        assert len(missing) == 0, f"Silver is missing {len(missing)} dates: {sorted(missing)[:5]}..."

    def test_dates_sorted(self, silver):
        assert silver["date"].is_monotonic_increasing, "Silver dates must be sorted ascending"

    def test_no_nan_after_imputation(self, silver):
        core_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
        for col in core_cols:
            nans = silver[col].isna().sum()
            assert nans == 0, f"Silver column '{col}' has {nans} NaN values after imputation"

    def test_value_ranges(self, silver):
        for col, (lo, hi) in VALUE_RANGES.items():
            below = (silver[col] < lo).sum()
            above = (silver[col] > hi).sum()
            assert below == 0, f"{col}: {below} values below {lo}"
            assert above == 0, f"{col}: {above} values above {hi}"

    def test_derived_features_present(self, silver):
        expected = {"meantemp_rolling_7d", "meantemp_lag_1", "meantemp_lag_7", "day_of_year"}
        assert expected.issubset(set(silver.columns)), (
            f"Missing derived features: {expected - set(silver.columns)}"
        )

    def test_validation_report_passed(self, silver_report):
        assert silver_report["validation_passed"] is True


# ---------------------------------------------------------------------------
# Gold layer tests
# ---------------------------------------------------------------------------
class TestGold:
    def test_no_nans(self, gold):
        total = gold.isna().sum().sum()
        assert total == 0, f"Gold dataset has {total} NaN values"

    def test_not_empty(self, gold):
        assert len(gold) > 0

    def test_target_column_exists(self, gold):
        assert "target" in gold.columns, "Gold must have a 'target' column"

    def test_target_is_next_day_meantemp(self, gold):
        """Verify target == next row's meantemp (except the last row which is dropped)."""
        shifted = gold["meantemp"].shift(-1)
        valid = shifted.dropna().index
        pd.testing.assert_series_equal(
            gold.loc[valid, "target"].reset_index(drop=True),
            shifted.dropna().reset_index(drop=True),
            check_names=False,
        )

    def test_expected_feature_columns(self, gold, gold_report):
        for col in gold_report["feature_columns"]:
            assert col in gold.columns, f"Feature '{col}' listed in report but missing from data"

    def test_date_range_nontrivial(self, gold):
        date_range = gold["date"].max() - gold["date"].min()
        assert date_range.days >= 30, "Gold date range must span at least 30 days"

    def test_gold_report_consistency(self, gold, gold_report):
        assert gold_report["rows"] == len(gold)
        assert gold_report["target_column"] == "target"


# ---------------------------------------------------------------------------
# Cross-layer consistency
# ---------------------------------------------------------------------------
class TestCrossLayer:
    def test_silver_not_larger_than_bronze_date_range(self, bronze, silver):
        """Silver should not introduce dates outside the bronze range."""
        assert silver["date"].min() >= bronze["date"].min()
        assert silver["date"].max() <= bronze["date"].max()

    def test_gold_within_silver_date_range(self, silver, gold):
        assert gold["date"].min() >= silver["date"].min()
        assert gold["date"].max() <= silver["date"].max()

    def test_no_data_loss_bronze_to_silver(self, bronze, silver):
        """Silver should have at least as many unique dates as bronze (gaps are filled)."""
        bronze_unique = bronze["date"].nunique()
        silver_unique = silver["date"].nunique()
        assert silver_unique >= bronze_unique, (
            f"Silver has fewer unique dates ({silver_unique}) than bronze ({bronze_unique})"
        )
