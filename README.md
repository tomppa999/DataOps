# DataOps — Incremental Climate Data Pipeline

A DataOps pipeline that manages the [Daily Delhi Climate](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) dataset through a **Bronze–Silver–Gold** architecture with incremental batch ingestion, automated validation, and full version control via **DVC**.

## Project Structure

```
├── config/
│   └── selected_features.yaml   # Features kept in the Gold dataset
├── data/
│   ├── raw/                     # Original Kaggle CSV (DailyDelhiClimateTrain.csv)
│   ├── raw_batches/             # 5 time-ordered batches (DVC-tracked)
│   ├── bronze/                  # Append-only ingested data
│   ├── silver/                  # Cleaned, validated, enriched data
│   └── gold/                    # ML-ready dataset (next-day meantemp prediction)
├── src/
│   ├── split.py                 # One-off: split train.csv into 5 batches
│   ├── ingest.py                # Bronze: append batch to bronze.csv
│   ├── validate.py              # Silver: validate, clean, add derived features
│   ├── transform.py             # Gold: feature selection + target creation
│   └── analyze_correlations.py  # One-off: correlation analysis for feature selection
├── tests/
│   └── test_validation.py       # Data quality tests (Bronze/Silver/Gold)
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Pipeline parameter (batch_id)
└── requirements.txt
```

## Setup

```bash
# Create and activate conda environment
conda create -n dataops python=3.14
conda activate dataops

# Install dependencies
pip install -r requirements.txt

# Initialise DVC (already done — remote is configured in .dvc/config)
dvc pull   # fetch tracked data from remote
```

## Pipeline Overview

The DVC pipeline (`dvc.yaml`) chains three stages:

```
raw_batches → ingest → validate → transform
```

| Stage        | Script              | Input → Output                        | Description                                                        |
| ------------ | ------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| **ingest**   | `src/ingest.py`     | `raw_batches/batch_N.csv` → `bronze/` | Appends one batch to `bronze.csv`; idempotent via ingestion log    |
| **validate** | `src/validate.py`   | `bronze/` → `silver/`                 | Dedup, date-gap filling, imputation, value-range checks, enrichment |
| **transform**| `src/transform.py`  | `silver/` → `gold/`                   | Feature selection, next-day `meantemp` target, final validation     |

The active batch is controlled by `batch_id` in `params.yaml`. Changing it and running `dvc repro` triggers the full pipeline.

## Batch Splitting

`src/split.py` divides `DailyDelhiClimateTrain.csv` into 5 time-ordered batches with simulated data quality issues:

- 10 randomly removed rows (seed 42)
- 10 randomly duplicated rows (seed 43)
- 5 NaN values per numeric column (seed 44)
- 2-row overlaps between consecutive batches

Fixed seeds ensure the split is fully reproducible.

## Running the Pipeline (Incremental Simulation)

To simulate 5 incoming batches from a clean state:

```bash
# 1. Reset generated data
rm -f data/bronze/bronze.csv data/bronze/ingested_batches.json
rm -f data/silver/silver.csv data/silver/validation_report.json
rm -f data/gold/gold.csv data/gold/gold_report.json

# 2. Loop through batches
for i in 1 2 3 4 5; do
    echo "batch_id: $i" > params.yaml

    dvc repro                                    # run pipeline
    pytest tests/test_validation.py -v           # verify data quality

    dvc push                                     # push data to remote
    git add -A
    git commit -m "Ingest batch $i: pipeline run + tests pass"
done
```

Each commit captures updated `.dvc` lock files and metric reports, so any historical batch state can be reproduced with `git checkout <commit> && dvc checkout`.

## Data Quality Tests

```bash
pytest tests/test_validation.py -v
```

The test suite covers 21 checks across all layers:

- **Bronze**: file existence, expected schema, ingestion log consistency
- **Silver**: no duplicate dates, continuous time index, no NaNs, value ranges, derived features present, validation report status
- **Gold**: no NaNs, target column correctness (next-day shift), feature consistency with config, non-trivial date range, report/data agreement
- **Cross-layer**: date range containment (Silver ⊆ Bronze, Gold ⊆ Silver), no data loss from Bronze to Silver

## Correlation Analysis (One-Off)

```bash
python src/analyze_correlations.py
```

Produces `data/silver/correlation_report.json` with feature–feature and feature–target Pearson correlations. Results informed the feature selection in `config/selected_features.yaml`. This script is not part of the automated pipeline.

## Reproducibility

| Mechanism | Purpose |
| --- | --- |
| **DVC** | Content-addressed data versioning + remote storage (DagsHub S3) |
| **params.yaml** | Pipeline parameter (`batch_id`) tracked by DVC |
| **dvc.yaml** | Declarative pipeline DAG — `dvc repro` reruns only changed stages |
| **Git** | Tracks code, configs, `.dvc` metadata, and metric reports |
| **Fixed seeds** | `split.py` produces identical batches on any machine |
| **Idempotent ingestion** | `ingest.py` skips already-ingested batches |

To reproduce the state after batch *N*: check out the corresponding git commit and run `dvc checkout`.

## Assumptions

- Only `train.csv` from the Kaggle dataset is used (for this first assignment).
- Prediction task: next-day `meantemp` forecasting.
- No model training is performed. The Gold dataset is the final output.
- Feature selection (`config/selected_features.yaml`) was determined once via correlation analysis and then fixed for pipeline stability.
