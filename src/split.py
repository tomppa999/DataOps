"""
Split DailyDelhiClimateTrain.csv into 5 time-ordered batches.
Before splitting, introduces simulated data quality issues:
- Randomly remove 10 rows
- Randomly duplicate 10 rows
- Randomly introduce 5 missing values per numeric column
"""

import random
from pathlib import Path

import pandas as pd
# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "DailyDelhiClimateTrain.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_batches"

# Reproducibility
SEED_REMOVE = 42
SEED_DUPLICATE = 43
SEED_MISSING = 44


def main() -> None:
    # 1. Load CSV
    df = pd.read_csv(INPUT_PATH)

    # 2. Remove 10 rows
    df = df.drop(df.sample(n=10, random_state=SEED_REMOVE).index)

    # 3. Duplicate 10 rows
    duplicates = df.sample(n=10, random_state=SEED_DUPLICATE)
    df = pd.concat([df, duplicates], ignore_index=True)

    # 4. Introduce 5 NaNs per numeric column (keeps date intact for time-ordering)
    numeric_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    random.seed(SEED_MISSING)
    for col in numeric_cols:
        indices = random.sample(range(len(df)), 5)
        df.loc[df.index[indices], col] = float("nan")

    # 5. Sort by date and reset index
    df = df.sort_values("date").reset_index(drop=True)

    # 6. Split into 5 batches
    n = len(df)
    batch_size = n // 5
    remainder = n % 5
    batches = []
    start = 0
    for i in range(5):
        size = batch_size + (1 if i < remainder else 0)
        batches.append(df.iloc[start : start + size])
        start += size

    # 7. Add overlapping data: append first 2 rows of each batch to the previous one
    for i in range(4):
        overlap = batches[i + 1].head(2)
        batches[i] = pd.concat([batches[i], overlap], ignore_index=True)

    # 8. Write to data/raw_batches/batch_1.csv, etc.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batches, start=1):
        output_path = OUTPUT_DIR / f"batch_{i}.csv"
        batch.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({len(batch)} rows)")


if __name__ == "__main__":
    main()
