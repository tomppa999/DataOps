"""
Ingest raw batch data into bronze layer.
Reads batch_id from params.yaml, loads batch CSV, appends to bronze.csv.
Idempotent: skips batches already recorded in ingested_batches.json.
"""

import json
import re
from pathlib import Path


def _read_batch_id(params_path: Path) -> int:
    """Read batch_id from params.yaml (simple key: value parsing)."""
    text = params_path.read_text()
    m = re.search(r"batch_id\s*:\s*(\d+)", text)
    if not m:
        raise ValueError(f"batch_id not found in {params_path}")
    return int(m.group(1))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "params.yaml"
    bronze_dir = project_root / "data" / "bronze"
    ingested_log_path = bronze_dir / "ingested_batches.json"
    bronze_path = bronze_dir / "bronze.csv"

    # 1. Read batch_id from params.yaml
    batch_id = _read_batch_id(params_path)
    if not isinstance(batch_id, int) or batch_id < 1 or batch_id > 5:
        raise ValueError(f"batch_id must be an int in 1..5, got {batch_id!r}")

    # 2. Check idempotency: skip if already ingested
    bronze_dir.mkdir(parents=True, exist_ok=True)
    ingested: list[int] = []
    if ingested_log_path.exists():
        with open(ingested_log_path) as f:
            ingested = json.load(f)
    if batch_id in ingested:
        print(f"[INGEST] Batch {batch_id} already ingested. Skipping (idempotent).")
        return

    # 3. Read raw batch CSV (preserve schema/values, no cleaning)
    batch_path = project_root / "data" / "raw_batches" / f"batch_{batch_id}.csv"
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    with open(batch_path) as f:
        lines = f.readlines()
    if not lines:
        print(f"[INGEST] Batch {batch_id} is empty. Nothing to append.")
        ingested.append(batch_id)
        with open(ingested_log_path, "w") as f:
            json.dump(ingested, f, indent=2)
        return

    header = lines[0]
    data_lines = lines[1:]
    row_count = len(data_lines)

    # 4. Append to bronze (create if not exists)
    write_header = not bronze_path.exists()
    with open(bronze_path, "a") as f:
        if write_header:
            f.write(header)
        f.writelines(data_lines)

    # 5. Update ingestion log
    ingested.append(batch_id)
    with open(ingested_log_path, "w") as f:
        json.dump(ingested, f, indent=2)

    print(f"[INGEST] Ingested batch {batch_id}: {row_count} rows appended to bronze.csv")
    if write_header:
        print(f"[INGEST] Created bronze.csv (first batch)")
    else:
        print(f"[INGEST] Appended to existing bronze.csv")


if __name__ == "__main__":
    main()
