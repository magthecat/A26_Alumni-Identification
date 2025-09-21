#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Optional: for streaming write that does not require loading all rows at once
try:
    import pyarrow as pa
    import pyarrow.dataset as pads
except Exception:
    pa = None
    pads = None

# -------------------------------------------------------------------
# Paths (repo root → data/raw, data/processed)
# -------------------------------------------------------------------
# This file lives in repo/src/, so repo root is parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# If your Parquet lives in a named subfolder/file under data/raw, set it here.
# Leave as None to auto-detect the first Parquet folder/file found.
DATASET_NAME: Optional[str] = None  # e.g., "ucla_edu_full_parquet"

# Output CSV name
OUT_CSV = PROCESSED_DIR / "ucla_edu_full.csv"

# Toggle to stream-write CSV (memory-safe for very large datasets).
# Requires pyarrow >= 8.0
STREAM_WRITE = True


def _find_parquet_path(raw_dir: Path, dataset_name: Optional[str]) -> Path:
    """
    Return a path that can be a Parquet directory (partitioned) or a single file.
    """
    if dataset_name:
        p = raw_dir / dataset_name
        if p.exists():
            return p

    # Prefer a subdirectory that contains *.parquet files
    for sub in raw_dir.iterdir():
        if sub.is_dir() and any(sub.glob("*.parquet")):
            return sub

    # Otherwise take the first top-level *.parquet
    files = list(raw_dir.glob("*.parquet"))
    if files:
        return files[0]

    raise FileNotFoundError(
        f"No Parquet dataset found under {raw_dir}. "
        f"Set DATASET_NAME or place a Parquet file/folder in data/raw."
    )


def _iter_pa_batches(ds, batch_size: int = 100_000):
    """
    Yield RecordBatches from a pyarrow.dataset.Dataset across PyArrow versions.
    Prefers Dataset.to_batches; falls back to Dataset.scanner(...).to_batches().
    """
    # Newer PyArrow: Dataset.to_batches exists
    if hasattr(ds, "to_batches"):
        yield from ds.to_batches(batch_size=batch_size)
        return
    # Older API: use a Scanner
    if hasattr(ds, "scanner"):
        scanner = ds.scanner(batch_size=batch_size)
        yield from scanner.to_batches()
        return
    raise AttributeError("This PyArrow version lacks both Dataset.to_batches and Dataset.scanner.")

def write_csv_entire_dataset(parquet_src: Path, out_csv: Path) -> None:
    """
    Convert the entire Parquet dataset (folder or file) to a single CSV.
    Uses streaming mode if available; otherwise falls back to pandas.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Streaming path (handles partitioned datasets and large files)
    if STREAM_WRITE and pads is not None:
        ds = pads.dataset(str(parquet_src))  # file or directory

        # Overwrite if exists
        if out_csv.exists():
            out_csv.unlink()

        header_written = False
        for rec_batch in _iter_pa_batches(ds, batch_size=100_000):
            # Convert to pandas on the fly and append
            try:
                chunk_df = rec_batch.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception:
                # Fallback for older pandas without ArrowDtype
                chunk_df = rec_batch.to_pandas()
            chunk_df.to_csv(out_csv, mode="a", index=False, header=not header_written)
            header_written = True
        return

    # Fallback: load all rows via pandas (simple, needs RAM)
    df = pd.read_parquet(parquet_src, engine="pyarrow")
    df.to_csv(out_csv, index=False)



def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parquet_src = _find_parquet_path(RAW_DIR, DATASET_NAME)
    logging.info("Converting Parquet → CSV\n  Source: %s\n  Output: %s", parquet_src, OUT_CSV)

    write_csv_entire_dataset(parquet_src, OUT_CSV)
    logging.info("Done. Wrote CSV to %s", OUT_CSV)


if __name__ == "__main__":
    main()
