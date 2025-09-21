#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:58:19 2025

@author: rubbersole
"""

"""
Local categorization bootstrap for A26 (UCLA education slice).

Reads a Parquet dataset (folder or single file) from the project's `data/raw`
directory, prepares minimal columns for downstream matching, and writes a clean
Parquet to `data/processed/ucla_edu_clean.parquet`.

Usage:
    python -m src.A26_categorization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import os
import csv
import json
import concurrent.futures
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
# Load environment variables (including OPENAI_API_KEY)
dotenv_path = "-"  # specify your .env path if needed
load_dotenv(dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=api_key)
print(repr(os.environ.get("OPENAI_API_KEY")))
# -------------------------------------------------------------------
# project paths (change DATASET_NAME if your folder/file is named differently)
# -------------------------------------------------------------------
RAW_DIR = Path(
    "/Users/rubbersole/Desktop/2023-2028/Summer 2025/A26_Alumni-Identification/data/raw"
)
PROCESSED_DIR = RAW_DIR.parent / "processed"
DATASET_NAME: Optional[str] = "ucla_edu_full_parquet"  # e.g., folder created on the server

# strictly required columns for subsequent joins
REQUIRED_COLS: tuple[str, ...] = (
    "user_id",
    "school",
    "university_name",
    "degree",
    "field",
    "startdate",
    "enddate",
)


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def _find_parquet_path(raw_dir: Path, dataset_name: Optional[str]) -> Path:
    """
    Resolve the Parquet source path. Prefer an explicit dataset_name; otherwise,
    try common patterns in data/raw.

    Returns a Path that can be a directory (partitioned dataset) or a single file.
    pandas.read_parquet supports both with the pyarrow engine.  # pandas docs
    """
    if dataset_name:
        p = raw_dir / dataset_name
        if p.exists():
            return p

    # fallbacks: first subdir containing parquet files, or any top-level *.parquet
    for sub in raw_dir.iterdir():
        if sub.is_dir() and any(sub.glob("*.parquet")):
            return sub
    files = list(raw_dir.glob("*.parquet"))
    if files:
        return files[0]

    raise FileNotFoundError(
        f"No Parquet dataset found under {raw_dir}. "
        f"Set DATASET_NAME correctly or place the export there."
    )


def _normalize_ws(s: pd.Series) -> pd.Series:
    """Collapse internal whitespace and strip ends (safe on nullable strings)."""
    # Pandas string ops handle NA gracefully when dtype is string/nullable
    return (
        s.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_ucla_edu(raw_dir: Path, dataset_name: Optional[str], columns: Iterable[str]) -> pd.DataFrame:
    """
    Read the Parquet folder/file into a DataFrame with the requested columns.
    - Points pandas at a directory or a file; pyarrow engine handles both.
    - Passes `columns` to minimize IO/memory footprint.

    Returns a DataFrame (potentially large).
    """
    src = _find_parquet_path(raw_dir, dataset_name)

    # dtype_backend is available in pandas >= 2.0; keep optional for portability.
    read_kwargs = {
        "engine": "pyarrow",
        "columns": list(columns),
    }
    try:
        # Prefer pyarrow-backed dtypes when available (memory efficient, nullable)
        df = pd.read_parquet(src, dtype_backend="pyarrow", **read_kwargs)  # pandas>=2.0
    except TypeError:
        # Fallback for older pandas: same read, just without dtype_backend
        df = pd.read_parquet(src, **read_kwargs)

    return df


def prepare_ucla_edu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type the UCLA education slice for downstream matching:
    - ensure required columns exist,
    - trim/collapse whitespace,
    - build `school_text` = university_name filled by school,
    - parse start/end dates with errors='coerce',
    - cast user_id to nullable integer,
    - drop exact duplicates.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    # normalize text columns
    for col in ("school", "university_name", "degree", "field"):
        if col in df.columns:
            df[col] = _normalize_ws(df[col])

    # build helper string for later GPT adjudication or rule checks
    # combine_first keeps left if not NA, otherwise uses right
    df["school_text"] = df["university_name"].combine_first(df["school"])
    df["school_text"] = _normalize_ws(df["school_text"])

    # parse dates; coerce invalids to NaT (no exceptions)
    # (documented behavior of errors='coerce')
    df["startdate"] = pd.to_datetime(df["startdate"], errors="coerce")  # converts invalids to NaT
    df["enddate"] = pd.to_datetime(df["enddate"], errors="coerce")

    # id types; prefer nullable integer
    try:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    except TypeError:
        # If pyarrow-backed, Int64[pyarrow] is fine; otherwise keep as object/int
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")

    # stable column order
    ordered = list(REQUIRED_COLS) + ["school_text"]
    existing_ordered = [c for c in ordered if c in df.columns]
    df = df[existing_ordered + [c for c in df.columns if c not in existing_ordered]]

    # drop exact duplicates to reduce noise
    df = df.drop_duplicates()

    # optional: move to modern nullable dtypes across the frame
    try:
        df = df.convert_dtypes(dtype_backend="pyarrow")
    except TypeError:
        df = df.convert_dtypes()

    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("Loading Parquet dataset from %s", RAW_DIR)
    df = load_ucla_edu(RAW_DIR, DATASET_NAME, REQUIRED_COLS)
    logging.info("Loaded %s rows, %s columns", len(df), len(df.columns))

    logging.info("Preparing dataset for downstream analysis")
    df = prepare_ucla_edu(df)
    logging.info("Prepared %s rows (after de-dup)", len(df))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "ucla_edu_clean.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)
    logging.info("Wrote cleaned Parquet to %s", out_path)

    # optional quick CSV peek for manual inspection (comment in if desired)
    # df.head(200000).to_csv(PROCESSED_DIR / "ucla_edu_clean_sample.csv", index=False)


if __name__ == "__main__":
    main()
