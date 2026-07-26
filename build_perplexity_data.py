#!/usr/bin/env python3
"""
build_perplexity_data.py

Reads per-prompt inference logs for KV-cache compression experiments from ./Data
and reshapes them into a model-ready format: one row per prompt, with one column
per compression setting holding that setting's perplexity.

Target model spec:
    input  = prompt text
    output = vector of 6 perplexities, one per compression setting:
             [h2o_20, h2o_40, h2o_60, kvquant_2bit, kvquant_3bit, kvquant_4bit]

Input files live in   ./Data/           (expects up to 24: 6 settings x 4 datasets)
Output files written to ./Perplexity Data/

Filename convention expected (matches the uploaded data):
    h2o_budget_<20|40|60>pct_<dataset>_per_prompt.csv
    kvquant_<2|3|4>bit_<dataset>_per_prompt.csv
where <dataset> is one of: gsm8k, arc_challenge, hellaswag, wikitext103

The four datasets use four different schemas, but all of them contain a `prompt`
column and a `perplexity` column, plus a leading per-prompt index column named
one of: question_index / item_index / chunk_index. This script relies only on
those three fields, so it is robust to the schema differences.

Outputs (in ./Perplexity Data/):
    perplexity_wide_all.csv        every prompt, all 6 setting columns (NaN where missing)
    perplexity_wide_complete.csv   only prompts that have all 6 settings present
    perplexity_long.csv            tidy long form: one row per (prompt, setting)
    coverage_report.csv            which (dataset x setting) files were found / missing
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "Perplexity Data"

# The six compression settings, in the fixed order the output vector should use.
SETTINGS = [
    "h2o_20",
    "h2o_40",
    "h2o_60",
    "kvquant_2bit",
    "kvquant_3bit",
    "kvquant_4bit",
]

# The four evaluation datasets.
DATASETS = ["gsm8k", "arc_challenge", "hellaswag", "wikitext103"]

# Candidate names for the per-prompt index column across the four schemas.
INDEX_COL_CANDIDATES = ["question_index", "item_index", "chunk_index"]


# ----------------------------------------------------------------------------
# Filename parsing
# ----------------------------------------------------------------------------

def parse_filename(name: str):
    """
    Map a CSV filename to (dataset, setting).
    Returns (dataset, setting) or None if the name doesn't match a known pattern.
    """
    stem = name.lower()

    # Identify dataset (check longer names first to avoid partial matches).
    dataset = None
    for ds in sorted(DATASETS, key=len, reverse=True):
        if ds in stem:
            dataset = ds
            break
    if dataset is None:
        return None

    # Identify setting.
    setting = None
    m_h2o = re.search(r"h2o_budget_(\d+)pct", stem)
    m_kv = re.search(r"kvquant_(\d+)bit", stem)
    if m_h2o:
        setting = f"h2o_{int(m_h2o.group(1))}"
    elif m_kv:
        setting = f"kvquant_{int(m_kv.group(1))}bit"

    if setting is None or setting not in SETTINGS:
        return None

    return dataset, setting


def find_index_col(df: pd.DataFrame) -> str | None:
    for c in INDEX_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_one(path: Path, dataset: str, setting: str) -> pd.DataFrame | None:
    """
    Read a single compression CSV and return a tidy frame with columns:
        dataset, setting, prompt_index, prompt, perplexity
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed to read {path.name}: {e}", file=sys.stderr)
        return None

    if "prompt" not in df.columns or "perplexity" not in df.columns:
        print(f"  ! {path.name} missing prompt/perplexity column; skipping",
              file=sys.stderr)
        return None

    idx_col = find_index_col(df)
    if idx_col is None:
        # Fall back to row position as the index.
        df = df.reset_index().rename(columns={"index": "prompt_index"})
        idx_col = "prompt_index"

    out = pd.DataFrame({
        "dataset": dataset,
        "setting": setting,
        "prompt_index": df[idx_col].values,
        "prompt": df["prompt"].astype(str).values,
        "perplexity": pd.to_numeric(df["perplexity"], errors="coerce").values,
    })

    # Drop rows with no usable perplexity.
    n_before = len(out)
    out = out.dropna(subset=["perplexity"])
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"    ({n_dropped} rows dropped for non-numeric/empty perplexity)")

    return out


def load_all(data_dir: Path):
    """Load every recognizable CSV; return (long_df, coverage_df)."""
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data directory not found: {data_dir}")

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"ERROR: no CSV files found in {data_dir}")

    frames = []
    found = set()  # (dataset, setting) pairs actually loaded

    print(f"Scanning {len(csvs)} CSV files in {data_dir} ...")
    for path in csvs:
        parsed = parse_filename(path.name)
        if parsed is None:
            print(f"  - skip (unrecognized name): {path.name}")
            continue
        dataset, setting = parsed
        print(f"  + {path.name}  ->  dataset={dataset}, setting={setting}")
        frame = load_one(path, dataset, setting)
        if frame is not None and len(frame):
            frames.append(frame)
            found.add((dataset, setting))

    if not frames:
        sys.exit("ERROR: nothing loadable after parsing filenames.")

    long_df = pd.concat(frames, ignore_index=True)

    # Coverage report over the full expected grid.
    rows = []
    for ds in DATASETS:
        for st in SETTINGS:
            rows.append({
                "dataset": ds,
                "setting": st,
                "present": (ds, st) in found,
            })
    coverage = pd.DataFrame(rows)

    return long_df, coverage


# ----------------------------------------------------------------------------
# Reshaping
# ----------------------------------------------------------------------------

def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per (dataset, prompt_index, prompt) with one column per
    setting. Column order follows SETTINGS.
    """
    # Guard against duplicate (dataset, index, setting) keys: average them.
    dup = long_df.duplicated(
        subset=["dataset", "prompt_index", "setting"], keep=False
    )
    if dup.any():
        n = int(dup.sum())
        print(f"  note: {n} duplicate (dataset,index,setting) rows -> averaged")
        long_df = (
            long_df.groupby(
                ["dataset", "prompt_index", "prompt", "setting"], as_index=False
            )["perplexity"].mean()
        )

    wide = long_df.pivot_table(
        index=["dataset", "prompt_index", "prompt"],
        columns="setting",
        values="perplexity",
        aggfunc="mean",
    ).reset_index()

    wide.columns.name = None

    # Ensure all six setting columns exist and are ordered.
    for st in SETTINGS:
        if st not in wide.columns:
            wide[st] = np.nan

    ordered = ["dataset", "prompt_index", "prompt"] + SETTINGS
    wide = wide[ordered]

    return wide


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_df, coverage = load_all(DATA_DIR)

    # --- coverage report ---
    print("\nCoverage (dataset x setting):")
    pivot_cov = coverage.pivot(index="dataset", columns="setting",
                               values="present")[SETTINGS]
    print(pivot_cov.to_string())
    n_present = int(coverage["present"].sum())
    print(f"\n{n_present} / {len(coverage)} setting-files present "
          f"(full grid = {len(DATASETS) * len(SETTINGS)}).")

    # --- reshape ---
    wide = to_wide(long_df)

    complete_mask = wide[SETTINGS].notna().all(axis=1)
    wide_complete = wide[complete_mask].copy()

    # --- write outputs ---
    p_long = OUT_DIR / "perplexity_long.csv"
    p_wide = OUT_DIR / "perplexity_wide_all.csv"
    p_complete = OUT_DIR / "perplexity_wide_complete.csv"
    p_cov = OUT_DIR / "coverage_report.csv"

    long_df.to_csv(p_long, index=False)
    wide.to_csv(p_wide, index=False)
    wide_complete.to_csv(p_complete, index=False)
    coverage.to_csv(p_cov, index=False)

    # --- summary ---
    print("\nWrote:")
    print(f"  {p_long}          ({len(long_df):>7,} rows, long/tidy form)")
    print(f"  {p_wide}      ({len(wide):>7,} prompts, 6 setting columns, NaN where missing)")
    print(f"  {p_complete} ({len(wide_complete):>7,} prompts with all 6 settings present)")
    print(f"  {p_cov}")

    print("\nPer-dataset complete-vector counts (usable as 6-output targets):")
    if len(wide_complete):
        print(wide_complete["dataset"].value_counts().to_string())
    else:
        print("  (none — no prompt has all 6 settings; check coverage above)")

    # Quick target sanity stats on the complete set.
    if len(wide_complete):
        print("\nPerplexity summary on complete-vector prompts:")
        desc = wide_complete[SETTINGS].describe().loc[["mean", "min", "max"]]
        print(desc.round(3).to_string())


if __name__ == "__main__":
    main()
