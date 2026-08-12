#!/usr/bin/env python3
"""
remove_prompt_context.py

Undoes add_prompt_context.py: strips the GSM8K few-shot prefix and the
ARC-Challenge/HellaSwag "Answer choices:" block back out of the `prompt`
column in ./Data/*_per_prompt.csv, restoring the original text byte-for-byte.

It works by string surgery using the same constants add_prompt_context.py
used to add the text (imported from that file, so the two scripts can never
drift out of sync) -- it does NOT rely on the ./Data/_original_backup/ copies,
so it works even if you've since made other edits to the augmented CSVs that
you want to keep.

Rows that were never augmented (no marker found) are left untouched.

Usage:
    python remove_prompt_context.py             # modify ./Data in place
    python remove_prompt_context.py --dry-run    # report what would change, touch nothing
    python remove_prompt_context.py --data-dir path/to/Data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from add_prompt_context import (
    GSM8K_FEWSHOT_PREFIX,
    GSM8K_Q_LEAD,
    GSM8K_A_TAIL,
    CHOICES_BLOCK_HEADER,
    GSM8K_SUFFIX,
    ARC_SUFFIX,
    HELLASWAG_SUFFIX,
    DEFAULT_DATA_DIR,
)

GSM8K_FULL_LEAD = GSM8K_FEWSHOT_PREFIX + GSM8K_Q_LEAD


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def strip_gsm8k(prompt: str) -> tuple[str, bool]:
    """Returns (possibly-restored prompt, whether it changed)."""
    if not prompt.startswith(GSM8K_FULL_LEAD) or not prompt.endswith(GSM8K_A_TAIL):
        return prompt, False
    middle = prompt[len(GSM8K_FULL_LEAD):-len(GSM8K_A_TAIL)]
    return middle, True


def strip_choices(prompt: str) -> tuple[str, bool]:
    idx = prompt.find(CHOICES_BLOCK_HEADER)
    if idx == -1:
        return prompt, False
    return prompt[:idx], True


# ----------------------------------------------------------------------------
# Per-file processing
# ----------------------------------------------------------------------------

def process_gsm8k(path: Path, dry_run: bool) -> None:
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        print(f"  ! {path.name}: no 'prompt' column; skipping")
        return

    n_changed = 0
    new_prompts = df["prompt"].astype(str).copy()
    for i in df.index:
        restored, changed = strip_gsm8k(df.at[i, "prompt"])
        if changed:
            new_prompts.at[i] = restored
            n_changed += 1

    if n_changed == 0:
        print(f"  = {path.name}: no augmented rows found; skipping")
        return

    if dry_run:
        print(f"  + {path.name}: would restore {n_changed} rows")
        return

    df["prompt"] = new_prompts
    df.to_csv(path, index=False)
    print(f"  + {path.name}: restored {n_changed} rows")


def process_mc(path: Path, dry_run: bool) -> None:
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        print(f"  ! {path.name}: no 'prompt' column; skipping")
        return

    n_changed = 0
    new_prompts = df["prompt"].astype(str).copy()
    for i in df.index:
        restored, changed = strip_choices(df.at[i, "prompt"])
        if changed:
            new_prompts.at[i] = restored
            n_changed += 1

    if n_changed == 0:
        print(f"  = {path.name}: no augmented rows found; skipping")
        return

    if dry_run:
        print(f"  + {path.name}: would restore {n_changed} rows")
        return

    df["prompt"] = new_prompts
    df.to_csv(path, index=False)
    print(f"  + {path.name}: restored {n_changed} rows")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                     help="Path to the Data/ folder (default: ./Data next to this script)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report what would change without writing anything")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data directory not found: {data_dir}")

    csvs = sorted(p for p in data_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"ERROR: no CSV files found in {data_dir}")

    print(f"Scanning {len(csvs)} CSV files in {data_dir} "
          f"{'(dry run, no files will be modified)' if args.dry_run else ''}...")

    for path in csvs:
        name = path.name.lower()
        if name.endswith(GSM8K_SUFFIX):
            process_gsm8k(path, args.dry_run)
        elif name.endswith(ARC_SUFFIX) or name.endswith(HELLASWAG_SUFFIX):
            process_mc(path, args.dry_run)
        else:
            print(f"  - {path.name}: not gsm8k/arc_challenge/hellaswag; skipping")

    print("Done.")


if __name__ == "__main__":
    main()
