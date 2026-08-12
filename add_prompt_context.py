#!/usr/bin/env python3
"""
add_prompt_context.py

The `prompt` column in ./Data/*_per_prompt.csv is currently missing context
that was actually part of what got fed to the model at inference time (or is
otherwise necessary to make sense of the row on its own):

  * GSM8K       -- `prompt` is just the raw question. The real model input
                   (see generate_gsm8k_kvquant() in the Implementations
                   notebooks) was GSM8K_FEWSHOT_PREFIX + "\nQuestion: <q>\nAnswer:".
                   The 8-shot prefix is reproduced verbatim below (identical
                   across every notebook that produced a file in ./Data).
  * ARC-Challenge / HellaSwag -- `prompt` is just the question/context. The
                   answer choices live in their own `choices` column but
                   aren't merged into the prompt text, so anything that reads
                   only `prompt` (e.g. the text-feature classifier notebook)
                   never sees the options. This script appends them as a
                   clearly delimited "Answer choices:" block, using each
                   choice's own label exactly as stored in `choices`
                   (ARC labels are a mix of "A".."D" and "1".."4" in the
                   source dataset; HellaSwag has no native label so this
                   script assigns A, B, C, ... in list order).
  * WikiText-103 -- left untouched (chunks are raw corpus text; there's no
                   missing context to add).

This script is idempotent: re-running it on already-augmented files is a
no-op (detected via markers -- see remove_prompt_context.py for the same
markers used in reverse). A pristine copy of every file it touches is saved
once to ./Data/_original_backup/ before the first modification, as a safety
net independent of the undo script.

Usage:
    python add_prompt_context.py             # modify ./Data in place
    python add_prompt_context.py --dry-run    # report what would change, touch nothing
    python add_prompt_context.py --data-dir path/to/Data
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Constants (must stay byte-identical to remove_prompt_context.py)
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "Data"
BACKUP_DIRNAME = "_original_backup"

# Verbatim from the Implementations notebooks (GSM8K_FEWSHOT_PREFIX), confirmed
# byte-identical across every notebook that produced a file currently in ./Data.
GSM8K_FEWSHOT_PREFIX = (
    "You are solving grade-school math word problems.\n"
    "Show the calculation step by step, then end with exactly this format:\n"
    "#### <final number>\n\n"

    "Question: There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
    "Answer: There are 15 trees originally. After planting, there are 21 trees. So the workers planted 21 - 15 = 6 trees.\n"
    "#### 6\n\n"

    "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
    "Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5 cars.\n"
    "#### 5\n\n"

    "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
    "Answer: Leah and her sister started with 32 + 42 = 74 chocolates. After eating 35, they have 74 - 35 = 39 left.\n"
    "#### 39\n\n"

    "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
    "Answer: Jason started with 20 lollipops and now has 12. So he gave away 20 - 12 = 8 lollipops.\n"
    "#### 8\n\n"

    "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
    "Answer: Shawn started with 5 toys. He got 2 from mom and 2 from dad, which is 2 + 2 = 4 more toys. 5 + 4 = 9 toys total.\n"
    "#### 9\n\n"

    "Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?\n"
    "Answer: 4 days from Monday to Thursday, with 5 computers installed each day, is 4 * 5 = 20 computers added. 9 + 20 = 29 computers total.\n"
    "#### 29\n\n"

    "Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?\n"
    "Answer: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more on Wednesday, he had 35 - 2 = 33 golf balls.\n"
    "#### 33\n\n"

    "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
    "Answer: Five bagels at $3 each cost 5 * 3 = 15 dollars. Olivia started with $23, so she has 23 - 15 = 8 dollars left.\n"
    "#### 8\n"
)

# Text inserted right before the question, and right after it, when building
# the augmented GSM8K prompt: GSM8K_FEWSHOT_PREFIX + GSM8K_Q_LEAD + q + GSM8K_A_TAIL
# Note: unlike generate_gsm8k_kvquant() this does NOT .strip() the question,
# so the transform is exactly invertible (remove_prompt_context.py can recover
# the original byte-for-byte).
GSM8K_Q_LEAD = "\nQuestion: "
GSM8K_A_TAIL = "\nAnswer:"

# Marker that opens the appended answer-choices block for ARC/HellaSwag.
# Used both to detect "already augmented" and, by the undo script, as the
# cut point to restore the original prompt.
CHOICES_BLOCK_HEADER = "\n\nAnswer choices:\n"

# Filenames ending in these suffixes get the corresponding treatment.
GSM8K_SUFFIX = "_gsm8k_per_prompt.csv"
ARC_SUFFIX = "_arc_challenge_per_prompt.csv"
HELLASWAG_SUFFIX = "_hellaswag_per_prompt.csv"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def format_choices_block(choices) -> str:
    """
    Build the 'Answer choices:' block from a parsed `choices` cell.

    Accepts either:
      * a list of (label, text) tuples/lists  (ARC-Challenge)
      * a plain list of strings                (HellaSwag; labels A, B, C...
        are assigned in list order since HellaSwag has no native label)
    """
    lines = []
    for i, entry in enumerate(choices):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            label, text = entry
        else:
            label, text = chr(ord("A") + i), entry
        lines.append(f"{label}. {text}")
    return CHOICES_BLOCK_HEADER + "\n".join(lines)


def augment_gsm8k(prompt: str) -> str:
    return GSM8K_FEWSHOT_PREFIX + GSM8K_Q_LEAD + prompt + GSM8K_A_TAIL


def augment_mc(prompt: str, choices_cell: str) -> str | None:
    """Returns the augmented prompt, or None if `choices_cell` can't be parsed."""
    try:
        choices = ast.literal_eval(choices_cell)
    except (ValueError, SyntaxError):
        return None
    return prompt + format_choices_block(choices)


def backup_once(path: Path, data_dir: Path) -> None:
    backup_dir = data_dir / BACKUP_DIRNAME
    backup_dir.mkdir(exist_ok=True)
    target = backup_dir / path.name
    if not target.exists():
        shutil.copy2(path, target)


# ----------------------------------------------------------------------------
# Per-file processing
# ----------------------------------------------------------------------------

def process_gsm8k(path: Path, data_dir: Path, dry_run: bool) -> None:
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        print(f"  ! {path.name}: no 'prompt' column; skipping")
        return

    already = df["prompt"].astype(str).str.startswith(GSM8K_FEWSHOT_PREFIX)
    n_already = int(already.sum())
    to_do = ~already

    if not to_do.any():
        print(f"  = {path.name}: all {len(df)} rows already augmented; skipping")
        return

    if n_already:
        print(f"  ~ {path.name}: {n_already} rows already augmented, "
              f"{int(to_do.sum())} to augment")

    if dry_run:
        print(f"  + {path.name}: would augment {int(to_do.sum())} rows")
        return

    backup_once(path, data_dir)
    df.loc[to_do, "prompt"] = df.loc[to_do, "prompt"].astype(str).map(augment_gsm8k)
    df.to_csv(path, index=False)
    print(f"  + {path.name}: augmented {int(to_do.sum())} rows")


def process_mc(path: Path, data_dir: Path, dry_run: bool) -> None:
    df = pd.read_csv(path)
    if "prompt" not in df.columns or "choices" not in df.columns:
        print(f"  ! {path.name}: missing 'prompt' or 'choices' column; skipping")
        return

    prompt_str = df["prompt"].astype(str)
    already = prompt_str.str.contains(CHOICES_BLOCK_HEADER, regex=False)
    n_already = int(already.sum())
    to_do = ~already

    if not to_do.any():
        print(f"  = {path.name}: all {len(df)} rows already augmented; skipping")
        return

    if n_already:
        print(f"  ~ {path.name}: {n_already} rows already augmented, "
              f"{int(to_do.sum())} to augment")

    if dry_run:
        print(f"  + {path.name}: would augment {int(to_do.sum())} rows")
        return

    backup_once(path, data_dir)
    n_failed = 0
    new_prompts = df["prompt"].astype(str).copy()
    for i in df.index[to_do]:
        result = augment_mc(df.at[i, "prompt"], df.at[i, "choices"])
        if result is None:
            n_failed += 1
            continue
        new_prompts.at[i] = result
    df["prompt"] = new_prompts
    df.to_csv(path, index=False)

    msg = f"  + {path.name}: augmented {int(to_do.sum()) - n_failed} rows"
    if n_failed:
        msg += f" ({n_failed} rows had unparseable 'choices' and were left as-is)"
    print(msg)


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
            process_gsm8k(path, data_dir, args.dry_run)
        elif name.endswith(ARC_SUFFIX) or name.endswith(HELLASWAG_SUFFIX):
            process_mc(path, data_dir, args.dry_run)
        else:
            print(f"  - {path.name}: not gsm8k/arc_challenge/hellaswag; skipping")

    if not args.dry_run:
        print(f"\nOriginal copies preserved in {data_dir / BACKUP_DIRNAME}/ "
              "(only ever written once per file).")
    print("Done.")


if __name__ == "__main__":
    main()
