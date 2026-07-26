"""
kv_common.py  —  shared utilities for the perplexity-prediction notebooks.

Every model notebook imports from here so the data loading, the train/test
split, the target transform, and the metrics are IDENTICAL across models.
That is what makes the model comparison fair.

Design decisions baked in (discussed with the user):
  * Data is pulled directly from the public GitHub repo at runtime.
  * Target = the 6 perplexities, modeled in LOG space (log-perplexity),
    because raw perplexity spans ~1 to millions and the tail would otherwise
    dominate every error metric. Predictions are inverted back to raw
    perplexity for reporting.
  * Split = stratified random 70/30 BY DATASET, so each of the four datasets
    contributes 70% of its rows to train and 30% to test. random_state fixed
    for reproducibility.
  * Duplicate prompts are dropped before splitting to avoid train/test leakage.
"""

from __future__ import annotations

import io
import os
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The six compression settings, in fixed output-vector order.
SETTINGS = ["h2o_20", "h2o_40", "h2o_60",
            "kvquant_2bit", "kvquant_3bit", "kvquant_4bit"]

# Candidate URLs for the wide-complete table. The loader tries them in order.
# (Branch and space-encoding variants; falls back to a local path.)
DATA_URLS = [
    "https://raw.githubusercontent.com/yoshikodes/KVCacheCompression/main/perplexity_data/perplexity_wide_complete.csv",
    "https://raw.githubusercontent.com/yoshikodes/KVCacheCompression/master/perplexity_data/perplexity_wide_complete.csv",
]

# Local fallback candidates, tried in order. Covers running from the repo root
# OR from inside the notebooks/ subfolder (where ../ reaches the data folder).
LOCAL_FALLBACKS = [
    "perplexity_data/perplexity_wide_complete.csv",
    "../perplexity_data/perplexity_wide_complete.csv",
    "/content/KVCacheCompression/perplexity_data/perplexity_wide_complete.csv",
]
# Back-compat: single-string fallback still accepted by load_data().
LOCAL_FALLBACK = LOCAL_FALLBACKS[0]

RANDOM_STATE = 42
TEST_SIZE = 0.30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(urls=None, local_fallback=None, verbose=True) -> pd.DataFrame:
    """
    Fetch the wide-complete perplexity table from GitHub (or a local file).
    Returns a cleaned DataFrame with columns:
        dataset, prompt_index, prompt, <6 settings>
    Cleaning: drop rows with any missing target, drop duplicate prompts.

    Tries, in order: each URL in `urls` (default DATA_URLS), then each local
    path in LOCAL_FALLBACKS (plus `local_fallback` if you pass one). The local
    candidates cover running from the repo root OR the notebooks/ subfolder.
    """
    urls = urls or DATA_URLS
    text = None
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 0:
                text = r.text
                if verbose:
                    print(f"Loaded data from: {url}")
                break
            elif verbose:
                print(f"  (HTTP {r.status_code} from {url})")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  (failed {url}: {e})")

    if text is not None:
        df = pd.read_csv(io.StringIO(text))
    else:
        candidates = ([local_fallback] if local_fallback else []) + LOCAL_FALLBACKS
        found = next((p for p in candidates if p and os.path.exists(p)), None)
        if found:
            df = pd.read_csv(found)
            if verbose:
                print(f"Loaded data from local file: {found}")
        else:
            raise RuntimeError(
                "Could not load data from any GitHub URL or local fallback.\n"
                f"URLs tried: {urls}\n"
                f"Local paths tried: {candidates}\n"
                "Fix DATA_URLS in kv_common.py, or pass "
                "load_data(local_fallback='<path to perplexity_wide_complete.csv>')."
            )

    missing = [c for c in SETTINGS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing expected setting columns: {missing}")

    n0 = len(df)
    df = df.dropna(subset=SETTINGS)
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    if verbose:
        print(f"Rows: {n0} -> {len(df)} after dropping NaN targets and dup prompts.")
        print("Per-dataset counts:")
        print(df["dataset"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE,
                     verbose: bool = True):
    """
    Stratified random split by dataset: each dataset contributes (1-test_size)
    of its rows to train and test_size to test. Returns (train_df, test_df).
    """
    train_parts, test_parts = [], []
    for ds, g in df.groupby("dataset"):
        if len(g) < 2:
            # Too few rows to split; put them all in train.
            train_parts.append(g)
            continue
        tr, te = train_test_split(
            g, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_parts.append(tr)
        test_parts.append(te)
    train = pd.concat(train_parts).sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    if verbose:
        print(f"Train: {len(train)}  Test: {len(test)}")
        print("Train per-dataset:", train["dataset"].value_counts().to_dict())
        print("Test  per-dataset:", test["dataset"].value_counts().to_dict())
    return train, test


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def get_targets(df: pd.DataFrame, log_space: bool = True) -> np.ndarray:
    """Return the (n, 6) target matrix, in log space by default."""
    y = df[SETTINGS].values.astype(float)
    return np.log(y) if log_space else y


def invert_targets(y: np.ndarray, log_space: bool = True) -> np.ndarray:
    """Invert get_targets: log-perplexity -> raw perplexity."""
    return np.exp(y) if log_space else y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true_log: np.ndarray,
             y_pred_log: np.ndarray,
             settings=SETTINGS,
             report_raw: bool = True) -> pd.DataFrame:
    """
    Compute per-setting and overall metrics. Inputs are in LOG space.
    Returns a tidy DataFrame; also computes MAE in raw perplexity space,
    which is the interpretable number.
    """
    rows = []
    for i, s in enumerate(settings):
        yt, yp = y_true_log[:, i], y_pred_log[:, i]
        row = {
            "setting": s,
            "MAE_log": mean_absolute_error(yt, yp),
            "RMSE_log": np.sqrt(mean_squared_error(yt, yp)),
            "R2_log": r2_score(yt, yp),
            "Spearman": spearmanr(yt, yp).correlation,
        }
        if report_raw:
            row["MAE_raw"] = mean_absolute_error(np.exp(yt), np.exp(yp))
        rows.append(row)
    per = pd.DataFrame(rows)

    overall = {
        "setting": "OVERALL",
        "MAE_log": per["MAE_log"].mean(),
        "RMSE_log": per["RMSE_log"].mean(),
        "R2_log": per["R2_log"].mean(),
        "Spearman": per["Spearman"].mean(),
    }
    if report_raw:
        overall["MAE_raw"] = per["MAE_raw"].mean()
    per = pd.concat([per, pd.DataFrame([overall])], ignore_index=True)
    return per


def baseline_predict_mean(y_train_log: np.ndarray, n_test: int) -> np.ndarray:
    """Baseline: predict the per-setting train mean for every test row."""
    return np.tile(y_train_log.mean(axis=0), (n_test, 1))


def print_comparison(model_metrics: pd.DataFrame,
                     baseline_metrics: pd.DataFrame):
    """Print model vs. mean-baseline on the OVERALL row so you can see lift."""
    m = model_metrics[model_metrics.setting == "OVERALL"].iloc[0]
    b = baseline_metrics[baseline_metrics.setting == "OVERALL"].iloc[0]
    print("\n=== Model vs. mean-baseline (OVERALL, log space) ===")
    print(f"  MAE_log   model={m.MAE_log:.4f}   baseline={b.MAE_log:.4f}   "
          f"{'BEATS' if m.MAE_log < b.MAE_log else 'does NOT beat'} baseline")
    print(f"  R2_log    model={m.R2_log:.4f}   baseline={b.R2_log:.4f}")
