"""
preprocess.py
=============
Preprocessing pipeline for Spatio-Temporal GNN on cross-stock order-book data.

Reads the split parquet files produced by split_and_save.py:
    processed/train.parquet
    processed/test.parquet

Bucketing strategy (Option A):
    Input window  : seconds 0–299  → 3 buckets × 100s
    Target window : seconds 300–599 → log(RV) of the full 300s

Per-bucket features (4 per bucket → INPUT_DIM = 12 total):
    1. log_return    – net price move within bucket  (wap[-1] / wap[0])
    2. realized_vol  – sqrt(Σ r²) of second-level log returns in bucket
    3. mean_spread   – average bid-ask spread (liquidity proxy)
    4. total_volume  – total traded volume (activity proxy)

Output shape per sample:
    features : (stock_id, time_id, bucket_0..2 × 4 features)  →  12 columns
    targets  : (stock_id, time_id, log_rv_target)

Splits:
    Inherits train/test from the parquet files.
    Carves a validation set from train time_ids (10% of train).

Design notes:
    - 100s buckets make zero-RV far less likely than per-second or 10s buckets
    - A stock needs ZERO trades for a full 100s to produce rv=0 in a bucket
    - RV_FLOOR applied only to the TARGET, not to input realized_vol
      (zero input RV is meaningful signal — the stock was flat)
    - No normalisation applied — handled by the model's LayerNorm
    - Stocks processed in parallel via multiprocessing Pool
"""

from __future__ import annotations

import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────── Config ──

INPUT_DIR  = Path("processed")           # output of split_and_save.py
OUTPUT_DIR = Path("processed_v2")

BUCKET_SIZE   = 100    # seconds per bucket
N_BUCKETS     = 3      # 3 × 100s = 300s input window
INPUT_SECONDS = BUCKET_SIZE * N_BUCKETS  # 300
TARGET_START  = 300    # seconds 300–599 used for target RV

VAL_RATIO   = 0.1      # fraction of train time_ids held out for validation
RANDOM_SEED = 42
RV_FLOOR    = 1e-8     # floor for TARGET log(RV) only
N_WORKERS   = 8        # parallel workers (tune to your CPU)

# Feature names in order — 4 per bucket, 3 buckets = 12 total
BUCKET_FEATURE_NAMES = ["log_return", "realized_vol", "mean_spread", "total_volume"]

FEATURE_COLS = [
    f"{feat}_b{b}"
    for b in range(N_BUCKETS)
    for feat in BUCKET_FEATURE_NAMES
]   # ['log_return_b0', 'realized_vol_b0', ..., 'total_volume_b2']

INPUT_DIM = len(FEATURE_COLS)   # 12


# ─────────────────────────────────────────── Bucket maths ──

def second_level_log_returns(wap: np.ndarray) -> np.ndarray:
    """Compute log returns from WAP array. Returns empty array if < 2 points."""
    if len(wap) < 2:
        return np.array([], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(wap[1:] / wap[:-1])
    return np.where(np.isfinite(lr), lr, 0.0)


def bucket_features(bucket_df: pd.DataFrame) -> dict[str, float]:
    """
    Extract 4 features from a single 100-second bucket.

    Returns zeros for empty/dead buckets — zero realized_vol is a valid
    signal (the stock was completely flat during this bucket).
    """
    if bucket_df.empty:
        return {f: 0.0 for f in BUCKET_FEATURE_NAMES}

    wap    = bucket_df["wap"].to_numpy(dtype=np.float64)
    spread = bucket_df["bid_ask_spread"].to_numpy(dtype=np.float64)
    vol    = bucket_df["total_volume"].to_numpy(dtype=np.float64)

    lr = second_level_log_returns(wap)

    # 1. log_return — net price move across bucket
    if len(wap) >= 2 and wap[0] > 0 and wap[-1] > 0:
        log_return = float(np.log(wap[-1] / wap[0]))
    else:
        log_return = 0.0

    # 2. realized_vol — sqrt(Σ r²) — zero is valid, do NOT apply floor here
    realized_vol = float(np.sqrt(np.sum(lr ** 2))) if len(lr) > 0 else 0.0

    # 3. mean_spread — average bid-ask spread
    valid_spread = spread[np.isfinite(spread) & (spread > 0)]
    mean_spread  = float(np.mean(valid_spread)) if len(valid_spread) > 0 else 0.0

    # 4. total_volume — total traded volume in bucket
    total_volume = float(np.nansum(vol))

    return {
        "log_return"  : log_return,
        "realized_vol": realized_vol,
        "mean_spread" : mean_spread,
        "total_volume": total_volume,
    }


def extract_bucketed_features(input_df: pd.DataFrame) -> dict[str, float]:
    """
    Split the 300s input window into N_BUCKETS x BUCKET_SIZE second buckets.
    Returns a flat dict of {feat_bN: value} for all buckets and features.
    """
    result: dict[str, float] = {}

    for b in range(N_BUCKETS):
        start = b * BUCKET_SIZE       # 0, 100, 200
        end   = start + BUCKET_SIZE   # 100, 200, 300

        bucket_df = input_df[
            (input_df["seconds_in_bucket"] >= start) &
            (input_df["seconds_in_bucket"] <  end)
        ]

        feats = bucket_features(bucket_df)
        for feat_name, val in feats.items():
            result[f"{feat_name}_b{b}"] = val

    return result


def compute_target_log_rv(target_df: pd.DataFrame) -> float | None:
    """
    Compute log(RV) over the target window (seconds 300–599).
    Returns None if no target data is available.
    """
    if target_df.empty:
        return None

    wap = target_df["wap"].to_numpy(dtype=np.float64)
    lr  = second_level_log_returns(wap)
    rv  = float(np.sqrt(np.sum(lr ** 2))) if len(lr) > 0 else 0.0

    # Apply RV_FLOOR only to target — prevents log(0) = -inf
    return float(np.log(max(rv, RV_FLOOR)))


# ─────────────────────────── Per-stock worker function ──

def process_stock(args: tuple) -> pd.DataFrame | None:
    """
    Worker function — processes all time_ids for a single stock.

    Args:
        args: (stock_id, stock_df) where stock_df is the full data for that stock

    Returns:
        DataFrame with columns: time_id, stock_id, *FEATURE_COLS, log_rv_target
        or None if no valid rows.
    """
    stock_id, stock_df = args

    stock_df = stock_df.sort_values("seconds_in_bucket").reset_index(drop=True)

    rows    = []
    skipped = 0

    for tid, tid_df in stock_df.groupby("time_id"):
        tid_df = tid_df.sort_values("seconds_in_bucket").reset_index(drop=True)

        # Split into input (0–299s) and target (300–599s)
        input_df  = tid_df[tid_df["seconds_in_bucket"] <  INPUT_SECONDS]
        target_df = tid_df[tid_df["seconds_in_bucket"] >= TARGET_START]

        # Skip if no input data at all
        if input_df.empty:
            skipped += 1
            continue

        # Skip if no target data (can't train without a label)
        log_rv_target = compute_target_log_rv(target_df)
        if log_rv_target is None:
            skipped += 1
            continue

        feats = extract_bucketed_features(input_df)

        row = {
            "time_id"      : int(tid),
            "stock_id"     : int(stock_id),
            **feats,
            "log_rv_target": log_rv_target,
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


# ──────────────────────────────────────── Main pipeline ──

def process_split(raw_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Process one split (train/val/test) through bucketing in parallel.
    Groups by stock_id, dispatches each stock to a worker, collects results.
    """
    n_stocks  = raw_df["stock_id"].nunique()
    n_timeids = raw_df["time_id"].nunique()
    print(f"[{split_name}] Processing {n_stocks} stocks x {n_timeids} time_ids ...")

    stock_groups = [
        (sid, grp.copy())
        for sid, grp in raw_df.groupby("stock_id")
    ]

    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_stock, stock_groups)

    valid = [r for r in results if r is not None and not r.empty]
    if not valid:
        raise RuntimeError(f"[{split_name}] No data processed successfully.")

    combined = pd.concat(valid, ignore_index=True)
    print(
        f"[{split_name}] Done: {len(combined):,} rows | "
        f"{combined['stock_id'].nunique()} stocks | "
        f"{combined['time_id'].nunique()} time_ids"
    )
    return combined


def save_split(df: pd.DataFrame, split_name: str) -> None:
    """Save features and targets as separate parquet files."""
    id_cols = ["time_id", "stock_id"]

    feat_path = OUTPUT_DIR / f"{split_name}_features.parquet"
    targ_path = OUTPUT_DIR / f"{split_name}_targets.parquet"

    df[id_cols + FEATURE_COLS].to_parquet(
        feat_path, engine="pyarrow", index=False, compression="snappy"
    )
    df[id_cols + ["log_rv_target"]].to_parquet(
        targ_path, engine="pyarrow", index=False, compression="snappy"
    )

    print(f"  -> {feat_path.name}  ({len(df):,} rows, {len(FEATURE_COLS)} feature cols)")
    print(f"  -> {targ_path.name}")


def print_target_stats(df: pd.DataFrame, split_name: str) -> None:
    """Print target RV distribution stats and warn if too many floor values."""
    tgt = df["log_rv_target"]
    print(
        f"[{split_name}] log_rv_target stats: "
        f"min={tgt.min():.3f}  p5={tgt.quantile(0.05):.3f}  "
        f"median={tgt.median():.3f}  p95={tgt.quantile(0.95):.3f}  "
        f"max={tgt.max():.3f}"
    )
    floor_pct = (tgt <= np.log(RV_FLOOR) + 0.01).mean() * 100
    if floor_pct > 5:
        print(
            f"  WARNING [{split_name}]: {floor_pct:.1f}% of targets are at RV_FLOOR — "
            f"consider raising BUCKET_SIZE or RV_FLOOR"
        )


def main() -> None:
    print("=" * 60)
    print("Preprocess v2 — 100s bucket aggregation")
    print(f"  Buckets     : {N_BUCKETS} x {BUCKET_SIZE}s = {INPUT_SECONDS}s input window")
    print(f"  Target      : seconds {TARGET_START}-599 -> log(RV)")
    print(f"  Features    : {BUCKET_FEATURE_NAMES} x {N_BUCKETS} buckets = {INPUT_DIM} dims")
    print(f"  Workers     : {N_WORKERS}")
    print(f"  Input dir   : {INPUT_DIR}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print("=" * 60)

    # ── Step 1: Load raw parquet splits ────────────────────────────
    train_path = INPUT_DIR / "train.parquet"
    test_path  = INPUT_DIR / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path}")

    print("Loading parquet files ...")
    train_raw = pd.read_parquet(train_path, engine="fastparquet")
    test_raw  = pd.read_parquet(test_path,  engine="fastparquet")

    # Ensure stock_id is integer.
    # split_and_save.py may store it as 'stock_0', 'stock_1', ... or plain int.
    def parse_stock_id(s):
        s = str(s)
        return int(s.replace("stock_", "")) if "stock_" in s else int(s)

    train_raw["stock_id"] = train_raw["stock_id"].apply(parse_stock_id)
    test_raw["stock_id"]  = test_raw["stock_id"].apply(parse_stock_id)

    print(
        f"Train raw : {len(train_raw):,} rows | "
        f"{train_raw['stock_id'].nunique()} stocks | "
        f"{train_raw['time_id'].nunique()} time_ids"
    )
    print(
        f"Test  raw : {len(test_raw):,} rows | "
        f"{test_raw['stock_id'].nunique()} stocks | "
        f"{test_raw['time_id'].nunique()} time_ids"
    )

    # ── Step 2: Carve validation set from train time_ids ───────────
    all_train_tids = train_raw["time_id"].unique()
    rng            = np.random.default_rng(RANDOM_SEED)
    shuffled_tids  = rng.permutation(all_train_tids)

    n_val      = max(1, int(len(shuffled_tids) * VAL_RATIO))
    val_tids   = set(shuffled_tids[:n_val].tolist())
    train_tids = set(shuffled_tids[n_val:].tolist())

    print(f"Train/val split: {len(train_tids)} train | {len(val_tids)} val time_ids")

    train_sub = train_raw[train_raw["time_id"].isin(train_tids)].copy()
    val_sub   = train_raw[train_raw["time_id"].isin(val_tids)].copy()

    del train_raw   # free memory

    # ── Step 3: Process each split ─────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, raw_df in [
        ("train", train_sub),
        ("val",   val_sub),
        ("test",  test_raw),
    ]:
        processed = process_split(raw_df, split_name)
        print_target_stats(processed, split_name)
        save_split(processed, split_name)

    # ── Step 4: Save metadata ───────────────────────────────────────
    meta = {
        "bucket_size_s" : BUCKET_SIZE,
        "n_buckets"     : N_BUCKETS,
        "input_seconds" : INPUT_SECONDS,
        "target_start_s": TARGET_START,
        "input_dim"     : INPUT_DIM,
        "feature_cols"  : FEATURE_COLS,
        "rv_floor"      : RV_FLOOR,
        "val_ratio"     : VAL_RATIO,
        "random_seed"   : RANDOM_SEED,
    }
    meta_path = OUTPUT_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved -> {meta_path}")
    print("All splits processed. Done.")


if __name__ == "__main__":
    main()