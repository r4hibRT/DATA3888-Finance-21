

import gc
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────

DATA_DIR      = Path("C:\\Users\\ngdo0466\\Downloads\\processed")
N_FOLDS       = 1

INPUT_END     = 480
TARGET_START  = 480
TOTAL_SECONDS = 600
BUCKET_SIZE   = 60
BUCKET_COUNT  = INPUT_END // BUCKET_SIZE   # 8

# Per-bucket features:
#   0: log_return        (sum of log returns in bucket)
#   1: realized_vol      (sqrt of sum of squared log returns)
#   2: mean_wap          (mean WAP)
#   3: mean_spread       (mean bid-ask spread)
#   4: sum_volume        (total volume)
#   5: mean_price_spread (mean price spread)
#   6: mean_depth_imbal  (mean depth imbalance)
INPUT_DIM = 7

RV_FLOOR = 1e-4
EPS      = 1e-8


def log(msg):
    print(f"[PREPROCESS] {msg}", flush=True)


def parse_stock_id(s):
    s = str(s)
    return int(s.replace("stock_", "")) if "stock_" in s else int(s)


def preprocess_parquet(path: Path, out_path: Path):
    """
    Read raw per-second parquet, compute bucketed features and targets
    using vectorized pandas operations. Save as .npz.
    """
    t0 = _time.time()
    log(f"  Reading {path.name} ...")
    df = pd.read_parquet(path)

    # Parse stock_id
    uniq = df["stock_id"].unique()
    sid_map = {s: parse_stock_id(s) for s in uniq}
    df["stock_id"] = df["stock_id"].map(sid_map).astype(np.int32)
    num_stocks = int(df["stock_id"].max()) + 1

    # Sort and compute log returns
    df.sort_values(["stock_id", "time_id", "seconds_in_bucket"], inplace=True)
    df["log_wap"] = np.log(np.clip(df["wap"].values.astype(np.float32), EPS, None))
    df["log_ret"] = df.groupby(["stock_id", "time_id"])["log_wap"].diff().fillna(0).astype(np.float32)
    df["bucket"] = np.clip(
        df["seconds_in_bucket"].values // BUCKET_SIZE, 0, BUCKET_COUNT - 1
    ).astype(np.int8)
    df["log_ret_sq"] = (df["log_ret"] ** 2).astype(np.float32)

    time_ids = np.array(sorted(df["time_id"].unique()), dtype=np.int32)
    tid_to_idx = {int(t): i for i, t in enumerate(time_ids)}
    n_tids = len(time_ids)

    log(f"    {n_tids} time_ids, {num_stocks} stocks")

    # ── Input features (seconds 0-479) ────────────────────────────
    log(f"  Computing input bucket features ...")
    inp = df[df["seconds_in_bucket"] < INPUT_END].copy()

    # Ensure columns exist with defaults
    for col in ["bid_ask_spread", "total_volume", "price_spread", "depth_imbalance"]:
        if col not in inp.columns:
            inp[col] = 0.0

    # Vectorized groupby aggregation
    grp = inp.groupby(["time_id", "stock_id", "bucket"])
    agg = grp.agg(
        log_ret_sum   = ("log_ret", "sum"),
        log_ret_sq_sum = ("log_ret_sq", "sum"),
        wap_mean       = ("wap", "mean"),
        spread_mean    = ("bid_ask_spread", "mean"),
        volume_sum     = ("total_volume", "sum"),
        pspread_mean   = ("price_spread", "mean"),
        dimbal_mean    = ("depth_imbalance", "mean"),
    ).reset_index()

    del inp; gc.collect()

    # Map to indices
    agg["tid_idx"] = agg["time_id"].map(tid_to_idx).astype(np.int32)
    agg["sid"] = agg["stock_id"].values.astype(np.int32)
    agg["bkt"] = agg["bucket"].values.astype(np.int8)

    # Allocate output array
    X = np.zeros((n_tids, num_stocks, BUCKET_COUNT, INPUT_DIM), dtype=np.float32)

    # Fill vectorized — extract numpy arrays for speed
    ti = agg["tid_idx"].values
    si = agg["sid"].values
    bi = agg["bkt"].values

    X[ti, si, bi, 0] = agg["log_ret_sum"].values.astype(np.float32)
    X[ti, si, bi, 1] = np.sqrt(np.clip(agg["log_ret_sq_sum"].values, 0, None)).astype(np.float32)
    X[ti, si, bi, 2] = agg["wap_mean"].values.astype(np.float32)
    X[ti, si, bi, 3] = agg["spread_mean"].values.astype(np.float32)
    X[ti, si, bi, 4] = agg["volume_sum"].values.astype(np.float32)
    X[ti, si, bi, 5] = agg["pspread_mean"].values.astype(np.float32)
    X[ti, si, bi, 6] = agg["dimbal_mean"].values.astype(np.float32)

    del agg; gc.collect()
    log(f"    X shape: {X.shape}")

    # ── Target (seconds 480-599) ──────────────────────────────────
    log(f"  Computing targets ...")
    tgt = df[df["seconds_in_bucket"] >= TARGET_START].copy()

    tgt_agg = (
        tgt.groupby(["time_id", "stock_id"])["log_ret_sq"]
        .sum()
        .reset_index(name="rv_sq_sum")
    )
    tgt_agg["rv"] = np.sqrt(np.clip(tgt_agg["rv_sq_sum"].values, 0, None))
    tgt_agg["log_rv"] = np.log(np.clip(tgt_agg["rv"].values, RV_FLOOR, None)).astype(np.float32)
    tgt_agg["tid_idx"] = tgt_agg["time_id"].map(tid_to_idx).astype(np.int32)

    del tgt; gc.collect()

    y = np.zeros((n_tids, num_stocks), dtype=np.float32)
    y[tgt_agg["tid_idx"].values, tgt_agg["stock_id"].values.astype(np.int32)] = tgt_agg["log_rv"].values

    del tgt_agg, df; gc.collect()
    log(f"    y shape: {y.shape}")

    # ── Save ──────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        time_ids=time_ids,
        num_stocks=np.array(num_stocks, dtype=np.int32),
    )
    elapsed = _time.time() - t0
    log(f"    Saved -> {out_path}  ({X.nbytes / 1e6:.0f} MB X + {y.nbytes / 1e6:.0f} MB y)  [{elapsed:.1f}s]")


def main():
    log("=" * 60)
    log("GNN Preprocessing — Bucket Features + Targets")
    log(f"  Input window  : 0-{INPUT_END-1}s -> {BUCKET_COUNT} buckets x {INPUT_DIM} features")
    log(f"  Target window : {TARGET_START}-{TOTAL_SECONDS-1}s -> log(RV)")
    log(f"  Bucket size   : {BUCKET_SIZE}s")
    log(f"  Folds         : {N_FOLDS}")
    log("=" * 60)

    for fold in range(N_FOLDS):
        fold_dir = DATA_DIR / f"fold_{fold}"
        if not fold_dir.exists():
            log(f"\n  {fold_dir} not found -- skipping")
            continue

        log(f"\n{'─' * 40}")
        log(f"  FOLD {fold}")
        log(f"{'─' * 40}")

        for split in ["train", "test"]:
            parquet_path = fold_dir / f"{split}.parquet"
            npz_path = fold_dir / f"{split}_gnn.npz"

            if not parquet_path.exists():
                log(f"  {parquet_path} not found -- skipping")
                continue

            if npz_path.exists():
                log(f"  {npz_path.name} already exists -- skipping (delete to recompute)")
                continue

            preprocess_parquet(parquet_path, npz_path)
            gc.collect()

    log("\nDone.")
    log("Output structure:")
    log("  processed/fold_i/train_gnn.npz  — X(n_tids, n_stocks, 4, 7) + y(n_tids, n_stocks)")
    log("  processed/fold_i/test_gnn.npz")
    log("\nUpdate gnn_rv.py to load .npz instead of calling build_dataset_from_raw():")
    log("  data = np.load(fold_dir / 'train_gnn.npz')")
    log("  X, y, time_ids = data['X'], data['y'], data['time_ids']")


if __name__ == "__main__":
    main()