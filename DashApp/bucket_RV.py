"""
compute_bucket_rv.py
====================
Read full.parquet and compute realized volatility for each
120-second bucket per (stock_id, time_id).

Output: processed/bucket_rv.parquet
Columns: stock_id, time_id, rv_b0, rv_b1, rv_b2, rv_b3, rv_target

Buckets:
  rv_b0     : seconds   0-119
  rv_b1     : seconds 120-239
  rv_b2     : seconds 240-359
  rv_b3     : seconds 360-479
  rv_target : seconds 480-599

Run: python compute_bucket_rv.py
"""

import gc
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────

INPUT_PATH  = Path(r"C:\Users\Pc\OneDrive - The University of Sydney (Students)\Final\processed\full.parquet")
OUTPUT_PATH = Path("bucket_rv.parquet")
WORKERS     = cpu_count()-1

EPS = 1e-10
BUCKETS = [
    (0,   120, "rv_b0"),
    (120, 240, "rv_b1"),
    (240, 360, "rv_b2"),
    (360, 480, "rv_b3"),
    (480, 600, "rv_target"),
]


def log(msg):
    print(f"[BucketRV] {msg}", flush=True)


def parse_stock_id(s):
    s = str(s)
    return int(s.replace("stock_", "")) if "stock_" in s else int(s)


def compute_rv_for_stock(args):
    """
    Worker: receives (stock_id, df_stock).
    Returns DataFrame with rv_b0..rv_b3 + rv_target per time_id.
    """
    stock_id, df_stock = args

    df_stock = df_stock.sort_values(["time_id", "seconds_in_bucket"]).copy()
    df_stock["log_wap"] = np.log(df_stock["wap"].clip(lower=EPS))
    df_stock["log_ret"] = df_stock.groupby("time_id")["log_wap"].diff().fillna(0.0)
    df_stock["log_ret_sq"] = df_stock["log_ret"] ** 2

    # Assign bucket label
    df_stock["bucket"] = None
    for start, end, col in BUCKETS:
        mask = (df_stock["seconds_in_bucket"] >= start) & (df_stock["seconds_in_bucket"] < end)
        df_stock.loc[mask, "bucket"] = col
    df_stock = df_stock[df_stock["bucket"].notna()]

    # Vectorized: sum sq returns per (time_id, bucket), then sqrt
    grp = (
        df_stock.groupby(["time_id", "bucket"])["log_ret_sq"]
        .sum()
        .apply(np.sqrt)
        .reset_index()
        .rename(columns={"log_ret_sq": "rv"})
    )

    pivoted = grp.pivot(index="time_id", columns="bucket", values="rv").reset_index()
    pivoted.columns.name = None

    for _, _, col in BUCKETS:
        if col not in pivoted.columns:
            pivoted[col] = 0.0

    pivoted["stock_id"] = stock_id
    cols = ["stock_id", "time_id"] + [col for _, _, col in BUCKETS]
    return pivoted[cols]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"Loading {INPUT_PATH.name} ...")
    df = pd.read_parquet(INPUT_PATH, columns=["stock_id", "time_id", "seconds_in_bucket", "wap"])
    log(f"  {len(df):,} rows | {df['stock_id'].nunique()} stocks | {df['time_id'].nunique()} time_ids")

    df["stock_id"] = df["stock_id"].apply(parse_stock_id).astype(np.int32)

    stocks = sorted(df["stock_id"].unique())
    log(f"  Splitting into {len(stocks)} stock tasks → {WORKERS} workers ...")

    tasks = [(sid, df[df["stock_id"] == sid].copy()) for sid in stocks]
    del df; gc.collect()

    t1 = time.time()
    if WORKERS > 1:
        with Pool(processes=WORKERS) as pool:
            results = pool.map(compute_rv_for_stock, tasks, chunksize=4)
    else:
        results = [compute_rv_for_stock(t) for t in tasks]
    log(f"  Computed in {time.time()-t1:.1f}s")

    log("Concatenating and saving ...")
    out = pd.concat(results, ignore_index=True)
    del results; gc.collect()

    out = out.sort_values(["stock_id", "time_id"]).reset_index(drop=True)
    out["stock_id"] = out["stock_id"].astype(np.int32)
    out["time_id"]  = out["time_id"].astype(np.int32)
    for _, _, col in BUCKETS:
        out[col] = out[col].astype(np.float32)

    out.to_parquet(OUTPUT_PATH, index=False, compression="snappy")

    log(f"\nDone in {time.time()-t0:.1f}s")
    log(f"Shape: {out.shape}")
    log(f"Saved -> {OUTPUT_PATH}")
    log(f"\nSample:")
    print(out.head(10).to_string(index=False))
    log(f"\nrv_target stats:")
    print(out["rv_target"].describe().to_string())


if __name__ == "__main__":
    main()