"""
feature_engineering.py
======================
Memory-efficient, multi-core feature engineering for Optiver Realized Volatility.

CPU parallelism strategy (same peak RAM):
  - Per-stock processing via multiprocessing.Pool (fork shares read-only
    parent DataFrame at zero copy cost on Linux; each worker returns only
    the tiny aggregated chunk)
  - NumPy vectorised ops replace groupby().transform() lambdas where
    possible (releases GIL → implicit thread parallelism via BLAS/MKL)
  - Mutual information via joblib n_jobs=-1 (sklearn built-in)
  - KNN queries via n_jobs=-1 (sklearn built-in)
  - Correlation matrix computed in chunked float32 matmul (BLAS threads)

Pipeline:
  processed/{train,test}.parquet → this script → features/feature_store_{train,test}.parquet

Feature groups:
  A. Rolling (30s/60s/120s causal windows)
  B. Cumulative (expanding sum/mean/max/std, price drift)
  C. Interaction combos (sums, ratios, diffs, products — at agg level)
  D. Clustering (KMeans on volatility/liquidity profiles, train-fit)
  E. KNN neighbour stats (train-fit NN index)
  F. Value counts / rarity encodings (train-fit bin maps)
  G. Target encoding (5-fold OOF on train, global mean on test)
  H. 120s bucket features (per-bucket aggs + cross-bucket deltas)
  I. Feature selection & noise removal (MI + correlation filter)

Data structure notes:
  - time_id is NOT sequential (arbitrary IDs, no ordering implied)
  - time_id is consistent across stocks (same market moment)
  - Not every stock has every time_id (sparse stock×time grid)
  - stock_id is NOT sequential

Leakage prevention:
  - Target from seconds 480–599 only; all features from 0–479 only
  - Rolling/cumulative strictly causal (backward-looking)
  - Clustering, KNN, value-count maps, normalisation fit on train only
  - Target encoding: 5-fold OOF on train; train global mean on test

Target:
  log_rv_diff = log(RV_480–599) - log(RV_360–479)
  Predicts the *change* in volatility relative to the nearest input bucket.
  At inference: log_rv = log_rv_diff_pred + past_log_rv_bkt3
"""

import gc
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

DATA_DIR   = Path("processed")
OUTPUT_DIR = Path("features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_END     = 480
TARGET_START  = 480
TOTAL_SECONDS = 600
BUCKET_SIZE   = 120
N_BUCKETS     = INPUT_END // BUCKET_SIZE   # 4

EPS      = 1e-8
RV_FLOOR = 1e-4

N_WORKERS = max(1, cpu_count())

ROLL_COLS    = ["wap", "log_wap", "bid_ask_spread", "total_volume",
                "log_volume", "price_spread", "volume_imbalance"]
ROLL_WINDOWS = [30, 60, 120]

BUCKET_AGG_COLS = ["wap", "log_wap", "bid_ask_spread", "total_volume",
                   "log_volume", "price_spread", "volume_imbalance"]

INTERACT_BASE = ["wap", "bid_ask_spread", "total_volume",
                 "price_spread", "volume_imbalance", "log_volume"]


def log(msg: str) -> None:
    print(f"[FE] {msg}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────

def parse_stock_id(s) -> int:
    s = str(s)
    return int(s.replace("stock_", "")) if "stock_" in s else int(s)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast all float64 → float32 and int64 → int32."""
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == np.int64:
            df[c] = df[c].astype(np.int32)
    return df


# ══════════════════════════════════════════════════════════════════════
#  STEP 0: Load, split target, restrict to input window
# ══════════════════════════════════════════════════════════════════════

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["stock_id"] = df["stock_id"].apply(parse_stock_id)
    df.sort_values(["stock_id", "time_id", "seconds_in_bucket"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    wap = df["wap"].values
    df["log_wap"]    = np.log(np.clip(wap, 1e-10, None)).astype(np.float32)
    df["log_spread"] = np.log(np.clip(df["bid_ask_spread"].values, 1e-10, None) + 1).astype(np.float32)
    df["log_volume"] = np.log1p(np.clip(df["total_volume"].values, 0, None)).astype(np.float32)
    df["bucket_120"] = np.clip(df["seconds_in_bucket"].values // BUCKET_SIZE,
                               0, N_BUCKETS - 1).astype(np.int8)

    return downcast(df)


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target: log(RV_480–599) - log(RV_360–479)

    log-difference between the target window and the nearest input bucket
    (bucket 3: seconds 360–479). Forces the model to predict *change* in
    volatility relative to the most recent signal rather than the level.

    At inference: log_rv = log_rv_diff_pred + past_log_rv_bkt3
    """
    # ── Target RV (seconds 480–599) ───────────────────────────────
    tgt = df[df["seconds_in_bucket"] >= TARGET_START].copy()
    tgt["lr"] = tgt.groupby(["stock_id", "time_id"])["log_wap"].diff().fillna(0)
    rv = (
        tgt.groupby(["stock_id", "time_id"])["lr"]
        .apply(lambda x: np.sqrt((x ** 2).sum()))
        .reset_index(name="rv")
    )
    rv["log_rv"] = np.log(rv["rv"].clip(lower=RV_FLOOR)).astype(np.float32)

    # ── Nearest bucket RV (bucket 3: seconds 360–479) ─────────────
    BKT_START = (N_BUCKETS - 1) * BUCKET_SIZE   # 360
    BKT_END   = INPUT_END                        # 480

    bkt = df[
        (df["seconds_in_bucket"] >= BKT_START) &
        (df["seconds_in_bucket"] <  BKT_END)
    ].copy()
    bkt["lr"] = bkt.groupby(["stock_id", "time_id"])["log_wap"].diff().fillna(0)
    bkt_rv = (
        bkt.groupby(["stock_id", "time_id"])["lr"]
        .apply(lambda x: np.sqrt((x ** 2).sum()))
        .reset_index(name="bkt3_rv")
    )
    bkt_rv["log_bkt3_rv"] = np.log(
        bkt_rv["bkt3_rv"].clip(lower=RV_FLOOR)
    ).astype(np.float32)

    rv = rv.merge(
        bkt_rv[["stock_id", "time_id", "log_bkt3_rv"]],
        on=["stock_id", "time_id"], how="left"
    )

    # log_rv_diff = log(RV_target) - log(RV_nearest_bucket)
    rv["log_rv_diff"] = (rv["log_rv"] - rv["log_bkt3_rv"]).astype(np.float32)

    del tgt, bkt, bkt_rv; gc.collect()
    return rv[["stock_id", "time_id", "rv", "log_rv", "log_rv_diff"]]


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: Per-stock feature building + aggregation
#          Parallelised via ProcessPoolExecutor (fork-safe on Linux)
# ══════════════════════════════════════════════════════════════════════

def _fast_rolling_mean_std(vals: np.ndarray, groups: np.ndarray, win: int):
    """
    Vectorised causal rolling mean+std within groups using cumsum trick.
    Much faster than groupby().transform(lambda) for large DataFrames.
    groups: integer group labels (e.g. time_id encoded as contiguous ints).
    Returns (mean_arr, std_arr) both float32.
    """
    n = len(vals)
    mean_out = np.empty(n, dtype=np.float32)
    std_out  = np.empty(n, dtype=np.float32)

    group_ids, inverse = np.unique(groups, return_inverse=True)

    for gid in group_ids:
        mask = groups == gid
        v = vals[mask].astype(np.float64)
        m = len(v)

        cs  = np.cumsum(v)
        cs2 = np.cumsum(v * v)

        idx    = np.arange(m)
        starts = np.maximum(0, idx - win + 1)
        counts = idx - starts + 1

        s_at_start  = np.where(starts > 0, cs[starts - 1], 0.0)
        roll_sum    = cs - s_at_start
        s2_at_start = np.where(starts > 0, cs2[starts - 1], 0.0)
        roll_sum2   = cs2 - s2_at_start

        rm = roll_sum / counts
        var = np.where(
            counts > 1,
            (roll_sum2 - roll_sum * roll_sum / counts) / (counts - 1),
            0.0,
        )
        rs = np.sqrt(np.maximum(var, 0.0))

        mean_out[mask] = rm.astype(np.float32)
        std_out[mask]  = rs.astype(np.float32)

    return mean_out, std_out


def two_scale_rv(log_rets: np.ndarray, h: int = 2) -> float:
    """
    Two-scale realized volatility estimator (Zhang, Mykland, Ait-Sahalia 2005).
    Reduces microstructure noise by separating bounce from true price movement.
    Returns the 2-scale RV (not log, not sqrt — raw variance scale).
    """
    n = len(log_rets)
    if n < h * 2:
        return float(np.sum(log_rets ** 2))

    rv_all  = float(np.sum(log_rets ** 2))
    rv_slow = float(np.sum(log_rets[::h] ** 2))
    m       = len(log_rets[::h])

    correction = (float(n) / m - 1.0)
    if abs(correction) < 1e-8:
        return rv_all

    noise_var = (rv_all - rv_slow) / correction
    rk = rv_slow - noise_var
    return float(max(rk, 0.0))


def process_one_stock(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all per-second features for one stock, aggregate to (time_id)
    rows, and return the aggregated DataFrame.
    Keeps only the input window (seconds 0–479).
    No whole-window (480s) RV computed — per-bucket RVs only.
    """
    df = stock_df[stock_df["seconds_in_bucket"] < INPUT_END].copy()
    if len(df) == 0:
        return pd.DataFrame()

    stock_id = df["stock_id"].iloc[0]
    tid_vals = df["time_id"].values

    # ── A. Rolling features ───────────────────────────────────────
    for col in ROLL_COLS:
        v = df[col].values.astype(np.float32)
        for w in ROLL_WINDOWS:
            rm, rs = _fast_rolling_mean_std(v, tid_vals, w)
            df[f"roll_{col}_mean_{w}s"] = rm
            df[f"roll_{col}_std_{w}s"]  = rs

    # ── B. Cumulative features ────────────────────────────────────
    for col in ["total_volume", "bid_ask_spread", "price_spread"]:
        g = df.groupby("time_id")[col]
        df[f"cum_sum_{col}"]  = g.cumsum().astype(np.float32)
        df[f"cum_mean_{col}"] = g.transform(
            lambda s: s.expanding(min_periods=1).mean()
        ).astype(np.float32)
        df[f"cum_max_{col}"]  = g.cummax().astype(np.float32)

    df["cum_log_return"] = df.groupby("time_id")["log_wap"].transform(
        lambda s: s - s.iloc[0]
    ).astype(np.float32)

    # ── Log return features ───────────────────────────────────────
    df["log_ret"]     = df.groupby("time_id")["log_wap"].diff().fillna(0).astype(np.float32)
    df["log_ret_sq"]  = (df["log_ret"] ** 2).astype(np.float32)
    df["abs_log_ret"] = df["log_ret"].abs().astype(np.float32)

    # Rolling realized volatility
    for w in ROLL_WINDOWS:
        rm_sq, _ = _fast_rolling_mean_std(df["log_ret_sq"].values, tid_vals, w)
        df[f"roll_rv_{w}s"] = np.sqrt(np.clip(rm_sq * w, 0, None)).astype(np.float32)

    # ── Realized kernel (2-scale RV) — per bucket and last windows ─
    tid_unique = df["time_id"].unique()
    rk_bkt  = {b: {} for b in range(N_BUCKETS)}
    rk_w60  = {}
    rk_w120 = {}

    for tid in tid_unique:
        tmask = df["time_id"].values == tid
        rets  = df.loc[tmask, "log_ret"].values.astype(np.float64)

        rk_w60[tid]  = two_scale_rv(rets[-60:]  if len(rets) >= 60  else rets, h=2)
        rk_w120[tid] = two_scale_rv(rets[-120:] if len(rets) >= 120 else rets, h=2)

        for b in range(N_BUCKETS):
            bmask = tmask & (df["bucket_120"].values == b)
            brets = df.loc[bmask, "log_ret"].values.astype(np.float64)
            rk_bkt[b][tid] = two_scale_rv(brets, h=2) if len(brets) >= 4 else 0.0

    # ── Aggregate to (time_id) level ──────────────────────────────
    skip = {"stock_id", "time_id", "seconds_in_bucket", "bucket_120"}
    num_cols = [c for c in df.columns if c not in skip and
                df[c].dtype in (np.float32, np.float64, np.int32, np.int8)]

    g = df.groupby("time_id")
    agg_last  = g[num_cols].last().add_suffix("_last")
    agg_mean  = g[num_cols].mean().add_suffix("_wmean")
    agg_std   = g[num_cols].std().fillna(0).add_suffix("_wstd")
    agg_range = (g[num_cols].max() - g[num_cols].min()).add_suffix("_range")

    agg = pd.concat([agg_last, agg_mean, agg_std, agg_range], axis=1)
    agg = agg.reset_index()
    agg["stock_id"] = stock_id
    agg["n_tids_this_stock"] = np.int32(len(agg))

    # ── Per-bucket (120s) RV ──────────────────────────────────────
    for b in range(N_BUCKETS):
        bkt_mask = df["bucket_120"] == b
        if bkt_mask.sum() == 0:
            agg[f"past_rv_bkt{b}"]     = np.float32(0)
            agg[f"past_log_rv_bkt{b}"] = np.float32(np.log(RV_FLOOR))
            continue
        bkt_rv = (
            df[bkt_mask].groupby("time_id")["log_ret_sq"]
            .sum().apply(lambda x: np.sqrt(max(x, 0)))
        )
        bkt_rv.name = f"past_rv_bkt{b}"
        agg = agg.merge(bkt_rv.reset_index(), on="time_id", how="left")
        agg[f"past_rv_bkt{b}"]     = agg[f"past_rv_bkt{b}"].fillna(0).astype(np.float32)
        agg[f"past_log_rv_bkt{b}"] = np.log(
            agg[f"past_rv_bkt{b}"].clip(lower=RV_FLOOR)
        ).astype(np.float32)

    # RV trend: last bucket vs first bucket
    agg["rv_trend_ratio"] = (
        agg[f"past_rv_bkt{N_BUCKETS-1}"] / agg["past_rv_bkt0"].clip(lower=EPS)
    ).astype(np.float32)
    agg["rv_accel"] = (
        agg[f"past_rv_bkt{N_BUCKETS-1}"] - agg["past_rv_bkt0"]
    ).astype(np.float32)

    # ── Realized kernel features (windowed + per-bucket only) ─────
    tids = agg["time_id"].values

    rk_w60_arr  = np.array([rk_w60.get(t, 0.0)  for t in tids], dtype=np.float64)
    rk_w120_arr = np.array([rk_w120.get(t, 0.0) for t in tids], dtype=np.float64)

    agg["rk_60s"]      = np.sqrt(np.clip(rk_w60_arr,  0, None)).astype(np.float32)
    agg["rk_120s"]     = np.sqrt(np.clip(rk_w120_arr, 0, None)).astype(np.float32)
    agg["log_rk_60s"]  = np.log(agg["rk_60s"].clip(lower=RV_FLOOR)).astype(np.float32)
    agg["log_rk_120s"] = np.log(agg["rk_120s"].clip(lower=RV_FLOOR)).astype(np.float32)

    for b in range(N_BUCKETS):
        rk_b = np.array([rk_bkt[b].get(t, 0.0) for t in tids], dtype=np.float64)
        agg[f"rk_bkt{b}"]     = np.sqrt(np.clip(rk_b, 0, None)).astype(np.float32)
        agg[f"log_rk_bkt{b}"] = np.log(
            agg[f"rk_bkt{b}"].clip(lower=RV_FLOOR)
        ).astype(np.float32)

    # RK trend: last bucket vs first bucket
    agg["rk_trend_ratio"] = (
        agg[f"rk_bkt{N_BUCKETS-1}"] / agg["rk_bkt0"].clip(lower=EPS)
    ).astype(np.float32)

    # ── H. 120s bucket features ───────────────────────────────────
    bkt = (
        df.groupby(["time_id", "bucket_120"])[BUCKET_AGG_COLS]
        .agg(["mean", "std"])
    )
    bkt.columns = [f"bkt_{c[0]}_{c[1]}" for c in bkt.columns]
    bkt = bkt.reset_index()

    piv = bkt.pivot_table(
        index="time_id", columns="bucket_120",
        values=[c for c in bkt.columns if c.startswith("bkt_")],
    )
    piv.columns = [f"{c[0]}_b{int(c[1])}" for c in piv.columns]

    bases = {c[:-3] for c in piv.columns if c.endswith("_b0")}
    for base in sorted(bases):
        for i in range(N_BUCKETS - 1):
            c0 = f"{base}_b{i}"
            c1 = f"{base}_b{i+1}"
            if c0 in piv.columns and c1 in piv.columns:
                piv[f"bktd_{base}_{i}to{i+1}"] = piv[c1] - piv[c0]
        cf = f"{base}_b0"
        cl = f"{base}_b{N_BUCKETS - 1}"
        if cf in piv.columns and cl in piv.columns:
            piv[f"bktt_{base}"] = piv[cl] / (piv[cf].abs() + EPS)

    piv = piv.reset_index()
    agg = agg.merge(piv, on="time_id", how="left")

    return downcast(agg)


def _worker_process_stock(stock_df):
    return process_one_stock(stock_df)


def build_aggregated(raw_df: pd.DataFrame, label: str) -> pd.DataFrame:
    stocks    = raw_df["stock_id"].unique()
    total     = len(stocks)
    stock_dfs = [raw_df[raw_df["stock_id"] == sid] for sid in stocks]

    log(f"  {label}: processing {total} stocks across {N_WORKERS} workers ...")

    if N_WORKERS <= 1:
        chunks = []
        for i, sdf in enumerate(stock_dfs):
            chunk = process_one_stock(sdf)
            if len(chunk) > 0:
                chunks.append(chunk)
            if (i + 1) % 25 == 0 or (i + 1) == total:
                log(f"  {label}: {i+1}/{total} stocks")
    else:
        chunks = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            results = pool.map(_worker_process_stock, stock_dfs,
                               chunksize=max(1, total // (N_WORKERS * 4)))
            for i, chunk in enumerate(results):
                if chunk is not None and len(chunk) > 0:
                    chunks.append(chunk)
                if (i + 1) % 25 == 0 or (i + 1) == total:
                    log(f"  {label}: {i+1}/{total} stocks")

    del stock_dfs
    result = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    return result


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Interaction features
# ══════════════════════════════════════════════════════════════════════

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    new  = {}
    cols = [c + "_last" for c in INTERACT_BASE if c + "_last" in df.columns]

    for a, b in combinations(cols, 2):
        tag = f"{a}_x_{b}".replace("_last", "")
        va  = df[a].values
        vb  = df[b].values
        new[f"isum_{tag}"]  = (va + vb).astype(np.float32)
        new[f"idiff_{tag}"] = (va - vb).astype(np.float32)
        new[f"irat_{tag}"]  = (va / (np.abs(vb) + EPS)).astype(np.float32)
        new[f"iprod_{tag}"] = (va * vb).astype(np.float32)

    if "bid_ask_spread_last" in df.columns and "total_volume_last" in df.columns:
        bas = df["bid_ask_spread_last"].values
        tv  = df["total_volume_last"].values
        new["spread_per_vol"] = (bas / (tv + EPS)).astype(np.float32)
        new["vol_wt_spread"]  = (bas * tv).astype(np.float32)
    if "bid_ask_spread_last" in df.columns and "volume_imbalance_last" in df.columns:
        new["spread_x_imbal"] = (
            df["bid_ask_spread_last"].values * df["volume_imbalance_last"].values
        ).astype(np.float32)
    if "price_spread_last" in df.columns and "wap_last" in df.columns:
        new["pspread_ratio"] = (
            df["price_spread_last"].values / (df["wap_last"].values + EPS)
        ).astype(np.float32)

    result = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    del new; gc.collect()
    return result


# ══════════════════════════════════════════════════════════════════════
#  STEP 2b: Cross-stock features per time_id
# ══════════════════════════════════════════════════════════════════════

def add_cross_stock_features(train_agg: pd.DataFrame,
                             test_agg:  pd.DataFrame):
    cross_cols = ["wap_last", "bid_ask_spread_last", "total_volume_last",
                  "log_volume_last", "price_spread_last"]
    cross_cols = [c for c in cross_cols if c in train_agg.columns]

    tid_stats = train_agg.groupby("time_id")[cross_cols].agg(["mean", "std", "count"])
    tid_stats.columns = [f"tid_{c[0]}_{c[1]}" for c in tid_stats.columns]
    tid_stats["tid_stock_count"] = (
        train_agg.groupby("time_id")["stock_id"].nunique()
    )
    tid_stats = tid_stats.reset_index()

    for df in [train_agg, test_agg]:
        merged = df.merge(tid_stats, on="time_id", how="left")
        for col in cross_cols:
            m_col = f"tid_{col}_mean"
            s_col = f"tid_{col}_std"
            if m_col in merged.columns and s_col in merged.columns:
                merged[f"csz_{col}"] = (
                    (merged[col].values - merged[m_col].values) /
                    np.clip(merged[s_col].values, EPS, None)
                ).astype(np.float32)
        new_cols = [c for c in merged.columns if c not in df.columns]
        for c in new_cols:
            merged[c] = merged[c].fillna(0).astype(np.float32)
        for c in new_cols:
            df[c] = merged[c].values
        del merged

    cs_feats = [c for c in train_agg.columns if c.startswith(("tid_", "csz_"))]
    log(f"  Cross-stock features added: {len(cs_feats)}")
    gc.collect()
    return train_agg, test_agg


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: Clustering (train-fit, test-transform)
# ══════════════════════════════════════════════════════════════════════

def add_clusters(train_agg: pd.DataFrame, test_agg: pd.DataFrame):
    cluster_cols = [c for c in train_agg.columns
                    if c.endswith("_last") and train_agg[c].notna().mean() > 0.9][:25]

    def _scale(X, means=None, stds=None):
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        if means is None:
            means = X.mean(axis=0)
            stds  = X.std(axis=0) + 1e-8
        return (X - means) / stds, means, stds

    # Stock clusters
    sp_tr = train_agg.groupby("stock_id")[cluster_cols].mean()
    X_s, sm, ss = _scale(sp_tr.values)
    km_s = MiniBatchKMeans(n_clusters=8, random_state=SEED, batch_size=256, n_init=3)
    km_s.fit(X_s)
    sc_map = pd.Series(km_s.labels_, index=sp_tr.index, name="stock_cluster")

    sp_te = test_agg.groupby("stock_id")[cluster_cols].mean()
    X_s_te, _, _ = _scale(sp_te.values, sm, ss)
    sc_map_te = pd.Series(km_s.predict(X_s_te), index=sp_te.index, name="stock_cluster")
    sc_full = pd.concat([sc_map, sc_map_te])
    sc_full = sc_full[~sc_full.index.duplicated(keep="first")]

    train_agg = train_agg.merge(
        sc_full.reset_index().rename(columns={"index": "stock_id"}),
        on="stock_id", how="left")
    test_agg = test_agg.merge(
        sc_full.reset_index().rename(columns={"index": "stock_id"}),
        on="stock_id", how="left")

    # Time clusters
    tp_tr = train_agg.groupby("time_id")[cluster_cols].mean()
    X_t, tm, ts_ = _scale(tp_tr.values)
    km_t = MiniBatchKMeans(n_clusters=6, random_state=SEED, batch_size=256, n_init=3)
    km_t.fit(X_t)
    tc_map = pd.Series(km_t.labels_, index=tp_tr.index, name="time_cluster")

    tp_te = test_agg.groupby("time_id")[cluster_cols].mean()
    X_t_te, _, _ = _scale(tp_te.values, tm, ts_)
    tc_map_te = pd.Series(km_t.predict(X_t_te), index=tp_te.index, name="time_cluster")
    tc_full = pd.concat([tc_map, tc_map_te])
    tc_full = tc_full[~tc_full.index.duplicated(keep="first")]

    train_agg = train_agg.merge(
        tc_full.reset_index().rename(columns={"index": "time_id"}),
        on="time_id", how="left")
    test_agg = test_agg.merge(
        tc_full.reset_index().rename(columns={"index": "time_id"}),
        on="time_id", how="left")

    for col in ["stock_cluster", "time_cluster"]:
        for df in [train_agg, test_agg]:
            if col in df.columns and df[col].isna().any():
                mode_val = int(train_agg[col].mode().iloc[0])
                df[col] = df[col].fillna(mode_val).astype(np.int32)

    log(f"Stock clusters (8): {sc_map.value_counts().to_dict()}")
    log(f"Time  clusters (6): {tc_map.value_counts().to_dict()}")
    gc.collect()
    return train_agg, test_agg


# ══════════════════════════════════════════════════════════════════════
#  STEP 4: KNN features
# ══════════════════════════════════════════════════════════════════════

def add_knn_features(train_agg: pd.DataFrame, test_agg: pd.DataFrame):
    K       = 10
    knn_src = [c for c in train_agg.columns if c.endswith("_last")][:20]

    X_tr   = np.nan_to_num(train_agg[knn_src].values, nan=0.0).astype(np.float32)
    mn     = X_tr.mean(axis=0)
    sd     = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - mn) / sd

    nn = NearestNeighbors(n_neighbors=K + 1, metric="euclidean",
                          algorithm="ball_tree", n_jobs=-1)
    nn.fit(X_tr_s)

    stat_cols   = ["wap_last", "bid_ask_spread_last", "total_volume_last"]
    stat_cols   = [c for c in stat_cols if c in train_agg.columns]
    stat_arrays = {c: train_agg[c].values.astype(np.float32) for c in stat_cols}

    def _knn_feats(X_scaled, is_train):
        dists, idxs = nn.kneighbors(X_scaled)
        if is_train:
            dists, idxs = dists[:, 1:], idxs[:, 1:]
        else:
            dists, idxs = dists[:, :K], idxs[:, :K]
        out = {
            "knn_mean_dist": dists.mean(axis=1).astype(np.float32),
            "knn_std_dist":  dists.std(axis=1).astype(np.float32),
            "knn_min_dist":  dists.min(axis=1).astype(np.float32),
        }
        for c in stat_cols:
            vals = stat_arrays[c][idxs]
            out[f"knn_mean_{c}"] = vals.mean(axis=1).astype(np.float32)
            out[f"knn_std_{c}"]  = vals.std(axis=1).astype(np.float32)
        return pd.DataFrame(out)

    X_te   = np.nan_to_num(test_agg[knn_src].values, nan=0.0).astype(np.float32)
    X_te_s = (X_te - mn) / sd

    knn_tr = _knn_feats(X_tr_s, is_train=True)
    knn_te = _knn_feats(X_te_s, is_train=False)

    for c in knn_tr.columns:
        train_agg[c] = knn_tr[c].values
        test_agg[c]  = knn_te[c].values

    log(f"KNN features added: {len(knn_tr.columns)}")
    del X_tr, X_tr_s, X_te, X_te_s, knn_tr, knn_te, stat_arrays; gc.collect()
    return train_agg, test_agg


# ══════════════════════════════════════════════════════════════════════
#  STEP 5: Value counts / rarity encodings (train-fit)
# ══════════════════════════════════════════════════════════════════════

def add_value_counts(train_agg: pd.DataFrame, test_agg: pd.DataFrame):
    vcount_cols = ["wap_last", "bid_ask_spread_last", "total_volume_last",
                   "log_volume_last", "price_spread_last", "volume_imbalance_last"]
    vcount_cols = [c for c in vcount_cols if c in train_agg.columns]
    N_BINS = 50

    for col in vcount_cols:
        vals  = train_agg[col].dropna().values
        edges = np.unique(np.percentile(vals, np.linspace(0, 100, N_BINS + 1)))

        tr_bins = np.searchsorted(edges, train_agg[col].fillna(0).values, side="right")
        te_bins = np.searchsorted(edges, test_agg[col].fillna(0).values,  side="right")

        uniq, cnts = np.unique(tr_bins, return_counts=True)
        max_bin    = int(max(tr_bins.max(), te_bins.max())) + 1
        count_arr  = np.ones(max_bin + 1, dtype=np.int32)
        count_arr[uniq] = cnts.astype(np.int32)

        train_agg[f"vc_{col}"] = count_arr[tr_bins]
        test_agg[f"vc_{col}"]  = count_arr[te_bins]
        train_agg[f"vb_{col}"] = tr_bins.astype(np.int16)
        test_agg[f"vb_{col}"]  = te_bins.astype(np.int16)

    n = sum(1 for c in train_agg.columns if c.startswith("vc_") or c.startswith("vb_"))
    log(f"Value count features added: {n}")
    gc.collect()
    return train_agg, test_agg


# ══════════════════════════════════════════════════════════════════════
#  STEP 6: Target encoding — uses log_rv_diff as target
# ══════════════════════════════════════════════════════════════════════

def add_target_encoding(train_agg: pd.DataFrame, test_agg: pd.DataFrame):
    tenc_cols  = ["stock_id", "stock_cluster", "time_cluster"]
    tenc_cols += [c for c in train_agg.columns if c.startswith("vb_")]
    tenc_cols  = [c for c in tenc_cols if c in train_agg.columns and c in test_agg.columns]

    ALPHA  = 20
    NFOLDS = 5
    gm     = train_agg["log_rv_diff"].mean()   # global mean of log_rv_diff

    kf        = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    fold_idx  = list(kf.split(train_agg))
    target_vals = train_agg["log_rv_diff"].values.astype(np.float64)

    for col in tenc_cols:
        enc_tr   = np.full(len(train_agg), np.nan, dtype=np.float32)
        col_vals = train_agg[col].values

        for tr_i, va_i in fold_idx:
            tmp    = pd.DataFrame({"cat": col_vals[tr_i], "tgt": target_vals[tr_i]})
            stats  = tmp.groupby("cat")["tgt"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + ALPHA * gm) / (stats["count"] + ALPHA)
            smap   = smooth.to_dict()
            enc_tr[va_i] = np.array([smap.get(c, np.nan) for c in col_vals[va_i]],
                                    dtype=np.float32)

        enc_tr = np.where(np.isnan(enc_tr), gm, enc_tr)
        train_agg[f"te_{col}"] = enc_tr.astype(np.float32)

        full_stats  = train_agg.groupby(col)["log_rv_diff"].agg(["mean", "count"])
        full_smooth = (full_stats["count"] * full_stats["mean"] + ALPHA * gm) / (full_stats["count"] + ALPHA)
        test_agg[f"te_{col}"] = test_agg[col].map(full_smooth.to_dict()).fillna(gm).astype(np.float32)

    n = sum(1 for c in train_agg.columns if c.startswith("te_"))
    log(f"Target-encoded features added: {n}")
    gc.collect()
    return train_agg, test_agg


# ══════════════════════════════════════════════════════════════════════
#  STEP 7: Feature selection — MI against log_rv_diff
# ══════════════════════════════════════════════════════════════════════

def _chunked_corr_f32(X: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    n_samples, n_feat = X.shape
    means = X.mean(axis=0, keepdims=True)
    stds  = X.std(axis=0, keepdims=True)
    stds[stds < 1e-10] = 1.0
    X_std = (X - means) / stds

    corr = np.empty((n_feat, n_feat), dtype=np.float32)
    for i in range(0, n_feat, chunk_size):
        end = min(i + chunk_size, n_feat)
        corr[i:end, :] = X_std[:, i:end].T @ X_std / n_samples

    del X_std
    return np.abs(corr)


def feature_selection(train_agg: pd.DataFrame, test_agg: pd.DataFrame):
    # Exclude identifiers, both targets, and removed whole-window RV cols
    exclude = {
        "stock_id", "time_id",
        "log_rv", "log_rv_diff", "rv",
        "seconds_in_bucket", "bucket_120",
    }
    fcols = [c for c in train_agg.columns
             if c not in exclude
             and train_agg[c].dtype in (np.float32, np.float64,
                                        np.int32, np.int16, np.int8)]

    log(f"Starting features: {len(fcols)}")

    # Stage 1: drop near-constant
    stds  = train_agg[fcols].std()
    const = stds[stds < 1e-10].index.tolist()
    fcols = [c for c in fcols if c not in const]
    log(f"After dropping {len(const)} near-constant: {len(fcols)}")

    # Stage 2: mutual-info noise test against log_rv_diff
    NSUB = min(8_000, len(train_agg))
    si   = np.random.choice(len(train_agg), NSUB, replace=False)
    X_s  = train_agg[fcols].iloc[si].fillna(0).values.astype(np.float32)
    y_s  = train_agg["log_rv_diff"].iloc[si].values.astype(np.float32)

    N_NOISE = 5
    noise   = np.random.randn(NSUB, N_NOISE).astype(np.float32)
    X_wn    = np.hstack([X_s, noise])

    mi       = mutual_info_regression(X_wn, y_s, random_state=SEED,
                                      n_neighbors=5, n_jobs=-1)
    real_mi  = mi[:len(fcols)]
    noise_mi = mi[len(fcols):]
    thresh   = np.median(noise_mi)

    mask  = real_mi > thresh
    fcols = [c for c, k in zip(fcols, mask) if k]
    log(f"Noise threshold: {thresh:.6f} → {len(fcols)} surviving")

    mi_map = dict(zip(fcols, real_mi[mask]))
    del X_s, X_wn, noise; gc.collect()

    # Stage 3: drop correlated pairs
    X_corr = train_agg[fcols].iloc[si].fillna(0).values.astype(np.float32)
    corr   = _chunked_corr_f32(X_corr)
    del X_corr; gc.collect()

    drop = set()
    n_f  = len(fcols)
    for i in range(n_f):
        if fcols[i] in drop:
            continue
        for j in range(i + 1, n_f):
            if fcols[j] in drop:
                continue
            if corr[i, j] > 0.98:
                if mi_map.get(fcols[i], 0) < mi_map.get(fcols[j], 0):
                    drop.add(fcols[i])
                else:
                    drop.add(fcols[j])

    fcols = [c for c in fcols if c not in drop]
    log(f"After correlation filter: {len(fcols)}")

    del corr; gc.collect()
    return fcols, mi_map


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("Feature Engineering — Multi-Core Memory-Efficient Pipeline")
    log(f"  Input  : 0–{INPUT_END-1}s  |  Target : {TARGET_START}–{TOTAL_SECONDS-1}s")
    log(f"  Buckets: {N_BUCKETS} × {BUCKET_SIZE}s")
    log(f"  Target : log_rv_diff = log(RV_480-599) - log(RV_360-479)")
    log(f"  Workers: {N_WORKERS}  |  CPU cores: {cpu_count()}")
    log("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    log("\nLoading data ...")
    train_raw = load_raw(DATA_DIR / "train.parquet")
    test_raw  = load_raw(DATA_DIR / "test.parquet")
    log(f"  Train: {train_raw.shape}  |  Test: {test_raw.shape}")

    # ── Target ────────────────────────────────────────────────────
    log("\nComputing targets ...")
    train_targets = compute_target(train_raw)
    test_targets  = compute_target(test_raw)
    log(f"  Train targets: {len(train_targets)}  |  Test: {len(test_targets)}")
    log(f"  log_rv_diff mean (train): {train_targets['log_rv_diff'].mean():.4f}")
    log(f"  log_rv_diff std  (train): {train_targets['log_rv_diff'].std():.4f}")

    # ── Per-stock feature building + aggregation ───────────────────
    log("\nBuilding per-stock features + aggregation ...")

    _ckpt_tr = OUTPUT_DIR / "_checkpoint_train_agg.parquet"
    _ckpt_te = OUTPUT_DIR / "_checkpoint_test_agg.parquet"

    if _ckpt_tr.exists() and _ckpt_te.exists():
        log("  Found checkpoint files — loading from disk (skipping rebuild)")
        train_agg = pd.read_parquet(_ckpt_tr)
        test_agg  = pd.read_parquet(_ckpt_te)
        del train_raw, test_raw; gc.collect()
    else:
        train_agg = build_aggregated(train_raw, "Train")
        del train_raw; gc.collect()

        test_agg = build_aggregated(test_raw, "Test")
        del test_raw; gc.collect()

        train_agg.to_parquet(_ckpt_tr, index=False)
        test_agg.to_parquet(_ckpt_te, index=False)
        log(f"  Checkpoint saved → {_ckpt_tr.name}, {_ckpt_te.name}")

    log(f"  Train agg: {train_agg.shape}  |  Test agg: {test_agg.shape}")
    log(f"  Train agg RAM: ~{train_agg.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    log(f"  Test  agg RAM: ~{test_agg.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # ── Interactions ──────────────────────────────────────────────
    log("\nAdding interaction features ...")
    train_agg = add_interactions(train_agg)
    test_agg  = add_interactions(test_agg)
    n_int = sum(1 for c in train_agg.columns
                if c.startswith(("isum_", "idiff_", "irat_", "iprod_")))
    log(f"  Interaction features: {n_int}")

    # ── Cross-stock ───────────────────────────────────────────────
    log("\nCross-stock time_id features ...")
    train_agg, test_agg = add_cross_stock_features(train_agg, test_agg)

    # ── Clustering ────────────────────────────────────────────────
    log("\nClustering (train-fit) ...")
    train_agg, test_agg = add_clusters(train_agg, test_agg)

    # ── KNN ───────────────────────────────────────────────────────
    log("\nKNN features (train-fit index) ...")
    train_agg, test_agg = add_knn_features(train_agg, test_agg)

    # ── Value counts ──────────────────────────────────────────────
    log("\nValue counts / rarity encoding ...")
    train_agg, test_agg = add_value_counts(train_agg, test_agg)

    # ── Attach both targets to train ──────────────────────────────
    train_agg = train_agg.merge(
        train_targets[["stock_id", "time_id", "log_rv", "log_rv_diff"]],
        on=["stock_id", "time_id"], how="left"
    )

    # ── Target encoding (against log_rv_diff) ─────────────────────
    log("\nTarget encoding (5-fold OOF on log_rv_diff) ...")
    train_agg, test_agg = add_target_encoding(train_agg, test_agg)

    # ── Feature selection (MI against log_rv_diff) ────────────────
    log("\nFeature selection ...")
    feature_cols, mi_map = feature_selection(train_agg, test_agg)

    # ── Attach both targets to test ───────────────────────────────
    test_agg = test_agg.merge(
        test_targets[["stock_id", "time_id", "log_rv", "log_rv_diff"]],
        on=["stock_id", "time_id"], how="left"
    )

    # ── Save ──────────────────────────────────────────────────────
    log("\nSaving ...")
    # Save features + both targets (log_rv for eval, log_rv_diff for training)
    save_cols = ["stock_id", "time_id"] + feature_cols + ["log_rv", "log_rv_diff"]
    sc_tr = [c for c in save_cols if c in train_agg.columns]
    sc_te = [c for c in save_cols if c in test_agg.columns]

    train_agg[sc_tr].to_parquet(OUTPUT_DIR / "feature_store_train.parquet", index=False)
    test_agg[sc_te].to_parquet(OUTPUT_DIR / "feature_store_test.parquet",   index=False)

    with open(OUTPUT_DIR / "selected_features.txt", "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    log(f"  Train saved: {train_agg[sc_tr].shape}")
    log(f"  Test  saved: {test_agg[sc_te].shape}")
    log(f"  Features:    {len(feature_cols)}")

    # Clean up checkpoints
    for ckpt in [_ckpt_tr, _ckpt_te]:
        if ckpt.exists():
            ckpt.unlink()
            log(f"  Cleaned up {ckpt.name}")

    # ── Diagnostics ───────────────────────────────────────────────
    mi_df = (
        pd.DataFrame({"feature": feature_cols,
                      "MI": [mi_map.get(c, 0) for c in feature_cols]})
        .sort_values("MI", ascending=False)
    )
    log("\nTop 20 features by mutual information (vs log_rv_diff):")
    for _, row in mi_df.head(20).iterrows():
        log(f"  {row['MI']:.5f}  {row['feature']}")

    groups = {
        "Rolling":       [c for c in feature_cols if "roll_" in c],
        "Cumulative":    [c for c in feature_cols if "cum_" in c],
        "Interaction":   [c for c in feature_cols if c.startswith(
                            ("isum_", "idiff_", "irat_", "iprod_",
                             "spread_per", "vol_wt", "spread_x", "pspread"))],
        "Bucket (120s)": [c for c in feature_cols if c.startswith(("bkt_", "bktd_", "bktt_"))],
        "Cross-stock":   [c for c in feature_cols if c.startswith(("tid_", "csz_"))],
        "Cluster":       [c for c in feature_cols if "cluster" in c],
        "KNN":           [c for c in feature_cols if "knn_" in c],
        "Value count":   [c for c in feature_cols if c.startswith(("vc_", "vb_"))],
        "Target enc":    [c for c in feature_cols if c.startswith("te_")],
    }
    counted = set()
    for cols in groups.values():
        counted.update(cols)
    groups["Base agg"] = [c for c in feature_cols if c not in counted]

    log("\nFeature group counts:")
    for g, cols in groups.items():
        log(f"  {g:20s}: {len(cols):4d}")
    log(f"  {'TOTAL':20s}: {len(feature_cols):4d}")

    log("\nDone.")
    log("  Inference: log_rv_pred = model.predict(X) + past_log_rv_bkt3")


if __name__ == "__main__":
    main()