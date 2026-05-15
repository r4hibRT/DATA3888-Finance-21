"""
preprocess_stocks.py — Read stock CSVs → split → engineer features → save parquets
====================================================================================

Input:  Folder of CSVs, one per stock.
        Each CSV: time_id, seconds_in_bucket, wap, bid_ask_spread, total_volume
        Each time_id has 600 rows (seconds 0–599).

Windows:
        Observation : seconds [0, 480)
        Target      : seconds [480, 600)

Output: train.parquet, test.parquet
        One row per (stock, time_id), 100+ feature columns + target.

Usage:
    python preprocess_stocks.py --data_dir ./stocks --output_dir ./processed
"""

from email import parser
import os
import glob
import warnings
import argparse
import time as _time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────

OBS_SECONDS  = 480
TARGET_START = 480
TOTAL_LEN    = 600
TEST_RATIO   = 0.2
SEED         = 42
EPSILON      = 1e-10


# ═════════════════════════════════════════════════════════════════════
# STEP 1: SPLIT TIME_IDS
# ═════════════════════════════════════════════════════════════════════

def split_time_ids(data_dir: str,
                   test_ratio: float = TEST_RATIO,
                   seed: int = SEED):
    """Read one file to get all time_ids, shuffle-split into train/test."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    tids = pd.read_csv(files[0], usecols=["time_id"])["time_id"].unique()
    rng  = np.random.default_rng(seed)
    rng.shuffle(tids)
    n_test    = int(len(tids) * test_ratio)
    test_ids  = set(tids[:n_test])
    train_ids = set(tids[n_test:])
    print(f"  {len(tids)} time_ids → {len(train_ids)} train, {len(test_ids)} test")
    return train_ids, test_ids


# ═════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING — one stock at a time
# ═════════════════════════════════════════════════════════════════════

def _safe_log(x):
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), EPSILON))


def _autocorr(vals, lag):
    if len(vals) <= lag + 1:
        return 0.0
    c = np.corrcoef(vals[lag:], vals[:-lag])
    return float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0


def process_one_stock(path: str,
                      time_ids: set,
                      obs_seconds: int = OBS_SECONDS,
                      target_start: int = TARGET_START) -> pd.DataFrame:
    """
    Process a single stock CSV for a given set of time_ids.
    Returns one row per time_id with 100+ engineered features.
    """
    stock_name = os.path.basename(path).replace(".csv", "")
    try:
        df = pd.read_csv(path)
        df = df[df["time_id"].isin(time_ids)].copy()
        if df.empty:
            return pd.DataFrame()
        df.sort_values(["time_id", "seconds_in_bucket"], inplace=True)

        df_obs = df[df["seconds_in_bucket"] < obs_seconds].copy()
        df_tgt = df[df["seconds_in_bucket"] >= target_start].copy()
        if df_obs.empty or df_tgt.empty:
            return pd.DataFrame()

        # ── Base transforms ───────────────────────────────────────────
        df_obs["log_wap"]    = _safe_log(df_obs["wap"])
        df_obs["log_ret"]    = df_obs.groupby("time_id")["log_wap"].diff().fillna(0)
        df_obs["abs_ret"]    = df_obs["log_ret"].abs()
        df_obs["sq_ret"]     = df_obs["log_ret"] ** 2
        df_obs["log_spread"] = _safe_log(df_obs["bid_ask_spread"])
        df_obs["log_volume"] = _safe_log(df_obs["total_volume"].clip(lower=1))
        df_obs["signed_vol"] = np.sign(df_obs["log_ret"]) * df_obs["total_volume"]
        df_obs["spread_x_vol"] = df_obs["bid_ask_spread"] * df_obs["total_volume"]
        df_obs["vw_ret"]     = df_obs["log_ret"] * df_obs["total_volume"]
        df_obs["vw_spread"]  = df_obs["bid_ask_spread"] * df_obs["total_volume"]

        df_tgt["log_wap"] = _safe_log(df_tgt["wap"])
        df_tgt["log_ret"] = df_tgt.groupby("time_id")["log_wap"].diff().fillna(0)

        g_obs = df_obs.groupby("time_id")

        # ══════════════════════════════════════════════════════════════
        # TARGET: log(RV_future)
        # ══════════════════════════════════════════════════════════════
        target_rv  = df_tgt.groupby("time_id")["log_ret"].apply(
            lambda x: np.sqrt(np.sum(x**2))
        ).rename("target_rv")
        log_target = _safe_log(target_rv).rename("target_log_rv")

        # ══════════════════════════════════════════════════════════════
        # A. INTERVAL RV LAGS  (30-second buckets over 480s → 16 lags)
        # ══════════════════════════════════════════════════════════════
        n_intervals = obs_seconds // 30
        df_obs["interval"] = (df_obs["seconds_in_bucket"] // 30).astype(int).clip(upper=n_intervals - 1)

        interval_rv = (
            df_obs.groupby(["time_id", "interval"])["log_ret"]
            .apply(lambda x: np.sqrt(np.sum(x**2)))
            .unstack(level="interval")
        ).fillna(0)
        interval_rv.columns = [f"rv_lag_{int(c)+1}" for c in interval_rv.columns]

        # Log versions
        for c in list(interval_rv.columns):
            interval_rv[f"log_{c}"] = _safe_log(interval_rv[c])

        # ══════════════════════════════════════════════════════════════
        # B. HAR AGGREGATES (short / medium / long)
        # ══════════════════════════════════════════════════════════════
        log_rv_cols = sorted([c for c in interval_rv.columns if c.startswith("log_rv_lag_")])
        n_lags = len(log_rv_cols)

        short_cols = [f"log_rv_lag_{i}" for i in range(n_lags - 2, n_lags + 1) if f"log_rv_lag_{i}" in interval_rv]
        med_cols   = [f"log_rv_lag_{i}" for i in range(n_lags - 7, n_lags + 1) if f"log_rv_lag_{i}" in interval_rv]

        interval_rv["log_rv_short"]  = interval_rv[short_cols].mean(axis=1) if short_cols else 0
        interval_rv["log_rv_medium"] = interval_rv[med_cols].mean(axis=1) if med_cols else 0
        interval_rv["log_rv_long"]   = interval_rv[log_rv_cols].mean(axis=1)

        interval_rv["har_short_long"] = interval_rv["log_rv_short"] - interval_rv["log_rv_long"]
        interval_rv["har_short_med"]  = interval_rv["log_rv_short"] - interval_rv["log_rv_medium"]
        interval_rv["har_med_long"]   = interval_rv["log_rv_medium"] - interval_rv["log_rv_long"]

        # RV trend slope (linear regression over lag indices)
        if n_lags >= 3:
            X_idx  = np.arange(n_lags, dtype=np.float64)
            X_mean = X_idx.mean()
            X_var  = ((X_idx - X_mean) ** 2).sum()
            rv_v   = interval_rv[log_rv_cols].values
            Y_mean = rv_v.mean(axis=1, keepdims=True)
            interval_rv["rv_trend_slope"] = ((rv_v - Y_mean) * (X_idx - X_mean)).sum(axis=1) / (X_var + EPSILON)

        # Std across lags (vol-of-vol proxy)
        interval_rv["rv_lag_std"] = interval_rv[log_rv_cols].std(axis=1)

        # Max / min lag
        raw_rv_cols = [c for c in interval_rv.columns if c.startswith("rv_lag_") and not c.startswith("log_")]
        if raw_rv_cols:
            interval_rv["rv_lag_max"]       = interval_rv[raw_rv_cols].max(axis=1)
            interval_rv["rv_lag_min"]       = interval_rv[raw_rv_cols].min(axis=1)
            interval_rv["rv_lag_range"]     = interval_rv["rv_lag_max"] - interval_rv["rv_lag_min"]
            interval_rv["rv_lag_max_ratio"] = interval_rv["rv_lag_max"] / (interval_rv[raw_rv_cols].mean(axis=1) + EPSILON)

        # ══════════════════════════════════════════════════════════════
        # C. SESSION-LEVEL RV AT MULTIPLE WINDOWS
        # ══════════════════════════════════════════════════════════════
        def _rv(grp):
            return np.sqrt(np.sum(grp ** 2))

        rv_full    = g_obs["log_ret"].apply(_rv).rename("rv_full")
        rv_last60  = df_obs[df_obs["seconds_in_bucket"] >= obs_seconds - 60].groupby("time_id")["log_ret"].apply(_rv).rename("rv_last_60s")
        rv_last120 = df_obs[df_obs["seconds_in_bucket"] >= obs_seconds - 120].groupby("time_id")["log_ret"].apply(_rv).rename("rv_last_120s")
        rv_last240 = df_obs[df_obs["seconds_in_bucket"] >= obs_seconds - 240].groupby("time_id")["log_ret"].apply(_rv).rename("rv_last_240s")
        rv_first_h = df_obs[df_obs["seconds_in_bucket"] < obs_seconds // 2].groupby("time_id")["log_ret"].apply(_rv).rename("rv_first_half")
        rv_second_h = df_obs[df_obs["seconds_in_bucket"] >= obs_seconds // 2].groupby("time_id")["log_ret"].apply(_rv).rename("rv_second_half")

        log_rv_full     = _safe_log(rv_full).rename("log_rv_full")
        log_rv_last60   = _safe_log(rv_last60).rename("log_rv_last_60s")
        log_rv_last120  = _safe_log(rv_last120).rename("log_rv_last_120s")
        log_rv_last240  = _safe_log(rv_last240).rename("log_rv_last_240s")
        log_rv_first_h  = _safe_log(rv_first_h).rename("log_rv_first_half")
        log_rv_second_h = _safe_log(rv_second_h).rename("log_rv_second_half")

        # ══════════════════════════════════════════════════════════════
        # D. RV RATIOS & ACCELERATION
        # ══════════════════════════════════════════════════════════════
        rv_ratio_60    = (rv_last60  / (rv_full + EPSILON)).rename("rv_ratio_60_full")
        rv_ratio_120   = (rv_last120 / (rv_full + EPSILON)).rename("rv_ratio_120_full")
        rv_ratio_240   = (rv_last240 / (rv_full + EPSILON)).rename("rv_ratio_240_full")
        rv_ratio_halves = (rv_second_h / (rv_first_h + EPSILON)).rename("rv_ratio_halves")
        log_rv_ratio_60  = _safe_log(rv_ratio_60).rename("log_rv_ratio_60")
        log_rv_ratio_120 = _safe_log(rv_ratio_120).rename("log_rv_ratio_120")
        rv_accel = (_safe_log(rv_last60) - _safe_log(rv_full / (obs_seconds / 60))).rename("rv_acceleration")

        # ══════════════════════════════════════════════════════════════
        # E. BIPOWER VARIATION & JUMP
        # ══════════════════════════════════════════════════════════════
        def _bpv_jump(grp):
            r = grp["abs_ret"].values
            if len(r) < 3:
                return pd.Series({"bpv": 0.0, "jump": 0.0})
            bpv  = (np.pi / 2) * np.sum(r[1:] * r[:-1])
            rv   = np.sum(grp["sq_ret"].values)
            return pd.Series({"bpv": bpv, "jump": max(rv - bpv, 0.0)})

        bpv_df = g_obs.apply(_bpv_jump)
        bpv_df["log_bpv"]    = _safe_log(bpv_df["bpv"])
        bpv_df["log_jump"]   = _safe_log(bpv_df["jump"].clip(lower=EPSILON))
        bpv_df["jump_ratio"] = bpv_df["jump"] / (bpv_df["bpv"] + EPSILON)
        bpv_df["jump_frac"]  = bpv_df["jump"] / (bpv_df["bpv"] + bpv_df["jump"] + EPSILON)

        # ══════════════════════════════════════════════════════════════
        # F. RETURN DISTRIBUTION
        # ══════════════════════════════════════════════════════════════
        ret_mean  = g_obs["log_ret"].mean().rename("ret_mean")
        ret_std   = g_obs["log_ret"].std().rename("ret_std")
        ret_skew  = g_obs["log_ret"].skew().rename("ret_skew")
        ret_kurt  = g_obs["log_ret"].apply(lambda x: x.kurtosis()).rename("ret_kurt")
        ret_min   = g_obs["log_ret"].min().rename("ret_min")
        ret_max   = g_obs["log_ret"].max().rename("ret_max")
        ret_range = (ret_max - ret_min).rename("ret_range")
        ret_iqr   = g_obs["log_ret"].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).rename("ret_iqr")
        abs_ret_mean = g_obs["abs_ret"].mean().rename("abs_ret_mean")
        abs_ret_max  = g_obs["abs_ret"].max().rename("abs_ret_max")
        abs_ret_p95  = g_obs["abs_ret"].quantile(0.95).rename("abs_ret_p95")

        # ══════════════════════════════════════════════════════════════
        # G. AUTOCORRELATION
        # ══════════════════════════════════════════════════════════════
        ret_ac1  = g_obs.apply(lambda g: _autocorr(g["log_ret"].values, 1)).rename("ret_ac1")
        ret_ac5  = g_obs.apply(lambda g: _autocorr(g["log_ret"].values, 5)).rename("ret_ac5")
        ret_ac10 = g_obs.apply(lambda g: _autocorr(g["log_ret"].values, 10)).rename("ret_ac10")
        abs_ac1  = g_obs.apply(lambda g: _autocorr(g["abs_ret"].values, 1)).rename("abs_ret_ac1")
        abs_ac5  = g_obs.apply(lambda g: _autocorr(g["abs_ret"].values, 5)).rename("abs_ret_ac5")

        # ══════════════════════════════════════════════════════════════
        # H. SPREAD FEATURES
        # ══════════════════════════════════════════════════════════════
        spread_mean  = g_obs["bid_ask_spread"].mean().rename("spread_mean")
        spread_std   = g_obs["bid_ask_spread"].std().rename("spread_std")
        spread_max   = g_obs["bid_ask_spread"].max().rename("spread_max")
        spread_min   = g_obs["bid_ask_spread"].min().rename("spread_min")
        spread_range = (spread_max - spread_min).rename("spread_range")
        spread_cv    = (spread_std / (spread_mean + EPSILON)).rename("spread_cv")
        log_spread_mean = g_obs["log_spread"].mean().rename("log_spread_mean")
        log_spread_std  = g_obs["log_spread"].std().rename("log_spread_std")
        spread_last60   = df_obs[df_obs["seconds_in_bucket"] >= obs_seconds - 60].groupby("time_id")["bid_ask_spread"].mean().rename("spread_last_60s")
        spread_ratio    = (spread_last60 / (spread_mean + EPSILON)).rename("spread_ratio_60_full")
        spread_skew     = g_obs["bid_ask_spread"].skew().rename("spread_skew")

        # ══════════════════════════════════════════════════════════════
        # I. VOLUME FEATURES
        # ══════════════════════════════════════════════════════════════
        vol_mean     = g_obs["total_volume"].mean().rename("vol_mean")
        vol_std      = g_obs["total_volume"].std().rename("vol_std")
        vol_sum      = g_obs["total_volume"].sum().rename("vol_sum")
        vol_max      = g_obs["total_volume"].max().rename("vol_max")
        vol_cv       = (vol_std / (vol_mean + EPSILON)).rename("vol_cv")
        log_vol_mean = g_obs["log_volume"].mean().rename("log_vol_mean")
        log_vol_std  = g_obs["log_volume"].std().rename("log_vol_std")
        vol_last60   = df_obs[df_obs["seconds_in_bucket"] >= obs_seconds - 60].groupby("time_id")["total_volume"].mean().rename("vol_last_60s")
        vol_ratio    = (vol_last60 / (vol_mean + EPSILON)).rename("vol_ratio_60_full")
        vol_skew     = g_obs["total_volume"].skew().rename("vol_skew")

        # Signed volume / order flow imbalance
        signed_vol_sum = g_obs["signed_vol"].sum().rename("signed_vol_sum")
        abs_signed_vol = signed_vol_sum.abs().rename("abs_signed_vol")
        ofi            = (signed_vol_sum / (vol_sum + EPSILON)).rename("order_flow_imbalance")

        # ══════════════════════════════════════════════════════════════
        # J. VOLUME-WEIGHTED FEATURES
        # ══════════════════════════════════════════════════════════════
        vwap_ret   = (g_obs["vw_ret"].sum() / (g_obs["total_volume"].sum() + EPSILON)).rename("vwap_ret")
        vw_spread_ = (g_obs["vw_spread"].sum() / (g_obs["total_volume"].sum() + EPSILON)).rename("vw_spread")

        # ══════════════════════════════════════════════════════════════
        # K. PRICE PATTERN FEATURES
        # ══════════════════════════════════════════════════════════════
        wap_range = g_obs["wap"].apply(lambda x: np.log(x.max() / (x.min() + EPSILON))).rename("wap_range")
        wap_first = g_obs["wap"].first()
        wap_last  = g_obs["wap"].last()
        wap_drift = (_safe_log(wap_last) - _safe_log(wap_first)).rename("wap_drift")
        wap_cv    = (g_obs["wap"].std() / (g_obs["wap"].mean() + EPSILON)).rename("wap_cv")

        # ══════════════════════════════════════════════════════════════
        # L. QUIETNESS / ACTIVITY SIGNALS
        # ══════════════════════════════════════════════════════════════
        zero_ret_frac = g_obs["log_ret"].apply(lambda x: (x.abs() < 1e-10).mean()).rename("zero_ret_frac")
        wide_spread_frac = g_obs.apply(
            lambda g: (g["bid_ask_spread"] > 2 * g["bid_ask_spread"].median()).mean()
            if g["bid_ask_spread"].median() > 0 else 0.0
        ).rename("wide_spread_frac")
        high_vol_frac = g_obs.apply(
            lambda g: (g["total_volume"] > 2 * g["total_volume"].median()).mean()
            if g["total_volume"].median() > 0 else 0.0
        ).rename("high_vol_frac")

        # ══════════════════════════════════════════════════════════════
        # M. REALIZED SEMIVARIANCE
        # ══════════════════════════════════════════════════════════════
        def _semivar(grp):
            r = grp["log_ret"].values
            up   = np.sqrt(np.sum(r[r > 0] ** 2)) if np.any(r > 0) else 0.0
            down = np.sqrt(np.sum(r[r < 0] ** 2)) if np.any(r < 0) else 0.0
            return pd.Series({"rv_up": up, "rv_down": down})

        semi = g_obs.apply(_semivar)
        semi["log_rv_up"]    = _safe_log(semi["rv_up"])
        semi["log_rv_down"]  = _safe_log(semi["rv_down"])
        semi["rv_asymmetry"] = semi["rv_up"] / (semi["rv_down"] + EPSILON)
        semi["signed_jump"]  = semi["rv_up"] - semi["rv_down"]

        # ══════════════════════════════════════════════════════════════
        # N. REALIZED QUARTICITY
        # ══════════════════════════════════════════════════════════════
        rq     = g_obs["log_ret"].apply(lambda x: np.sum(x**4)).rename("realized_quarticity")
        log_rq = _safe_log(rq).rename("log_realized_quarticity")
        rq_rv_ratio = (rq / (rv_full**2 + EPSILON)).rename("rq_rv_ratio")

        # ══════════════════════════════════════════════════════════════
        # O. SPREAD-VOLUME INTERACTION
        # ══════════════════════════════════════════════════════════════
        sv_mean     = g_obs["spread_x_vol"].mean().rename("spread_vol_mean")
        log_sv_mean = _safe_log(sv_mean).rename("log_spread_vol_mean")

        # ══════════════════════════════════════════════════════════════
        # P. AMIHUD ILLIQUIDITY
        # ══════════════════════════════════════════════════════════════
        amihud = g_obs.apply(
            lambda g: (g["abs_ret"] / (g["total_volume"].clip(lower=1))).mean()
        ).rename("amihud_illiquidity")
        log_amihud = _safe_log(amihud).rename("log_amihud")

        # ══════════════════════════════════════════════════════════════
        # Q. KYLE'S LAMBDA (price impact proxy)
        # ══════════════════════════════════════════════════════════════
        kyle = g_obs.apply(
            lambda g: np.abs(np.corrcoef(g["log_ret"].values[1:],
                                          g["signed_vol"].values[1:])[0, 1])
            if len(g) > 2 else 0.0
        ).rename("kyle_lambda")

        # ══════════════════════════════════════════════════════════════
        # R. SPREAD-RETURN CORRELATION
        # ══════════════════════════════════════════════════════════════
        spread_ret_corr = g_obs.apply(
            lambda g: np.corrcoef(g["abs_ret"].values[1:],
                                   g["bid_ask_spread"].values[1:])[0, 1]
            if len(g) > 2 else 0.0
        ).rename("spread_ret_corr")

        # ══════════════════════════════════════════════════════════════
        # COMBINE
        # ══════════════════════════════════════════════════════════════
        features = pd.concat([
            # targets
            target_rv, log_target,
            # C: session RV
            rv_full, rv_last60, rv_last120, rv_last240, rv_first_h, rv_second_h,
            log_rv_full, log_rv_last60, log_rv_last120, log_rv_last240,
            log_rv_first_h, log_rv_second_h,
            # D: RV ratios
            rv_ratio_60, rv_ratio_120, rv_ratio_240, rv_ratio_halves,
            log_rv_ratio_60, log_rv_ratio_120, rv_accel,
            # E: bipower & jump
            bpv_df[["bpv", "jump", "log_bpv", "log_jump", "jump_ratio", "jump_frac"]],
            # F: return distribution
            ret_mean, ret_std, ret_skew, ret_kurt,
            ret_min, ret_max, ret_range, ret_iqr,
            abs_ret_mean, abs_ret_max, abs_ret_p95,
            # G: autocorrelation
            ret_ac1, ret_ac5, ret_ac10, abs_ac1, abs_ac5,
            # H: spread
            spread_mean, spread_std, spread_max, spread_min, spread_range, spread_cv,
            log_spread_mean, log_spread_std, spread_last60, spread_ratio, spread_skew,
            # I: volume
            vol_mean, vol_std, vol_sum, vol_max, vol_cv,
            log_vol_mean, log_vol_std, vol_last60, vol_ratio, vol_skew,
            signed_vol_sum, abs_signed_vol, ofi,
            # J: volume-weighted
            vwap_ret, vw_spread_,
            # K: price patterns
            wap_range, wap_drift, wap_cv,
            # L: quietness
            zero_ret_frac, wide_spread_frac, high_vol_frac,
            # M: semivariance
            semi[["rv_up", "rv_down", "log_rv_up", "log_rv_down", "rv_asymmetry", "signed_jump"]],
            # N: quarticity
            rq, log_rq, rq_rv_ratio,
            # O: spread-volume
            sv_mean, log_sv_mean,
            # P: amihud
            amihud, log_amihud,
            # Q: kyle
            kyle,
            # R: spread-return correlation
            spread_ret_corr,
        ], axis=1).reset_index()

        # Merge interval RV features (A + B)
        features = features.merge(interval_rv.reset_index(), on="time_id", how="left")

        features["stock_id"] = stock_name
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        return features

    except Exception as e:
        print(f"  FAILED {stock_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════
# STEP 3: PROCESS BOTH SPLITS
# ═════════════════════════════════════════════════════════════════════

def process_split(data_dir: str, time_ids: set,
                  split_name: str, n_jobs: int = -1) -> pd.DataFrame:
    """Run feature engineering for one split across all stock files."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"\n  {split_name}: {len(files)} stocks × {len(time_ids)} time_ids")
    t0 = _time.time()

    dfs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_one_stock)(p, time_ids)
        for p in files
    )

    result = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    elapsed = _time.time() - t0
    feat_cols = [c for c in result.columns if c not in ("time_id", "stock_id", "target_rv", "target_log_rv")]
    print(f"  {split_name} done: {result.shape[0]} rows, {len(feat_cols)} features, {elapsed:.1f}s")
    return result


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def run_preprocessing(data_dir: str,
                      output_dir: str,
                      test_ratio: float = TEST_RATIO,
                      seed: int = SEED,
                      n_jobs: int = -1):
    """
    Full pipeline: split → process → save.

    Returns (train_df, test_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stock Feature Engineering Pipeline")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Obs window : [0, {OBS_SECONDS})  Target: [{TARGET_START}, {TOTAL_LEN})")
    print("=" * 60)

    # Split
    print("\nStep 1 — Splitting time_ids …")
    train_ids, test_ids = split_time_ids(data_dir, test_ratio, seed)

    # Process each split
    print("\nStep 2 — Feature engineering …")
    train_df = process_split(data_dir, train_ids, "TRAIN", n_jobs)
    test_df  = process_split(data_dir, test_ids,  "TEST",  n_jobs)

    # Save
    print("\nStep 3 — Saving …")
    train_path = os.path.join(output_dir, "train.parquet")
    test_path  = os.path.join(output_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"  {train_path}  ({train_df.shape})")
    print(f"  {test_path}   ({test_df.shape})")

    # Summary
    feat_cols = [c for c in train_df.columns if c not in ("time_id", "stock_id", "target_rv", "target_log_rv")]
    print(f"\n{'=' * 60}")
    print(f"Done — {len(feat_cols)} features engineered")
    print(f"  Train: {train_df.shape[0]} rows ({len(train_ids)} time_ids)")
    print(f"  Test : {test_df.shape[0]} rows ({len(test_ids)} time_ids)")
    print(f"\n  Feature categories:")
    print(f"    A. Interval RV lags (30s buckets)  : rv_lag_*, log_rv_lag_*")
    print(f"    B. HAR aggregates                  : log_rv_short/medium/long, har_*, rv_trend_*")
    print(f"    C. Session RV (multiple windows)   : rv_full, rv_last_*, rv_*_half")
    print(f"    D. RV ratios & acceleration        : rv_ratio_*, rv_acceleration")
    print(f"    E. Bipower variation & jump         : bpv, jump, jump_ratio, jump_frac")
    print(f"    F. Return distribution             : ret_mean/std/skew/kurt/iqr/range")
    print(f"    G. Autocorrelation                 : ret_ac*, abs_ret_ac*")
    print(f"    H. Spread features                 : spread_mean/std/max/cv/ratio/skew")
    print(f"    I. Volume features                 : vol_mean/std/sum/cv/ratio/skew, ofi")
    print(f"    J. Volume-weighted                 : vwap_ret, vw_spread")
    print(f"    K. Price patterns                  : wap_range, wap_drift, wap_cv")
    print(f"    L. Quietness signals               : zero_ret_frac, wide_spread_frac")
    print(f"    M. Realized semivariance           : rv_up/down, rv_asymmetry, signed_jump")
    print(f"    N. Realized quarticity             : rq, log_rq, rq_rv_ratio")
    print(f"    O. Spread-volume interaction        : spread_vol_mean")
    print(f"    P. Amihud illiquidity              : amihud_illiquidity, log_amihud")
    print(f"    Q. Kyle's lambda                   : kyle_lambda")
    print(f"    R. Spread-return correlation        : spread_ret_corr")
    print(f"{'=' * 60}")

    return train_df, test_df


if __name__ == "__main__":
    data_dir   = "./stocks"
    output_dir = "./processed"
    test_ratio = 0.2
    seed       = 42
    n_jobs     = -1

    run_preprocessing(
        data_dir   = data_dir,
        output_dir = output_dir,
        test_ratio = test_ratio,
        seed       = seed,
        n_jobs     = n_jobs,
    )
