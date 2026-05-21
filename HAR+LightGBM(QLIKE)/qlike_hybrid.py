# qlike_lgbm.py
# Self-contained QLIKE-optimised HAR+LightGBM hybrid

import gc
import json
import warnings
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import optuna
import shap
import psutil

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths — change these three ────────────────────────────────────────────────
DENORM_DIR = Path(r"E:\Optiver\individual_book_train_denorm")
DATA_DIR   = Path(r"E:\Optiver\processed")
OUTPUT_DIR = Path(r"E:\Optiver\outputs")
# ─────────────────────────────────────────────────────────────────────────────

HAR_DIR    = Path(r"E:\Optiver\outputs\har_predictions")
FEAT_DIR   = Path(r"E:\Optiver\outputs\features")
ZSCORE_DIR = Path(r"E:\Optiver\outputs\zscore_stats")
MODEL_DIR  = Path(r"E:\Optiver\outputs\stock_lgbm_qlike")
PRED_DIR   = Path(r"E:\Optiver\outputs\stock_lgbm_qlike")

for d in [HAR_DIR, FEAT_DIR, ZSCORE_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Window config ─────────────────────────────────────────────────────────────
OBS_END      = 480
TARGET_START = 480
TARGET_END   = 600
BUCKET_SIZE  = 120
N_BUCKETS    = OBS_END // BUCKET_SIZE
EPS          = 1e-10
RV_FLOOR     = 1e-4

# ── Model config ──────────────────────────────────────────────────────────────
SEED            = 42
N_INNER_FOLDS   = 5
N_OPTUNA_TRIALS = 10
N_ROUNDS        = 5000
EARLY_STOPPING  = 50
N_JOBS          = 4
TARGET_COL      = "har_residual"

np.random.seed(SEED)

# ── Fold paths ────────────────────────────────────────────────────────────────
FOLD_PATHS = [(DATA_DIR / f"fold_{i}" / "train.parquet",
               DATA_DIR / f"fold_{i}" / "test.parquet")
              for i in range(5)]

# ── Cluster map ───────────────────────────────────────────────────────────────
STOCK_CLUSTER_MAP = {
    0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1,
    10:1, 11:1, 13:0, 14:1, 15:1, 16:1, 17:1, 18:1, 19:1, 20:1,
    21:0, 22:1, 23:1, 26:1, 27:1, 28:1, 29:1, 30:1, 31:2, 32:0,
    33:1, 34:1, 35:2, 36:1, 37:1, 38:1, 39:1, 40:1, 41:0, 42:1,
    43:1, 44:2, 46:0, 47:0, 48:1, 50:1, 51:1, 52:1, 53:1, 55:1,
    56:1, 58:1, 59:1, 60:1, 61:1, 62:1, 63:2, 64:1, 66:1, 67:1,
    68:1, 69:1, 70:1, 72:1, 73:1, 74:2, 75:1, 76:1, 77:0, 78:1,
    80:1, 81:1, 82:1, 83:1, 84:1, 85:2, 86:2, 87:1, 88:1, 89:2,
    90:1, 93:1, 94:1, 95:1, 96:1, 97:1, 98:1, 99:2, 100:1, 101:1,
    102:1, 103:1, 104:1, 105:2, 107:1, 108:0, 109:1, 110:1, 111:0,
    112:1, 113:1, 114:1, 115:1, 116:1, 118:1, 119:2, 120:1, 122:1,
    123:1, 124:2, 125:0, 126:1,
}

# ── Selected features (Cell 7 output) ────────────────────────────────────────
SELECTED_FEATURES = [
    "spread_last60", "log_rv_last60", "rv_lag_13", "ret_kurt",
    "log_rv_lag_14", "log_rv_ratio_60", "log_rv_lag_15", "rv_ratio_240",
    "ret_iqr", "rv_lag_std", "jump_ratio", "log_vol_mean",
    "log_rv_ratio_120", "spread_trend", "log_rv_lag_16", "vol_max",
    "har_short_long", "rv_ratio_halves", "har_med_long", "log_rv_lag_12",
    "rv_trend_slope",
]
ZSCORE_COLS    = [f"{f}_zscore" for f in SELECTED_FEATURES]
FINAL_FEATURES = SELECTED_FEATURES + ZSCORE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_stock_id(fname):
    return int(Path(fname).stem.replace("stock_", ""))

def get_fold_time_ids(fold_idx):
    train_path, test_path = FOLD_PATHS[fold_idx]
    train_tids = set(pd.read_parquet(train_path, columns=["time_id"])["time_id"].unique())
    test_tids  = set(pd.read_parquet(test_path,  columns=["time_id"])["time_id"].unique())
    return train_tids, test_tids

def safe_log(x):
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), EPS))

def rmspe(y_true, y_pred):
    y_true = np.maximum(y_true, EPS)
    y_pred = np.maximum(y_pred, EPS)
    return float(np.sqrt(np.mean(((y_true - y_pred) / y_true)**2)))

def qlike(y_true, y_pred):
    y_true = np.maximum(y_true, EPS)
    y_pred = np.maximum(y_pred, EPS)
    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING (mirrors Cell 3)
# ═══════════════════════════════════════════════════════════════════════════════

def vec_autocorr_lag1(values, tid_mapped, n_tids):
    out = np.zeros(n_tids, dtype=np.float64)
    for i in range(n_tids):
        mask = tid_mapped == i
        v = values[mask]
        if len(v) < 3:
            continue
        x, y   = v[:-1], v[1:]
        mx, my = x.mean(), y.mean()
        num    = ((x - mx) * (y - my)).sum()
        den    = np.sqrt(((x - mx)**2).sum() * ((y - my)**2).sum())
        out[i] = num / (den + EPS)
    return out

def vec_spread_trend(seconds, spread, tid_mapped, n_tids):
    out = np.zeros(n_tids, dtype=np.float64)
    for i in range(n_tids):
        mask   = tid_mapped == i
        t      = seconds[mask].astype(np.float64)
        s      = spread[mask].astype(np.float64)
        if len(t) < 3:
            continue
        t_mean = t.mean()
        s_mean = s.mean()
        var_t  = ((t - t_mean)**2).sum()
        if var_t < EPS:
            continue
        out[i] = ((t - t_mean) * (s - s_mean)).sum() / var_t
    return out

def vec_bpv(log_ret, tid_mapped, n_tids):
    bpv_out        = np.zeros(n_tids)
    jump_out       = np.zeros(n_tids)
    log_bpv_out    = np.full(n_tids, np.log(RV_FLOOR))
    log_jump_out   = np.full(n_tids, np.log(RV_FLOOR))
    jump_ratio_out = np.zeros(n_tids)
    jump_frac_out  = np.zeros(n_tids)
    for i in range(n_tids):
        mask  = tid_mapped == i
        r     = log_ret[mask]
        if len(r) < 2:
            continue
        rv    = float(np.sum(r**2))
        bpv   = float((np.pi / 2) * np.mean(np.abs(r[1:]) * np.abs(r[:-1])) * len(r))
        jump  = max(rv - bpv, 0.0)
        bpv_out[i]        = np.sqrt(bpv)
        jump_out[i]       = np.sqrt(jump)
        log_bpv_out[i]    = float(safe_log([max(bpv,  RV_FLOOR)])[0])
        log_jump_out[i]   = float(safe_log([max(jump, RV_FLOOR)])[0])
        jump_ratio_out[i] = jump / (rv + EPS)
        jump_frac_out[i]  = 1.0 if jump > 0 else 0.0
    return pd.DataFrame({
        "bpv": bpv_out, "jump": jump_out,
        "log_bpv": log_bpv_out, "log_jump": log_jump_out,
        "jump_ratio": jump_ratio_out, "jump_frac": jump_frac_out,
    })

def vec_semi(log_ret, tid_mapped, n_tids):
    rv_up_out   = np.zeros(n_tids)
    rv_down_out = np.zeros(n_tids)
    asym_out    = np.zeros(n_tids)
    for i in range(n_tids):
        mask     = tid_mapped == i
        r        = log_ret[mask]
        rv_up    = float(np.sum(r[r > 0]**2))
        rv_down  = float(np.sum(r[r < 0]**2))
        rv_tot   = rv_up + rv_down + EPS
        rv_up_out[i]   = np.sqrt(rv_up)
        rv_down_out[i] = np.sqrt(rv_down)
        asym_out[i]    = (rv_up - rv_down) / rv_tot
    return pd.DataFrame({
        "rv_up": rv_up_out, "rv_down": rv_down_out,
        "log_rv_up":    np.log(np.clip(rv_up_out,   RV_FLOOR, None)),
        "log_rv_down":  np.log(np.clip(rv_down_out, RV_FLOOR, None)),
        "rv_asymmetry": asym_out,
    })

def vec_spread_rv_corr(spread, abs_ret, tid_mapped, n_tids):
    out = np.zeros(n_tids, dtype=np.float64)
    for i in range(n_tids):
        mask = tid_mapped == i
        s    = spread[mask][1:]
        r    = abs_ret[mask][1:]
        if len(s) < 3:
            continue
        c      = np.corrcoef(s, r)
        out[i] = float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0
    return out

def process_one_stock(stock_df):
    """Full feature engineering for one stock — mirrors Cell 3 exactly."""
    stock_id = stock_df["stock_id"].iloc[0]
    df_obs   = stock_df[stock_df["seconds_in_bucket"] < OBS_END].copy()
    if df_obs.empty:
        return pd.DataFrame()

    df_obs = df_obs.sort_values(["time_id", "seconds_in_bucket"])
    df_obs["log_wap"]    = safe_log(df_obs["wap"].values)
    df_obs["log_ret"]    = df_obs.groupby("time_id")["log_wap"].diff().fillna(0)
    df_obs["abs_ret"]    = df_obs["log_ret"].abs()
    df_obs["sq_ret"]     = df_obs["log_ret"] ** 2
    df_obs["log_spread"] = safe_log(df_obs["bid_ask_spread"].values)
    df_obs["log_volume"] = safe_log(df_obs["total_volume"].clip(lower=1).values)
    df_obs["signed_vol"] = np.sign(df_obs["log_ret"].values) * df_obs["total_volume"].values
    df_obs["vw_ret"]     = df_obs["log_ret"].values * df_obs["total_volume"].values
    df_obs["vw_spread"]  = df_obs["bid_ask_spread"].values * df_obs["total_volume"].values

    time_ids   = np.sort(df_obs["time_id"].unique())
    n_tids     = len(time_ids)
    tid_index  = {t: i for i, t in enumerate(time_ids)}
    tid_mapped = np.array([tid_index[t] for t in df_obs["time_id"].values], dtype=np.int32)
    g_obs      = df_obs.groupby("time_id")

    # Interval RV lags
    n_intervals        = OBS_END // 30
    df_obs["interval"] = (df_obs["seconds_in_bucket"] // 30).clip(upper=n_intervals - 1).astype(int)
    interval_rv = (
        df_obs.groupby(["time_id", "interval"])["log_ret"]
        .apply(lambda x: np.sqrt(np.sum(x**2)))
        .unstack(level="interval")
    ).fillna(0)
    interval_rv.columns = [f"rv_lag_{int(c)+1}" for c in interval_rv.columns]
    for c in list(interval_rv.columns):
        interval_rv[f"log_{c}"] = safe_log(interval_rv[c].values)

    log_rv_cols = sorted([c for c in interval_rv.columns if c.startswith("log_rv_lag_")])
    n_lags      = len(log_rv_cols)
    short_cols  = [f"log_rv_lag_{i}" for i in range(n_lags - 2, n_lags + 1) if f"log_rv_lag_{i}" in interval_rv]
    med_cols    = [f"log_rv_lag_{i}" for i in range(n_lags - 7, n_lags + 1) if f"log_rv_lag_{i}" in interval_rv]
    interval_rv["log_rv_short"]   = interval_rv[short_cols].mean(axis=1) if short_cols else 0
    interval_rv["log_rv_medium"]  = interval_rv[med_cols].mean(axis=1)   if med_cols  else 0
    interval_rv["log_rv_long"]    = interval_rv[log_rv_cols].mean(axis=1)
    interval_rv["har_short_long"] = interval_rv["log_rv_short"]  - interval_rv["log_rv_long"]
    interval_rv["har_short_med"]  = interval_rv["log_rv_short"]  - interval_rv["log_rv_medium"]
    interval_rv["har_med_long"]   = interval_rv["log_rv_medium"] - interval_rv["log_rv_long"]
    if n_lags >= 3:
        X_idx   = np.arange(n_lags, dtype=np.float64)
        X_mean  = X_idx.mean()
        X_var   = ((X_idx - X_mean)**2).sum()
        rv_v    = interval_rv[log_rv_cols].values
        Y_mean  = rv_v.mean(axis=1, keepdims=True)
        interval_rv["rv_trend_slope"] = ((rv_v - Y_mean) * (X_idx - X_mean)).sum(axis=1) / (X_var + EPS)
        interval_rv["rv_lag_std"]     = interval_rv[log_rv_cols].std(axis=1)

    def make_log_rv(series, name):
        return pd.Series(safe_log(series.reindex(time_ids).fillna(0).values), index=time_ids, name=name)

    rv_full     = g_obs["sq_ret"].sum().apply(np.sqrt).rename("rv_full")
    rv_last60   = df_obs[df_obs["seconds_in_bucket"] >= 420].groupby("time_id")["sq_ret"].sum().apply(np.sqrt).rename("rv_last60")
    rv_last120  = df_obs[df_obs["seconds_in_bucket"] >= 360].groupby("time_id")["sq_ret"].sum().apply(np.sqrt).rename("rv_last120")
    rv_last240  = df_obs[df_obs["seconds_in_bucket"] >= 240].groupby("time_id")["sq_ret"].sum().apply(np.sqrt).rename("rv_last240")
    rv_first_h  = df_obs[df_obs["seconds_in_bucket"] < 240].groupby("time_id")["sq_ret"].sum().apply(np.sqrt).rename("rv_first_half")
    rv_second_h = df_obs[df_obs["seconds_in_bucket"] >= 240].groupby("time_id")["sq_ret"].sum().apply(np.sqrt).rename("rv_second_half")

    log_rv_full     = make_log_rv(rv_full,     "log_rv_full")
    log_rv_last60   = make_log_rv(rv_last60,   "log_rv_last60")
    log_rv_last120  = make_log_rv(rv_last120,  "log_rv_last120")
    log_rv_last240  = make_log_rv(rv_last240,  "log_rv_last240")
    log_rv_first_h  = make_log_rv(rv_first_h,  "log_rv_first_half")
    log_rv_second_h = make_log_rv(rv_second_h, "log_rv_second_half")

    rv_full_s     = rv_full.reindex(time_ids).fillna(0)
    rv_first_h_s  = rv_first_h.reindex(time_ids).fillna(0)
    rv_second_h_s = rv_second_h.reindex(time_ids).fillna(0)

    rv_ratio_60     = (rv_last60.reindex(time_ids).fillna(0)  / rv_full_s.clip(lower=EPS)).rename("rv_ratio_60")
    rv_ratio_120    = (rv_last120.reindex(time_ids).fillna(0) / rv_full_s.clip(lower=EPS)).rename("rv_ratio_120")
    rv_ratio_240    = (rv_last240.reindex(time_ids).fillna(0) / rv_full_s.clip(lower=EPS)).rename("rv_ratio_240")
    rv_ratio_halves = (rv_second_h_s / rv_first_h_s.clip(lower=EPS)).rename("rv_ratio_halves")
    log_rv_ratio_60  = pd.Series(safe_log(rv_ratio_60.values),  index=time_ids, name="log_rv_ratio_60")
    log_rv_ratio_120 = pd.Series(safe_log(rv_ratio_120.values), index=time_ids, name="log_rv_ratio_120")
    rv_accel         = (rv_second_h_s - rv_first_h_s).rename("rv_accel")

    bpv_df       = vec_bpv(df_obs["log_ret"].values, tid_mapped, n_tids)
    bpv_df.index = time_ids

    ret_mean     = g_obs["log_ret"].mean().rename("ret_mean")
    ret_std      = g_obs["log_ret"].std().rename("ret_std")
    ret_skew     = g_obs["log_ret"].skew().rename("ret_skew")
    ret_kurt     = g_obs["log_ret"].apply(pd.Series.kurt).rename("ret_kurt")
    ret_min      = g_obs["log_ret"].min().rename("ret_min")
    ret_max      = g_obs["log_ret"].max().rename("ret_max")
    ret_range    = (ret_max - ret_min).rename("ret_range")
    ret_iqr      = (g_obs["log_ret"].quantile(0.75) - g_obs["log_ret"].quantile(0.25)).rename("ret_iqr")
    abs_ret_mean = g_obs["abs_ret"].mean().rename("abs_ret_mean")
    abs_ret_max  = g_obs["abs_ret"].max().rename("abs_ret_max")
    abs_ret_p95  = g_obs["abs_ret"].quantile(0.95).rename("abs_ret_p95")

    ret_ac1     = pd.Series(vec_autocorr_lag1(df_obs["log_ret"].values, tid_mapped, n_tids), index=time_ids, name="ret_ac1")
    abs_ret_ac1 = pd.Series(vec_autocorr_lag1(df_obs["abs_ret"].values, tid_mapped, n_tids), index=time_ids, name="abs_ret_ac1")

    spread_mean   = g_obs["bid_ask_spread"].mean().rename("spread_mean")
    spread_std    = g_obs["bid_ask_spread"].std().rename("spread_std")
    spread_max    = g_obs["bid_ask_spread"].max().rename("spread_max")
    spread_min    = g_obs["bid_ask_spread"].min().rename("spread_min")
    spread_range  = (spread_max - spread_min).rename("spread_range")
    spread_cv     = (spread_std / spread_mean.clip(lower=EPS)).rename("spread_cv")
    spread_last60 = df_obs[df_obs["seconds_in_bucket"] >= 420].groupby("time_id")["bid_ask_spread"].mean().rename("spread_last60")
    spread_ratio  = (spread_last60.reindex(time_ids).fillna(0) / spread_mean.clip(lower=EPS)).rename("spread_ratio")
    spread_skew   = g_obs["bid_ask_spread"].skew().rename("spread_skew")
    log_spread_mean     = g_obs["log_spread"].mean().rename("log_spread_mean")
    log_spread_std      = g_obs["log_spread"].std().rename("log_spread_std")
    spread_trend        = pd.Series(vec_spread_trend(df_obs["seconds_in_bucket"].values, df_obs["bid_ask_spread"].values, tid_mapped, n_tids), index=time_ids, name="spread_trend")
    spread_first60      = df_obs[df_obs["seconds_in_bucket"] < 60].groupby("time_id")["bid_ask_spread"].mean()
    spread_change_ratio = (spread_last60.reindex(time_ids).fillna(0) / spread_first60.reindex(time_ids).fillna(EPS).clip(lower=EPS)).rename("spread_change_ratio")
    spread_rv_corr      = pd.Series(vec_spread_rv_corr(df_obs["bid_ask_spread"].values, df_obs["abs_ret"].values, tid_mapped, n_tids), index=time_ids, name="spread_rv_corr")

    vol_mean     = g_obs["total_volume"].mean().rename("vol_mean")
    vol_std      = g_obs["total_volume"].std().rename("vol_std")
    vol_sum      = g_obs["total_volume"].sum().rename("vol_sum")
    vol_max      = g_obs["total_volume"].max().rename("vol_max")
    vol_cv       = (vol_std / vol_mean.clip(lower=EPS)).rename("vol_cv")
    log_vol_mean = g_obs["log_volume"].mean().rename("log_vol_mean")
    log_vol_std  = g_obs["log_volume"].std().rename("log_vol_std")
    vol_last60   = df_obs[df_obs["seconds_in_bucket"] >= 420].groupby("time_id")["total_volume"].mean().rename("vol_last60")
    vol_ratio    = (vol_last60.reindex(time_ids).fillna(0) / vol_mean.clip(lower=EPS)).rename("vol_ratio")
    vol_skew     = g_obs["total_volume"].skew().rename("vol_skew")
    ofi          = g_obs["signed_vol"].sum().rename("ofi")

    tot_vol   = g_obs["total_volume"].sum().clip(lower=EPS)
    vwap_ret  = g_obs["vw_ret"].sum().div(tot_vol).rename("vwap_ret")
    vw_spread = g_obs["vw_spread"].sum().div(tot_vol).rename("vw_spread")

    semi_df       = vec_semi(df_obs["log_ret"].values, tid_mapped, n_tids)
    semi_df.index = time_ids

    df_obs["bucket_120"] = (df_obs["seconds_in_bucket"] // BUCKET_SIZE).clip(upper=N_BUCKETS - 1).astype(int)
    bucket_frames = []
    for b in range(N_BUCKETS):
        bkt = (df_obs[df_obs["bucket_120"] == b]
               .groupby("time_id")["sq_ret"]
               .sum().apply(np.sqrt)
               .rename(f"past_rv_bkt{b}"))
        bucket_frames.append(bkt)
    bucket_rv_df = pd.concat(bucket_frames, axis=1).reindex(time_ids).fillna(0)
    for b in range(N_BUCKETS):
        bucket_rv_df[f"past_log_rv_bkt{b}"] = safe_log(bucket_rv_df[f"past_rv_bkt{b}"].clip(lower=RV_FLOOR).values)
    bucket_rv_df["rv_trend_ratio"] = (bucket_rv_df[f"past_rv_bkt{N_BUCKETS-1}"] / bucket_rv_df["past_rv_bkt0"].clip(lower=EPS))

    features = pd.concat([
        rv_full, rv_last60, rv_last120, rv_last240, rv_first_h, rv_second_h,
        log_rv_full, log_rv_last60, log_rv_last120, log_rv_last240, log_rv_first_h, log_rv_second_h,
        rv_ratio_60, rv_ratio_120, rv_ratio_240, rv_ratio_halves,
        log_rv_ratio_60, log_rv_ratio_120, rv_accel,
        bpv_df, ret_mean, ret_std, ret_skew, ret_kurt,
        ret_min, ret_max, ret_range, ret_iqr,
        abs_ret_mean, abs_ret_max, abs_ret_p95,
        ret_ac1, abs_ret_ac1,
        spread_mean, spread_std, spread_max, spread_min, spread_range,
        spread_cv, log_spread_mean, log_spread_std,
        spread_last60, spread_ratio, spread_skew,
        spread_trend, spread_change_ratio, spread_rv_corr,
        vol_mean, vol_std, vol_sum, vol_max, vol_cv,
        log_vol_mean, log_vol_std, vol_last60, vol_ratio, vol_skew, ofi,
        vwap_ret, vw_spread, semi_df,
    ], axis=1)
    features.index.name = "time_id"
    features = features.reset_index()
    features = features.merge(interval_rv.reset_index(),  on="time_id", how="left")
    features = features.merge(bucket_rv_df.reset_index(), on="time_id", how="left")
    features["stock_id"] = stock_id
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HAR FITTING (mirrors Cell 5)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rv_window(group, sec_min, sec_max):
    mask    = (group["seconds_in_bucket"] >= sec_min) & \
              (group["seconds_in_bucket"] <  sec_max)
    sub     = group[mask]
    if sub.empty:
        return RV_FLOOR
    log_ret = np.diff(np.log(np.maximum(sub["wap"].values, EPS)))
    return float(max(np.sqrt(np.sum(log_ret**2)), RV_FLOOR))

def make_har_X(df):
    return np.column_stack([
        np.ones(len(df)),
        df["log_rv_in"].values,
        df["log_rv_last_window"].values,
        df["log_rv_ratio"].values,
    ])

def fit_ols(X, y):
    try:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

def fit_predict_har_stock(stock_file, fold_idx, train_tids, test_tids):
    stock_id   = parse_stock_id(stock_file)
    ckpt_train = HAR_DIR / f"har_fold_{fold_idx}_stock_{stock_id}_train.parquet"
    ckpt_test  = HAR_DIR / f"har_fold_{fold_idx}_stock_{stock_id}_test.parquet"

    if ckpt_train.exists() and ckpt_test.exists():
        return pd.read_parquet(ckpt_train), pd.read_parquet(ckpt_test)

    df             = pd.read_csv(stock_file)
    df["stock_id"] = stock_id
    records        = []

    for tid, group in df.groupby("time_id"):
        rv_in          = compute_rv_window(group, 0,   480)
        rv_last_window = compute_rv_window(group, 360, 480)
        rv_fut         = compute_rv_window(group, 480, 600)
        records.append({
            "time_id":            tid,
            "stock_id":           stock_id,
            "rv_in":              rv_in,
            "rv_last_window":     rv_last_window,
            "rv_fut":             rv_fut,
            "log_rv_in":          np.log(max(rv_in,          RV_FLOOR)),
            "log_rv_last_window": np.log(max(rv_last_window, RV_FLOOR)),
            "log_rv_ratio":       np.log(max(rv_last_window, RV_FLOOR)) - np.log(max(rv_in, RV_FLOOR)),
            "log_rv_fut":         np.log(max(rv_fut,         RV_FLOOR)),
        })

    rv_df    = pd.DataFrame(records)
    train_rv = rv_df[rv_df["time_id"].isin(train_tids)].copy().reset_index(drop=True)
    test_rv  = rv_df[rv_df["time_id"].isin(test_tids)].copy().reset_index(drop=True)

    if len(train_rv) < N_INNER_FOLDS * 2 or test_rv.empty:
        return pd.DataFrame(), pd.DataFrame()

    train_rv["har_pred_log_rv"] = np.nan
    train_rv["har_residual"]    = np.nan

    gkf = GroupKFold(n_splits=N_INNER_FOLDS)
    for inner_train_idx, inner_val_idx in gkf.split(train_rv, groups=train_rv["time_id"]):
        beta = fit_ols(make_har_X(train_rv.iloc[inner_train_idx]),
                       train_rv.iloc[inner_train_idx]["log_rv_fut"].values)
        if beta is None:
            continue
        preds = make_har_X(train_rv.iloc[inner_val_idx]) @ beta
        train_rv.loc[inner_val_idx, "har_pred_log_rv"] = preds
        train_rv.loc[inner_val_idx, "har_residual"]    = (
            train_rv.iloc[inner_val_idx]["log_rv_fut"].values - preds
        )

    train_rv["har_pred_rv"] = np.exp(train_rv["har_pred_log_rv"].fillna(0))
    train_rv["fold"]        = fold_idx
    train_rv                = train_rv.dropna(subset=["har_residual"])

    beta_full = fit_ols(make_har_X(train_rv), train_rv["log_rv_fut"].values)
    if beta_full is None:
        return pd.DataFrame(), pd.DataFrame()

    test_rv["har_pred_log_rv"] = make_har_X(test_rv) @ beta_full
    test_rv["har_residual"]    = test_rv["log_rv_fut"] - test_rv["har_pred_log_rv"]
    test_rv["har_pred_rv"]     = np.exp(test_rv["har_pred_log_rv"])
    test_rv["fold"]            = fold_idx

    out_cols  = ["time_id", "stock_id", "fold", "log_rv_fut",
                 "har_pred_log_rv", "har_pred_rv", "har_residual"]
    train_out = train_rv[out_cols].copy()
    test_out  = test_rv[out_cols].copy()

    train_out.to_parquet(ckpt_train, index=False)
    test_out.to_parquet(ckpt_test,   index=False)

    del df, rv_df, train_rv, test_rv
    gc.collect()
    return train_out, test_out

def build_har_fold(fold_idx, train_tids, test_tids, stock_files):
    train_out = HAR_DIR / f"har_fold_{fold_idx}_train.parquet"
    test_out  = HAR_DIR / f"har_fold_{fold_idx}_test.parquet"

    if train_out.exists() and test_out.exists():
        print(f"  Fold {fold_idx} HAR already complete — loading from disk")
        return pd.read_parquet(train_out), pd.read_parquet(test_out)

    results    = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=5)(
        delayed(fit_predict_har_stock)(sf, fold_idx, train_tids, test_tids)
        for sf in stock_files
    )
    train_df   = pd.concat([r[0] for r in results if len(r[0]) > 0], ignore_index=True)
    test_df    = pd.concat([r[1] for r in results if len(r[1]) > 0], ignore_index=True)

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out,   index=False)
    print(f"  Fold {fold_idx} HAR saved — train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE RUNNER + ZSCORES (mirrors Cells 4 and 9)
# ═══════════════════════════════════════════════════════════════════════════════

def build_features_fold(fold_idx, train_tids, test_tids, stock_files):
    train_out = FEAT_DIR / f"features_fold_{fold_idx}_train.parquet"
    test_out  = FEAT_DIR / f"features_fold_{fold_idx}_test.parquet"

    if train_out.exists() and test_out.exists():
        print(f"  Fold {fold_idx} features already complete — loading from disk")
        return pd.read_parquet(train_out), pd.read_parquet(test_out)

    def process_and_collect(stock_file):
        stock_id       = parse_stock_id(stock_file)
        df             = pd.read_csv(stock_file)
        df["stock_id"] = stock_id
        train_raw      = df[df["time_id"].isin(train_tids)]
        test_raw       = df[df["time_id"].isin(test_tids)]
        if train_raw.empty or test_raw.empty:
            return pd.DataFrame(), pd.DataFrame()
        return process_one_stock(train_raw), process_one_stock(test_raw)

    results   = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=5)(
        delayed(process_and_collect)(sf) for sf in stock_files
    )
    train_df  = pd.concat([r[0] for r in results if len(r[0]) > 0], ignore_index=True)
    test_df   = pd.concat([r[1] for r in results if len(r[1]) > 0], ignore_index=True)

    train_df["cluster_id"] = train_df["stock_id"].map(STOCK_CLUSTER_MAP)
    test_df["cluster_id"]  = test_df["stock_id"].map(STOCK_CLUSTER_MAP)

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out,   index=False)
    print(f"  Fold {fold_idx} features saved — train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df

def apply_zscores(fold_idx, split, feat_df):
    out_path   = FEAT_DIR / f"features_fold_{fold_idx}_{split}_with_zscores.parquet"
    stats_path = ZSCORE_DIR / f"zscore_stats_fold_{fold_idx}_{split}.parquet"

    if out_path.exists():
        print(f"  Fold {fold_idx} {split} zscores already applied — loading")
        return pd.read_parquet(out_path)

    means = feat_df.groupby("time_id")[SELECTED_FEATURES].mean()
    stds  = feat_df.groupby("time_id")[SELECTED_FEATURES].std().clip(lower=EPS)
    means.columns = [f"{c}_mean" for c in means.columns]
    stds.columns  = [f"{c}_std"  for c in stds.columns]
    stats = pd.concat([means, stds], axis=1).reset_index()
    stats.to_parquet(stats_path, index=False)

    df = feat_df.merge(stats, on="time_id", how="left")
    for f in SELECTED_FEATURES:
        df[f"{f}_zscore"] = (df[f] - df[f"{f}_mean"]) / df[f"{f}_std"]
    drop_cols = [f"{f}_mean" for f in SELECTED_FEATURES] + [f"{f}_std" for f in SELECTED_FEATURES]
    df        = df.drop(columns=drop_cols)
    df        = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    df.to_parquet(out_path, index=False)
    print(f"  Fold {fold_idx} {split} zscores applied — {df.shape[0]} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QLIKE OBJECTIVE AND FEVAL (mirrors Cell 10 + Cell 12)
# ═══════════════════════════════════════════════════════════════════════════════

def qlike_obj(preds, dataset):
    """Custom QLIKE gradient and hessian — passed as params['objective']."""
    labels  = dataset.get_label()
    anchor  = dataset.anchor_log_rv
    pred_rv = np.exp(np.clip(preds + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp(labels + anchor).clip(min=EPS)
    ratio   = true_rv / pred_rv
    grad    = -(ratio - 1)
    hess    = np.clip(ratio, 0.01, 100)
    return grad, hess

def anchor_qlike_feval(y_pred, dtrain):
    """Anchor-aware QLIKE feval for early stopping and Optuna scoring."""
    y_residual  = dtrain.get_label()
    anchor      = dtrain.anchor_log_rv
    pred_log_rv = np.clip(y_pred + anchor, -20, 5)
    true_log_rv = y_residual + anchor
    pred_rv     = np.exp(pred_log_rv).clip(min=EPS)
    true_rv     = np.exp(true_log_rv).clip(min=EPS)
    score       = float(np.mean(true_rv / pred_rv - np.log(true_rv / pred_rv) - 1))
    return "qlike", score, False

def load_fold_with_zscores(fold_idx, split):
    feat_path = FEAT_DIR / f"features_fold_{fold_idx}_{split}_with_zscores.parquet"
    df        = pd.read_parquet(feat_path)
    df["cluster_id"] = df["stock_id"].map(STOCK_CLUSTER_MAP)
    har_path  = HAR_DIR / f"har_fold_{fold_idx}_{split}.parquet"
    har_df    = pd.read_parquet(har_path)
    df        = df.merge(har_df, on=["time_id", "stock_id"], how="inner")
    true_rv   = np.exp(df["log_rv_fut"].values).clip(min=RV_FLOOR)
    raw_w     = 1.0 / true_rv
    weight_cap = np.percentile(raw_w, 99)
    df["sample_weight"] = np.minimum(raw_w, weight_cap)
    return df

def make_optuna_objective(X_train, y_train, w_train, anchor_train, time_ids_train):
    def objective(trial):
        params = {
            "objective":         qlike_obj,
            "metric":            "None",
            "boosting_type":     "gbdt",
            "verbosity":         -1,
            "seed":              SEED,
            "learning_rate":     trial.suggest_float("learning_rate",      0.01,  0.15,  log=True),
            "num_leaves":        trial.suggest_int(  "num_leaves",         15,    200),
            "max_depth":         trial.suggest_int(  "max_depth",          3,     12),
            "min_child_samples": trial.suggest_int(  "min_child_samples",  10,    100),
            "feature_fraction":  trial.suggest_float("feature_fraction",   0.4,   1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction",   0.4,   1.0),
            "bagging_freq":      trial.suggest_int(  "bagging_freq",       1,     7),
            "reg_alpha":         trial.suggest_float("reg_alpha",          1e-3,  10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",         1e-3,  10.0, log=True),
        }
        gkf    = GroupKFold(n_splits=5)
        scores = []
        for tr_idx, va_idx in gkf.split(X_train, groups=time_ids_train):
            d_tr = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx],
                               weight=w_train[tr_idx],
                               feature_name=FINAL_FEATURES, free_raw_data=False)
            d_va = lgb.Dataset(X_train[va_idx], label=y_train[va_idx],
                               feature_name=FINAL_FEATURES,
                               reference=d_tr, free_raw_data=False)
            d_tr.anchor_log_rv = anchor_train[tr_idx]
            d_va.anchor_log_rv = anchor_train[va_idx]
            m = lgb.train(
                params, d_tr,
                num_boost_round=N_ROUNDS,
                valid_sets=[d_va],
                feval=anchor_qlike_feval,
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            va_pred     = m.predict(X_train[va_idx])
            pred_log_rv = np.clip(va_pred + anchor_train[va_idx], -20, 5)
            true_log_rv = y_train[va_idx] + anchor_train[va_idx]
            pred_rv     = np.exp(pred_log_rv).clip(min=EPS)
            true_rv     = np.exp(true_log_rv).clip(min=EPS)
            scores.append(float(np.mean(true_rv / pred_rv - np.log(true_rv / pred_rv) - 1)))
            del d_tr, d_va, m
            gc.collect()
        return float(np.mean(scores))
    return objective

def train_qlike_fold(fold_idx):
    pred_path  = PRED_DIR  / f"predictions_fold_{fold_idx}.parquet"
    model_path = MODEL_DIR / f"model_fold_{fold_idx}.txt"
    params_path = MODEL_DIR / f"best_params_fold_{fold_idx}.json"

    if pred_path.exists() and model_path.exists():
        print(f"  Fold {fold_idx} already complete — loading from disk")
        return {"predictions": pd.read_parquet(pred_path)}

    available_gb = psutil.virtual_memory().available / 1e9
    print(f"  RAM before fold {fold_idx}: {available_gb:.2f} GB")
    if available_gb < 1.0:
        raise MemoryError(f"Only {available_gb:.2f} GB available")

    train_df     = load_fold_with_zscores(fold_idx, "train")
    test_df      = load_fold_with_zscores(fold_idx, "test")

    X_train      = train_df[FINAL_FEATURES].values.astype(np.float32)
    y_train      = train_df[TARGET_COL].values.astype(np.float32)
    w_train      = train_df["sample_weight"].values.astype(np.float32)
    anchor_train = train_df["har_pred_log_rv"].values.astype(np.float32)
    time_ids_tr  = train_df["time_id"].values

    X_test       = test_df[FINAL_FEATURES].values.astype(np.float32)
    anchor_test  = test_df["har_pred_log_rv"].values.astype(np.float32)

    # Optuna
    print(f"  Running Optuna ({N_OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(
        make_optuna_objective(X_train, y_train, w_train, anchor_train, time_ids_tr),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False,
    )

    serialisable_params = dict(study.best_params)
    best_params         = dict(study.best_params)
    best_params["objective"]     = qlike_obj
    best_params["metric"]        = "None"
    best_params["boosting_type"] = "gbdt"
    best_params["verbosity"]     = -1
    best_params["seed"]          = SEED

    print(f"  Best inner QLIKE: {study.best_value:.6f}")

    # Final fit with 90/10 holdout
    rng    = np.random.default_rng(SEED)
    idx    = rng.permutation(len(X_train))
    n_es   = max(1, int(len(X_train) * 0.1))
    es_idx = idx[:n_es]
    tr_idx = idx[n_es:]

    d_tr = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx],
                       weight=w_train[tr_idx],
                       feature_name=FINAL_FEATURES, free_raw_data=False)
    d_es = lgb.Dataset(X_train[es_idx], label=y_train[es_idx],
                       feature_name=FINAL_FEATURES,
                       reference=d_tr, free_raw_data=False)
    d_tr.anchor_log_rv = anchor_train[tr_idx]
    d_es.anchor_log_rv = anchor_train[es_idx]

    model_final = lgb.train(
        best_params, d_tr,
        num_boost_round=N_ROUNDS,
        valid_sets=[d_es],
        feval=anchor_qlike_feval,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    print(f"  Best iteration: {model_final.best_iteration}")
    model_final.save_model(str(model_path))

    # Predictions
    lgb_residual_pred = model_final.predict(X_test)
    har_pred_log_rv   = test_df["har_pred_log_rv"].values
    final_pred_log_rv = np.clip(har_pred_log_rv + lgb_residual_pred, -20, 5)
    actual_log_rv     = test_df["log_rv_fut"].values
    actual_rv_arr     = np.exp(actual_log_rv).clip(min=EPS)
    final_pred_rv_arr = np.exp(final_pred_log_rv).clip(min=EPS)
    har_pred_rv_arr   = np.exp(har_pred_log_rv).clip(min=EPS)

    rmspe_hybrid = rmspe(actual_rv_arr, final_pred_rv_arr)
    qlike_hybrid = qlike(actual_rv_arr, final_pred_rv_arr)
    rmspe_har    = rmspe(actual_rv_arr, har_pred_rv_arr)
    qlike_har    = qlike(actual_rv_arr, har_pred_rv_arr)

    print(f"\n  Fold {fold_idx} Results:")
    print(f"    HAR           — RMSPE: {rmspe_har:.6f}    QLIKE: {qlike_har:.6f}")
    print(f"    QLIKE-HAR+LGB — RMSPE: {rmspe_hybrid:.6f}  QLIKE: {qlike_hybrid:.6f}")
    print(f"    RMSPE vs HAR: {(rmspe_har - rmspe_hybrid)/rmspe_har*100:+.2f}%")
    print(f"    QLIKE vs HAR: {(qlike_har - qlike_hybrid)/qlike_har*100:+.2f}%")

    # SHAP
    explainer  = shap.TreeExplainer(model_final)
    shap_vals  = explainer.shap_values(X_test)
    shap_df    = pd.DataFrame(shap_vals, columns=[f"shap_{f}" for f in FINAL_FEATURES])
    shap_summary = pd.DataFrame({
        "feature":        FINAL_FEATURES,
        "mean_abs_shap":  np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    print(f"\n  Top 10 features by SHAP:")
    print(shap_summary.head(10).to_string(index=False))
    shap_summary.to_parquet(MODEL_DIR / f"shap_summary_fold_{fold_idx}.parquet", index=False)

    # Save predictions
    pred_df = test_df[["time_id", "stock_id", "cluster_id",
                        "log_rv_fut", "har_pred_log_rv",
                        "har_pred_rv", "har_residual"]].copy()
    pred_df["lgb_residual_pred"] = lgb_residual_pred
    pred_df["final_pred_log_rv"] = final_pred_log_rv
    pred_df["final_pred_rv"]     = final_pred_rv_arr
    pred_df["actual_rv"]         = actual_rv_arr
    pred_df["fold"]              = fold_idx
    pred_df = pd.concat([pred_df.reset_index(drop=True),
                         shap_df.reset_index(drop=True)], axis=1)
    pred_df.to_parquet(pred_path, index=False)

    with open(params_path, "w") as f:
        json.dump({
            "best_params":      serialisable_params,
            "best_inner_qlike": study.best_value,
            "n_optuna_trials":  N_OPTUNA_TRIALS,
            "best_iteration":   model_final.best_iteration,
        }, f, indent=2, default=str)

    print(f"  Predictions saved to {pred_path}")

    del train_df, test_df, X_train, y_train, w_train, X_test
    del d_tr, d_es, model_final, shap_vals, shap_df
    gc.collect()

    return {
        "predictions": pred_df,
        "metrics": {
            "rmspe_har":    rmspe_har,    "qlike_har":    qlike_har,
            "rmspe_hybrid": rmspe_hybrid, "qlike_hybrid": qlike_hybrid,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    stock_files = sorted(glob.glob(str(DENORM_DIR / "stock_*.csv")))
    print(f"Found {len(stock_files)} stock files")

    # Step 1: HAR
    print("\nStep 1: Fitting HAR models across all folds...")
    for fold_idx in range(5):
        print(f"\n  Fold {fold_idx}")
        train_tids, test_tids = get_fold_time_ids(fold_idx)
        build_har_fold(fold_idx, train_tids, test_tids, stock_files)

    # Step 2: Feature engineering
    print("\nStep 2: Building features across all folds...")
    for fold_idx in range(5):
        print(f"\n  Fold {fold_idx}")
        train_tids, test_tids = get_fold_time_ids(fold_idx)
        train_df, test_df     = build_features_fold(fold_idx, train_tids, test_tids, stock_files)
        apply_zscores(fold_idx, "train", train_df)
        apply_zscores(fold_idx, "test",  test_df)
        del train_df, test_df
        gc.collect()

    # Step 3: QLIKE-LGB training
    print("\nStep 3: Training QLIKE-optimised HAR+LightGBM across all folds...")
    available_gb = psutil.virtual_memory().available / 1e9
    print(f"Available RAM: {available_gb:.2f} GB")
    if available_gb < 1.0:
        raise MemoryError(f"Only {available_gb:.2f} GB available — free memory and rerun")

    results = []
    for fold_idx in range(5):
        print(f"\n  Fold {fold_idx}")
        result = train_qlike_fold(fold_idx)
        results.append(result)

    # Step 4: Pooled evaluation
    all_preds = pd.concat(
        [r["predictions"] for r in results],
        ignore_index=True
    )
    all_preds.to_parquet(PRED_DIR / "predictions_all_folds.parquet", index=False)

    actual_rv_all   = all_preds["actual_rv"].values
    har_rv_all      = all_preds["har_pred_rv"].values
    hybrid_rv_all   = all_preds["final_pred_rv"].values
    fold_id         = all_preds["fold"].values

    print(f"\n{'='*60}")
    print(f"QLIKE-HAR+LGB — Pooled OOF Results ({len(all_preds):,} rows)")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'RMSPE':>10} {'QLIKE':>10}")
    print(f"  {'='*42}")
    print(f"  {'HAR':<20} {rmspe(actual_rv_all, har_rv_all):>10.6f} {qlike(actual_rv_all, har_rv_all):>10.6f}")
    print(f"  {'QLIKE-HAR+LGB':<20} {rmspe(actual_rv_all, hybrid_rv_all):>10.6f} {qlike(actual_rv_all, hybrid_rv_all):>10.6f}")
    print(f"\n  RMSPE improvement: {(rmspe(actual_rv_all, har_rv_all) - rmspe(actual_rv_all, hybrid_rv_all)) / rmspe(actual_rv_all, har_rv_all)*100:+.2f}%")
    print(f"  QLIKE improvement: {(qlike(actual_rv_all, har_rv_all) - qlike(actual_rv_all, hybrid_rv_all)) / qlike(actual_rv_all, har_rv_all)*100:+.2f}%")

    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':<6} {'RMSPE-HAR':>10} {'RMSPE-HYB':>10} {'QLIKE-HAR':>10} {'QLIKE-HYB':>10}")
    for f in range(5):
        mask = fold_id == f
        print(f"  {f:<6} "
              f"{rmspe(actual_rv_all[mask], har_rv_all[mask]):>10.6f} "
              f"{rmspe(actual_rv_all[mask], hybrid_rv_all[mask]):>10.6f} "
              f"{qlike(actual_rv_all[mask], har_rv_all[mask]):>10.6f} "
              f"{qlike(actual_rv_all[mask], hybrid_rv_all[mask]):>10.6f}")

    fold_rmspe = [rmspe(actual_rv_all[fold_id==f], hybrid_rv_all[fold_id==f]) for f in range(5)]
    fold_qlike = [qlike(actual_rv_all[fold_id==f], hybrid_rv_all[fold_id==f]) for f in range(5)]
    print(f"\n  Fold std — RMSPE: {np.std(fold_rmspe):.6f}  QLIKE: {np.std(fold_qlike):.6f}")
    print(f"\nDone. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
