"""
lightgbm_model.py
=================
Standalone LightGBM model for realized volatility forecasting.

Reads the same processed_v2 parquet files as the GNN model.
Engineers richer per-stock features from the 3 × 100s buckets.
Predicts log(RV) target directly.

Features engineered:
  - HAR log-RV levels and trends
  - Spread regime (level, trend, volatility)
  - Volume regime (level, trend, acceleration)
  - Return momentum and direction
  - Volatility trajectory (trend, acceleration)
  - Cross-feature interactions (Amihud, vol-weighted activity)
  - Stock identity (categorical)

Outputs:
  - models/lgbm_model.pkl         trained model
  - models/lgbm_predictions.csv   test set predictions
  - models/lgbm_feature_importance.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb

EPS          = 1e-8
LOG_RV_FLOOR = -18.0
RV_FLOOR     = 1e-8

FEATURES_DIR   = Path("processed_v2")
OUTPUT_DIR     = Path("models")
BEST_PARAMS_PATH = OUTPUT_DIR / "lgbm_best_params.json"

N_BUCKETS    = 3
BUCKET_FEATS = ["log_return", "realized_vol", "mean_spread", "total_volume"]

# Default params — used only if lgbm_best_params.json is not found
LGBM_PARAMS_DEFAULT = {
    "objective"        : "regression",
    "metric"           : "rmse",
    "boosting_type"    : "gbdt",
    "num_leaves"       : 63,
    "max_depth"        : -1,
    "learning_rate"    : 0.05,
    "n_estimators"     : 1000,
    "min_child_samples": 50,
    "subsample"        : 0.8,
    "subsample_freq"   : 1,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbose"          : -1,
}


def load_params() -> dict:
    """
    Load tuned params from lgbm_best_params.json if available,
    otherwise fall back to LGBM_PARAMS_DEFAULT.
    Always ensures required keys (objective, metric, verbose, n_jobs)
    are present so the model trains correctly regardless of what
    lgbm_tune.py saved.
    """
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            params = json.load(f)
        print(f"Loaded tuned params from {BEST_PARAMS_PATH}")

        # Ensure required keys that tune script may not include
        params.setdefault("objective",  "regression")
        params.setdefault("metric",     "rmse")
        params.setdefault("verbose",    -1)
        params.setdefault("n_jobs",     -1)
        params.setdefault("random_state", 42)
        params.setdefault("subsample_freq", 1)

        # n_estimators from tuning is the best_iteration — use it directly
        # (no early stopping needed since we already know the right value)
        print(f"  boosting_type  : {params.get('boosting_type', 'gbdt')}")
        print(f"  num_leaves     : {params.get('num_leaves')}")
        print(f"  learning_rate  : {params.get('learning_rate', 'N/A'):.4f}")
        print(f"  n_estimators   : {params.get('n_estimators')}")
        print(f"  reg_alpha      : {params.get('reg_alpha', 'N/A'):.4f}")
        print(f"  reg_lambda     : {params.get('reg_lambda', 'N/A'):.4f}")
        return params

    print(f"lgbm_best_params.json not found — using default params.")
    print("Run lgbm_tune.py first to get optimised hyperparameters.")
    return LGBM_PARAMS_DEFAULT.copy()


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(pred_log: np.ndarray, true_log: np.ndarray) -> dict:
    pred_log  = np.clip(pred_log, -20, 5)
    mse_log   = float(np.mean((pred_log - true_log) ** 2))
    pred_rv   = np.clip(np.exp(pred_log), EPS, None)
    true_rv   = np.clip(np.exp(true_log), EPS, None)
    resid     = pred_rv - true_rv
    rmse      = float(np.sqrt(np.mean(resid ** 2)))
    mape      = float(np.mean(np.abs(resid) / true_rv) * 100)
    rmspe     = float(np.sqrt(np.mean((resid / true_rv) ** 2)) * 100)
    ratio     = true_rv / pred_rv
    qlike     = float(np.mean(ratio - np.log(ratio) - 1))
    return {
        "MSE (log-RV)": mse_log,
        "QLIKE"       : qlike,
        "RMSE"        : rmse,
        "MAPE%"       : mape,
        "RMSPE%"      : rmspe,
    }


# ── Feature engineering ───────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature matrix from the 3-bucket parquet features.

    All features are computed per (time_id, stock_id) row.
    LightGBM handles non-linearity so we provide both raw values
    and derived interaction terms — the tree will select what matters.
    """
    out = pd.DataFrame()
    out["time_id"]  = df["time_id"]
    out["stock_id"] = df["stock_id"]

    # ── Raw bucket features ───────────────────────────────────────
    for b in range(N_BUCKETS):
        for feat in BUCKET_FEATS:
            col = f"{feat}_b{b}"
            if col in df.columns:
                out[col] = df[col].fillna(0.0)

    # ── Log-scale RV (matches HAR specification) ──────────────────
    for b in range(N_BUCKETS):
        col = f"realized_vol_b{b}"
        if col in df.columns:
            out[f"log_rv_b{b}"] = np.log(df[col].clip(lower=RV_FLOOR))

    # ── HAR volatility features ───────────────────────────────────
    # Short / medium / long RV in log space
    if all(f"log_rv_b{b}" in out for b in range(N_BUCKETS)):
        out["rv_level"]  = out[["log_rv_b0", "log_rv_b1", "log_rv_b2"]].mean(axis=1)
        out["rv_trend"]  = out["log_rv_b2"] - out["log_rv_b0"]   # recent vs old
        out["rv_accel"]  = (out["log_rv_b2"] - out["log_rv_b1"]) \
                         - (out["log_rv_b1"] - out["log_rv_b0"])  # curvature
        out["rv_max"]    = out[["log_rv_b0", "log_rv_b1", "log_rv_b2"]].max(axis=1)
        out["rv_min"]    = out[["log_rv_b0", "log_rv_b1", "log_rv_b2"]].min(axis=1)
        out["rv_range"]  = out["rv_max"] - out["rv_min"]          # intra-window spread

    # ── Spread features ───────────────────────────────────────────
    spread_cols = [f"mean_spread_b{b}" for b in range(N_BUCKETS)
                   if f"mean_spread_b{b}" in df.columns]
    if len(spread_cols) == N_BUCKETS:
        s0, s1, s2 = [df[c].fillna(0.0) for c in spread_cols]
        out["spread_level"]  = s1                    # mid-window spread
        out["spread_trend"]  = s2 - s0               # tightening or widening
        out["spread_accel"]  = (s2 - s1) - (s1 - s0) # curvature
        out["spread_mean"]   = (s0 + s1 + s2) / 3
        out["spread_max"]    = pd.concat([s0, s1, s2], axis=1).max(axis=1)

    # ── Volume features ───────────────────────────────────────────
    vol_cols = [f"total_volume_b{b}" for b in range(N_BUCKETS)
                if f"total_volume_b{b}" in df.columns]
    if len(vol_cols) == N_BUCKETS:
        v0, v1, v2 = [df[c].fillna(0.0) for c in vol_cols]
        out["volume_level"]  = v1
        out["volume_trend"]  = v2 - v0
        out["volume_accel"]  = (v2 - v1) - (v1 - v0)
        out["volume_total"]  = v0 + v1 + v2
        out["volume_max"]    = pd.concat([v0, v1, v2], axis=1).max(axis=1)
        # Log volume (more normally distributed)
        out["log_vol_b0"]    = np.log(v0.clip(lower=EPS))
        out["log_vol_b1"]    = np.log(v1.clip(lower=EPS))
        out["log_vol_b2"]    = np.log(v2.clip(lower=EPS))
        out["log_vol_trend"] = out["log_vol_b2"] - out["log_vol_b0"]

    # ── Return features ───────────────────────────────────────────
    ret_cols = [f"log_return_b{b}" for b in range(N_BUCKETS)
                if f"log_return_b{b}" in df.columns]
    if len(ret_cols) == N_BUCKETS:
        r0, r1, r2 = [df[c].fillna(0.0) for c in ret_cols]
        out["return_sum"]    = r0 + r1 + r2           # net price move
        out["return_trend"]  = r2 - r0                # momentum
        out["return_abs_b0"] = r0.abs()
        out["return_abs_b1"] = r1.abs()
        out["return_abs_b2"] = r2.abs()
        out["return_abs_sum"] = r0.abs() + r1.abs() + r2.abs()
        # Direction consistency (all same sign = trending)
        out["return_sign_b0"] = np.sign(r0)
        out["return_sign_b1"] = np.sign(r1)
        out["return_sign_b2"] = np.sign(r2)
        out["direction_consistent"] = (
            (np.sign(r0) == np.sign(r1)) & (np.sign(r1) == np.sign(r2))
        ).astype(float)

    # ── Cross-feature interactions ────────────────────────────────
    # Amihud illiquidity: |return| / volume (price impact per unit vol)
    for b in range(N_BUCKETS):
        ret_col = f"log_return_b{b}"
        vol_col = f"total_volume_b{b}"
        if ret_col in df.columns and vol_col in df.columns:
            out[f"amihud_b{b}"] = (
                df[ret_col].abs() / (df[vol_col].clip(lower=EPS))
            ).fillna(0.0)

    # Volatility × volume (active volatile period or just noise?)
    if "realized_vol_b1" in df.columns and "total_volume_b1" in df.columns:
        out["vol_times_volume"] = (
            df["realized_vol_b1"] * df["total_volume_b1"]
        ).fillna(0.0)

    # Spread × RV (liquidity-adjusted volatility)
    if "mean_spread_b1" in df.columns and "realized_vol_b1" in df.columns:
        out["spread_adj_rv"] = (
            df["realized_vol_b1"] / df["mean_spread_b1"].clip(lower=EPS)
        ).fillna(0.0)

    # Noise floor estimate: sqrt(2 * zeta * 300)
    # zeta = (spread/2)^2 per second
    if "mean_spread_b1" in df.columns:
        zeta = (df["mean_spread_b1"] / 2) ** 2
        out["noise_floor"]    = np.sqrt(2 * zeta * 300)
        out["log_noise_floor"] = np.log(out["noise_floor"].clip(lower=EPS))

    # SNR: realized_vol / noise_floor
    if "realized_vol_b1" in df.columns and "noise_floor" in out.columns:
        out["snr_b1"] = (
            df["realized_vol_b1"] / out["noise_floor"].clip(lower=EPS)
        ).fillna(0.0)

    # Stock identity — LightGBM uses this as a categorical split feature
    # Captures per-stock fixed effects (always illiquid, always volatile, etc.)
    out["stock_id_cat"] = df["stock_id"].astype("category")

    tid_rv_mean = df.groupby("time_id")["realized_vol_b2"].transform("mean")
    tid_rv_std  = df.groupby("time_id")["realized_vol_b2"].transform("std")

    out["market_rv_mean"] = tid_rv_mean   # market-wide volatility level
    out["market_rv_std"]  = tid_rv_std    # market-wide volatility dispersion
    out["stock_vs_market"] = (
        df["realized_vol_b2"] - tid_rv_mean
    ) / tid_rv_std.clip(lower=1e-8)       # this stock's deviation from market


    return out


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature columns (exclude id and target columns)."""
    exclude = {"time_id", "stock_id", "log_rv_target"}
    return [c for c in df.columns if c not in exclude]


# ── Live mask ────────────────────────────────────────────────────────

def is_live(series: pd.Series) -> pd.Series:
    """True for stocks with meaningful target RV (not dead or floor)."""
    return (series != 0.0) & (series > LOG_RV_FLOOR)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LightGBM Realized Volatility Forecaster")
    print(f"  Data dir : {FEATURES_DIR}")
    print(f"  Output   : {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load parquet files ────────────────────────────────
    print("\nLoading parquet files ...")
    train_feat = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_targ = pd.read_parquet(FEATURES_DIR / "train_targets.parquet")
    val_feat   = pd.read_parquet(FEATURES_DIR / "val_features.parquet")
    val_targ   = pd.read_parquet(FEATURES_DIR / "val_targets.parquet")
    test_feat  = pd.read_parquet(FEATURES_DIR / "test_features.parquet")
    test_targ  = pd.read_parquet(FEATURES_DIR / "test_targets.parquet")

    print(f"  Train: {len(train_feat):,} rows")
    print(f"  Val  : {len(val_feat):,} rows")
    print(f"  Test : {len(test_feat):,} rows")

    # ── Step 2: Merge features + targets ─────────────────────────
    def merge(feat, targ):
        return feat.merge(targ[["time_id", "stock_id", "log_rv_target"]],
                          on=["time_id", "stock_id"], how="inner")

    train_df = merge(train_feat, train_targ)
    val_df   = merge(val_feat,   val_targ)
    test_df  = merge(test_feat,  test_targ)

    # ── Step 3: Engineer features ─────────────────────────────────
    print("\nEngineering features ...")
    train_eng = engineer_features(train_df)
    val_eng   = engineer_features(val_df)
    test_eng  = engineer_features(test_df)

    # Add target back
    train_eng["log_rv_target"] = train_df["log_rv_target"].values
    val_eng["log_rv_target"]   = val_df["log_rv_target"].values
    test_eng["log_rv_target"]  = test_df["log_rv_target"].values

    # ── Step 4: Filter live stocks ────────────────────────────────
    train_live = train_eng[is_live(train_eng["log_rv_target"])].copy()
    val_live   = val_eng[is_live(val_eng["log_rv_target"])].copy()
    test_live  = test_eng[is_live(test_eng["log_rv_target"])].copy()

    print(f"\nLive rows after filtering:")
    print(f"  Train: {len(train_live):,} / {len(train_eng):,}")
    print(f"  Val  : {len(val_live):,} / {len(val_eng):,}")
    print(f"  Test : {len(test_live):,} / {len(test_eng):,}")

    # ── Step 5: Prepare matrices ──────────────────────────────────
    feat_cols = get_feature_cols(train_live)
    cat_cols  = ["stock_id_cat"] if "stock_id_cat" in feat_cols else []

    print(f"\nFeature count: {len(feat_cols)}")

    X_train = train_live[feat_cols]
    y_train = train_live["log_rv_target"].values

    X_val   = val_live[feat_cols]
    y_val   = val_live["log_rv_target"].values

    X_test  = test_live[feat_cols]
    y_test  = test_live["log_rv_target"].values

    # ── Step 6: Load params and train LightGBM ─────────────────
    params = load_params()
    using_tuned = BEST_PARAMS_PATH.exists()

    print("\nTraining LightGBM ...")
    model = lgb.LGBMRegressor(**params)

    if using_tuned:
        # Tuned params already have the right n_estimators from best_iteration.
        # No early stopping needed — train for exactly that many trees.
        model.fit(
            X_train, y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.log_evaluation(period=50)],
        )
        print(f"\nTrained for {params.get('n_estimators')} iterations (from tuning).")
    else:
        # Default params — use early stopping to find best n_estimators
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50),
            ],
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        print(f"\nBest iteration: {model.best_iteration_}")

    # ── Step 7: Evaluate ──────────────────────────────────────────
    for split_name, X, y in [
        ("Train", X_train, y_train),
        ("Val",   X_val,   y_val),
        ("Test",  X_test,  y_test),
    ]:
        pred    = model.predict(X)
        metrics = compute_metrics(pred, y)
        print(f"\n{split_name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    # ── Step 8: Bias / variance check ────────────────────────────
    pred_test = model.predict(X_test)
    pred_rv   = np.exp(np.clip(pred_test, -20, 5))
    true_rv   = np.exp(y_test)

    print(f"\nBias / variance check (test set):")
    print(f"  Mean pred RV : {pred_rv.mean():.6f}")
    print(f"  Mean true RV : {true_rv.mean():.6f}")
    print(f"  Ratio        : {(pred_rv / true_rv.clip(EPS)).mean():.3f}")
    print(f"  Std pred RV  : {pred_rv.std():.6f}")
    print(f"  Std true RV  : {true_rv.std():.6f}")
    print(f"  Var ratio    : {pred_rv.std() / true_rv.std():.3f}")

    # ── Step 9: Per-RV-tier metrics ───────────────────────────────
    print(f"\nMetrics by RV tier:")
    for name, mask in [
        ("Low  (<0.005) ", true_rv < 0.005),
        ("High (>=0.005)", true_rv >= 0.005),
    ]:
        if mask.sum() == 0:
            continue
        m = compute_metrics(pred_test[mask], y_test[mask])
        print(f"  {name} n={mask.sum():6,}: "
              f"QLIKE={m['QLIKE']:.4f}  MAPE%={m['MAPE%']:.1f}")

    # ── Step 10: Feature importance ───────────────────────────────
    importance = pd.DataFrame({
        "feature"   : feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nTop 20 features:")
    print(importance.head(20).to_string(index=False))

    importance.to_csv(OUTPUT_DIR / "lgbm_feature_importance.csv", index=False)

    # ── Step 11: Save model ───────────────────────────────────────
    model_path = OUTPUT_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {model_path}")

    # ── Step 12: Export predictions ───────────────────────────────
    out_df = test_live[["time_id", "stock_id"]].copy()
    out_df["pred_log_rv"] = np.clip(pred_test, -20, 5)
    out_df["true_log_rv"] = y_test
    out_df["pred_rv"]     = np.exp(out_df["pred_log_rv"])
    out_df["true_rv"]     = np.exp(out_df["true_log_rv"])

    out_path = OUTPUT_DIR / "lgbm_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved -> {out_path} ({len(out_df):,} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()