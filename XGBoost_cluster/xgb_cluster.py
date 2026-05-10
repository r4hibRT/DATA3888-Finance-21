"""
xgb_rv.py
=========
XGBoost model for realized volatility forecasting at cluster level.
Predicts the average log_rv_diff across all stocks in a cluster.

Pipeline:
  feature_engineering.py → features/feature_store_{train,test}.parquet
                         → features/selected_features.txt
                         → this script → models/xgb_rv_predictions.csv

Prediction unit: (cluster_id, time_id)
  Features : mean of each feature across all stocks in the cluster
  Target   : mean log_rv_diff across stocks in the cluster
  Anchor   : mean past_log_rv_bkt3 across stocks in the cluster
  Inference: log_rv = model.predict(X) + mean(past_log_rv_bkt3)
"""

import gc
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

FEAT_DIR   = Path("features")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS        = 5
EPS            = 1e-8
N_ROUNDS       = 5000
EARLY_STOPPING = 200
ANCHOR_COL     = "past_log_rv_bkt3"

# ── Cluster definitions ───────────────────────────────────────────────

CLUSTER_STOCKS = {
    0: [1, 8, 17, 21, 23, 48, 61, 63, 77, 95, 104, 110, 111, 113, 120, 122],
    1: [3, 6, 18, 27, 80, 97],
    2: [0, 4, 9, 30, 31, 58, 60, 74, 78, 82, 83, 87, 89, 90, 94, 102, 115, 118, 126],
    3: [2, 7, 13, 20, 28, 29, 32, 34, 39, 41, 42, 43, 46, 47, 51, 64, 67, 93, 99, 125],
    4: [10, 14, 15, 35, 36, 44, 50, 56, 69, 73, 76, 84, 96, 101, 109, 119, 124],
    5: [19, 26, 52, 53, 59, 68, 70, 72, 85, 86, 105, 107, 108, 114, 123],
    6: [5, 11, 16, 22, 33, 37, 38, 40, 55, 62, 66, 75, 81, 88, 98, 100, 103, 112, 116],
}
STOCK_CLUSTER_MAP = {s: c for c, stocks in CLUSTER_STOCKS.items() for s in stocks}


def _has_gpu() -> bool:
    try:
        _tmp = xgb.DMatrix(np.zeros((2, 2)), label=np.zeros(2))
        xgb.train({"tree_method": "gpu_hist", "device": "cuda",
                   "verbosity": 0}, _tmp, num_boost_round=1)
        return True
    except Exception:
        return False


USE_GPU = _has_gpu()

REG_PARAMS = {
    "tree_method":      "gpu_hist" if USE_GPU else "hist",
    "device":           "cuda" if USE_GPU else "cpu",
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "learning_rate":    0.05,
    "max_depth":        8,
    "min_child_weight": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "gamma":            0.01,
    "verbosity":        0,
    "nthread":          -1,
    "seed":             SEED,
}


def log(msg: str) -> None:
    print(f"[XGB] {msg}", flush=True)


# ── Weights & metrics ─────────────────────────────────────────────────

def make_qlike_weights(true_log_rv: np.ndarray) -> np.ndarray:
    """1/true_rv weights — QLIKE-proxy; tilts toward high-RV samples."""
    true_rv = np.exp(true_log_rv).clip(min=EPS)
    w       = 1.0 / true_rv
    cap     = float(np.percentile(w, 99))
    return np.clip(w, 0.0, cap).astype(np.float32)


def compute_metrics(pred_log: np.ndarray, true_log: np.ndarray) -> dict:
    pred_log = np.clip(pred_log, -20, 5)
    pred_rv  = np.clip(np.exp(pred_log), EPS, None)
    true_rv  = np.clip(np.exp(true_log), EPS, None)
    ratio    = true_rv / pred_rv
    qlike    = float(np.mean(ratio - np.log(ratio) - 1))
    resid    = pred_rv - true_rv
    rmse     = float(np.sqrt(np.mean(resid ** 2)))
    mape     = float(np.mean(np.abs(resid) / true_rv) * 100)
    rmspe    = float(np.sqrt(np.mean((resid / true_rv) ** 2)) * 100)
    mae_log  = float(np.mean(np.abs(pred_log - true_log)))
    return {"QLIKE": qlike, "RMSE": rmse, "MAPE%": mape,
            "RMSPE%": rmspe, "MAE_log": mae_log}


def xgb_qlike_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
    """QLIKE + RMSPE eval; anchor reconstructs absolute log-RV from diff."""
    labels  = dtrain.get_label()
    anchor  = dtrain.anchor_log_rv
    pred_rv = np.exp(np.clip(preds  + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp((labels + anchor)).clip(min=EPS)
    ratio   = true_rv / pred_rv
    qlike   = float(np.mean(ratio - np.log(ratio) - 1))
    rmspe   = float(np.sqrt(np.mean(((pred_rv - true_rv) / true_rv) ** 2)))
    return [("qlike", qlike), ("rmspe", rmspe)]


# ── Cluster aggregation ───────────────────────────────────────────────

def aggregate_to_clusters(df: pd.DataFrame, feature_cols: list,
                           extra_cols: list) -> pd.DataFrame:
    """
    Collapse stock-level rows → (cluster_id, time_id) rows by taking
    the mean of every feature and target across stocks in the cluster.

    extra_cols  : non-feature columns to also average (targets, anchor).
    Deduplication prevents duplicate columns when ANCHOR_COL appears
    in both feature_cols and extra_cols.
    n_stocks_contributing : how many stocks were present for that
                            (cluster, time_id) — signals sparse coverage.
    """
    df = df.copy()
    df["cluster_id"] = df["stock_id"].map(STOCK_CLUSTER_MAP)

    # Drop any stock not in a cluster (safety valve)
    df = df[df["cluster_id"].notna()].copy()
    df["cluster_id"] = df["cluster_id"].astype(np.int32)

    # Deduplicate — ANCHOR_COL may appear in both feature_cols and extra_cols
    cols = list(dict.fromkeys(
        c for c in feature_cols + extra_cols if c in df.columns
    ))

    cluster_agg = (
        df.groupby(["cluster_id", "time_id"])[cols]
        .mean()
        .reset_index()
    )

    n_stocks = (
        df.groupby(["cluster_id", "time_id"])["stock_id"]
        .count()
        .rename("n_stocks_contributing")
        .reset_index()
    )
    cluster_agg = cluster_agg.merge(
        n_stocks, on=["cluster_id", "time_id"], how="left"
    )
    return cluster_agg


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("XGBoost — cluster-level average volatility prediction")
    log(f"  Prediction unit : (cluster_id, time_id)")
    log(f"  Target          : mean(log_rv_diff) across cluster stocks")
    log(f"  Anchor          : mean({ANCHOR_COL}) across cluster stocks")
    log(f"  Clusters        : {len(CLUSTER_STOCKS)}")
    log(f"  Device          : {'GPU (gpu_hist)' if USE_GPU else 'CPU (hist)'}")
    log(f"  Folds           : {N_FOLDS}  (KFold, shuffled)")
    log(f"  Rounds          : {N_ROUNDS}  (early stop: {EARLY_STOPPING})")
    log("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    log("\nLoading feature store ...")
    train = pd.read_parquet(FEAT_DIR / "feature_store_train.parquet")
    test  = pd.read_parquet(FEAT_DIR / "feature_store_test.parquet")

    feat_path = FEAT_DIR / "selected_features.txt"
    if feat_path.exists():
        with open(feat_path) as f:
            feature_cols = [l.strip() for l in f if l.strip()]
        feature_cols = [c for c in feature_cols
                        if c in train.columns and c in test.columns]
    else:
        exclude = {"stock_id", "time_id", "log_rv", "rv", "log_rv_diff"}
        feature_cols = [c for c in train.columns if c not in exclude
                        and train[c].dtype in (np.float32, np.float64,
                                               np.int32, np.int16, np.int8)]

    log(f"  Train (stock-level): {train.shape}")
    log(f"  Test  (stock-level): {test.shape}")
    log(f"  Features: {len(feature_cols)}")

    for split, df in [("train", train), ("test", test)]:
        if ANCHOR_COL not in df.columns:
            raise ValueError(
                f"Anchor column '{ANCHOR_COL}' missing from {split}. "
                f"Re-run feature_engineering.py."
            )

    # ── Aggregate stocks → clusters ───────────────────────────────
    log("\nAggregating to cluster level (mean per cluster per time_id) ...")
    extra_cols = [ANCHOR_COL, "log_rv", "log_rv_diff"]

    train = aggregate_to_clusters(train, feature_cols, extra_cols)
    test  = aggregate_to_clusters(test,  feature_cols, extra_cols)

    # n_stocks_contributing added as an extra feature
    feature_cols = feature_cols + ["n_stocks_contributing"]

    log(f"  Train (cluster-level): {train.shape}")
    log(f"  Test  (cluster-level): {test.shape}")
    log(f"  Rows per cluster (train):")
    for cid, n in train.groupby("cluster_id").size().items():
        log(f"    Cluster {cid} ({len(CLUSTER_STOCKS[cid]):2d} stocks): {n:,} time_ids")

    # ── Prepare arrays ────────────────────────────────────────────
    anc_train = train[ANCHOR_COL].values.astype(np.float32)
    anc_test  = test[ANCHOR_COL].values.astype(np.float32)

    if "log_rv_diff" in train.columns:
        y_train = train["log_rv_diff"].values.astype(np.float32)
        log("\n  Using log_rv_diff from feature store")
    else:
        y_train = (train["log_rv"].values - anc_train).astype(np.float32)
        log(f"\n  Computing log_rv_diff on-the-fly from log_rv - {ANCHOR_COL}")

    if "log_rv_diff" in test.columns:
        y_test = test["log_rv_diff"].values.astype(np.float32)
    else:
        y_test = (test["log_rv"].values - anc_test).astype(np.float32)

    true_log_rv_train = (y_train + anc_train).astype(np.float32)
    true_log_rv_test  = (y_test  + anc_test).astype(np.float32)

    cluster_ids_train = train["cluster_id"].values
    cluster_ids_test  = test["cluster_id"].values
    time_ids_train    = train["time_id"].values
    time_ids_test     = test["time_id"].values

    X_train = train[feature_cols].values.astype(np.float32)
    X_test  = test[feature_cols].values.astype(np.float32)

    log(f"\n  Target (mean log_rv_diff) stats:")
    log(f"    mean={y_train.mean():.4f}  std={y_train.std():.4f}  "
        f"min={y_train.min():.4f}  max={y_train.max():.4f}")
    log(f"\n  Anchor (mean {ANCHOR_COL}) stats:")
    log(f"    mean={anc_train.mean():.4f}  std={anc_train.std():.4f}")

    # Pre-build test DMatrix once — shared across folds
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    dtest.anchor_log_rv = anc_test

    # ── Cross-validation ──────────────────────────────────────────
    kf        = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_diff  = np.zeros(len(X_train), dtype=np.float64)
    test_diff = np.zeros(len(X_test),  dtype=np.float64)

    models      = []
    fold_scores = []
    fold_curves = []

    log(f"\n{N_FOLDS}-fold KFold ...")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        log(f"\n── Fold {fold + 1}/{N_FOLDS} ──────────────────────────")

        X_tr, X_va     = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va     = y_train[tr_idx],  y_train[va_idx]
        anc_tr, anc_va = anc_train[tr_idx], anc_train[va_idx]

        log(f"  Train: {len(tr_idx):,}  Val: {len(va_idx):,}")

        w_tr = make_qlike_weights(y_tr + anc_tr)

        d_tr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr,
                           feature_names=feature_cols)
        d_va = xgb.DMatrix(X_va, label=y_va,
                           feature_names=feature_cols)

        d_tr.anchor_log_rv = anc_tr
        d_va.anchor_log_rv = anc_va

        eval_history = {}
        reg = xgb.train(
            REG_PARAMS, d_tr,
            num_boost_round=N_ROUNDS,
            evals=[(d_tr, "train"), (d_va, "val")],
            custom_metric=xgb_qlike_eval,
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=100,
            evals_result=eval_history,
        )

        va_diff = reg.predict(d_va)
        te_diff = reg.predict(dtest)

        oof_diff[va_idx] = va_diff
        test_diff       += te_diff / N_FOLDS

        fold_m = compute_metrics(va_diff + anc_va, y_va + anc_va)
        fold_scores.append(fold_m)
        fold_curves.append(eval_history)

        log(f"  QLIKE={fold_m['QLIKE']:.6f}  "
            f"RMSPE%={fold_m['RMSPE%']:.2f}  "
            f"best_iter={reg.best_iteration}")

        models.append(reg)
        del d_tr, d_va; gc.collect()

    # ── OOF metrics ───────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("OOF metrics (cluster-level):")
    oof_log_rv = oof_diff + anc_train
    oof_m = compute_metrics(oof_log_rv, true_log_rv_train)
    for k, v in oof_m.items():
        log(f"  {k}: {v:.6f}")

    pred_std = float(np.std(oof_log_rv))
    true_std = float(np.std(true_log_rv_train))
    log(f"\n  Spread diagnostic:")
    log(f"    pred std = {pred_std:.4f}  true std = {true_std:.4f}  "
        f"ratio = {pred_std / true_std:.3f}")

    log("\nPer-fold summary:")
    for i, fm in enumerate(fold_scores):
        log(f"  Fold {i+1}: QLIKE={fm['QLIKE']:.6f}  RMSPE%={fm['RMSPE%']:.2f}")
    log(f"  Mean QLIKE: {np.mean([f['QLIKE'] for f in fold_scores]):.6f}")
    log(f"  Std  QLIKE: {np.std([f['QLIKE'] for f in fold_scores]):.6f}")

    # ── Test metrics ──────────────────────────────────────────────
    log("\nTest metrics (ensembled, cluster-level):")
    test_log_rv = test_diff + anc_test
    test_m = compute_metrics(test_log_rv, true_log_rv_test)
    for k, v in test_m.items():
        log(f"  {k}: {v:.6f}")

    pred_rv = np.exp(np.clip(test_log_rv, -20, 5))
    true_rv = np.exp(true_log_rv_test)
    log(f"\nBias/variance (test):")
    log(f"  Mean pred RV: {pred_rv.mean():.6f}  Mean true RV: {true_rv.mean():.6f}")
    log(f"  Bias ratio  : {(pred_rv / true_rv.clip(EPS)).mean():.3f}  "
        f"Var ratio: {pred_rv.std() / true_rv.std():.3f}")

    log(f"\nPer-cluster QLIKE (test):")
    for cid in sorted(np.unique(cluster_ids_test)):
        mask = cluster_ids_test == cid
        if mask.sum() < 5:
            continue
        m = compute_metrics(test_log_rv[mask], true_log_rv_test[mask])
        log(f"  Cluster {cid} ({len(CLUSTER_STOCKS[cid]):2d} stocks, "
            f"{mask.sum():4d} time_ids): "
            f"QLIKE={m['QLIKE']:.6f}  RMSPE%={m['RMSPE%']:.2f}")

    log(f"\nMetrics by RV tier (test):")
    thresholds = [0.0005, 0.001, 0.002, 0.005]
    for lo, hi in zip([0] + thresholds, thresholds + [np.inf]):
        mask = (true_rv >= lo) & (true_rv < hi)
        if mask.sum() == 0:
            continue
        m = compute_metrics(test_log_rv[mask], true_log_rv_test[mask])
        log(f"  RV [{lo:.4f}, {hi:.4f}) n={mask.sum():5,}: "
            f"QLIKE={m['QLIKE']:.4f}  RMSPE%={m['RMSPE%']:.1f}")

    # ── Feature importance ────────────────────────────────────────
    gain_maps  = [m.get_score(importance_type="gain")  for m in models]
    wt_maps    = [m.get_score(importance_type="weight") for m in models]
    imp_gain   = np.zeros(len(feature_cols))
    imp_weight = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        for gm in gain_maps:
            imp_gain[i]   += gm.get(col, 0)
        for wm in wt_maps:
            imp_weight[i] += wm.get(col, 0)
    imp_gain   /= N_FOLDS
    imp_weight /= N_FOLDS
    imp_df = pd.DataFrame({
        "feature":     feature_cols,
        "gain":        imp_gain,
        "weight":      imp_weight,
        "gain_rank":   np.argsort(-imp_gain)   + 1,
        "weight_rank": np.argsort(-imp_weight) + 1,
    }).sort_values("gain", ascending=False)
    log("\nTop 25 features (gain):")
    for _, row in imp_df.head(25).iterrows():
        log(f"  {row['gain']:10.1f}  {row['feature']}")
    imp_df.to_csv(OUTPUT_DIR / "xgb_rv_importance.csv", index=False)

    # ── Save ──────────────────────────────────────────────────────
    log("\nSaving ...")
    pd.DataFrame({
        "time_id":     time_ids_test,
        "cluster_id":  cluster_ids_test,
        "pred_log_rv": np.clip(test_log_rv, -20, 5).astype(np.float32),
        "true_log_rv": true_log_rv_test.astype(np.float32),
        "pred_rv":     pred_rv.astype(np.float32),
        "true_rv":     true_rv.astype(np.float32),
        "pred_diff":   test_diff.astype(np.float32),
        "true_diff":   y_test.astype(np.float32),
        "anchor":      anc_test.astype(np.float32),
    }).to_csv(OUTPUT_DIR / "xgb_rv_predictions.csv", index=False)

    pd.DataFrame({
        "time_id":     time_ids_train,
        "cluster_id":  cluster_ids_train,
        "pred_log_rv": np.clip(oof_log_rv, -20, 5).astype(np.float32),
        "true_log_rv": true_log_rv_train.astype(np.float32),
        "pred_rv":     np.exp(np.clip(oof_log_rv, -20, 5)).astype(np.float32),
        "true_rv":     np.exp(true_log_rv_train).astype(np.float32),
        "pred_diff":   oof_diff.astype(np.float32),
        "true_diff":   y_train.astype(np.float32),
        "anchor":      anc_train.astype(np.float32),
    }).to_csv(OUTPUT_DIR / "xgb_rv_oof.csv", index=False)

    for i, m in enumerate(models):
        m.save_model(str(OUTPUT_DIR / f"xgb_reg_fold{i+1}.json"))

    with open(OUTPUT_DIR / "xgb_rv_params.json", "w") as f:
        json.dump({
            "reg_params":     REG_PARAMS,
            "target":         "mean(log_rv_diff) per cluster per time_id",
            "anchor":         ANCHOR_COL,
            "n_clusters":     len(CLUSTER_STOCKS),
            "cluster_stocks": CLUSTER_STOCKS,
            "n_folds":        N_FOLDS,
            "n_rounds":       N_ROUNDS,
            "early_stopping": EARLY_STOPPING,
            "n_features":     len(feature_cols),
            "use_gpu":        USE_GPU,
        }, f, indent=2, default=str)

    log(f"  Saved → {OUTPUT_DIR}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, eh in enumerate(fold_curves):
        val = eh.get("val", {})
        if "qlike" in val:
            axes[0].plot(val["qlike"], label=f"Fold {i+1}", alpha=0.7)
        if "rmspe" in val:
            axes[1].plot(val["rmspe"], label=f"Fold {i+1}", alpha=0.7)
    for ax, t, yl in zip(axes, ["Val QLIKE", "Val RMSPE"], ["QLIKE", "RMSPE"]):
        ax.set_title(t); ax.set_xlabel("Round")
        ax.set_ylabel(yl); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgb_rv_fold_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    log("\n" + "=" * 60)
    log("Summary:")
    log(f"  Device       : {'GPU' if USE_GPU else 'CPU'}")
    log(f"  OOF  QLIKE   : {oof_m['QLIKE']:.6f}")
    log(f"  OOF  RMSPE%  : {oof_m['RMSPE%']:.2f}")
    log(f"  Test QLIKE   : {test_m['QLIKE']:.6f}")
    log(f"  Test RMSPE%  : {test_m['RMSPE%']:.2f}")
    log(f"  Pred/true std: {pred_std / true_std:.3f}")
    log("=" * 60)
    log("Done.")
    log(f"  Inference: log_rv = model.predict(X) + mean({ANCHOR_COL})")


if __name__ == "__main__":
    main()