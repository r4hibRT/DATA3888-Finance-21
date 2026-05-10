"""
lgbm_rv.py
==========
LightGBM single-stage model for realized volatility forecasting.

Two key changes vs naive MSE baseline:
  1. Target  : log_rv_diff = log(RV_480-599) - log(RV_360-479)
               Forces the model to predict the *change* in volatility
               relative to the nearest input bucket (360–479s), not the
               level. Avoids mean-collapse caused by past RV dominating.
               At inference: log_rv = pred_diff + past_log_rv_bkt3

  2. Weights : 1/true_rv (QLIKE-proxy) — tilts toward high-RV samples,
               matching QLIKE's asymmetric underprediction penalty.

Anchor for reconstruction:
  past_log_rv_bkt3 = log(RV_360–479) — nearest input bucket to target.

Pipeline:
  feature_engineering.py → features/feature_store_{train,test}.parquet
                         → features/selected_features.txt
                         → this script → models/lgbm_rv_predictions.csv

Outputs:
  models/lgbm_rv_predictions.csv
  models/lgbm_rv_oof.csv
  models/lgbm_rv_importance.csv
  models/lgbm_rv_fold_curves.png
  models/lgbm_rv_params.json
  models/lgbm_reg_fold*.txt
"""

import gc
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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

# Anchor column: log(RV_360–479), nearest input bucket to the target window.
# Used to reconstruct absolute log-RV from the diff prediction.
ANCHOR_COL = "past_log_rv_bkt3"

REG_PARAMS = {
    "objective":         "regression",
    "metric":            "None",       # custom feval used instead
    "boosting_type":     "gbdt",
    "learning_rate":     0.05,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 50,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "min_gain_to_split": 0.01,
    "verbose":           -1,
    "n_jobs":            -1,
    "seed":              SEED,
}


def log(msg: str) -> None:
    print(f"[LGBM] {msg}", flush=True)


# ── Weights & metrics ─────────────────────────────────────────────────

def make_qlike_weights(true_log_rv: np.ndarray) -> np.ndarray:
    """
    Sample weights approximating the QLIKE gradient direction.

    QLIKE penalises underprediction asymmetrically. We approximate this
    by weighting MSE loss by 1/true_rv:
      - High-RV samples contribute more to each gradient step,
        pulling predictions away from the mean toward the tails.
      - Capped at the 99th percentile to prevent extreme outliers
        from dominating.

    true_log_rv: absolute log-RV (diff + anchor), not the diff alone.
    """
    true_rv = np.exp(true_log_rv).clip(min=EPS)
    w       = 1.0 / true_rv
    cap     = float(np.percentile(w, 99))
    return np.clip(w, 0.0, cap).astype(np.float32)


def lgb_qlike_eval(preds, dataset):
    """
    QLIKE eval metric for LightGBM.
    Preds and labels are in diff space; anchor (past_log_rv_bkt3)
    reconstructs absolute log-RV for meaningful metric computation.
    """
    labels  = dataset.get_label()
    anchor  = dataset.anchor_log_rv           # past_log_rv_bkt3
    pred_rv = np.exp(np.clip(preds + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp(labels + anchor).clip(min=EPS)
    ratio   = true_rv / pred_rv
    return "qlike", float(np.mean(ratio - np.log(ratio) - 1)), False


def lgb_rmspe_eval(preds, dataset):
    """
    RMSPE eval metric for LightGBM.
    Preds and labels are in diff space; anchor reconstructs absolute log-RV.
    """
    labels  = dataset.get_label()
    anchor  = dataset.anchor_log_rv           # past_log_rv_bkt3
    pred_rv = np.exp(np.clip(preds + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp(labels + anchor).clip(min=EPS)
    return "rmspe", float(np.sqrt(np.mean(
        ((pred_rv - true_rv) / true_rv) ** 2))), False


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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("LightGBM — diff target + QLIKE-proxy weights")
    log(f"  Target    : log_rv_diff = log(RV_480-599) - log(RV_360-479)")
    log(f"  Anchor    : {ANCHOR_COL}  (nearest input bucket)")
    log(f"  Weights   : 1/true_rv (QLIKE-proxy)")
    log(f"  Folds     : {N_FOLDS}")
    log(f"  Rounds    : {N_ROUNDS}  (early stop: {EARLY_STOPPING})")
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

    log(f"  Train: {train.shape}  |  Test: {test.shape}")
    log(f"  Features: {len(feature_cols)}")

    # Verify anchor column is present in both splits
    for split, df in [("train", train), ("test", test)]:
        if ANCHOR_COL not in df.columns:
            raise ValueError(
                f"Anchor column '{ANCHOR_COL}' missing from {split} feature store. "
                f"Re-run feature_engineering.py to regenerate."
            )

    anc_train = train[ANCHOR_COL].values.astype(np.float32)   # log(RV_360–479)
    anc_test  = test[ANCHOR_COL].values.astype(np.float32)

    # Target: log_rv_diff stored directly in the feature store
    if "log_rv_diff" in train.columns:
        y_train = train["log_rv_diff"].values.astype(np.float32)
        log("  Using log_rv_diff column from feature store")
    else:
        # Fallback: reconstruct from log_rv and anchor
        y_train = (train["log_rv"].values - anc_train).astype(np.float32)
        log(f"  log_rv_diff not found — computing on-the-fly from log_rv - {ANCHOR_COL}")

    if "log_rv_diff" in test.columns:
        y_test = test["log_rv_diff"].values.astype(np.float32)
    else:
        y_test = (test["log_rv"].values - anc_test).astype(np.float32)

    # Absolute log-RV for metric computation (diff + anchor)
    true_log_rv_train = (y_train + anc_train).astype(np.float32)
    true_log_rv_test  = (y_test  + anc_test).astype(np.float32)

    groups    = train["stock_id"].values
    stock_ids = train["stock_id"].values
    time_ids  = train["time_id"].values
    test_sids = test["stock_id"].values
    test_tids = test["time_id"].values

    X_train = train[feature_cols].values.astype(np.float32)
    X_test  = test[feature_cols].values.astype(np.float32)

    log(f"\n  Target (log_rv_diff) stats:")
    log(f"    mean={y_train.mean():.4f}  std={y_train.std():.4f}  "
        f"min={y_train.min():.4f}  max={y_train.max():.4f}")
    log(f"\n  Anchor ({ANCHOR_COL}) stats:")
    log(f"    mean={anc_train.mean():.4f}  std={anc_train.std():.4f}")

    # ── Cross-validation ──────────────────────────────────────────
    gkf       = GroupKFold(n_splits=N_FOLDS)
    oof_diff  = np.zeros(len(X_train), dtype=np.float64)
    test_diff = np.zeros(len(X_test),  dtype=np.float64)

    models      = []
    fold_scores = []
    fold_curves = []

    log(f"\n{N_FOLDS}-fold GroupKFold on stock_id ...")

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train, y_train, groups)):
        log(f"\n── Fold {fold + 1}/{N_FOLDS} ──────────────────────────")

        X_tr, X_va     = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va     = y_train[tr_idx],  y_train[va_idx]
        anc_tr, anc_va = anc_train[tr_idx], anc_train[va_idx]

        log(f"  Train: {len(tr_idx):,}  Val: {len(va_idx):,}")

        # QLIKE-proxy weights: need absolute log-RV, so add anchor back
        w_tr = make_qlike_weights(y_tr + anc_tr)

        eval_log = {}
        d_tr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                           feature_name=feature_cols, free_raw_data=False)
        d_va = lgb.Dataset(X_va, label=y_va,
                           feature_name=feature_cols, free_raw_data=False)

        # Attach anchors so feval can reconstruct absolute log-RV
        d_tr.anchor_log_rv = anc_tr
        d_va.anchor_log_rv = anc_va

        reg = lgb.train(
            REG_PARAMS, d_tr,
            num_boost_round=N_ROUNDS,
            valid_sets=[d_tr, d_va],
            valid_names=["train", "val"],
            feval=[lgb_qlike_eval, lgb_rmspe_eval],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(eval_log),
            ],
        )

        va_diff = reg.predict(X_va)
        te_diff = reg.predict(X_test)

        oof_diff[va_idx] = va_diff
        test_diff       += te_diff / N_FOLDS

        # Reconstruct absolute log-RV for metrics
        va_log_rv_pred = va_diff + anc_va
        va_log_rv_true = y_va   + anc_va

        fold_m = compute_metrics(va_log_rv_pred, va_log_rv_true)
        fold_scores.append(fold_m)
        fold_curves.append(eval_log)

        log(f"  QLIKE={fold_m['QLIKE']:.6f}  "
            f"RMSPE%={fold_m['RMSPE%']:.2f}  "
            f"best_iter={reg.best_iteration}")

        models.append(reg)
        del d_tr, d_va; gc.collect()

    # ── OOF metrics ───────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("OOF metrics:")
    oof_log_rv = oof_diff + anc_train       # reconstruct: diff + log(RV_360–479)
    oof_m = compute_metrics(oof_log_rv, true_log_rv_train)
    for k, v in oof_m.items():
        log(f"  {k}: {v:.6f}")

    pred_std = float(np.std(oof_log_rv))
    true_std = float(np.std(true_log_rv_train))
    log(f"\n  Prediction spread diagnostic (log-RV space):")
    log(f"    pred std = {pred_std:.4f}")
    log(f"    true std = {true_std:.4f}")
    log(f"    ratio    = {pred_std / true_std:.3f}  "
        f"(1.0 = perfect spread, <1 = still collapsing)")

    log("\nPer-fold summary:")
    for i, fm in enumerate(fold_scores):
        log(f"  Fold {i+1}: QLIKE={fm['QLIKE']:.6f}  RMSPE%={fm['RMSPE%']:.2f}")
    log(f"  Mean QLIKE: {np.mean([f['QLIKE'] for f in fold_scores]):.6f}")
    log(f"  Std  QLIKE: {np.std([f['QLIKE'] for f in fold_scores]):.6f}")

    # ── Test metrics ──────────────────────────────────────────────
    log("\nTest metrics (ensembled):")
    test_log_rv = test_diff + anc_test      # reconstruct: diff + log(RV_360–479)
    test_m = compute_metrics(test_log_rv, true_log_rv_test)
    for k, v in test_m.items():
        log(f"  {k}: {v:.6f}")

    pred_rv = np.exp(np.clip(test_log_rv, -20, 5))
    true_rv = np.exp(true_log_rv_test)
    log(f"\nBias/variance (test):")
    log(f"  Mean pred RV : {pred_rv.mean():.6f}")
    log(f"  Mean true RV : {true_rv.mean():.6f}")
    log(f"  Bias ratio   : {(pred_rv / true_rv.clip(EPS)).mean():.3f}")
    log(f"  Var ratio    : {pred_rv.std() / true_rv.std():.3f}")

    log(f"\nMetrics by RV tier (test):")
    thresholds = [0.0005, 0.001, 0.002, 0.005]
    for lo, hi in zip([0] + thresholds, thresholds + [np.inf]):
        mask = (true_rv >= lo) & (true_rv < hi)
        if mask.sum() == 0:
            continue
        m = compute_metrics(test_log_rv[mask], true_log_rv_test[mask])
        log(f"  RV [{lo:.4f}, {hi:.4f}) n={mask.sum():6,}: "
            f"QLIKE={m['QLIKE']:.4f}  RMSPE%={m['RMSPE%']:.1f}")

    log(f"\nPer-stock QLIKE — best and worst:")
    stock_qlike = {}
    for sid in np.unique(test_sids):
        mask = test_sids == sid
        if mask.sum() < 10:
            continue
        m = compute_metrics(test_log_rv[mask], true_log_rv_test[mask])
        stock_qlike[int(sid)] = m["QLIKE"]
    sorted_s = sorted(stock_qlike.items(), key=lambda x: x[1])
    for label, items in [("Best 5", sorted_s[:5]), ("Worst 5", sorted_s[-5:])]:
        log(f"  {label}:")
        for sid, q in items:
            log(f"    stock_{sid:3d}: QLIKE={q:.6f}")

    # ── Feature importance ────────────────────────────────────────
    imp_gain  = np.zeros(len(feature_cols))
    imp_split = np.zeros(len(feature_cols))
    for m in models:
        imp_gain  += m.feature_importance("gain")
        imp_split += m.feature_importance("split")
    imp_gain  /= N_FOLDS
    imp_split /= N_FOLDS
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "gain":       imp_gain,
        "split":      imp_split,
        "gain_rank":  np.argsort(-imp_gain)  + 1,
        "split_rank": np.argsort(-imp_split) + 1,
    }).sort_values("gain", ascending=False)
    log("\nTop 25 features (gain):")
    for _, row in imp_df.head(25).iterrows():
        log(f"  {row['gain']:10.1f}  {row['feature']}")
    imp_df.to_csv(OUTPUT_DIR / "lgbm_rv_importance.csv", index=False)

    # ── Save predictions ──────────────────────────────────────────
    log("\nSaving ...")
    pd.DataFrame({
        "time_id":     test_tids,
        "stock_id":    test_sids,
        "pred_log_rv": np.clip(test_log_rv, -20, 5).astype(np.float32),
        "true_log_rv": true_log_rv_test.astype(np.float32),
        "pred_rv":     pred_rv.astype(np.float32),
        "true_rv":     true_rv.astype(np.float32),
        "pred_diff":   test_diff.astype(np.float32),
        "true_diff":   y_test.astype(np.float32),
        "anchor":      anc_test.astype(np.float32),        # past_log_rv_bkt3
    }).to_csv(OUTPUT_DIR / "lgbm_rv_predictions.csv", index=False)

    pd.DataFrame({
        "time_id":     time_ids,
        "stock_id":    stock_ids,
        "pred_log_rv": np.clip(oof_log_rv, -20, 5).astype(np.float32),
        "true_log_rv": true_log_rv_train.astype(np.float32),
        "pred_rv":     np.exp(np.clip(oof_log_rv, -20, 5)).astype(np.float32),
        "true_rv":     np.exp(true_log_rv_train).astype(np.float32),
        "pred_diff":   oof_diff.astype(np.float32),
        "true_diff":   y_train.astype(np.float32),
        "anchor":      anc_train.astype(np.float32),       # past_log_rv_bkt3
    }).to_csv(OUTPUT_DIR / "lgbm_rv_oof.csv", index=False)

    for i, m in enumerate(models):
        m.save_model(str(OUTPUT_DIR / f"lgbm_reg_fold{i+1}.txt"))

    with open(OUTPUT_DIR / "lgbm_rv_params.json", "w") as f:
        json.dump({
            "reg_params":     REG_PARAMS,
            "objective":      "mse + qlike_proxy_weights",
            "target":         "log_rv_diff = log(RV_480-599) - log(RV_360-479)",
            "anchor":         ANCHOR_COL,
            "n_folds":        N_FOLDS,
            "n_rounds":       N_ROUNDS,
            "early_stopping": EARLY_STOPPING,
            "n_features":     len(feature_cols),
        }, f, indent=2, default=str)

    log(f"  Predictions + models saved → {OUTPUT_DIR}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, ev in enumerate(fold_curves):
        if "val" in ev and "qlike" in ev["val"]:
            axes[0].plot(ev["val"]["qlike"], label=f"Fold {i+1}", alpha=0.7)
        if "val" in ev and "rmspe" in ev["val"]:
            axes[1].plot(ev["val"]["rmspe"], label=f"Fold {i+1}", alpha=0.7)
    for ax, t, yl in zip(axes, ["Val QLIKE", "Val RMSPE"], ["QLIKE", "RMSPE"]):
        ax.set_title(t); ax.set_xlabel("Round")
        ax.set_ylabel(yl); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lgbm_rv_fold_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    log("\n" + "=" * 60)
    log("Summary:")
    log(f"  Objective    : MSE + QLIKE-proxy weights (1/true_rv)")
    log(f"  Target       : log_rv_diff = log(RV_480-599) - log(RV_360-479)")
    log(f"  Anchor       : {ANCHOR_COL}")
    log(f"  OOF  QLIKE   : {oof_m['QLIKE']:.6f}")
    log(f"  OOF  RMSPE%  : {oof_m['RMSPE%']:.2f}")
    log(f"  Test QLIKE   : {test_m['QLIKE']:.6f}")
    log(f"  Test RMSPE%  : {test_m['RMSPE%']:.2f}")
    log(f"  Pred/true std: {pred_std / true_std:.3f}")
    log("=" * 60)
    log("Done.")
    log(f"  Inference: log_rv = model.predict(X) + {ANCHOR_COL}")


if __name__ == "__main__":
    main()