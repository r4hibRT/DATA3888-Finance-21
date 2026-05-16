"""
lgbm_rv.py
==========
LightGBM model for realized volatility forecasting at cluster level.
Predicts the average log_rv_diff across all stocks in a cluster.

Pipeline:
  eda.ipynb → processed/fold_{0..4}/{train,test}.parquet
  feature_engineer.py → processed/fold_{0..4}/feature_store_{train,test}.parquet
  → this script → processed/fold_{0..4}/lgbm_* outputs

Nested CV:
  Outer loop : 5 folds from eda.ipynb (GroupShuffleSplit on time_id)
  Inner loop : Optuna tunes hyperparameters via 5-fold CV on outer train
  Final      : one model per outer fold, trained on full outer train with best params
  Evaluation : RMSPE (optimization), QLIKE, MSE, MAPE (reporting)

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

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

DATA_DIR   = Path("processed")
OUTPUT_DIR = Path("processed")   # save into each fold dir

N_OUTER_FOLDS  = 5
N_INNER_FOLDS  = 5
N_OPTUNA_TRIALS = 50
N_ROUNDS        = 5000
EARLY_STOPPING  = 200
EPS             = 1e-8
ANCHOR_COL      = "past_log_rv_bkt3"

# ── Cluster definitions ───────────────────────────────────────────────

CLUSTER_STOCKS = {
    0: [13, 21, 32, 41, 46, 47, 77, 108, 111, 125],
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 22, 23,
        26, 27, 28, 29, 30, 33, 34, 36, 37, 38, 39, 40, 42, 43, 48, 50, 51, 52, 53,
        55, 56, 58, 59, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 73, 75, 76, 78, 80,
        81, 82, 83, 84, 87, 88, 90, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104,
        107, 109, 110, 112, 113, 114, 115, 116, 118, 120, 122, 123, 126],
    2: [31, 35, 44, 63, 74, 85, 86, 89, 99, 105, 119, 124],
}
STOCK_CLUSTER_MAP = {s: c for c, stocks in CLUSTER_STOCKS.items() for s in stocks}


def log(msg: str) -> None:
    print(f"[LGBM] {msg}", flush=True)


# ── Metrics ───────────────────────────────────────────────────────────

def make_qlike_weights(true_log_rv: np.ndarray) -> np.ndarray:
    """1/true_rv weights — QLIKE-proxy; tilts toward high-RV samples."""
    true_rv = np.exp(true_log_rv).clip(min=EPS)
    w       = 1.0 / true_rv
    cap     = float(np.percentile(w, 99))
    return np.clip(w, 0.0, cap).astype(np.float32)


def lgb_qlike_eval(preds, dataset):
    labels  = dataset.get_label()
    anchor  = dataset.anchor_log_rv
    pred_rv = np.exp(np.clip(preds + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp(labels + anchor).clip(min=EPS)
    ratio   = true_rv / pred_rv
    return "qlike", float(np.mean(ratio - np.log(ratio) - 1)), False


def lgb_rmspe_eval(preds, dataset):
    labels  = dataset.get_label()
    anchor  = dataset.anchor_log_rv
    pred_rv = np.exp(np.clip(preds + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp(labels + anchor).clip(min=EPS)
    return "rmspe", float(np.sqrt(np.mean(
        ((pred_rv - true_rv) / true_rv) ** 2))), False


def lgb_mse_eval(preds, dataset):
    labels = dataset.get_label()
    return "mse", float(np.mean((preds - labels) ** 2)), False


def compute_metrics(pred_log: np.ndarray, true_log: np.ndarray,
                    pred_diff: np.ndarray = None,
                    true_diff: np.ndarray = None) -> dict:
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

    result = {"QLIKE": qlike, "RMSE": rmse, "MAPE%": mape,
              "RMSPE%": rmspe, "MAE_log": mae_log}

    # MSE on log_rv_diff (the direct model output)
    if pred_diff is not None and true_diff is not None:
        result["MSE_diff"] = float(np.mean((pred_diff - true_diff) ** 2))

    # MSE on log_rv level
    result["MSE_log_rv"] = float(np.mean((pred_log - true_log) ** 2))
    result["MSE_rv"]     = float(np.mean((pred_rv - true_rv) ** 2))

    return result


# ── Cluster aggregation ───────────────────────────────────────────────

def aggregate_to_clusters(df: pd.DataFrame, feature_cols: list,
                          extra_cols: list) -> pd.DataFrame:
    df = df.copy()
    df["cluster_id"] = df["stock_id"].map(STOCK_CLUSTER_MAP)
    df = df[df["cluster_id"].notna()].copy()
    df["cluster_id"] = df["cluster_id"].astype(np.int32)

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


# ── Optuna objective ──────────────────────────────────────────────────

def make_objective(X, y, anchor, feature_cols, n_inner_folds, seed):
    """Return an Optuna objective that minimises RMSPE via inner CV."""
    n_samples = len(X)

    def objective(trial):
        params = {
            "objective":         "regression",
            "metric":            "None",
            "boosting_type":     "gbdt",
            "verbosity":         -1,
            "n_jobs":            -1,
            "seed":              seed,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15,
                                                    min(255, max(31, n_samples // 100))),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples",
                                                    max(5, n_samples // 500), 100),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }

        kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        rmspe_scores = []

        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va     = X[tr_idx], X[va_idx]
            y_tr, y_va     = y[tr_idx], y[va_idx]
            anc_tr, anc_va = anchor[tr_idx], anchor[va_idx]

            w_tr = make_qlike_weights(y_tr + anc_tr)

            d_tr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                               feature_name=feature_cols, free_raw_data=False)
            d_va = lgb.Dataset(X_va, label=y_va,
                               feature_name=feature_cols, free_raw_data=False)

            d_tr.anchor_log_rv = anc_tr
            d_va.anchor_log_rv = anc_va

            model = lgb.train(
                params, d_tr,
                num_boost_round=N_ROUNDS,
                valid_sets=[d_va],
                valid_names=["val"],
                feval=[lgb_rmspe_eval],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING, verbose=False),
                ],
            )

            va_diff = model.predict(X_va)
            pred_rv = np.exp(np.clip(va_diff + anc_va, -20, 5)).clip(min=EPS)
            true_rv = np.exp(y_va + anc_va).clip(min=EPS)
            rmspe   = float(np.sqrt(np.mean(((pred_rv - true_rv) / true_rv) ** 2)))
            rmspe_scores.append(rmspe)

            del d_tr, d_va, model; gc.collect()

        return float(np.mean(rmspe_scores))

    return objective


# ── Run one outer fold ────────────────────────────────────────────────

def run_outer_fold(fold: int, fold_dir: Path):
    """
    Load fold data, tune hyperparams via Optuna inner CV,
    train final model on full outer train, evaluate on outer test.
    """
    log(f"\n{'━' * 60}")
    log(f"  OUTER FOLD {fold}")
    log(f"  Data: {fold_dir}")
    log(f"{'━' * 60}")

    # ── Load ──────────────────────────────────────────────────────
    train = pd.read_parquet(fold_dir / "feature_store_train.parquet")
    test  = pd.read_parquet(fold_dir / "feature_store_test.parquet")

    feat_path = fold_dir / "selected_features.txt"
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

    # ── Aggregate to cluster level ────────────────────────────────
    log("\n  Aggregating to cluster level ...")
    extra_cols = [ANCHOR_COL, "log_rv", "log_rv_diff"]
    train = aggregate_to_clusters(train, feature_cols, extra_cols)
    test  = aggregate_to_clusters(test,  feature_cols, extra_cols)
    feature_cols = feature_cols + ["n_stocks_contributing"]

    log(f"  Train (cluster-level): {train.shape}")
    log(f"  Test  (cluster-level): {test.shape}")

    # ── Prepare arrays ────────────────────────────────────────────
    anc_train = train[ANCHOR_COL].values.astype(np.float32)
    anc_test  = test[ANCHOR_COL].values.astype(np.float32)

    y_train = train["log_rv_diff"].values.astype(np.float32) if "log_rv_diff" in train.columns \
        else (train["log_rv"].values - anc_train).astype(np.float32)
    y_test = test["log_rv_diff"].values.astype(np.float32) if "log_rv_diff" in test.columns \
        else (test["log_rv"].values - anc_test).astype(np.float32)

    true_log_rv_train = (y_train + anc_train).astype(np.float32)
    true_log_rv_test  = (y_test  + anc_test).astype(np.float32)

    cluster_ids_test = test["cluster_id"].values
    time_ids_test    = test["time_id"].values

    X_train = train[feature_cols].values.astype(np.float32)
    X_test  = test[feature_cols].values.astype(np.float32)

    log(f"\n  Target stats: mean={y_train.mean():.4f}  std={y_train.std():.4f}")
    log(f"  Anchor stats: mean={anc_train.mean():.4f}  std={anc_train.std():.4f}")

    # ── Optuna hyperparameter tuning (inner CV) ───────────────────
    log(f"\n  Optuna: {N_OPTUNA_TRIALS} trials, {N_INNER_FOLDS}-fold inner CV, "
        f"optimising RMSPE ...")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    objective = make_objective(
        X_train, y_train, anc_train, feature_cols, N_INNER_FOLDS, SEED
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_rmspe  = study.best_value
    log(f"\n  Best inner RMSPE: {best_rmspe:.4f}")
    log(f"  Best params:")
    for k, v in best_params.items():
        log(f"    {k}: {v}")

    # ── Train final model on full outer train ─────────────────────
    # Hold out 10% of outer train for early stopping (never touch test)
    log(f"\n  Training final model (90/10 train/early-stop split) ...")

    final_params = {
        "objective":     "regression",
        "metric":        "None",
        "boosting_type": "gbdt",
        "verbosity":     -1,
        "n_jobs":        -1,
        "seed":          SEED,
        **best_params,
    }

    n_es = max(1, int(len(X_train) * 0.1))
    rng  = np.random.default_rng(SEED)
    idx  = rng.permutation(len(X_train))
    es_idx, tr_idx = idx[:n_es], idx[n_es:]

    w_tr = make_qlike_weights(y_train[tr_idx] + anc_train[tr_idx])

    d_tr = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx], weight=w_tr,
                       feature_name=feature_cols, free_raw_data=False)
    d_es = lgb.Dataset(X_train[es_idx], label=y_train[es_idx],
                       feature_name=feature_cols, free_raw_data=False)
    d_tr.anchor_log_rv = anc_train[tr_idx]
    d_es.anchor_log_rv = anc_train[es_idx]

    eval_log = {}
    final_model = lgb.train(
        final_params, d_tr,
        num_boost_round=N_ROUNDS,
        valid_sets=[d_tr, d_es],
        valid_names=["train", "early_stop"],
        feval=[lgb_qlike_eval, lgb_rmspe_eval, lgb_mse_eval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=200),
            lgb.record_evaluation(eval_log),
        ],
    )

    log(f"  Best iteration: {final_model.best_iteration}")

    # ── Predictions & metrics ─────────────────────────────────────
    test_diff   = final_model.predict(X_test)
    test_log_rv = test_diff + anc_test
    pred_rv     = np.exp(np.clip(test_log_rv, -20, 5))
    true_rv     = np.exp(true_log_rv_test)

    test_m = compute_metrics(test_log_rv, true_log_rv_test,
                             pred_diff=test_diff, true_diff=y_test)

    log(f"\n  Outer test metrics:")
    for k, v in test_m.items():
        log(f"    {k}: {v:.6f}")

    log(f"\n  Per-cluster RMSPE:")
    for cid in sorted(np.unique(cluster_ids_test)):
        mask = cluster_ids_test == cid
        if mask.sum() < 5:
            continue
        m = compute_metrics(test_log_rv[mask], true_log_rv_test[mask])
        log(f"    Cluster {cid} ({len(CLUSTER_STOCKS[cid]):2d} stocks, "
            f"{mask.sum():4d} time_ids): "
            f"RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}  "
            f"MSE_log_rv={m['MSE_log_rv']:.6f}")

    # ── Feature importance ────────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain":    final_model.feature_importance("gain"),
        "split":   final_model.feature_importance("split"),
    }).sort_values("gain", ascending=False)

    log(f"\n  Top 15 features (gain):")
    for _, row in imp_df.head(15).iterrows():
        log(f"    {row['gain']:10.1f}  {row['feature']}")

    # ── Save fold outputs ─────────────────────────────────────────
    log(f"\n  Saving ...")

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
    }).to_csv(fold_dir / "lgbm_predictions.csv", index=False)

    imp_df.to_csv(fold_dir / "lgbm_importance.csv", index=False)
    final_model.save_model(str(fold_dir / "lgbm_model.txt"))

    with open(fold_dir / "lgbm_best_params.json", "w") as f:
        json.dump({
            "best_params":     best_params,
            "best_inner_rmspe": best_rmspe,
            "best_iteration":  final_model.best_iteration,
            "test_metrics":    test_m,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_inner_folds":   N_INNER_FOLDS,
        }, f, indent=2, default=str)

    log(f"  Saved → {fold_dir}/lgbm_*")

    del d_tr, d_es, final_model
    gc.collect()

    return test_m, best_params


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("LightGBM — Nested CV with Optuna Hyperparameter Tuning")
    log(f"  Prediction unit  : (cluster_id, time_id)")
    log(f"  Target           : mean(log_rv_diff) per cluster")
    log(f"  Anchor           : mean({ANCHOR_COL}) per cluster")
    log(f"  Outer folds      : {N_OUTER_FOLDS}")
    log(f"  Inner folds      : {N_INNER_FOLDS}")
    log(f"  Optuna trials    : {N_OPTUNA_TRIALS}")
    log(f"  Optimisation     : RMSPE")
    log(f"  Eval metrics     : RMSPE, QLIKE, MSE, MAPE, RMSE")
    log(f"  Clusters         : {len(CLUSTER_STOCKS)}")
    log("=" * 60)

    all_metrics    = []
    all_params     = []

    for fold in range(N_OUTER_FOLDS):
        fold_dir = DATA_DIR / f"fold_{fold}"
        if not (fold_dir / "feature_store_train.parquet").exists():
            log(f"\n  ⚠ {fold_dir}/feature_store_train.parquet not found — skipping")
            continue

        test_m, best_params = run_outer_fold(fold, fold_dir)
        all_metrics.append(test_m)
        all_params.append(best_params)
        gc.collect()

    # ── Summary across all outer folds ────────────────────────────
    log("\n" + "=" * 60)
    log("NESTED CV SUMMARY")
    log("=" * 60)

    log("\nPer-fold outer test metrics:")
    metric_keys = list(all_metrics[0].keys()) if all_metrics else []
    for i, m in enumerate(all_metrics):
        parts = "  ".join(f"{k}={v:.6f}" for k, v in m.items())
        log(f"  Fold {i}: {parts}")

    if all_metrics:
        log(f"\nAggregated (mean ± std across {len(all_metrics)} outer folds):")
        for k in metric_keys:
            vals = [m[k] for m in all_metrics]
            log(f"  {k:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

    log(f"\nBest params per fold:")
    for i, p in enumerate(all_params):
        log(f"  Fold {i}:")
        for k, v in p.items():
            log(f"    {k}: {v}")

    # Check param stability across folds
    if len(all_params) >= 2:
        log(f"\nParam stability (are folds finding similar params?):")
        for k in all_params[0].keys():
            vals = [p[k] for p in all_params]
            if isinstance(vals[0], (int, float)):
                log(f"  {k:22s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                    f"range=[{min(vals):.4f}, {max(vals):.4f}]")

    # ── Save global summary ───────────────────────────────────────
    summary = {
        "n_outer_folds":    len(all_metrics),
        "n_inner_folds":    N_INNER_FOLDS,
        "n_optuna_trials":  N_OPTUNA_TRIALS,
        "optimisation":     "RMSPE",
        "per_fold_metrics": all_metrics,
        "per_fold_params":  all_params,
        "mean_metrics": {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in metric_keys
        } if all_metrics else {},
        "std_metrics": {
            k: float(np.std([m[k] for m in all_metrics]))
            for k in metric_keys
        } if all_metrics else {},
    }

    with open(DATA_DIR / "lgbm_nested_cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nSaved summary → {DATA_DIR}/lgbm_nested_cv_summary.json")
    log("=" * 60)
    log("Done.")
    log(f"  Inference: log_rv = model.predict(X) + mean({ANCHOR_COL})")


if __name__ == "__main__":
    main()