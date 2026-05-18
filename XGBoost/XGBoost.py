"""
xgb_rv.py
=========
XGBoost model for realized volatility forecasting at individual stock level.
Predicts log_rv_diff for each (stock_id, time_id).

Pipeline:
  eda.ipynb -> processed/fold_{0..4}/{train,test}.parquet
  feature_engineer.py -> processed/fold_{0..4}/feature_store_{train,test}.parquet
  -> this script -> processed/fold_{0..4}/xgb_* outputs

Nested CV:
  Outer loop : 5 folds from eda.ipynb (GroupShuffleSplit on time_id)
  Inner loop : Optuna tunes hyperparameters via 5-fold CV on outer train
  Final      : one model per outer fold, trained on full outer train with best params
  Evaluation : RMSPE (optimization), QLIKE, MSE, MAPE (reporting)

Prediction unit: (stock_id, time_id)
  Target   : log_rv_diff
  Anchor   : past_log_rv_bkt3
  Inference: log_rv = model.predict(X) + past_log_rv_bkt3
"""

import gc
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

DATA_DIR        = Path("C:\\Users\\ngdo0466\\Downloads\\processed")
N_OUTER_FOLDS   = 5
N_INNER_FOLDS   = 3
N_OPTUNA_TRIALS = 10
N_ROUNDS        = 50
EARLY_STOPPING  = 5
EPS             = 1e-8
ANCHOR_COL      = "past_log_rv_bkt3"


def _has_gpu():
    try:
        _tmp = xgb.DMatrix(np.zeros((2, 2)), label=np.zeros(2))
        xgb.train({"tree_method": "gpu_hist", "device": "cuda", "verbosity": 0},
                  _tmp, num_boost_round=1)
        return True
    except Exception:
        return False

USE_GPU = _has_gpu()


def log(msg):
    print(f"[XGB] {msg}", flush=True)


def make_qlike_weights(true_log_rv):
    true_rv = np.exp(true_log_rv).clip(min=EPS)
    w = 1.0 / true_rv
    return np.clip(w, 0.0, float(np.percentile(w, 99))).astype(np.float32)


def xgb_qlike_eval(preds, dtrain):
    labels = dtrain.get_label(); anchor = dtrain.anchor_log_rv
    pred_rv = np.exp(np.clip(preds + anchor, -20, 5)).clip(min=EPS)
    true_rv = np.exp((labels + anchor)).clip(min=EPS)
    ratio = true_rv / pred_rv
    qlike = float(np.mean(ratio - np.log(ratio) - 1))
    rmspe = float(np.sqrt(np.mean(((pred_rv - true_rv) / true_rv) ** 2)))
    mse   = float(np.mean((preds - labels) ** 2))
    return [("qlike", qlike), ("rmspe", rmspe), ("mse", mse)]


def compute_metrics(pred_log, true_log, pred_diff=None, true_diff=None):
    pred_log = np.clip(pred_log, -20, 5)
    pred_rv = np.clip(np.exp(pred_log), EPS, None)
    true_rv = np.clip(np.exp(true_log), EPS, None)
    ratio = true_rv / pred_rv
    resid = pred_rv - true_rv
    result = {
        "QLIKE":    float(np.mean(ratio - np.log(ratio) - 1)),
        "RMSE":     float(np.sqrt(np.mean(resid ** 2))),
        "MAPE%":    float(np.mean(np.abs(resid) / true_rv) * 100),
        "RMSPE%":   float(np.sqrt(np.mean((resid / true_rv) ** 2)) * 100),
        "MAE_log":  float(np.mean(np.abs(pred_log - true_log))),
    }
    if pred_diff is not None and true_diff is not None:
        result["MSE_diff"] = float(np.mean((pred_diff - true_diff) ** 2))
    result["MSE_log_rv"] = float(np.mean((pred_log - true_log) ** 2))
    result["MSE_rv"]     = float(np.mean((pred_rv - true_rv) ** 2))
    return result


def make_objective(X, y, anchor, feature_cols, n_inner_folds, seed, use_gpu):
    n_samples = len(X)
    def objective(trial):
        params = {
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "device": "cuda" if use_gpu else "cpu",
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "verbosity": 0, "nthread": -1, "seed": seed,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        }
        kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        scores = []
        for tr_idx, va_idx in kf.split(X):
            w_tr = make_qlike_weights(y[tr_idx] + anchor[tr_idx])
            d_tr = xgb.DMatrix(X[tr_idx], label=y[tr_idx], weight=w_tr, feature_names=feature_cols)
            d_va = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=feature_cols)
            d_tr.anchor_log_rv = anchor[tr_idx]
            d_va.anchor_log_rv = anchor[va_idx]
            model = xgb.train(params, d_tr, num_boost_round=N_ROUNDS,
                              evals=[(d_va, "val")], custom_metric=xgb_qlike_eval,
                              early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
            va_diff = model.predict(d_va)
            pred_rv = np.exp(np.clip(va_diff + anchor[va_idx], -20, 5)).clip(min=EPS)
            true_rv = np.exp(y[va_idx] + anchor[va_idx]).clip(min=EPS)
            scores.append(float(np.sqrt(np.mean(((pred_rv - true_rv) / true_rv) ** 2))))
            del d_tr, d_va, model; gc.collect()
        return float(np.mean(scores))
    return objective


def run_outer_fold(fold, fold_dir):
    log(f"\n{'=' * 60}")
    log(f"  OUTER FOLD {fold}  |  {fold_dir}  |  {'GPU' if USE_GPU else 'CPU'}")
    log(f"{'=' * 60}")

    train = pd.read_parquet(fold_dir / "feature_store_train.parquet")
    test  = pd.read_parquet(fold_dir / "feature_store_test.parquet")

    feat_path = fold_dir / "selected_features.txt"
    if feat_path.exists():
        with open(feat_path) as f:
            feature_cols = [l.strip() for l in f if l.strip()]
        feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]
    else:
        exclude = {"stock_id", "time_id", "log_rv", "rv", "log_rv_diff"}
        feature_cols = [c for c in train.columns if c not in exclude
                        and train[c].dtype in (np.float32, np.float64, np.int32, np.int16, np.int8)]

    log(f"  Train: {train.shape}  |  Test: {test.shape}  |  Features: {len(feature_cols)}")

    anc_train = train[ANCHOR_COL].values.astype(np.float32)
    anc_test  = test[ANCHOR_COL].values.astype(np.float32)
    y_train = train["log_rv_diff"].values.astype(np.float32) if "log_rv_diff" in train.columns \
        else (train["log_rv"].values - anc_train).astype(np.float32)
    y_test = test["log_rv_diff"].values.astype(np.float32) if "log_rv_diff" in test.columns \
        else (test["log_rv"].values - anc_test).astype(np.float32)
    true_log_rv_test = (y_test + anc_test).astype(np.float32)
    stock_ids_test = test["stock_id"].values
    time_ids_test  = test["time_id"].values
    X_train = train[feature_cols].values.astype(np.float32)
    X_test  = test[feature_cols].values.astype(np.float32)

    log(f"  Target: mean={y_train.mean():.4f}  std={y_train.std():.4f}")

    # ── Optuna ────────────────────────────────────────────────────
    log(f"\n  Optuna: {N_OPTUNA_TRIALS} trials, {N_INNER_FOLDS}-fold, optimising RMSPE ...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(X_train, y_train, anc_train, feature_cols,
                                  N_INNER_FOLDS, SEED, USE_GPU),
                   n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    log(f"  Best inner RMSPE: {study.best_value:.4f}")
    for k, v in best_params.items():
        log(f"    {k}: {v}")

    # ── Final model (90/10 early-stop holdout) ────────────────────
    log(f"\n  Training final model ...")
    final_params = {"tree_method": "gpu_hist" if USE_GPU else "hist",
                    "device": "cuda" if USE_GPU else "cpu",
                    "objective": "regression", "eval_metric": "rmse",
                    "verbosity": 0, "nthread": -1, "seed": SEED, **best_params}
    n_es = max(1, int(len(X_train) * 0.1))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_train))
    es_idx, tr_idx = idx[:n_es], idx[n_es:]

    w_tr = make_qlike_weights(y_train[tr_idx] + anc_train[tr_idx])
    d_tr = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx], weight=w_tr, feature_names=feature_cols)
    d_es = xgb.DMatrix(X_train[es_idx], label=y_train[es_idx], feature_names=feature_cols)
    d_tr.anchor_log_rv = anc_train[tr_idx]
    d_es.anchor_log_rv = anc_train[es_idx]

    eval_log = {}
    final_model = xgb.train(final_params, d_tr, num_boost_round=N_ROUNDS,
                            evals=[(d_tr, "train"), (d_es, "early_stop")],
                            custom_metric=xgb_qlike_eval,
                            early_stopping_rounds=EARLY_STOPPING,
                            verbose_eval=200, evals_result=eval_log)
    log(f"  Best iteration: {final_model.best_iteration}")

    # ── Evaluate ──────────────────────────────────────────────────
    d_test = xgb.DMatrix(X_test, feature_names=feature_cols)
    test_diff = final_model.predict(d_test)
    test_log_rv = test_diff + anc_test
    pred_rv = np.exp(np.clip(test_log_rv, -20, 5))
    true_rv = np.exp(true_log_rv_test)
    test_m = compute_metrics(test_log_rv, true_log_rv_test, pred_diff=test_diff, true_diff=y_test)

    log(f"\n  Outer test metrics:")
    for k, v in test_m.items():
        log(f"    {k}: {v:.6f}")

    # Per-stock breakdown
    stock_metrics = {}
    for sid in np.unique(stock_ids_test):
        mask = stock_ids_test == sid
        if mask.sum() >= 5:
            stock_metrics[sid] = compute_metrics(test_log_rv[mask], true_log_rv_test[mask])
    if stock_metrics:
        sorted_stocks = sorted(stock_metrics.items(), key=lambda x: x[1]["RMSPE%"])
        log(f"\n  Best 5 stocks by RMSPE%:")
        for sid, m in sorted_stocks[:5]:
            log(f"    Stock {sid}: RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}")
        log(f"  Worst 5 stocks by RMSPE%:")
        for sid, m in sorted_stocks[-5:]:
            log(f"    Stock {sid}: RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}")

    # Feature importance
    gain_map = final_model.get_score(importance_type="gain")
    weight_map = final_model.get_score(importance_type="weight")
    imp_df = pd.DataFrame({"feature": feature_cols,
                           "gain": [gain_map.get(c, 0) for c in feature_cols],
                           "weight": [weight_map.get(c, 0) for c in feature_cols]}).sort_values("gain", ascending=False)
    log(f"\n  Top 15 features (gain):")
    for _, row in imp_df.head(15).iterrows():
        log(f"    {row['gain']:10.1f}  {row['feature']}")

    # ── Save ──────────────────────────────────────────────────────
    pd.DataFrame({"stock_id": stock_ids_test, "time_id": time_ids_test,
                  "pred_log_rv": np.clip(test_log_rv, -20, 5).astype(np.float32),
                  "true_log_rv": true_log_rv_test.astype(np.float32),
                  "pred_rv": pred_rv.astype(np.float32), "true_rv": true_rv.astype(np.float32),
                  "pred_diff": test_diff.astype(np.float32), "true_diff": y_test.astype(np.float32),
                  "anchor": anc_test.astype(np.float32)}
                 ).to_csv(fold_dir / "xgb_predictions.csv", index=False)
    imp_df.to_csv(fold_dir / "xgb_importance.csv", index=False)
    final_model.save_model(str(fold_dir / "xgb_model.json"))
    with open(fold_dir / "xgb_best_params.json", "w") as f:
        json.dump({"best_params": best_params, "best_inner_rmspe": study.best_value,
                   "best_iteration": final_model.best_iteration, "test_metrics": test_m,
                   "n_optuna_trials": N_OPTUNA_TRIALS, "n_inner_folds": N_INNER_FOLDS,
                   "use_gpu": USE_GPU}, f, indent=2, default=str)
    log(f"  Saved -> {fold_dir}/xgb_*")
    del d_tr, d_es, d_test, final_model; gc.collect()
    return test_m, best_params


def main():
    log("=" * 60)
    log("XGBoost — Nested CV with Optuna (individual stock)")
    log(f"  Prediction unit  : (stock_id, time_id)")
    log(f"  Target           : log_rv_diff")
    log(f"  Anchor           : {ANCHOR_COL}")
    log(f"  Device           : {'GPU' if USE_GPU else 'CPU'}")
    log(f"  Outer folds      : {N_OUTER_FOLDS}")
    log(f"  Inner folds      : {N_INNER_FOLDS}")
    log(f"  Optuna trials    : {N_OPTUNA_TRIALS}")
    log(f"  Optimisation     : RMSPE")
    log("=" * 60)

    all_metrics, all_params = [], []
    for fold in range(N_OUTER_FOLDS):
        fold_dir = DATA_DIR / f"fold_{fold}"
        if not (fold_dir / "feature_store_train.parquet").exists():
            log(f"\n  {fold_dir}/feature_store_train.parquet not found -- skipping"); continue
        test_m, bp = run_outer_fold(fold, fold_dir)
        all_metrics.append(test_m); all_params.append(bp); gc.collect()

    log("\n" + "=" * 60)
    log("NESTED CV SUMMARY")
    log("=" * 60)
    metric_keys = list(all_metrics[0].keys()) if all_metrics else []
    for i, m in enumerate(all_metrics):
        log(f"  Fold {i}: " + "  ".join(f"{k}={v:.6f}" for k, v in m.items()))
    if all_metrics:
        log(f"\nAggregated (mean +/- std):")
        for k in metric_keys:
            vals = [m[k] for m in all_metrics]
            log(f"  {k:12s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")
    if len(all_params) >= 2:
        log(f"\nParam stability:")
        for k in all_params[0]:
            vals = [p[k] for p in all_params]
            if isinstance(vals[0], (int, float)):
                log(f"  {k:22s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    with open(DATA_DIR / "xgb_nested_cv_summary.json", "w") as f:
        json.dump({"n_outer_folds": len(all_metrics), "n_inner_folds": N_INNER_FOLDS,
                   "n_optuna_trials": N_OPTUNA_TRIALS, "optimisation": "RMSPE", "use_gpu": USE_GPU,
                   "per_fold_metrics": all_metrics, "per_fold_params": all_params,
                   "mean_metrics": {k: float(np.mean([m[k] for m in all_metrics])) for k in metric_keys} if all_metrics else {},
                   "std_metrics": {k: float(np.std([m[k] for m in all_metrics])) for k in metric_keys} if all_metrics else {}},
                  f, indent=2, default=str)
    log(f"\nDone. Inference: log_rv = model.predict(X) + {ANCHOR_COL}")

if __name__ == "__main__":
    main()