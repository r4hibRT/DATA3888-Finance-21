"""
feature_pipeline.py — Feature Engineering + Selection for Per-Stock Volatility
================================================================================

Pipeline
--------
  1. Feature selection:
     a. Mutual information — kill useless features
     b. LightGBM + SHAP — rank survivors
     c. Hierarchical clustering on |corr| — pick best per cluster
     d. Permutation importance — final sanity check
  2. Save selected feature list + full report

Target: log(RV_future)
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────

def run_feature_selection(train_df: pd.DataFrame,
                           target_col: str = "target_log_rv",
                           mi_threshold: float = 0.01,
                           corr_threshold: float = 0.7,
                           n_lgb_rounds: int = 500,
                           seed: int = 42) -> dict:
    """
    Full feature selection pipeline:
        1. Mutual information — kill useless features
        2. LightGBM + SHAP — rank survivors
        3. Correlation clustering — de-duplicate, keep best per cluster
        4. Permutation importance — final sanity check
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance

    import lightgbm as lgb
    import shap

    meta_cols    = ["time_id", "stock_id", target_col]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    y = y.fillna(y.median())

    print(f"\nStarting feature selection on {X.shape[1]} features, {X.shape[0]} samples")
    results = {"initial_features": list(feature_cols)}

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Mutual Information
    # ─────────────────────────────────────────────────────────────────
    print("\n═══ Step 1: Mutual Information ═══")
    mi        = mutual_info_regression(X, y, random_state=seed, n_neighbors=5)
    mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)

    survivors_mi = mi_series[mi_series >= mi_threshold].index.tolist()
    dropped_mi   = mi_series[mi_series <  mi_threshold].index.tolist()
    print(f"  MI threshold: {mi_threshold}")
    print(f"  Survivors: {len(survivors_mi)} | Dropped: {len(dropped_mi)}")
    if dropped_mi:
        print(f"  Dropped: {dropped_mi[:10]}{'…' if len(dropped_mi) > 10 else ''}")
    print(f"  Top 10 MI:\n{mi_series.head(10).to_string()}")

    results["mi_scores"]    = mi_series.to_dict()
    results["mi_survivors"] = survivors_mi
    results["mi_dropped"]   = dropped_mi

    X_mi = X[survivors_mi]

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: LightGBM + SHAP
    # ─────────────────────────────────────────────────────────────────
    print("\n═══ Step 2: LightGBM + SHAP ═══")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_mi, y, test_size=0.2, random_state=seed
    )

    lgb_params = {
        "objective":         "regression",
        "metric":            "rmse",
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         8,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "verbose":           -1,
        "seed":              seed,
    }

    dtrain = lgb.Dataset(X_tr, y_tr)
    dval   = lgb.Dataset(X_val, y_val, reference=dtrain)

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=n_lgb_rounds,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(50),
        ],
    )

    gain_importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=survivors_mi,
    ).sort_values(ascending=False)
    print(f"\n  Top 10 by gain:\n{gain_importance.head(10).to_string()}")

    print("\n  Computing SHAP values …")
    explainer       = shap.TreeExplainer(model)
    shap_values     = explainer.shap_values(X_val)
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=survivors_mi,
    ).sort_values(ascending=False)
    print(f"  Top 10 by SHAP:\n{shap_importance.head(10).to_string()}")

    results["lgb_model"]       = model
    results["gain_importance"] = gain_importance.to_dict()
    results["shap_importance"] = shap_importance.to_dict()

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Correlation Clustering — pick best per cluster
    # ─────────────────────────────────────────────────────────────────
    print(f"\n═══ Step 3: Correlation Clustering (|r| > {corr_threshold} grouped) ═══")

    corr_matrix = np.abs(X_mi.corr().values)
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, 0, 1)
    distance    = 1.0 - corr_matrix
    distance    = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    distance    = np.clip(distance, 0, None)

    condensed = squareform(distance, checks=False)
    Z         = linkage(condensed, method="average")
    clusters  = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")

    cluster_map = pd.DataFrame({
        "feature": survivors_mi,
        "cluster": clusters,
        "shap":    [shap_importance.get(f, 0.0) for f in survivors_mi],
    })

    # From each cluster keep the feature with highest SHAP
    selected_decorr = []
    for cid, grp in cluster_map.groupby("cluster"):
        best = grp.sort_values("shap", ascending=False).iloc[0]
        selected_decorr.append(best["feature"])

    n_corr_clusters = cluster_map["cluster"].nunique()
    print(f"  {n_corr_clusters} correlation clusters from {len(survivors_mi)} features")
    print(f"  Selected {len(selected_decorr)} de-correlated features")

    for cid, grp in cluster_map.groupby("cluster"):
        if len(grp) > 1:
            feats = grp.sort_values("shap", ascending=False)["feature"].tolist()
            print(f"  Cluster {cid}: kept '{feats[0]}', dropped {feats[1:]}")

    results["correlation_clusters"]  = cluster_map.to_dict(orient="records")
    results["decorrelated_features"] = selected_decorr

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Permutation Importance
    # ─────────────────────────────────────────────────────────────────
    print(f"\n═══ Step 4: Permutation Importance ═══")

    X_decorr = train_df[selected_decorr].replace([np.inf, -np.inf], np.nan)
    X_decorr = X_decorr.fillna(X_decorr.median())

    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_decorr, y, test_size=0.2, random_state=seed
    )

    dtrain_f = lgb.Dataset(X_tr_f, y_tr_f)
    dval_f   = lgb.Dataset(X_val_f, y_val_f, reference=dtrain_f)
    model_f  = lgb.train(
        lgb_params, dtrain_f,
        num_boost_round=n_lgb_rounds,
        valid_sets=[dtrain_f, dval_f],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    class _LGBWrapper:
        def __init__(self, m): self.m = m
        def fit(self, X, y):   return self
        def predict(self, X):  return self.m.predict(X)

    perm = permutation_importance(
        _LGBWrapper(model_f), X_val_f, y_val_f,
        n_repeats=10, random_state=seed,
        scoring="neg_mean_squared_error",
    )
    perm_importance = pd.Series(
        perm.importances_mean, index=selected_decorr
    ).sort_values(ascending=False)

    perm_survivors = perm_importance[perm_importance > 0].index.tolist()

    print(f"  Features with positive permutation importance: "
          f"{len(perm_survivors)}/{len(selected_decorr)}")
    print(f"  Top 10:\n{perm_importance.head(10).to_string()}")

    neg_perm = perm_importance[perm_importance <= 0].index.tolist()
    if neg_perm:
        print(f"  Dropped (negative perm importance): {neg_perm}")

    results["perm_importance"] = perm_importance.to_dict()
    results["perm_survivors"]  = perm_survivors
    results["final_features"]  = perm_survivors
    results["final_model"]     = model_f

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"Feature Selection Summary")
    print(f"{'═' * 60}")
    print(f"  Initial features             : {len(feature_cols)}")
    print(f"  After MI filter              : {len(survivors_mi)}")
    print(f"  After correlation clustering : {len(selected_decorr)}")
    print(f"  After permutation importance : {len(perm_survivors)}")
    print(f"\n  FINAL FEATURES ({len(perm_survivors)}):")
    for i, f in enumerate(perm_survivors):
        shap_v = shap_importance.get(f, 0)
        perm_v = perm_importance.get(f, 0)
        print(f"    {i+1:>3}. {f:<40s} SHAP={shap_v:.4f}  Perm={perm_v:.6f}")
    print(f"{'═' * 60}")

    return results


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main(seed: int,
         output_dir: str,
         run_selection: bool = True):

    train_df = pd.read_parquet(os.path.join(output_dir, "train.parquet"))

    if not run_selection:
        return train_df, None

    selection_results = run_feature_selection(train_df, seed=seed)

    # Save feature list
    final_feats = selection_results["final_features"]
    feat_path   = os.path.join(output_dir, "selected_features.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(final_feats))
    print(f"\n  Saved feature list: {feat_path}")

    # Save report
    def _clean(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, dict):        return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):        return [_clean(v) for v in obj]
        return obj

    report = {k: v for k, v in selection_results.items()
              if k not in ("lgb_model", "final_model")}
    report_path = os.path.join(output_dir, "feature_selection_report.json")
    with open(report_path, "w") as f:
        json.dump(_clean(report), f, indent=2, default=str)
    print(f"  Saved report: {report_path}")

    return train_df, selection_results


if __name__ == "__main__":
    OUTPUT_DIR = "processed"

    main(
        seed          = 42,
        output_dir    = OUTPUT_DIR,
        run_selection = True,
    )