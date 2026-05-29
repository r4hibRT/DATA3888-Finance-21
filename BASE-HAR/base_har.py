# har_model.py
# Run: python har_model.py

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

# ── Config — change these paths ───────────────────────────────────────────────
DENORM_DIR    = Path(r"E:\Optiver\individual_book_train_denorm")
OUTPUT_DIR    = Path(r"E:\Optiver\outputs\har_predictions")
DATA_DIR      = Path(r"E:\Optiver\processed")
# ─────────────────────────────────────────────────────────────────────────────

FOLD_PATHS    = [(DATA_DIR / f"fold_{i}" / "train.parquet",
                  DATA_DIR / f"fold_{i}" / "test.parquet")
                 for i in range(5)]
OBS_START     = 0
OBS_END       = 480
TARGET_START  = 480
TARGET_END    = 600
N_INNER_FOLDS = 5
EPS           = 1e-10
RV_FLOOR      = 1e-4
N_JOBS        = 4


def parse_stock_id(fname):
    return int(Path(fname).stem.replace("stock_", ""))

def get_fold_time_ids(fold_idx):
    train_path, test_path = FOLD_PATHS[fold_idx]
    train_tids = set(pd.read_parquet(train_path, columns=["time_id"])["time_id"].unique())
    test_tids  = set(pd.read_parquet(test_path,  columns=["time_id"])["time_id"].unique())
    return train_tids, test_tids

def compute_rv(group, sec_min, sec_max):
    mask = (group["seconds_in_bucket"] >= sec_min) & \
           (group["seconds_in_bucket"] <  sec_max)
    sub = group[mask]
    if sub.empty:
        return RV_FLOOR
    log_ret = np.diff(np.log(np.maximum(sub["wap"].values, EPS)))
    return float(max(np.sqrt(np.sum(log_ret**2)), RV_FLOOR))

def make_X(df):
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

def rmspe(y_true, y_pred):
    y_true = np.maximum(y_true, EPS)
    y_pred = np.maximum(y_pred, EPS)
    return float(np.sqrt(np.mean(((y_true - y_pred) / y_true)**2)))

def qlike(y_true, y_pred):
    y_true = np.maximum(y_true, EPS)
    y_pred = np.maximum(y_pred, EPS)
    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))

def build_rv_df(stock_file, stock_id):
    df = pd.read_csv(stock_file)
    records = []
    for tid, group in df.groupby("time_id"):
        rv_in          = compute_rv(group, OBS_START,       OBS_END)
        rv_last_window = compute_rv(group, OBS_END - 120,   OBS_END)
        rv_fut         = compute_rv(group, TARGET_START,     TARGET_END)
        records.append({
            "time_id":            tid,
            "stock_id":           stock_id,
            "rv_in":              rv_in,
            "rv_last_window":     rv_last_window,
            "rv_fut":             rv_fut,
            "log_rv_in":          np.log(max(rv_in,          RV_FLOOR)),
            "log_rv_last_window": np.log(max(rv_last_window, RV_FLOOR)),
            "log_rv_ratio":       np.log(max(rv_last_window, RV_FLOOR)) -
                                  np.log(max(rv_in,          RV_FLOOR)),
            "log_rv_fut":         np.log(max(rv_fut,         RV_FLOOR)),
        })
    return pd.DataFrame(records)

def fit_predict_har_stock(stock_file, fold_idx, train_tids, test_tids):
    stock_id = parse_stock_id(stock_file)
    rv_df    = build_rv_df(stock_file, stock_id)

    train_rv = rv_df[rv_df["time_id"].isin(train_tids)].copy().reset_index(drop=True)
    test_rv  = rv_df[rv_df["time_id"].isin(test_tids)].copy().reset_index(drop=True)

    if len(train_rv) < N_INNER_FOLDS * 2 or test_rv.empty:
        return pd.DataFrame(), pd.DataFrame()

    train_rv["har_pred_log_rv"] = np.nan
    train_rv["har_residual"]    = np.nan

    gkf = GroupKFold(n_splits=N_INNER_FOLDS)
    for inner_train_idx, inner_val_idx in gkf.split(train_rv, groups=train_rv["time_id"]):
        beta = fit_ols(make_X(train_rv.iloc[inner_train_idx]),
                       train_rv.iloc[inner_train_idx]["log_rv_fut"].values)
        if beta is None:
            continue
        preds = make_X(train_rv.iloc[inner_val_idx]) @ beta
        train_rv.loc[inner_val_idx, "har_pred_log_rv"] = preds
        train_rv.loc[inner_val_idx, "har_residual"]    = (
            train_rv.iloc[inner_val_idx]["log_rv_fut"].values - preds
        )

    train_rv["har_pred_rv"] = np.exp(train_rv["har_pred_log_rv"].fillna(0))
    train_rv["fold"]        = fold_idx
    train_rv = train_rv.dropna(subset=["har_residual"])

    beta_full = fit_ols(make_X(train_rv), train_rv["log_rv_fut"].values)
    if beta_full is None:
        return pd.DataFrame(), pd.DataFrame()

    test_rv["har_pred_log_rv"] = make_X(test_rv) @ beta_full
    test_rv["har_residual"]    = test_rv["log_rv_fut"] - test_rv["har_pred_log_rv"]
    test_rv["har_pred_rv"]     = np.exp(test_rv["har_pred_log_rv"])
    test_rv["fold"]            = fold_idx

    out_cols = ["time_id", "stock_id", "fold",
                "rv_fut", "log_rv_fut",
                "har_pred_log_rv", "har_pred_rv", "har_residual"]
    return train_rv[out_cols].copy(), test_rv[out_cols].copy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stock_files = sorted(glob.glob(str(DENORM_DIR / "stock_*.csv")))
    print(f"Found {len(stock_files)} stock files")

    train_folds, test_folds = [], []

    for fold_idx in range(5):
        train_out = OUTPUT_DIR / f"har_fold_{fold_idx}_train.parquet"
        test_out  = OUTPUT_DIR / f"har_fold_{fold_idx}_test.parquet"

        if train_out.exists() and test_out.exists():
            print(f"Fold {fold_idx} already complete — loading from disk")
            train_folds.append(pd.read_parquet(train_out))
            test_folds.append(pd.read_parquet(test_out))
            continue

        print(f"\nFold {fold_idx}...")
        train_tids, test_tids = get_fold_time_ids(fold_idx)
        print(f"  Train time_ids: {len(train_tids)} | Test time_ids: {len(test_tids)}")

        results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=5)(
            delayed(fit_predict_har_stock)(sf, fold_idx, train_tids, test_tids)
            for sf in stock_files
        )

        train_df = pd.concat([r[0] for r in results if len(r[0]) > 0], ignore_index=True)
        test_df  = pd.concat([r[1] for r in results if len(r[1]) > 0], ignore_index=True)

        train_df.to_parquet(train_out, index=False)
        test_df.to_parquet(test_out,   index=False)
        print(f"  Saved — train: {train_df.shape}, test: {test_df.shape}")

        train_folds.append(train_df)
        test_folds.append(test_df)

    # Pooled evaluation
    all_test        = pd.concat(test_folds, ignore_index=True)
    actual_rv_all   = np.exp(all_test["log_rv_fut"].values).clip(min=EPS)
    har_pred_rv_all = all_test["har_pred_rv"].values.clip(min=EPS)
    fold_id         = all_test["fold"].values

    all_test.to_parquet(OUTPUT_DIR / "har_all_folds_test.parquet", index=False)

    print(f"\n{'='*50}")
    print(f"HAR — Pooled OOF Results ({len(all_test):,} rows)")
    print(f"{'='*50}")
    print(f"  RMSPE : {rmspe(actual_rv_all, har_pred_rv_all):.6f}")
    print(f"  QLIKE : {qlike(actual_rv_all, har_pred_rv_all):.6f}")

    print(f"\nPer-fold breakdown:")
    print(f"  {'Fold':<6} {'RMSPE':>10} {'QLIKE':>10} {'N rows':>8}")
    for f in range(5):
        mask = fold_id == f
        print(f"  {f:<6} {rmspe(actual_rv_all[mask], har_pred_rv_all[mask]):>10.6f} "
              f"{qlike(actual_rv_all[mask], har_pred_rv_all[mask]):>10.6f} "
              f"{mask.sum():>8,}")

    fold_rmspe = [rmspe(actual_rv_all[fold_id==f], har_pred_rv_all[fold_id==f]) for f in range(5)]
    fold_qlike = [qlike(actual_rv_all[fold_id==f], har_pred_rv_all[fold_id==f]) for f in range(5)]
    print(f"\nFold std — RMSPE: {np.std(fold_rmspe):.6f}")
    print(f"Fold std — QLIKE: {np.std(fold_qlike):.6f}")
    print(f"\nDone. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()