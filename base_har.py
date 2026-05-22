"""Base HAR model that runs directly on processed Optiver CV folds.

Default run:
    python3 base_har.py --folds-root processed

The model is intentionally simple and close to the original base HAR script:
- one separate OLS HAR model per stock
- input window: seconds 0-479
- target window: seconds 480-599
- features: log input RV, log last-window RV, and their log ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold


OBS_START = 0
OBS_END = 480
TARGET_START = 480
TARGET_END = 600
DEFAULT_INNER_FOLDS = 5
DEFAULT_EPS = 1e-10
DEFAULT_RV_FLOOR = 1e-4
DEFAULT_N_JOBS = 4


def normalise_stock_id(series: pd.Series, source: Path) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    extracted = series.astype(str).str.extract(r"(\d+)$")[0]
    if extracted.isna().any():
        bad_values = series[extracted.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{source} has stock_id values that do not end in an integer: {bad_values}"
        )
    return extracted.astype(int)


def compute_rv(
    group: pd.DataFrame,
    sec_min: int,
    sec_max: int,
    eps: float,
    rv_floor: float,
) -> float:
    mask = group["seconds_in_bucket"].ge(sec_min) & group["seconds_in_bucket"].lt(sec_max)
    sub = group.loc[mask].sort_values("seconds_in_bucket")
    if sub.empty:
        return rv_floor

    log_wap = np.log(np.maximum(sub["wap"].to_numpy(dtype=float), eps))
    log_ret = np.diff(log_wap)
    return float(max(np.sqrt(np.sum(log_ret**2)), rv_floor))


def make_har_row(
    stock_id: int,
    time_id: int,
    group: pd.DataFrame,
    eps: float,
    rv_floor: float,
) -> dict[str, float | int]:
    rv_in = compute_rv(group, OBS_START, OBS_END, eps, rv_floor)
    rv_last_window = compute_rv(group, OBS_END - 120, OBS_END, eps, rv_floor)
    rv_fut = compute_rv(group, TARGET_START, TARGET_END, eps, rv_floor)

    log_rv_in = np.log(max(rv_in, rv_floor))
    log_rv_last_window = np.log(max(rv_last_window, rv_floor))
    return {
        "time_id": int(time_id),
        "stock_id": int(stock_id),
        "rv_in": rv_in,
        "rv_last_window": rv_last_window,
        "rv_fut": rv_fut,
        "log_rv_in": log_rv_in,
        "log_rv_last_window": log_rv_last_window,
        "log_rv_ratio": log_rv_last_window - log_rv_in,
        "log_rv_fut": np.log(max(rv_fut, rv_floor)),
    }


def build_rv_df_from_frame(
    frame: pd.DataFrame,
    source: Path,
    eps: float,
    rv_floor: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame["stock_id"] = normalise_stock_id(frame["stock_id"], source)
    frame = frame.sort_values(["stock_id", "time_id", "seconds_in_bucket"])
    rows = [
        make_har_row(stock_id, time_id, group, eps, rv_floor)
        for (stock_id, time_id), group in frame.groupby(["stock_id", "time_id"], sort=False)
    ]
    return pd.DataFrame(rows)


def build_rv_df_from_processed_parquet(
    path: Path,
    eps: float,
    rv_floor: float,
    max_stocks: int | None = None,
) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading processed parquet folds requires pyarrow. "
            "Install pyarrow or use a raw-CSV workflow instead."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    required_cols = {"stock_id", "time_id", "seconds_in_bucket", "wap"}
    available_cols = set(parquet_file.schema.names)
    missing = sorted(required_cols - available_cols)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    features = []
    current_stock_id: int | None = None
    current_chunks: list[pd.DataFrame] = []
    completed_stock_ids: set[int] = set()

    def flush_current_stock() -> None:
        nonlocal current_stock_id, current_chunks
        if current_stock_id is None or not current_chunks:
            return
        stock_frame = pd.concat(current_chunks, ignore_index=True)
        features.append(build_rv_df_from_frame(stock_frame, path, eps, rv_floor))
        completed_stock_ids.add(current_stock_id)
        print(
            f"  {path.name}: built base HAR rows for stock {current_stock_id} "
            f"({len(completed_stock_ids)} stocks)"
        )
        current_stock_id = None
        current_chunks = []

    read_cols = ["stock_id", "time_id", "seconds_in_bucket", "wap"]
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=read_cols)
        chunk = table.to_pandas()
        chunk["stock_id"] = normalise_stock_id(chunk["stock_id"], path)

        for stock_id, stock_chunk in chunk.groupby("stock_id", sort=False):
            stock_id = int(stock_id)
            if stock_id in completed_stock_ids:
                raise ValueError(
                    f"{path} is not sorted by stock_id: stock {stock_id} appeared again "
                    "after being flushed."
                )

            if current_stock_id is None:
                current_stock_id = stock_id
            elif stock_id != current_stock_id:
                flush_current_stock()
                if max_stocks is not None and len(completed_stock_ids) >= max_stocks:
                    break
                current_stock_id = stock_id

            current_chunks.append(stock_chunk.copy())

        if max_stocks is not None and len(completed_stock_ids) >= max_stocks:
            break

    if max_stocks is None or len(completed_stock_ids) < max_stocks:
        flush_current_stock()

    if not features:
        raise ValueError(f"{path} did not contain any rows after filtering.")

    return pd.concat(features, ignore_index=True)


def make_x(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df)),
            df["log_rv_in"].to_numpy(dtype=float),
            df["log_rv_last_window"].to_numpy(dtype=float),
            df["log_rv_ratio"].to_numpy(dtype=float),
        ]
    )


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


def rmspe(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    return float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))


def fit_predict_har_stock(
    stock_id: int,
    fold_idx: int,
    train_rv: pd.DataFrame,
    test_rv: pd.DataFrame,
    n_inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rv = train_rv.copy().reset_index(drop=True)
    test_rv = test_rv.copy().reset_index(drop=True)

    n_splits = min(n_inner_folds, train_rv["time_id"].nunique(), len(train_rv))
    if n_splits < 2 or test_rv.empty:
        return pd.DataFrame(), pd.DataFrame()

    train_rv["har_pred_log_rv"] = np.nan
    train_rv["har_residual"] = np.nan

    gkf = GroupKFold(n_splits=n_splits)
    for inner_train_idx, inner_val_idx in gkf.split(train_rv, groups=train_rv["time_id"]):
        beta = fit_ols(
            make_x(train_rv.iloc[inner_train_idx]),
            train_rv.iloc[inner_train_idx]["log_rv_fut"].to_numpy(dtype=float),
        )
        if beta is None:
            continue
        preds = make_x(train_rv.iloc[inner_val_idx]) @ beta
        train_rv.loc[inner_val_idx, "har_pred_log_rv"] = preds
        train_rv.loc[inner_val_idx, "har_residual"] = (
            train_rv.iloc[inner_val_idx]["log_rv_fut"].to_numpy(dtype=float) - preds
        )

    train_rv["har_pred_rv"] = np.exp(train_rv["har_pred_log_rv"])
    train_rv["fold"] = fold_idx
    train_rv = train_rv.dropna(subset=["har_residual"]).copy()
    if len(train_rv) < 2:
        return pd.DataFrame(), pd.DataFrame()

    beta_full = fit_ols(make_x(train_rv), train_rv["log_rv_fut"].to_numpy(dtype=float))
    if beta_full is None:
        return pd.DataFrame(), pd.DataFrame()

    test_rv["har_pred_log_rv"] = make_x(test_rv) @ beta_full
    test_rv["har_residual"] = test_rv["log_rv_fut"] - test_rv["har_pred_log_rv"]
    test_rv["har_pred_rv"] = np.exp(test_rv["har_pred_log_rv"])
    test_rv["fold"] = fold_idx

    out_cols = [
        "time_id",
        "stock_id",
        "fold",
        "rv_fut",
        "log_rv_fut",
        "har_pred_log_rv",
        "har_pred_rv",
        "har_residual",
    ]
    train_rv["stock_id"] = stock_id
    test_rv["stock_id"] = stock_id
    return train_rv[out_cols].copy(), test_rv[out_cols].copy()


def fit_predict_fold(
    fold_idx: int,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    n_inner_folds: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_by_stock = {
        int(stock_id): stock_df
        for stock_id, stock_df in test_features.groupby("stock_id", sort=False)
    }

    jobs = [
        delayed(fit_predict_har_stock)(
            int(stock_id),
            fold_idx,
            train_stock,
            test_by_stock.get(int(stock_id), pd.DataFrame()),
            n_inner_folds,
        )
        for stock_id, train_stock in train_features.groupby("stock_id", sort=False)
    ]
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=5)(jobs)

    train_frames = [result[0] for result in results if len(result[0]) > 0]
    test_frames = [result[1] for result in results if len(result[1]) > 0]
    if not train_frames or not test_frames:
        return pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(train_frames, ignore_index=True),
        pd.concat(test_frames, ignore_index=True),
    )


def metric_row(
    fold: int | str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float,
) -> dict[str, float | int | str]:
    return {
        "fold": fold,
        "n_rows": int(len(y_true)),
        "RMSPE": rmspe(y_true, y_pred, eps),
        "QLIKE": qlike(y_true, y_pred, eps),
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
    }


def evaluate_folds(test_folds: list[pd.DataFrame], eps: float) -> pd.DataFrame:
    rows = []
    for fold_df in test_folds:
        fold = int(fold_df["fold"].iloc[0])
        y_true = np.exp(fold_df["log_rv_fut"].to_numpy(dtype=float)).clip(min=eps)
        y_pred = fold_df["har_pred_rv"].to_numpy(dtype=float).clip(min=eps)
        rows.append(metric_row(fold, y_true, y_pred, eps))

    all_test = pd.concat(test_folds, ignore_index=True)
    y_true_all = np.exp(all_test["log_rv_fut"].to_numpy(dtype=float)).clip(min=eps)
    y_pred_all = all_test["har_pred_rv"].to_numpy(dtype=float).clip(min=eps)
    rows.append(metric_row("all", y_true_all, y_pred_all, eps))

    fold_metrics = pd.DataFrame(rows[:-1])
    metric_cols = ["RMSPE", "QLIKE", "MSE"]
    rows.append(
        {
            "fold": "mean",
            "n_rows": float(fold_metrics["n_rows"].mean()),
            **{col: float(fold_metrics[col].mean()) for col in metric_cols},
        }
    )
    rows.append(
        {
            "fold": "std",
            "n_rows": float(fold_metrics["n_rows"].std(ddof=0)),
            **{col: float(fold_metrics[col].std(ddof=0)) for col in metric_cols},
        }
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the base per-stock HAR model on processed CV folds."
    )
    parser.add_argument("--folds-root", type=Path, default=Path("processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("base_har_outputs"))
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--rv-floor", type=float, default=DEFAULT_RV_FLOOR)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of stocks read from each parquet file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute fold outputs even when output parquet files already exist.",
    )
    return parser.parse_args()


def discover_folds(folds_root: Path, selected_folds: list[int] | None) -> list[int]:
    if selected_folds:
        folds = selected_folds
    else:
        folds = []
        for path in sorted(folds_root.glob("fold_*")):
            try:
                folds.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue

    if not folds:
        raise FileNotFoundError(f"No fold_* directories found in {folds_root.resolve()}")

    for fold in folds:
        fold_dir = folds_root / f"fold_{fold}"
        missing = [
            path for path in [fold_dir / "train.parquet", fold_dir / "test.parquet"]
            if not path.exists()
        ]
        if missing:
            missing_text = ", ".join(str(path.resolve()) for path in missing)
            raise FileNotFoundError(f"Missing fold parquet file(s): {missing_text}")
    return folds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = discover_folds(args.folds_root, args.folds)

    train_folds = []
    test_folds = []
    for fold_idx in folds:
        fold_dir = args.folds_root / f"fold_{fold_idx}"
        train_out = args.output_dir / f"base_har_fold_{fold_idx}_train.parquet"
        test_out = args.output_dir / f"base_har_fold_{fold_idx}_test.parquet"

        if train_out.exists() and test_out.exists() and not args.overwrite:
            print(f"Fold {fold_idx} already complete; loading from disk")
            train_folds.append(pd.read_parquet(train_out))
            test_folds.append(pd.read_parquet(test_out))
            continue

        print(f"\nFold {fold_idx}: {fold_dir}")
        print("  Building train HAR rows")
        train_features = build_rv_df_from_processed_parquet(
            fold_dir / "train.parquet",
            eps=args.eps,
            rv_floor=args.rv_floor,
            max_stocks=args.max_stocks,
        )
        print("  Building test HAR rows")
        test_features = build_rv_df_from_processed_parquet(
            fold_dir / "test.parquet",
            eps=args.eps,
            rv_floor=args.rv_floor,
            max_stocks=args.max_stocks,
        )

        print(
            f"  Fitting per-stock base HAR models: "
            f"train={train_features.shape}, test={test_features.shape}"
        )
        train_df, test_df = fit_predict_fold(
            fold_idx,
            train_features,
            test_features,
            n_inner_folds=args.inner_folds,
            n_jobs=args.n_jobs,
        )
        if train_df.empty or test_df.empty:
            raise ValueError(f"Fold {fold_idx} produced no predictions.")

        train_df.to_parquet(train_out, index=False)
        test_df.to_parquet(test_out, index=False)
        print(f"  Saved train: {train_out} {train_df.shape}")
        print(f"  Saved test:  {test_out} {test_df.shape}")

        train_folds.append(train_df)
        test_folds.append(test_df)

    all_test = pd.concat(test_folds, ignore_index=True)
    all_test_out = args.output_dir / "base_har_all_folds_test.parquet"
    all_test.to_parquet(all_test_out, index=False)

    metrics = evaluate_folds(test_folds, args.eps)
    metrics_out = args.output_dir / "base_har_metrics.csv"
    metrics.to_csv(metrics_out, index=False)

    print(f"\n{'=' * 50}")
    print(f"Base HAR results ({len(all_test):,} test rows)")
    print(f"{'=' * 50}")
    print(metrics.to_string(index=False))
    print(f"\nSaved all test predictions: {all_test_out.resolve()}")
    print(f"Saved metrics: {metrics_out.resolve()}")


if __name__ == "__main__":
    main()
