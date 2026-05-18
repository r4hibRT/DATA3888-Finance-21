"""Cluster-level HAR-RV benchmark.

This script is self-contained for HAR-RV:
- stock clusters come from the clustering output CSV
- features are built from raw or preprocessed stock source files
- rows are evaluated at cluster_id x time_id level
- target is future cluster-average realised volatility

The model is intentionally simple and interpretable. Each cluster gets a
separate linear regression using only realised-volatility HAR features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
PREDICTION_FLOOR = 1e-5
INPUT_END = 480
LAST_WINDOW_START = 360
TARGET_START = 480
TARGET_END = 600
DEFAULT_DATA_DIR = Path("individual_book_train")
DEFAULT_CLUSTER_LABELS = Path(
    "Clustering+Feature engineering/processed/clustering/best_cluster_labels.csv"
)
DEFAULT_CLUSTER_FEATURES = Path("har_rv_cluster_features.csv")
DEFAULT_FEATURES = [
    "log_rv_input",
    "log_rv_last_window",
    "rv_last_window_to_input",
]


def stock_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def sorted_stock_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("stock_*.csv"), key=stock_id_from_path)


def load_cluster_labels(path: Path, expected_clusters: int = 3) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cluster labels file not found: {path.resolve()}")

    labels = pd.read_csv(path)
    required_cols = {"stock_id", "cluster_id"}
    if not required_cols.issubset(labels.columns):
        raise ValueError(
            "Cluster labels must contain stock_id and cluster_id columns. "
            f"Found: {list(labels.columns)}"
        )

    labels = labels[["stock_id", "cluster_id"]].copy()
    labels["stock_id"] = labels["stock_id"].astype(int)
    labels["cluster_id"] = labels["cluster_id"].astype(int)

    if labels["stock_id"].duplicated().any():
        dupes = labels.loc[labels["stock_id"].duplicated(), "stock_id"].tolist()
        raise ValueError(f"Duplicate stock_id values in cluster labels: {dupes}")

    cluster_ids = sorted(labels["cluster_id"].unique().tolist())
    expected_ids = list(range(expected_clusters))
    if cluster_ids != expected_ids:
        raise ValueError(
            f"Cluster labels contain IDs {cluster_ids}; expected {expected_ids}."
        )

    return labels


def realised_volatility(frame: pd.DataFrame, start: int, end: int, name: str) -> pd.Series:
    mask = frame["seconds_in_bucket"].ge(start) & frame["seconds_in_bucket"].lt(end)
    return (
        frame.loc[mask]
        .groupby("time_id")["log_return"]
        .apply(lambda x: np.sqrt(np.square(x.dropna()).sum()))
        .rename(name)
    )


def realised_volatility_by_stock(
    frame: pd.DataFrame,
    start: int,
    end: int,
    name: str,
) -> pd.Series:
    mask = frame["seconds_in_bucket"].ge(start) & frame["seconds_in_bucket"].lt(end)
    return (
        frame.loc[mask]
        .groupby(["stock_id", "time_id"])["log_return"]
        .apply(lambda x: np.sqrt(np.square(x.dropna()).sum()))
        .rename(name)
    )


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


def make_stock_features(path: Path) -> pd.DataFrame:
    stock_id = stock_id_from_path(path)
    columns = set(pd.read_csv(path, nrows=0).columns)
    base_cols = {"time_id", "seconds_in_bucket"}
    if not base_cols.issubset(columns):
        raise ValueError(f"{path} must contain time_id and seconds_in_bucket columns")

    if "wap" in columns:
        data = pd.read_csv(path, usecols=["time_id", "seconds_in_bucket", "wap"])
    else:
        raw_cols = [
            "time_id",
            "seconds_in_bucket",
            "bid_price1",
            "ask_price1",
            "bid_size1",
            "ask_size1",
        ]
        missing = sorted(set(raw_cols) - columns)
        if missing:
            raise ValueError(
                f"{path} must contain either wap or raw order-book columns. "
                f"Missing: {missing}"
            )
        data = pd.read_csv(path, usecols=raw_cols)
        depth = (data["bid_size1"] + data["ask_size1"]).replace(0, np.nan)
        data["wap"] = (
            data["bid_price1"] * data["ask_size1"]
            + data["ask_price1"] * data["bid_size1"]
        ) / depth

    data = data.sort_values(["time_id", "seconds_in_bucket"]).copy()
    data["log_wap"] = np.log(data["wap"].clip(lower=EPS))
    data["log_return"] = data.groupby("time_id")["log_wap"].diff()

    features = pd.DataFrame({"time_id": np.sort(data["time_id"].unique())})
    for start, end, name in [
        (0, INPUT_END, "rv_in"),
        (LAST_WINDOW_START, INPUT_END, "rv_last_window"),
        (TARGET_START, TARGET_END, "rv_fut"),
    ]:
        features = features.merge(
            realised_volatility(data, start, end, name).reset_index(),
            on="time_id",
            how="left",
        )

    features.insert(0, "stock_id", stock_id)
    fill_cols = ["rv_in", "rv_last_window", "rv_fut"]
    features[fill_cols] = features[fill_cols].fillna(0.0)
    return features


def make_table_stock_features(path: Path, max_files: int | None = None) -> pd.DataFrame:
    required_cols = ["stock_id", "time_id", "seconds_in_bucket", "wap"]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading shared fold parquet files requires a parquet engine such as "
            "pyarrow. Install pyarrow or run HAR from CSV source files instead."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available_cols = set(parquet_file.schema.names)
    missing = sorted(set(required_cols) - available_cols)
    if missing:
        raise ValueError(
            f"{path} must contain columns {required_cols}. "
            f"Missing: {missing}. Run the updated preprocess.ipynb to generate "
            "the shared fold files."
        )

    features = []
    current_stock_id: int | None = None
    current_chunks: list[pd.DataFrame] = []
    completed_stocks = 0

    def flush_current_stock() -> None:
        nonlocal current_stock_id, current_chunks, completed_stocks
        if current_stock_id is None or not current_chunks:
            return
        stock_frame = pd.concat(current_chunks, ignore_index=True)
        features.append(table_stock_frame_to_features(stock_frame))
        completed_stocks += 1
        print(
            f"  {path.name}: built HAR features for stock {current_stock_id} "
            f"({completed_stocks} stocks)"
        )
        current_stock_id = None
        current_chunks = []

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=required_cols)
        chunk = table.to_pandas()
        chunk["stock_id"] = normalise_stock_id(chunk["stock_id"], path)

        for stock_id, stock_chunk in chunk.groupby("stock_id", sort=False):
            stock_id = int(stock_id)
            if current_stock_id is None:
                current_stock_id = stock_id
            elif stock_id != current_stock_id:
                flush_current_stock()
                current_stock_id = stock_id

            if max_files is None or completed_stocks < max_files:
                current_chunks.append(stock_chunk.copy())

        if row_group_idx + 1 == parquet_file.num_row_groups:
            flush_current_stock()

        if max_files is not None and completed_stocks >= max_files:
            break

    if not features:
        raise ValueError(f"{path} did not contain any rows after filtering.")

    return pd.concat(features, ignore_index=True)


def table_stock_frame_to_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).copy()
    data["log_wap"] = np.log(data["wap"].clip(lower=EPS))
    data["log_return"] = data.groupby(["stock_id", "time_id"])["log_wap"].diff()

    features = (
        data[["stock_id", "time_id"]]
        .drop_duplicates()
        .sort_values(["stock_id", "time_id"])
    )
    for start, end, name in [
        (0, INPUT_END, "rv_in"),
        (LAST_WINDOW_START, INPUT_END, "rv_last_window"),
        (TARGET_START, TARGET_END, "rv_fut"),
    ]:
        features = features.merge(
            realised_volatility_by_stock(data, start, end, name).reset_index(),
            on=["stock_id", "time_id"],
            how="left",
        )

    fill_cols = ["rv_in", "rv_last_window", "rv_fut"]
    features[fill_cols] = features[fill_cols].fillna(0.0)
    return features.reset_index(drop=True)


def build_stock_features(data_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    files = sorted_stock_files(data_dir)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No stock_*.csv files found in {data_dir.resolve()}. "
            "Use --data-dir individual_book_train or omit --data-dir for raw CSVs. "
            "If you want the shared preprocess split, run preprocess.ipynb and use "
            "--fold-dir processed/fold_0."
        )

    frames = []
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Building HAR features for {path.name}")
        frames.append(make_stock_features(path))
    return pd.concat(frames, ignore_index=True)


def aggregate_cluster_features(
    stock_features: pd.DataFrame,
    cluster_labels: pd.DataFrame,
) -> pd.DataFrame:
    labelled = stock_features.merge(
        cluster_labels,
        on="stock_id",
        how="left",
        validate="many_to_one",
    )
    if labelled["cluster_id"].isna().any():
        missing = sorted(labelled.loc[labelled["cluster_id"].isna(), "stock_id"].unique())
        raise ValueError(f"Stocks missing cluster labels: {missing}")

    cluster_sizes = cluster_labels.groupby("cluster_id")["stock_id"].nunique()
    cluster_features = (
        labelled.groupby(["cluster_id", "time_id"], as_index=False)
        .agg(
            cluster_size_observed=("stock_id", "nunique"),
            rv_in=("rv_in", "mean"),
            rv_last_window=("rv_last_window", "mean"),
            rv_fut=("rv_fut", "mean"),
        )
        .sort_values(["time_id", "cluster_id"])
        .reset_index(drop=True)
    )
    cluster_features["cluster_size"] = (
        cluster_features["cluster_id"].map(cluster_sizes).astype(int)
    )
    cluster_features["log_rv_input"] = np.log(cluster_features["rv_in"].clip(lower=EPS))
    cluster_features["log_rv_last_window"] = np.log(
        cluster_features["rv_last_window"].clip(lower=EPS)
    )
    cluster_features["rv_last_window_to_input"] = (
        cluster_features["rv_last_window"] / cluster_features["rv_in"].clip(lower=EPS)
    )
    cluster_features["y_log"] = np.log(
        cluster_features["rv_fut"].clip(lower=EPS)
        / cluster_features["rv_in"].clip(lower=EPS)
    )

    return cluster_features[
        [
            "time_id",
            "cluster_id",
            "cluster_size",
            "cluster_size_observed",
            "rv_in",
            "rv_last_window",
            "rv_fut",
            "y_log",
            *DEFAULT_FEATURES,
        ]
    ]


def load_or_build_cluster_features(
    features_in: Path | None,
    features_out: Path,
    data_dir: Path,
    cluster_labels_path: Path,
    expected_clusters: int,
    max_files: int | None,
) -> pd.DataFrame:
    if features_in is not None:
        print(f"Loaded cluster features: {features_in.resolve()}")
        return pd.read_csv(features_in)

    labels = load_cluster_labels(cluster_labels_path, expected_clusters)
    stock_features = build_stock_features(data_dir, max_files=max_files)
    cluster_features = aggregate_cluster_features(stock_features, labels)
    cluster_features.to_csv(features_out, index=False)
    print(f"Saved cluster features: {features_out.resolve()}")
    return cluster_features


def load_fold_cluster_features(
    fold_dir: Path,
    cluster_labels_path: Path,
    expected_clusters: int,
    max_files: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = fold_dir / "train.parquet"
    test_path = fold_dir / "test.parquet"
    missing = [path for path in [train_path, test_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.resolve()) for path in missing)
        raise FileNotFoundError(
            f"Missing shared fold parquet file(s): {missing_text}. "
            "Run preprocess.ipynb first, then pass a folder such as "
            "--fold-dir processed/fold_0."
        )

    labels = load_cluster_labels(cluster_labels_path, expected_clusters)
    print(f"Building cluster features from shared fold: {fold_dir.resolve()}")
    train_stock_features = make_table_stock_features(train_path, max_files=max_files)
    test_stock_features = make_table_stock_features(test_path, max_files=max_files)
    train = aggregate_cluster_features(train_stock_features, labels)
    test = aggregate_cluster_features(test_stock_features, labels)
    val = train.iloc[0:0].copy()
    return train, val, test


def split_by_time_id(
    data: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_ids = np.sort(data["time_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(time_ids)

    n_test = int(len(time_ids) * test_ratio)
    n_val = int(len(time_ids) * val_ratio)
    test_ids = set(time_ids[:n_test])
    val_ids = set(time_ids[n_test : n_test + n_val])

    test = data[data["time_id"].isin(test_ids)].copy()
    val = data[data["time_id"].isin(val_ids)].copy()
    train = data[~data["time_id"].isin(test_ids | val_ids)].copy()
    return train, val, test


def regression_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return coef


def regression_predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x)), x])
    return x_design @ coef


def rv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.maximum(np.asarray(y_true, dtype=float), EPS)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), EPS)
    error = y_pred - y_true
    pct = error / y_true
    ratio = y_true / (y_pred + EPS)
    mse = float(np.mean(np.square(error)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "rmspe": float(np.sqrt(np.mean(np.square(pct)))),
        "mape": float(np.mean(np.abs(pct))),
        "qlike": float(np.mean(ratio - np.log(ratio + EPS) - 1.0)),
    }


def fit_predict_cluster_har(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_val = pd.concat([train, val], ignore_index=True)
    predicted_rv = np.zeros(len(test), dtype=float)
    coef_rows = []

    for cluster_id in sorted(train_val["cluster_id"].unique()):
        train_mask = train_val["cluster_id"] == cluster_id
        test_mask = test["cluster_id"].to_numpy() == cluster_id

        x_train = train_val.loc[train_mask, feature_names].to_numpy(dtype=float)
        y_train = train_val.loc[train_mask, "y_log"].to_numpy(dtype=float)
        x_test = test.loc[test_mask, feature_names].to_numpy(dtype=float)

        coef = regression_fit(x_train, y_train)
        predicted_log_ratio = regression_predict(coef, x_test)
        predicted_rv[test_mask] = (
            test.loc[test_mask, "rv_in"].to_numpy(dtype=float)
            * np.exp(predicted_log_ratio)
        )

        coef_rows.append(
            {
                "cluster_id": int(cluster_id),
                "feature": "intercept",
                "coefficient": float(coef[0]),
            }
        )
        for feature, value in zip(feature_names, coef[1:]):
            coef_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "feature": feature,
                    "coefficient": float(value),
                }
            )

    predicted_rv = np.clip(predicted_rv, PREDICTION_FLOOR, None)
    predictions = test[
        ["time_id", "cluster_id", "cluster_size", "rv_in", "rv_fut"]
    ].copy()
    predictions = predictions.rename(columns={"rv_fut": "actual_rv_480_600"})
    predictions["predicted_rv_480_600"] = predicted_rv
    predictions["absolute_error"] = (
        predictions["predicted_rv_480_600"] - predictions["actual_rv_480_600"]
    ).abs()
    predictions["percentage_error"] = (
        predictions["absolute_error"] / predictions["actual_rv_480_600"].clip(lower=EPS)
    )
    coefficients = pd.DataFrame(coef_rows)
    return predictions.sort_values(["time_id", "cluster_id"]), coefficients


def summarise_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = [
        {
            "evaluation_level": "cluster_time",
            "rows": len(predictions),
            "clusters": int(predictions["cluster_id"].nunique()),
            **rv_metrics(
                predictions["actual_rv_480_600"].to_numpy(),
                predictions["predicted_rv_480_600"].to_numpy(),
            ),
        }
    ]

    per_cluster_rows = []
    for cluster_id, group in predictions.groupby("cluster_id", sort=True):
        per_cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "rows": len(group),
                "cluster_size": int(group["cluster_size"].iloc[0]),
                **rv_metrics(
                    group["actual_rv_480_600"].to_numpy(),
                    group["predicted_rv_480_600"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(per_cluster_rows)


def discover_fold_dirs(
    folds_root: Path,
    selected_folds: list[int] | None,
) -> list[tuple[int, Path]]:
    if selected_folds:
        fold_dirs = [(fold_id, folds_root / f"fold_{fold_id}") for fold_id in selected_folds]
    else:
        fold_dirs = []
        for path in sorted(folds_root.glob("fold_*")):
            try:
                fold_id = int(path.name.split("_")[-1])
            except ValueError:
                continue
            fold_dirs.append((fold_id, path))

    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* folders found in {folds_root.resolve()}")

    missing = [
        path
        for _, fold_dir in fold_dirs
        for path in [fold_dir / "train.parquet", fold_dir / "test.parquet"]
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path.resolve()) for path in missing)
        raise FileNotFoundError(f"Missing fold parquet file(s): {missing_text}")

    return fold_dirs


def fit_and_summarise(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, coefficients = fit_predict_cluster_har(
        train,
        val,
        test,
        DEFAULT_FEATURES,
    )
    metrics, per_cluster = summarise_metrics(predictions)
    return predictions, metrics, per_cluster, coefficients


def summarise_cv_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = ["mse", "rmse", "mae", "rmspe", "mape", "qlike"]
    pooled = {
        "evaluation_level": "cv_pooled_cluster_time",
        "fold_id": "all",
        "rows": len(predictions),
        "clusters": int(predictions["cluster_id"].nunique()),
        **rv_metrics(
            predictions["actual_rv_480_600"].to_numpy(),
            predictions["predicted_rv_480_600"].to_numpy(),
        ),
    }
    fold_mean = {
        "evaluation_level": "cv_fold_mean",
        "fold_id": "mean",
        "rows": float(fold_metrics["rows"].mean()),
        "clusters": float(fold_metrics["clusters"].mean()),
        **{col: float(fold_metrics[col].mean()) for col in metric_cols},
    }
    fold_std = {
        "evaluation_level": "cv_fold_std",
        "fold_id": "std",
        "rows": float(fold_metrics["rows"].std(ddof=0)),
        "clusters": float(fold_metrics["clusters"].std(ddof=0)),
        **{col: float(fold_metrics[col].std(ddof=0)) for col in metric_cols},
    }
    return pd.concat(
        [fold_metrics, pd.DataFrame([pooled, fold_mean, fold_std])],
        ignore_index=True,
    )


def summarise_cv_per_cluster(per_cluster: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mse", "rmse", "mae", "rmspe", "mape", "qlike"]
    summary_rows = []
    for cluster_id, group in per_cluster.groupby("cluster_id", sort=True):
        for summary, func in [("mean", "mean"), ("std", "std")]:
            row = {
                "fold_id": summary,
                "cluster_id": int(cluster_id),
                "rows": (
                    float(group["rows"].std(ddof=0))
                    if summary == "std"
                    else float(group["rows"].mean())
                ),
                "cluster_size": int(group["cluster_size"].iloc[0]),
            }
            for col in metric_cols:
                if summary == "std":
                    row[col] = float(group[col].std(ddof=0))
                else:
                    row[col] = float(group[col].mean())
            summary_rows.append(row)

    return pd.concat(
        [per_cluster, pd.DataFrame(summary_rows)],
        ignore_index=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a cluster-level HAR-RV benchmark from stock source files."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--cluster-labels",
        type=Path,
        default=DEFAULT_CLUSTER_LABELS,
        help="CSV containing stock_id and cluster_id columns.",
    )
    parser.add_argument("--expected-clusters", type=int, default=3)
    parser.add_argument(
        "--cluster-features-in",
        type=Path,
        default=None,
        help="Load prebuilt cluster HAR features instead of rebuilding from source files.",
    )
    parser.add_argument(
        "--cluster-features-out",
        type=Path,
        default=DEFAULT_CLUSTER_FEATURES,
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=None,
        help=(
            "Use shared preprocess fold files from this folder, for example "
            "processed/fold_0 containing train.parquet and test.parquet."
        ),
    )
    parser.add_argument(
        "--folds-root",
        type=Path,
        default=None,
        help=(
            "Run the fixed HAR model over all shared outer CV folds under this "
            "folder, for example processed/fold_0 through processed/fold_4. "
            "No hyperparameter tuning is performed."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Optional fold IDs to run with --folds-root, for example --folds 0 1 2 3 4.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("har_rv_predictions.csv"),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("har_rv_metrics.csv"),
    )
    parser.add_argument(
        "--per-cluster-out",
        type=Path,
        default=Path("har_rv_per_cluster_metrics.csv"),
    )
    parser.add_argument(
        "--coefficients-out",
        type=Path,
        default=Path("har_rv_coefficients.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold_dir is not None and args.folds_root is not None:
        raise ValueError("Use either --fold-dir for one fold or --folds-root for CV, not both.")

    if args.folds_root is not None:
        if args.cluster_features_in is not None:
            raise ValueError("--folds-root cannot be used with --cluster-features-in.")

        fold_dirs = discover_fold_dirs(args.folds_root, args.folds)
        all_predictions = []
        all_fold_metrics = []
        all_per_cluster = []
        all_coefficients = []

        print("Cluster-level HAR-RV outer CV")
        print(f"Cluster labels: {args.cluster_labels}")
        print(f"Folds root: {args.folds_root}")
        print("Hyperparameter tuning: skipped (fixed HAR specification)")

        for fold_id, fold_dir in fold_dirs:
            print(f"\n=== Fold {fold_id}: {fold_dir} ===")
            train, val, test = load_fold_cluster_features(
                fold_dir=fold_dir,
                cluster_labels_path=args.cluster_labels,
                expected_clusters=args.expected_clusters,
                max_files=args.max_files,
            )
            predictions, metrics, per_cluster, coefficients = fit_and_summarise(
                train,
                val,
                test,
            )

            for frame in [predictions, metrics, per_cluster, coefficients]:
                frame.insert(0, "fold_id", fold_id)

            all_predictions.append(predictions)
            all_fold_metrics.append(metrics)
            all_per_cluster.append(per_cluster)
            all_coefficients.append(coefficients)

            print("Fold metrics")
            print(metrics.to_string(index=False))
            print("Fold per-cluster metrics")
            print(per_cluster.to_string(index=False))

        predictions = pd.concat(all_predictions, ignore_index=True)
        fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
        per_cluster = pd.concat(all_per_cluster, ignore_index=True)
        coefficients = pd.concat(all_coefficients, ignore_index=True)
        metrics = summarise_cv_metrics(predictions, fold_metrics)
        per_cluster = summarise_cv_per_cluster(per_cluster)

        predictions.to_csv(args.predictions_out, index=False)
        metrics.to_csv(args.metrics_out, index=False)
        per_cluster.to_csv(args.per_cluster_out, index=False)
        coefficients.to_csv(args.coefficients_out, index=False)

        print("\nOuter-CV metrics")
        print(metrics.to_string(index=False))
        print("\nOuter-CV per-cluster metrics")
        print(per_cluster.to_string(index=False))
        print(f"\nSaved predictions: {args.predictions_out.resolve()}")
        print(f"Saved metrics: {args.metrics_out.resolve()}")
        print(f"Saved per-cluster metrics: {args.per_cluster_out.resolve()}")
        print(f"Saved coefficients: {args.coefficients_out.resolve()}")
        return

    if args.fold_dir is not None:
        if args.cluster_features_in is not None:
            raise ValueError("--fold-dir cannot be used with --cluster-features-in.")
        train, val, test = load_fold_cluster_features(
            fold_dir=args.fold_dir,
            cluster_labels_path=args.cluster_labels,
            expected_clusters=args.expected_clusters,
            max_files=args.max_files,
        )
        cluster_features = pd.concat([train, test], ignore_index=True)
    else:
        cluster_features = load_or_build_cluster_features(
            features_in=args.cluster_features_in,
            features_out=args.cluster_features_out,
            data_dir=args.data_dir,
            cluster_labels_path=args.cluster_labels,
            expected_clusters=args.expected_clusters,
            max_files=args.max_files,
        )
        train, val, test = split_by_time_id(
            cluster_features,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    predictions, metrics, per_cluster, coefficients = fit_and_summarise(train, val, test)

    predictions.to_csv(args.predictions_out, index=False)
    metrics.to_csv(args.metrics_out, index=False)
    per_cluster.to_csv(args.per_cluster_out, index=False)
    coefficients.to_csv(args.coefficients_out, index=False)

    cluster_sizes = (
        cluster_features[["cluster_id", "cluster_size"]]
        .drop_duplicates()
        .sort_values("cluster_id")
    )

    print("Cluster-level HAR-RV")
    print(f"Cluster labels: {args.cluster_labels}")
    if args.fold_dir is not None:
        print(f"Shared fold: {args.fold_dir}")
    print(f"Features: {', '.join(DEFAULT_FEATURES)}")
    print("Cluster sizes")
    print(cluster_sizes.to_string(index=False))
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Test rows: {len(test):,}")
    print("\nOverall metrics")
    print(metrics.to_string(index=False))
    print("\nPer-cluster metrics")
    print(per_cluster.to_string(index=False))
    print(f"\nSaved predictions: {args.predictions_out.resolve()}")
    print(f"Saved metrics: {args.metrics_out.resolve()}")
    print(f"Saved per-cluster metrics: {args.per_cluster_out.resolve()}")
    print(f"Saved coefficients: {args.coefficients_out.resolve()}")


if __name__ == "__main__":
    main()
