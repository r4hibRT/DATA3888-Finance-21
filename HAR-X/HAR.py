"""HAR-RV volatility model for the Optiver order book data.

- input window: first 8 minutes of each 10 minute time_id bucket
- target window: final 2 minutes of each bucket
- model: interpretable HAR-RV linear regression on log realised volatility
- metrics: RMSPE, QLIKE, MSE
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


EPS = 1e-12
MIN_TARGET_RV = 1e-4
PREDICTION_FLOOR = MIN_TARGET_RV
TARGET_COL = "target_rv_480_600"

HAR_RV_FEATURES = ["rv_360_480", "rv_240_480", "rv_0_480"]
LIQUIDITY_FEATURES = [
    "spread_mean_0_480",
    "spread_max_0_480",
    "volume_sum_0_480",
    "volume_imbalance_mean_0_480",
]

PURE_HAR_MODEL_FEATURES = ["log_rv_360_480", "log_rv_240_480", "log_rv_0_480"]
HAR_WITH_LIQUIDITY_FEATURES = PURE_HAR_MODEL_FEATURES + [
    "log_spread_mean_0_480",
    "log_spread_max_0_480",
    "log_volume_sum_0_480",
    "volume_imbalance_mean_0_480",
]
LOG_TARGET_COL = f"log_{TARGET_COL}"
HAR_MODEL_NAME = "HAR"
NAIVE_MODEL_NAME = "naive_last_2min"


def stock_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def sorted_stock_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("stock_*.csv"), key=stock_id_from_path)


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


def make_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create WAP, spread, volume and log-return features from the order book."""
    df = df.sort_values(["time_id", "seconds_in_bucket"]).copy()

    best_bid_price = np.maximum(df["bid_price1"], df["bid_price2"])
    best_ask_price = np.minimum(df["ask_price1"], df["ask_price2"])
    best_bid_size = np.where(
        df["bid_price1"] >= df["bid_price2"],
        df["bid_size1"],
        df["bid_size2"],
    )
    best_ask_size = np.where(
        df["ask_price1"] <= df["ask_price2"],
        df["ask_size1"],
        df["ask_size2"],
    )
    best_depth = pd.Series(best_bid_size + best_ask_size, index=df.index).replace(0, np.nan)

    df["wap"] = (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / best_depth
    df["bid_ask_spread"] = (best_ask_price / best_bid_price) - 1
    df["total_volume"] = df[["bid_size1", "bid_size2", "ask_size1", "ask_size2"]].sum(axis=1)
    df["volume_imbalance"] = (best_bid_size - best_ask_size) / best_depth
    df["log_wap"] = np.log(df["wap"])
    df["log_return"] = df.groupby("time_id")["log_wap"].diff()

    return df[
        [
            "time_id",
            "seconds_in_bucket",
            "wap",
            "log_return",
            "bid_ask_spread",
            "total_volume",
            "volume_imbalance",
        ]
    ]


def window_realised_volatility(
    df: pd.DataFrame,
    start_second: int,
    end_second: int,
    output_col: str,
) -> pd.DataFrame:
    mask = df["seconds_in_bucket"].ge(start_second) & df["seconds_in_bucket"].lt(end_second)
    return (
        df.loc[mask]
        .groupby("time_id")["log_return"]
        .apply(lambda x: np.sqrt(np.square(x.dropna()).sum()))
        .rename(output_col)
        .reset_index()
    )


def window_realised_volatility_by_stock(
    df: pd.DataFrame,
    start_second: int,
    end_second: int,
    output_col: str,
) -> pd.DataFrame:
    mask = df["seconds_in_bucket"].ge(start_second) & df["seconds_in_bucket"].lt(end_second)
    return (
        df.loc[mask]
        .groupby(["stock_id", "time_id"])["log_return"]
        .apply(lambda x: np.sqrt(np.square(x.dropna()).sum()))
        .rename(output_col)
        .reset_index()
    )


def input_window_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    input_df = df[df["seconds_in_bucket"].lt(480)]
    return (
        input_df.groupby("time_id")
        .agg(
            spread_mean_0_480=("bid_ask_spread", "mean"),
            spread_max_0_480=("bid_ask_spread", "max"),
            volume_sum_0_480=("total_volume", "sum"),
            volume_imbalance_mean_0_480=("volume_imbalance", "mean"),
        )
        .reset_index()
    )


def input_window_microstructure_by_stock(df: pd.DataFrame) -> pd.DataFrame:
    input_df = df[df["seconds_in_bucket"].lt(480)]
    return (
        input_df.groupby(["stock_id", "time_id"])
        .agg(
            spread_mean_0_480=("bid_ask_spread", "mean"),
            spread_max_0_480=("bid_ask_spread", "max"),
            volume_sum_0_480=("total_volume", "sum"),
            volume_imbalance_mean_0_480=("volume_imbalance", "mean"),
        )
        .reset_index()
    )


def make_har_features_for_stock(path: Path) -> pd.DataFrame:
    stock_id = stock_id_from_path(path)
    usecols = [
        "time_id",
        "seconds_in_bucket",
        "bid_price1",
        "ask_price1",
        "bid_price2",
        "ask_price2",
        "bid_size1",
        "ask_size1",
        "bid_size2",
        "ask_size2",
    ]
    raw = pd.read_csv(path, usecols=usecols)
    book = make_book_features(raw)
    features = pd.DataFrame({"time_id": np.sort(book["time_id"].unique())})

    windows = [
        (360, 480, "rv_360_480"),
        (240, 480, "rv_240_480"),
        (0, 480, "rv_0_480"),
        (480, 600, TARGET_COL),
    ]
    for start_second, end_second, output_col in windows:
        features = features.merge(
            window_realised_volatility(book, start_second, end_second, output_col),
            on="time_id",
            how="left",
        )

    features = features.merge(input_window_microstructure(book), on="time_id", how="left")
    features.insert(0, "stock_id", stock_id)

    fill_zero_cols = HAR_RV_FEATURES + [TARGET_COL] + LIQUIDITY_FEATURES
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0.0)
    return features.replace([np.inf, -np.inf], np.nan).dropna(subset=fill_zero_cols)


def make_preprocessed_book_features(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Use shared preprocessed fold rows that already contain WAP and liquidity columns."""
    df = df.copy()
    df["stock_id"] = normalise_stock_id(df["stock_id"], source)
    if "volume_imbalance" not in df.columns:
        if "depth_imbalance" in df.columns:
            df["volume_imbalance"] = df["depth_imbalance"]
        else:
            df["volume_imbalance"] = 0.0

    df = df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).copy()
    df["wap"] = df["wap"].clip(lower=EPS)
    df["bid_ask_spread"] = df["bid_ask_spread"].clip(lower=0)
    df["total_volume"] = df["total_volume"].clip(lower=0)
    df["log_wap"] = np.log(df["wap"])
    df["log_return"] = df.groupby(["stock_id", "time_id"])["log_wap"].diff()
    return df[
        [
            "stock_id",
            "time_id",
            "seconds_in_bucket",
            "wap",
            "log_return",
            "bid_ask_spread",
            "total_volume",
            "volume_imbalance",
        ]
    ]


def make_har_features_for_preprocessed_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    book = make_preprocessed_book_features(frame, source)
    features = (
        book[["stock_id", "time_id"]]
        .drop_duplicates()
        .sort_values(["stock_id", "time_id"])
    )

    windows = [
        (360, 480, "rv_360_480"),
        (240, 480, "rv_240_480"),
        (0, 480, "rv_0_480"),
        (480, 600, TARGET_COL),
    ]
    for start_second, end_second, output_col in windows:
        features = features.merge(
            window_realised_volatility_by_stock(
                book,
                start_second,
                end_second,
                output_col,
            ),
            on=["stock_id", "time_id"],
            how="left",
        )

    features = features.merge(
        input_window_microstructure_by_stock(book),
        on=["stock_id", "time_id"],
        how="left",
    )

    fill_zero_cols = HAR_RV_FEATURES + [TARGET_COL] + LIQUIDITY_FEATURES
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0.0)
    return features.replace([np.inf, -np.inf], np.nan).dropna(subset=fill_zero_cols)


def make_har_features_for_preprocessed_parquet(
    path: Path,
    max_files: int | None = None,
) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading shared fold parquet files requires pyarrow. "
            "Install pyarrow or build features from raw CSV source files instead."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available_cols = set(parquet_file.schema.names)
    required_cols = {
        "stock_id",
        "time_id",
        "seconds_in_bucket",
        "wap",
        "bid_ask_spread",
        "total_volume",
    }
    missing = sorted(required_cols - available_cols)
    if missing:
        raise ValueError(f"{path} is missing required preprocessed columns: {missing}")

    read_cols = sorted(required_cols)
    if "volume_imbalance" in available_cols:
        read_cols.append("volume_imbalance")
    elif "depth_imbalance" in available_cols:
        read_cols.append("depth_imbalance")

    features = []
    current_stock_id: int | None = None
    current_chunks: list[pd.DataFrame] = []
    completed_stocks = 0

    def flush_current_stock() -> None:
        nonlocal current_stock_id, current_chunks, completed_stocks
        if current_stock_id is None or not current_chunks:
            return
        stock_frame = pd.concat(current_chunks, ignore_index=True)
        features.append(make_har_features_for_preprocessed_frame(stock_frame, path))
        completed_stocks += 1
        print(
            f"  {path.name}: built HAR features for stock {current_stock_id} "
            f"({completed_stocks} stocks)"
        )
        current_stock_id = None
        current_chunks = []

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=read_cols)
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


def load_fold_har_rv_dataset(
    fold_dir: Path,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = fold_dir / "train.parquet"
    test_path = fold_dir / "test.parquet"
    missing = [path for path in [train_path, test_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.resolve()) for path in missing)
        raise FileNotFoundError(f"Missing fold parquet file(s): {missing_text}")

    print(f"Building HAR features from shared fold: {fold_dir.resolve()}")
    train = make_har_features_for_preprocessed_parquet(train_path, max_files=max_files)
    test = make_har_features_for_preprocessed_parquet(test_path, max_files=max_files)
    return train, test


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


def build_har_rv_dataset(
    data_dir: Path,
    max_files: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    files = sorted_stock_files(data_dir)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No stock_*.csv files found in {data_dir.resolve()}")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            frames = list(executor.map(make_har_features_for_stock, files))
    else:
        frames = []
        for i, path in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] Building features for {path.name}")
            frames.append(make_har_features_for_stock(path))

    return pd.concat(frames, ignore_index=True)


def add_har_model_columns(data: pd.DataFrame) -> pd.DataFrame:
    model_data = data.copy()
    positive_cols = HAR_RV_FEATURES + [TARGET_COL, "spread_mean_0_480", "spread_max_0_480"]
    for col in positive_cols:
        model_data[f"log_{col}"] = np.log(model_data[col].clip(lower=EPS))
    model_data["log_volume_sum_0_480"] = np.log1p(model_data["volume_sum_0_480"].clip(lower=0))
    return model_data.replace([np.inf, -np.inf], np.nan).dropna()


def prepare_har_model_data(
    data: pd.DataFrame,
    min_target_rv: float = MIN_TARGET_RV,
) -> tuple[pd.DataFrame, int]:
    model_data = add_har_model_columns(data)
    before = len(model_data)
    model_data = model_data[model_data[TARGET_COL].gt(min_target_rv)].copy()
    dropped = before - len(model_data)
    if len(model_data) < 2:
        raise ValueError(
            f"Need at least two rows with {TARGET_COL} > {min_target_rv} "
            "to train and evaluate the model."
        )
    return model_data, dropped


def fit_linear_regression(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, object]:
    x = train[feature_cols].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    coefs, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return {
        "intercept": coefs[0],
        "coef": coefs[1:],
        "feature_cols": list(feature_cols),
        "target_col": target_col,
    }


def predict_linear_regression(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    feature_cols = model["feature_cols"]
    x = frame[feature_cols].to_numpy(dtype=float)
    return model["intercept"] + x @ model["coef"]


def inverse_log_rv(log_rv: np.ndarray, prediction_floor: float = PREDICTION_FLOOR) -> np.ndarray:
    return np.clip(np.exp(log_rv) - EPS, prediction_floor, None)


def naive_last_2min_prediction(
    frame: pd.DataFrame,
    prediction_floor: float = PREDICTION_FLOOR,
) -> np.ndarray:
    return np.clip(frame["rv_360_480"].to_numpy(dtype=float), prediction_floor, None)


def prediction_latency_summary(
    model: dict[str, object],
    frame: pd.DataFrame,
    prediction_floor: float,
    repeats: int,
) -> dict[str, float]:
    repeats = max(1, repeats)
    if len(frame) == 0:
        raise ValueError("Need at least one row to benchmark prediction latency.")

    # Warm up numpy/pandas paths once so the repeated timings focus on prediction.
    inverse_log_rv(predict_linear_regression(model, frame), prediction_floor)

    timings = []
    for _ in range(repeats):
        start = perf_counter()
        inverse_log_rv(predict_linear_regression(model, frame), prediction_floor)
        timings.append(perf_counter() - start)

    timings_array = np.asarray(timings, dtype=float)
    mean_seconds = float(timings_array.mean())
    return {
        "prediction_rows": float(len(frame)),
        "latency_repeats": float(repeats),
        "prediction_seconds_mean": mean_seconds,
        "prediction_seconds_median": float(np.median(timings_array)),
        "prediction_seconds_min": float(timings_array.min()),
        "prediction_seconds_max": float(timings_array.max()),
        "prediction_microseconds_per_row": (mean_seconds / len(frame)) * 1_000_000,
        "prediction_rows_per_second": len(frame) / mean_seconds if mean_seconds > 0 else np.inf,
    }


def latency_row(
    fold_id: int | str,
    feature_build_seconds: float,
    fit_evaluate_seconds: float,
    pipeline_seconds_before_csv: float,
    latency: dict[str, float],
) -> dict[str, float | int | str]:
    return {
        "fold_id": fold_id,
        "feature_build_seconds": feature_build_seconds,
        "fit_evaluate_seconds": fit_evaluate_seconds,
        "pipeline_seconds_before_csv": pipeline_seconds_before_csv,
        **latency,
    }


def print_prediction_latency(latency: dict[str, float]) -> None:
    print("\nPrediction latency")
    print(f"Rows predicted: {int(latency['prediction_rows']):,}")
    print(f"Mean prediction time: {latency['prediction_seconds_mean']:.6f} sec")
    print(f"Median prediction time: {latency['prediction_seconds_median']:.6f} sec")
    print(
        "Mean per-row latency: "
        f"{latency['prediction_microseconds_per_row']:.6f} microseconds"
    )
    print(
        "Throughput: "
        f"{latency['prediction_rows_per_second']:,.2f} rows/sec"
    )


def volatility_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    eps: float = EPS,
) -> dict[str, float]:
    y_true = np.clip(np.asarray(y_true, dtype=float), eps, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    percentage_error = (y_true - y_pred) / y_true
    ratio = y_true / y_pred

    return {
        "RMSPE": float(np.sqrt(np.mean(np.square(percentage_error)))),
        "QLIKE": float(np.mean(ratio - np.log(ratio) - 1.0)),
        "MSE": float(np.mean(np.square(y_true - y_pred))),
    }


def train_test_split_frame(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) < 2:
        raise ValueError("Need at least two rows to create a train/test split.")
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(indices) * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return data.iloc[train_idx].copy(), data.iloc[test_idx].copy()


def fit_and_evaluate_har_rv(
    data: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    min_target_rv: float = MIN_TARGET_RV,
    prediction_floor: float = PREDICTION_FLOOR,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = HAR_WITH_LIQUIDITY_FEATURES if feature_cols is None else feature_cols
    model_data, dropped_rows = prepare_har_model_data(data, min_target_rv=min_target_rv)
    train, test = train_test_split_frame(model_data, test_size=test_size, random_state=random_state)

    model = fit_linear_regression(train, feature_cols, LOG_TARGET_COL)
    train_pred = inverse_log_rv(predict_linear_regression(model, train), prediction_floor)
    test_pred = inverse_log_rv(predict_linear_regression(model, test), prediction_floor)
    train_naive_pred = naive_last_2min_prediction(train, prediction_floor)
    test_naive_pred = naive_last_2min_prediction(test, prediction_floor)

    metrics = pd.DataFrame(
        [
            {
                "model": HAR_MODEL_NAME,
                "split": "train",
                "n_rows": len(train),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(train[TARGET_COL], train_pred),
            },
            {
                "model": HAR_MODEL_NAME,
                "split": "test",
                "n_rows": len(test),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(test[TARGET_COL], test_pred),
            },
            {
                "model": NAIVE_MODEL_NAME,
                "split": "train",
                "n_rows": len(train),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(train[TARGET_COL], train_naive_pred),
            },
            {
                "model": NAIVE_MODEL_NAME,
                "split": "test",
                "n_rows": len(test),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(test[TARGET_COL], test_naive_pred),
            },
        ]
    )
    coefficients = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model["coef"],
        }
    )
    coefficients = coefficients.reindex(
        coefficients["coefficient"].abs().sort_values(ascending=False).index
    )

    predictions = test[["stock_id", "time_id", TARGET_COL]].copy()
    predictions["predicted_rv_480_600"] = test_pred
    predictions["absolute_error"] = (
        predictions[TARGET_COL] - predictions["predicted_rv_480_600"]
    ).abs()
    predictions["percentage_error"] = (
        predictions["absolute_error"] / predictions[TARGET_COL].clip(lower=EPS)
    )
    predictions["naive_rv_360_480"] = test_naive_pred
    predictions["naive_absolute_error"] = (
        predictions[TARGET_COL] - predictions["naive_rv_360_480"]
    ).abs()
    predictions["naive_percentage_error"] = (
        predictions["naive_absolute_error"] / predictions[TARGET_COL].clip(lower=EPS)
    )
    return model, metrics, coefficients, predictions


def fit_and_evaluate_har_rv_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str] | None = None,
    min_target_rv: float = MIN_TARGET_RV,
    prediction_floor: float = PREDICTION_FLOOR,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = HAR_WITH_LIQUIDITY_FEATURES if feature_cols is None else feature_cols
    train, dropped_train_rows = prepare_har_model_data(
        train_data,
        min_target_rv=min_target_rv,
    )
    test, dropped_test_rows = prepare_har_model_data(
        test_data,
        min_target_rv=min_target_rv,
    )

    model = fit_linear_regression(train, feature_cols, LOG_TARGET_COL)
    train_pred = inverse_log_rv(predict_linear_regression(model, train), prediction_floor)
    test_pred = inverse_log_rv(predict_linear_regression(model, test), prediction_floor)
    train_naive_pred = naive_last_2min_prediction(train, prediction_floor)
    test_naive_pred = naive_last_2min_prediction(test, prediction_floor)

    metrics = pd.DataFrame(
        [
            {
                "model": HAR_MODEL_NAME,
                "split": "train",
                "n_rows": len(train),
                "filtered_target_rows": dropped_train_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(train[TARGET_COL], train_pred),
            },
            {
                "model": HAR_MODEL_NAME,
                "split": "test",
                "n_rows": len(test),
                "filtered_target_rows": dropped_test_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(test[TARGET_COL], test_pred),
            },
            {
                "model": NAIVE_MODEL_NAME,
                "split": "train",
                "n_rows": len(train),
                "filtered_target_rows": dropped_train_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(train[TARGET_COL], train_naive_pred),
            },
            {
                "model": NAIVE_MODEL_NAME,
                "split": "test",
                "n_rows": len(test),
                "filtered_target_rows": dropped_test_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(test[TARGET_COL], test_naive_pred),
            },
        ]
    )
    coefficients = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model["coef"],
        }
    )
    coefficients = coefficients.reindex(
        coefficients["coefficient"].abs().sort_values(ascending=False).index
    )

    predictions = test[["stock_id", "time_id", TARGET_COL]].copy()
    predictions["predicted_rv_480_600"] = test_pred
    predictions["absolute_error"] = (
        predictions[TARGET_COL] - predictions["predicted_rv_480_600"]
    ).abs()
    predictions["percentage_error"] = (
        predictions["absolute_error"] / predictions[TARGET_COL].clip(lower=EPS)
    )
    predictions["naive_rv_360_480"] = test_naive_pred
    predictions["naive_absolute_error"] = (
        predictions[TARGET_COL] - predictions["naive_rv_360_480"]
    ).abs()
    predictions["naive_percentage_error"] = (
        predictions["naive_absolute_error"] / predictions[TARGET_COL].clip(lower=EPS)
    )
    return model, metrics, coefficients, predictions


def summarise_fold_cv_metrics(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_floor: float,
) -> pd.DataFrame:
    test_metrics = fold_metrics[fold_metrics["split"].eq("test")]
    metric_cols = ["RMSPE", "QLIKE", "MSE"]
    if "model" not in test_metrics.columns:
        test_metrics = test_metrics.assign(model=HAR_MODEL_NAME)
        fold_metrics = fold_metrics.assign(model=HAR_MODEL_NAME)

    prediction_cols = {
        HAR_MODEL_NAME: "predicted_rv_480_600",
        NAIVE_MODEL_NAME: "naive_rv_360_480",
    }
    summary_rows = []
    for model_name, model_test_metrics in test_metrics.groupby("model", sort=False):
        prediction_col = prediction_cols.get(model_name)
        if prediction_col is None or prediction_col not in predictions.columns:
            continue

        summary_rows.extend(
            [
                {
                    "fold_id": "all",
                    "model": model_name,
                    "split": "test_pooled",
                    "n_rows": len(predictions),
                    "filtered_target_rows": float(
                        model_test_metrics["filtered_target_rows"].sum()
                    ),
                    "prediction_floor": prediction_floor,
                    **volatility_metrics(
                        predictions[TARGET_COL],
                        predictions[prediction_col],
                    ),
                },
                {
                    "fold_id": "mean",
                    "model": model_name,
                    "split": "test_fold_mean",
                    "n_rows": float(model_test_metrics["n_rows"].mean()),
                    "filtered_target_rows": float(
                        model_test_metrics["filtered_target_rows"].mean()
                    ),
                    "prediction_floor": prediction_floor,
                    **{col: float(model_test_metrics[col].mean()) for col in metric_cols},
                },
                {
                    "fold_id": "std",
                    "model": model_name,
                    "split": "test_fold_std",
                    "n_rows": float(model_test_metrics["n_rows"].std(ddof=0)),
                    "filtered_target_rows": float(
                        model_test_metrics["filtered_target_rows"].std(ddof=0)
                    ),
                    "prediction_floor": prediction_floor,
                    **{col: float(model_test_metrics[col].std(ddof=0)) for col in metric_cols},
                },
            ]
        )
    return pd.concat(
        [fold_metrics, pd.DataFrame(summary_rows)],
        ignore_index=True,
    )


def cross_validate_har_rv(
    data: pd.DataFrame,
    n_splits: int = 5,
    feature_cols: list[str] | None = None,
    group_by_stock: bool = False,
    random_state: int = 42,
    min_target_rv: float = MIN_TARGET_RV,
    prediction_floor: float = PREDICTION_FLOOR,
) -> pd.DataFrame:
    feature_cols = HAR_WITH_LIQUIDITY_FEATURES if feature_cols is None else feature_cols
    model_data, dropped_rows = prepare_har_model_data(data, min_target_rv=min_target_rv)
    model_data = model_data.reset_index(drop=True)
    n_splits = min(n_splits, len(model_data))
    if n_splits < 2:
        raise ValueError("Need at least two rows for cross-validation.")

    rng = np.random.default_rng(random_state)
    all_indices = np.arange(len(model_data))
    if group_by_stock:
        groups = model_data["stock_id"].to_numpy()
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        fold_groups = np.array_split(unique_groups, min(n_splits, len(unique_groups)))
        folds = [np.flatnonzero(np.isin(groups, group_values)) for group_values in fold_groups]
    else:
        shuffled = all_indices.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, n_splits)

    rows = []
    for fold, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        train = model_data.iloc[train_idx]
        test = model_data.iloc[test_idx]
        model = fit_linear_regression(train, feature_cols, LOG_TARGET_COL)
        pred = inverse_log_rv(predict_linear_regression(model, test), prediction_floor)
        rows.append(
            {
                "fold": fold,
                "n_rows": len(test),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(test[TARGET_COL], pred),
            }
        )
    return pd.DataFrame(rows)


def save_model_summary(
    model: dict[str, object],
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = [{"feature": "intercept", "coefficient": model["intercept"]}]
    rows.extend(coefficients.to_dict("records"))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)

    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.csv")
    metrics.to_csv(metrics_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HAR-RV volatility model.")
    parser.add_argument("--data-dir", type=Path, default=Path("individual_book_train"))
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=None,
        help=(
            "Use a shared preprocess fold directory containing train.parquet "
            "and test.parquet, for example processed/fold_0."
        ),
    )
    parser.add_argument(
        "--folds-root",
        type=Path,
        default=None,
        help=(
            "Run HAR-RV over shared CV fold directories under this root, "
            "for example processed/fold_0 through processed/fold_4."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Optional fold IDs to run with --folds-root, for example --folds 0 1 2 3 4.",
    )
    parser.add_argument(
        "--features-in",
        type=Path,
        default=None,
        help="Load an existing HAR-RV feature table instead of rebuilding from raw book files.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--group-cv-by-stock", action="store_true")
    parser.add_argument("--no-liquidity", action="store_true")
    parser.add_argument(
        "--min-target-rv",
        type=float,
        default=MIN_TARGET_RV,
        help=(
            "Exclude rows with final-window realised volatility at or below this value. "
            "RMSPE is undefined at zero and unstable for extremely small targets."
        ),
    )
    parser.add_argument(
        "--prediction-floor",
        type=float,
        default=PREDICTION_FLOOR,
        help="Minimum predicted realised volatility used for predictions and metric calculation.",
    )
    parser.add_argument("--features-out", type=Path, default=Path("har_rv_features.csv"))
    parser.add_argument("--predictions-out", type=Path, default=Path("har_rv_predictions.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("har_rv_metrics.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("har_rv_model_coefficients.csv"))
    parser.add_argument("--cv-out", type=Path, default=Path("har_rv_cv_metrics.csv"))
    parser.add_argument("--latency-out", type=Path, default=Path("har_rv_latency.csv"))
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=25,
        help="Number of repeated prediction-only timings to run on the test rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cols = PURE_HAR_MODEL_FEATURES if args.no_liquidity else HAR_WITH_LIQUIDITY_FEATURES
    if args.fold_dir is not None and args.folds_root is not None:
        raise ValueError("Use either --fold-dir for one fold or --folds-root for CV, not both.")

    if args.folds_root is not None:
        if args.features_in is not None:
            raise ValueError("--folds-root cannot be combined with --features-in.")

        all_predictions = []
        all_metrics = []
        all_coefficients = []
        all_latency = []
        fold_dirs = discover_fold_dirs(args.folds_root, args.folds)

        print("HAR-RV shared-fold CV")
        print(f"Folds root: {args.folds_root}")
        for fold_id, fold_dir in fold_dirs:
            print(f"\n=== Fold {fold_id}: {fold_dir} ===")
            fold_start = perf_counter()
            feature_start = perf_counter()
            train_data, test_data = load_fold_har_rv_dataset(
                fold_dir,
                max_files=args.max_files,
            )
            feature_build_seconds = perf_counter() - feature_start

            fit_start = perf_counter()
            model, metrics, coefficients, predictions = fit_and_evaluate_har_rv_split(
                train_data,
                test_data,
                feature_cols=feature_cols,
                min_target_rv=args.min_target_rv,
                prediction_floor=args.prediction_floor,
            )
            fit_evaluate_seconds = perf_counter() - fit_start

            test_model_data, _ = prepare_har_model_data(
                test_data,
                min_target_rv=args.min_target_rv,
            )
            latency = prediction_latency_summary(
                model,
                test_model_data,
                prediction_floor=args.prediction_floor,
                repeats=args.latency_repeats,
            )
            pipeline_seconds_before_csv = perf_counter() - fold_start

            metrics.insert(0, "fold_id", fold_id)
            predictions.insert(0, "fold_id", fold_id)
            coefficients = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "feature": "intercept",
                                "coefficient": model["intercept"],
                            }
                        ]
                    ),
                    coefficients,
                ],
                ignore_index=True,
            )
            coefficients.insert(0, "fold_id", fold_id)

            all_predictions.append(predictions)
            all_metrics.append(metrics)
            all_coefficients.append(coefficients)
            all_latency.append(
                latency_row(
                    fold_id,
                    feature_build_seconds,
                    fit_evaluate_seconds,
                    pipeline_seconds_before_csv,
                    latency,
                )
            )

            print("Fold metrics")
            print(metrics.to_string(index=False))
            print_prediction_latency(latency)

        predictions = pd.concat(all_predictions, ignore_index=True)
        fold_metrics = pd.concat(all_metrics, ignore_index=True)
        coefficients = pd.concat(all_coefficients, ignore_index=True)
        latency = pd.DataFrame(all_latency)
        cv_metrics = summarise_fold_cv_metrics(
            fold_metrics,
            predictions,
            prediction_floor=args.prediction_floor,
        )

        predictions.to_csv(args.predictions_out, index=False)
        cv_metrics.to_csv(args.metrics_out, index=False)
        cv_metrics.to_csv(args.cv_out, index=False)
        coefficients.to_csv(args.model_out, index=False)
        latency.to_csv(args.latency_out, index=False)
        model_metrics_out = args.model_out.with_name(f"{args.model_out.stem}_metrics.csv")
        cv_metrics.to_csv(model_metrics_out, index=False)

        print("\nCV metrics")
        print(cv_metrics.to_string(index=False))
        print("\nLatency summary")
        print(latency.to_string(index=False))
        print(f"\nSaved predictions: {args.predictions_out.resolve()}")
        print(f"Saved metrics: {args.metrics_out.resolve()}")
        print(f"Saved cross-validation metrics: {args.cv_out.resolve()}")
        print(f"Saved fold coefficients: {args.model_out.resolve()}")
        print(f"Saved coefficient metrics: {model_metrics_out.resolve()}")
        print(f"Saved latency metrics: {args.latency_out.resolve()}")
        return

    if args.fold_dir is not None:
        if args.features_in is not None:
            raise ValueError("--fold-dir cannot be combined with --features-in.")

        fold_start = perf_counter()
        feature_start = perf_counter()
        train_data, test_data = load_fold_har_rv_dataset(
            args.fold_dir,
            max_files=args.max_files,
        )
        feature_build_seconds = perf_counter() - feature_start

        fit_start = perf_counter()
        model, metrics, coefficients, predictions = fit_and_evaluate_har_rv_split(
            train_data,
            test_data,
            feature_cols=feature_cols,
            min_target_rv=args.min_target_rv,
            prediction_floor=args.prediction_floor,
        )
        fit_evaluate_seconds = perf_counter() - fit_start

        test_model_data, _ = prepare_har_model_data(
            test_data,
            min_target_rv=args.min_target_rv,
        )
        latency = prediction_latency_summary(
            model,
            test_model_data,
            prediction_floor=args.prediction_floor,
            repeats=args.latency_repeats,
        )
        pipeline_seconds_before_csv = perf_counter() - fold_start
        latency = pd.DataFrame(
            [
                latency_row(
                    args.fold_dir.name,
                    feature_build_seconds,
                    fit_evaluate_seconds,
                    pipeline_seconds_before_csv,
                    latency,
                )
            ]
        )

        predictions.to_csv(args.predictions_out, index=False)
        metrics.to_csv(args.metrics_out, index=False)
        latency.to_csv(args.latency_out, index=False)
        save_model_summary(model, coefficients, metrics, args.model_out)

        print(f"HAR-RV train dataset shape: {train_data.shape}")
        print(f"HAR-RV test dataset shape: {test_data.shape}")
        print(f"Minimum target RV for fitting/evaluation: {args.min_target_rv:g}")
        print(f"Prediction floor: {args.prediction_floor:g}")
        print(f"Saved predictions: {args.predictions_out.resolve()}")
        print(f"Saved metrics: {args.metrics_out.resolve()}")
        print(f"Saved model coefficients: {args.model_out.resolve()}")
        print(f"Saved latency metrics: {args.latency_out.resolve()}")
        print("Cross-validation skipped: explicit fold train/test split was used.")
        print("\nFold holdout metrics")
        print(metrics.to_string(index=False))
        print("\nModel coefficients")
        print(coefficients.to_string(index=False))
        print("\nRuntime summary")
        print(latency.to_string(index=False))
        print_prediction_latency(latency.iloc[0].to_dict())
        return

    if args.features_in is not None:
        har_data = pd.read_csv(args.features_in)
        print(f"Loaded existing features: {args.features_in.resolve()}")
    else:
        har_data = build_har_rv_dataset(
            data_dir=args.data_dir,
            max_files=args.max_files,
            workers=args.workers,
        )
        har_data.to_csv(args.features_out, index=False)

    model, metrics, coefficients, predictions = fit_and_evaluate_har_rv(
        har_data,
        feature_cols=feature_cols,
        test_size=args.test_size,
        random_state=args.random_state,
        min_target_rv=args.min_target_rv,
        prediction_floor=args.prediction_floor,
    )
    predictions.to_csv(args.predictions_out, index=False)
    metrics.to_csv(args.metrics_out, index=False)
    save_model_summary(model, coefficients, metrics, args.model_out)

    cv_metrics = cross_validate_har_rv(
        har_data,
        n_splits=args.cv_splits,
        feature_cols=feature_cols,
        group_by_stock=args.group_cv_by_stock,
        random_state=args.random_state,
        min_target_rv=args.min_target_rv,
        prediction_floor=args.prediction_floor,
    )
    cv_metrics.to_csv(args.cv_out, index=False)

    print(f"HAR-RV dataset shape: {har_data.shape}")
    print(f"Minimum target RV for fitting/evaluation: {args.min_target_rv:g}")
    print(f"Prediction floor: {args.prediction_floor:g}")
    if args.features_in is None:
        print(f"Saved features: {args.features_out.resolve()}")
    print(f"Saved predictions: {args.predictions_out.resolve()}")
    print(f"Saved metrics: {args.metrics_out.resolve()}")
    print(f"Saved model coefficients: {args.model_out.resolve()}")
    print(f"Saved cross-validation metrics: {args.cv_out.resolve()}")
    print("\nHoldout metrics")
    print(metrics.to_string(index=False))
    print("\nModel coefficients")
    print(coefficients.to_string(index=False))
    print("\nCross-validation metrics")
    print(cv_metrics.to_string(index=False))


if __name__ == "__main__":
    main()