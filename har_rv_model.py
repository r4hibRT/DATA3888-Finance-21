"""HAR-RV volatility model for the Optiver order book data.

The modelling setup matches the project framework:
- input window: first 8 minutes of each 10 minute time_id bucket
- target window: final 2 minutes of each bucket
- model: interpretable HAR-RV linear regression on log realised volatility
- metrics: RMSPE, QLIKE, MSE
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
MIN_TARGET_RV = 1e-5
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


def stock_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def sorted_stock_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("stock_*.csv"), key=stock_id_from_path)


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


def volatility_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    eps: float = EPS,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    percentage_error = (y_true - y_pred) / np.clip(y_true, eps, None)

    true_var = np.square(np.clip(y_true, eps, None))
    pred_var = np.square(y_pred)
    ratio = true_var / np.clip(pred_var, eps, None)

    return {
        "RMSPE": float(np.sqrt(np.mean(np.square(percentage_error)))),
        "QLIKE": float(np.mean(ratio - np.log(np.clip(ratio, eps, None)) - 1.0)),
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

    metrics = pd.DataFrame(
        [
            {
                "split": "train",
                "n_rows": len(train),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(train[TARGET_COL], train_pred),
            },
            {
                "split": "test",
                "n_rows": len(test),
                "filtered_target_rows": dropped_rows,
                "prediction_floor": prediction_floor,
                **volatility_metrics(test[TARGET_COL], test_pred),
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
    return model, metrics, coefficients, predictions


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
    parser.add_argument("--model-out", type=Path, default=Path("har_rv_model_coefficients.csv"))
    parser.add_argument("--cv-out", type=Path, default=Path("har_rv_cv_metrics.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cols = PURE_HAR_MODEL_FEATURES if args.no_liquidity else HAR_WITH_LIQUIDITY_FEATURES

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
