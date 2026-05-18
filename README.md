# Stock-Level HAR-RV Volatility Prediction

## 1. Project Aim

The task is to predict **individual stock realised volatility** for the final 2 minutes of each 10 minute `time_id` bucket.

- Input window: seconds `0-479`
- Target window: seconds `480-599`
- Evaluation unit: `stock_id x time_id`
- Stocks: 112 individual stocks
- Model: pooled HAR-style linear regression over stock-level rows

This version predicts at the individual stock level and does **not** use cluster labels or cluster aggregation.

## 2. Files Used

| File | Purpose |
|---|---|
| `har_model_StockLevel.py` | Final HAR-RV script for stock-level rows. |
| `preprocess.ipynb` | Optional preprocessing notebook that creates shared parquet folds. |
| `processed/fold_*/train.parquet` | Shared fold training data. |
| `processed/fold_*/test.parquet` | Shared fold test data. |
| `har_rv_predictions.csv` | HAR and naive predictions for each evaluated `stock_id x time_id`. |
| `har_rv_metrics.csv` | Metrics for the latest run. |
| `har_rv_cv_metrics.csv` | Five-fold CV metrics when `--folds-root` is used. |
| `har_rv_model_coefficients.csv` | Fitted HAR coefficients. |
| `har_rv_model_coefficients_metrics.csv` | Metrics copied beside the coefficient output. |

## 3. Method

Each modelling row is one stock in one `time_id`. The script can read either raw order-book CSV files or shared preprocessed parquet folds that already contain `wap`, spread, and volume fields.

The HAR-RV model uses log-transformed realised-volatility features from the first 8 minutes:

- `log_rv_360_480`
- `log_rv_240_480`
- `log_rv_0_480`

It also includes liquidity features from the same input window:

- `log_spread_mean_0_480`
- `log_spread_max_0_480`
- `log_volume_sum_0_480`
- `volume_imbalance_mean_0_480`

The fitted target is:

```text
log_target_rv_480_600 = log(target_rv_480_600)
```

Predictions are transformed back onto realised-volatility scale and clipped at the prediction floor:

```text
predicted_rv_480_600 = max(exp(predicted_log_rv) - EPS, prediction_floor)
```

Rows with `target_rv_480_600 <= 1e-5` are excluded from fitting and evaluation because RMSPE is unstable near zero. The default prediction floor is also `1e-5`.

## 4. Naive Baseline

The script also reports a simple persistence baseline:

```text
naive_last_2min prediction = rv_360_480
```

This means the realised volatility from the final 2 minutes of the input window is used to predict the realised volatility in the next 2 minutes.

For fold 0, the comparison is:

| Model | Split | Rows | RMSPE | QLIKE | MSE |
|---|---|---:|---:|---:|---:|
| HAR | test | 85,782 | 0.985506 | 0.250614 | 4.6551e-07 |
| naive_last_2min | test | 85,782 | 1.160154 | 3.092463 | 6.5560e-07 |

On fold 0, HAR beats the naive last-2-minute baseline on RMSPE.

## 5. Results

Full CV finished against `processed/fold_0` through `processed/fold_4` and regenerated the outputs.

### 5.1 Five-Fold HAR-RV CV Metrics

| Summary | Test Rows | Filtered Target Rows | RMSPE | QLIKE | MSE |
|---|---:|---:|---:|---:|---:|
| Pooled test rows | 428,899 | 33 | 0.960275 | 0.893083 | 4.9699e-07 |
| Mean across folds | 85,779.8 | 6.6 | 0.957173 | 0.893150 | 4.9699e-07 |
| Std across folds | 4.445 | 2.245 | 0.077098 | 1.319412 | 3.2054e-08 |

Fold-level HAR test metrics:

| Fold | Test Rows | Filtered Target Rows | RMSPE | QLIKE | MSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 85,782 | 4 | 0.985506 | 0.250614 | 4.6551e-07 |
| 1 | 85,781 | 9 | 1.070420 | 0.226772 | 5.0126e-07 |
| 2 | 85,782 | 7 | 0.957635 | 0.236665 | 4.8521e-07 |
| 3 | 85,783 | 4 | 0.940943 | 0.219807 | 4.7630e-07 |
| 4 | 85,771 | 9 | 0.831359 | 3.531893 | 5.5667e-07 |

## 6. How to Run

Run one shared fold:

```bash
python3 har_rv_model.py --fold-dir processed/fold_0
```

Run the full five-fold CV:

```bash
python3 har_rv_model.py --folds-root processed --folds 0 1 2 3 4
```

Run from raw stock files:

```bash
python3 har_rv_model.py --data-dir individual_book_train
```

If needed, run `preprocess.ipynb` first to create:

```text
processed/full.parquet
processed/fold_0/train.parquet
processed/fold_0/test.parquet
...
processed/fold_4/train.parquet
processed/fold_4/test.parquet
```

Reading parquet files requires `pyarrow` or `fastparquet`.

## 7. Outputs

The main output files are:

```text
har_rv_predictions.csv
har_rv_metrics.csv
har_rv_cv_metrics.csv
har_rv_model_coefficients.csv
har_rv_model_coefficients_metrics.csv
```

When `--fold-dir` is used, `har_rv_metrics.csv` contains HAR and naive baseline train/test rows for that fold.

When `--folds-root` is used, the outputs include `fold_id`, and `har_rv_cv_metrics.csv` includes fold rows plus pooled, fold-mean, and fold-standard-deviation summaries.

## 8. Notes

- This is a stock-level prediction task, not a cluster-level task.
- The current HAR model is pooled across stock-level rows; it is not one separate regression per stock.
- Cluster labels are not used.
- The reported RMSPE should be compared with other stock-level models using the same target filter and prediction floor.

## 9. References

- Corsi, F. (2009). [A Simple Approximate Long-Memory Model of Realized Volatility](https://doi.org/10.1093/jjfinec/nbp001).
- Patton, A. (2011). [Volatility forecast comparison using imperfect volatility proxies](https://EconPapers.repec.org/RePEc:eee:econom:v:160:y:2011:i:1:p:246-256).
