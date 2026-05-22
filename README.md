# Stock-Level HAR-RV Volatility Prediction

## 1. Project Aim

The task is to predict **individual stock realised volatility** for the final 2 minutes of each 10 minute `time_id` bucket.

- Input window: seconds `0-479`
- Target window: seconds `480-599`
- Evaluation unit: `stock_id x time_id`
- Stocks: 112 individual stocks
- Main model: pooled stock-level HAR+ linear regression

This project predicts at the individual stock level. It does **not** use cluster labels or cluster aggregation.

## 2. Files Used

| File | Purpose |
|---|---|
| `har_model_StockLevel.py` | Main pooled stock-level HAR+ model. |
| `preprocess.ipynb` | Optional preprocessing notebook that creates shared parquet folds. |
| `processed/fold_*/train.parquet` | Shared fold training data. |
| `processed/fold_*/test.parquet` | Shared fold test data. |
| `har_rv_predictions.csv` | HAR+ and naive predictions for evaluated `stock_id x time_id` rows. |
| `har_rv_metrics.csv` | Metrics for the latest HAR+ run. |
| `har_rv_cv_metrics.csv` | Five-fold HAR+ CV metrics when `--folds-root` is used. |
| `har_rv_model_coefficients.csv` | Fitted HAR+ coefficients. |
| `har_rv_latency.csv` | Runtime and prediction-latency metrics. |

## 3. Main Model: HAR+

Each modelling row is one stock in one `time_id`. The script reads shared preprocessed parquet folds that already contain `wap`, spread, and volume fields.

The main model is labelled **HAR+** because it combines HAR realised-volatility features with liquidity features.

HAR realised-volatility features:

- `log_rv_360_480`
- `log_rv_240_480`
- `log_rv_0_480`

Liquidity features:

- `log_spread_mean_0_480`
- `log_spread_max_0_480`
- `log_volume_sum_0_480`
- `volume_imbalance_mean_0_480`

The fitted target is:

```text
log_target_rv_480_600 = log(target_rv_480_600)
```

Predictions are transformed back onto realised-volatility scale:

```text
predicted_rv_480_600 = max(exp(predicted_log_rv) - EPS, prediction_floor)
```

The current reported HAR+ run uses:

```text
min_target_rv = 1e-4
prediction_floor = 1e-4
```

Rows with `target_rv_480_600 <= min_target_rv` are excluded from fitting and evaluation because RMSPE is unstable near zero.

## 4. Metrics

Report **test** metrics as the model result. Train metrics are only used to check overfitting.

RMSPE:

```text
sqrt(mean(((y_true - y_pred) / y_true)^2))
```

QLIKE uses the direct realised-volatility ratio:

```text
ratio = y_true / y_pred
QLIKE = mean(ratio - log(ratio) - 1)
```

This is not the variance-squared QLIKE version.

## 5. Baseline

`har_model_StockLevel.py` also reports a simple persistence baseline:

```text
naive_last_2min prediction = rv_360_480
```

This uses realised volatility from the last 2 minutes of the input window to predict the next 2 minutes.

## 6. Results

Full CV finished against `processed/fold_0` through `processed/fold_4` and regenerated the HAR+ outputs.

The final reported result should use **test** metrics, not train metrics.

### 6.1 Headline Result

| Model | Evaluation | Rows | RMSPE | QLIKE | MSE |
|---|---|---:|---:|---:|---:|
| HAR+ | pooled 5-fold test | 428,513 | 0.449822 | 0.050025 | 4.9693e-07 |
| naive_last_2min | pooled 5-fold test | 428,513 | 0.520140 | 0.081341 | 6.7978e-07 |

HAR+ outperforms the naive last-2-minute baseline on all three metrics:

- RMSPE improves by about 13.5%.
- QLIKE improves by about 38.5%.
- MSE improves by about 26.9%.

### 6.2 Fold Stability

| Model | Mean Fold RMSPE | Std Fold RMSPE | Mean Fold QLIKE | Std Fold QLIKE |
|---|---:|---:|---:|---:|
| HAR+ | 0.449666 | 0.011846 | 0.050025 | 0.000745 |
| naive_last_2min | 0.519887 | 0.016232 | 0.081340 | 0.001379 |

### 6.3 Fold Test Results

| Fold | HAR+ RMSPE | HAR+ QLIKE | HAR+ MSE | Naive RMSPE | Naive QLIKE | Naive MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.457491 | 0.050891 | 4.6568e-07 | 0.529393 | 0.083510 | 6.5539e-07 |
| 1 | 0.442609 | 0.049939 | 5.0114e-07 | 0.507010 | 0.081737 | 6.6800e-07 |
| 2 | 0.461208 | 0.049796 | 4.8533e-07 | 0.535980 | 0.080890 | 6.4913e-07 |
| 3 | 0.429698 | 0.048796 | 4.7633e-07 | 0.494535 | 0.079233 | 6.3491e-07 |
| 4 | 0.457324 | 0.050701 | 5.5619e-07 | 0.532518 | 0.081333 | 7.9153e-07 |

HAR+ beats the naive baseline on every test fold.

### 6.4 Run Used

```bash
python3 har_model_StockLevel.py --folds-root processed --folds 0 1 2 3 4
```

## 7. How to Run

Run HAR+ on one shared fold:

```bash
python3 har_model_StockLevel.py --fold-dir processed/fold_0
```

Run full HAR+ five-fold CV:

```bash
python3 har_model_StockLevel.py --folds-root processed --folds 0 1 2 3 4
```

## 8. Runtime and Prediction Latency

Latency was measured during the full five-fold HAR+ run. The table below reports the **average prediction-only latency** after HAR features are built and the linear model is fitted.

Run the same latency-producing CV command:

```bash
python3 har_model_StockLevel.py --folds-root processed --folds 0 1 2 3 4
```

| Metric | Five-Fold Average |
|---|---:|
| Prediction rows per fold | 85,702.6 |
| Latency repeats per fold | 25 |
| Mean prediction time | 0.002006 sec |
| Median prediction time | 0.001688 sec |
| Minimum prediction time | 0.001018 sec |
| Maximum prediction time | 0.005767 sec |
| Mean per-row prediction latency | 0.023404 microseconds |
| Average rows per second | 44.49M |

On average, HAR+ predicts about **85.7k rows in 2.0 milliseconds**. The detailed per-fold latency output is saved to:

```text
har_rv_latency.csv
```

## 9. Outputs

HAR+ outputs:

```text
har_rv_predictions.csv
har_rv_metrics.csv
har_rv_cv_metrics.csv
har_rv_model_coefficients.csv
har_rv_model_coefficients_metrics.csv
har_rv_latency.csv
```

When `--fold-dir` is used, `har_rv_metrics.csv` contains train/test rows for HAR+ and `naive_last_2min`.

When `--folds-root` is used, `har_rv_cv_metrics.csv` includes fold rows plus pooled, fold-mean, and fold-standard-deviation summaries.

## 10. Notes

- This is a stock-level prediction task, not a cluster-level task.
- The current HAR+ model is pooled across stock-level rows; it is not one separate regression per stock.
- Cluster labels are not used.
- Compare models using the same target filter, prediction floor, and QLIKE formula.

## 11. References

- Corsi, F. (2009). [A Simple Approximate Long-Memory Model of Realized Volatility](https://doi.org/10.1093/jjfinec/nbp001).
- Patton, A. (2011). [Volatility forecast comparison using imperfect volatility proxies](https://EconPapers.repec.org/RePEc:eee:econom:v:160:y:2011:i:1:p:246-256).
