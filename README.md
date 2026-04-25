# HAR-RV Volatility Prediction 

## 1. Project Aim

The aim of this project is to build an interpretable volatility prediction tool for order book data. The final modelling task is a within-session prediction problem:

- Each `time_id` represents a 10 minute trading interval.
- The first 8 minutes are used as the information window.
- The final 2 minutes are used as the prediction target.
- The model predicts realised volatility for the final 2 minutes.

This setup avoids assuming that `time_id` values are sequential across trading sessions. It also produces a trader-friendly forecast: given the first 8 minutes of order book behaviour, estimate how volatile the stock will be in the next 2 minutes.

## 2. Repository Structure

| File | Purpose |
|---|---|
| `eda.ipynb` | Exploratory data analysis and order book preprocessing experiments. |
| `har_rv_model.py` | Standalone HAR-RV modelling pipeline. |
| `individual_book_train/stock_*.csv` | Raw order book data for 112 anonymous stocks. |
| `har_rv_features.csv` | Feature table created by the model script. |
| `har_rv_predictions.csv` | Holdout predictions from the HAR-RV model. |
| `har_rv_model_coefficients.csv` | Fitted model coefficients. |
| `har_rv_model_coefficients_metrics.csv` | Train/test performance metrics. |
| `har_rv_cv_metrics.csv` | Cross-validation metrics. |

## 3. EDA Workflow

The EDA notebook focuses on understanding and transforming ultra-high-frequency order book data into interpretable market features.

### 3.1 Raw Data

Each stock file contains order book snapshots with:

- `time_id`
- `seconds_in_bucket`
- level 1 and level 2 bid prices
- level 1 and level 2 ask prices
- bid sizes
- ask sizes
- `stock_id`

The data is not guaranteed to contain every second from 0 to 599 for each `time_id`, so missing seconds may need to be forward-filled or back-filled for analyses requiring a regular time grid.

### 3.2 EDA Feature Engineering

The notebook computes the following core order book features.

**Best bid and best ask**

The best bid is the highest available bid price across levels 1 and 2. The best ask is the lowest available ask price across levels 1 and 2.

```text
best_bid = max(bid_price1, bid_price2)
best_ask = min(ask_price1, ask_price2)
```

**Weighted average price**

The weighted average price, WAP, estimates the mid-market execution price using the best bid/ask prices and their opposite-side sizes:

```text
WAP = (best_bid_price * best_ask_size + best_ask_price * best_bid_size)
      / (best_bid_size + best_ask_size)
```

WAP is the central price series used to calculate log returns and realised volatility.

**Bid-ask spread**

```text
bid_ask_spread = best_ask / best_bid - 1
```

The spread measures trading cost and liquidity. Wider spreads usually indicate less liquid or more uncertain markets.

**Total volume**

```text
total_volume = bid_size1 + bid_size2 + ask_size1 + ask_size2
```

This captures visible order book depth.

**Missing seconds**

The EDA notebook includes a function to reindex each `time_id` to all seconds from 0 to 599, then forward-fill and back-fill missing observations. This is useful when comparing stocks on a regular second-by-second grid.

### 3.3 EDA Outputs

The notebook also creates a wide WAP matrix and visualises denormalised price distributions across stocks. This helps compare price levels and detect differences between stocks before modelling.

For the final model, the script does not require a fully regular 600-row grid per `time_id`; it computes realised volatility directly from observed WAP log returns inside each window.

## 4. Model Workflow

The modelling pipeline is implemented in `har_rv_model.py`.

```text
raw stock files
    -> order book features
    -> realised volatility windows
    -> HAR-RV feature table
    -> log transformation
    -> train/test split and cross-validation
    -> predictions, coefficients, and metrics
```

### 4.1 Input and Target Windows

For each stock and `time_id`, the 10 minute bucket is split as follows:

| Window | Seconds | Role |
|---|---:|---|
| Full input window | 0-480 | Available information |
| Long HAR feature | 0-480 | 8 minute realised volatility |
| Medium HAR feature | 240-480 | Recent 4 minute realised volatility |
| Short HAR feature | 360-480 | Recent 2 minute realised volatility |
| Target window | 480-600 | Future 2 minute realised volatility |

The model only uses information from seconds 0-479 to predict volatility over seconds 480-599.

### 4.2 Realised Volatility

For a WAP price series, log returns are:

```text
r_t = log(WAP_t) - log(WAP_{t-1})
```

Realised volatility over a window is:

```text
RV = sqrt(sum(r_t^2))
```

This is calculated separately for:

- `rv_360_480`
- `rv_240_480`
- `rv_0_480`
- `target_rv_480_600`

### 4.3 Additional Liquidity Features

The model also includes optional liquidity controls from the first 8 minutes:

- mean bid-ask spread
- maximum bid-ask spread
- total displayed volume
- mean volume imbalance

Volume imbalance is calculated from the best bid and ask sizes:

```text
volume_imbalance = (best_bid_size - best_ask_size)
                   / (best_bid_size + best_ask_size)
```

These features are included because volatility can be related to liquidity stress, widening spreads, and one-sided order book pressure.

## 5. HAR-RV Model Logic

HAR-RV stands for Heterogeneous Autoregressive Realised Volatility. The key idea is that volatility has memory over different horizons. Short-term, medium-term, and longer-term realised volatility can all help predict future volatility.

The model uses a linear regression on log realised volatility:

```text
log(target_rv_480_600)
    = beta_0
    + beta_1 log(rv_360_480)
    + beta_2 log(rv_240_480)
    + beta_3 log(rv_0_480)
    + beta_4 log(spread_mean_0_480)
    + beta_5 log(spread_max_0_480)
    + beta_6 log(volume_sum_0_480)
    + beta_7 volume_imbalance_mean_0_480
    + error
```

The log transformation is used because realised volatility is positive and right-skewed. It also makes the coefficients easier to interpret as proportional effects.

The prediction is transformed back to realised volatility scale:

```text
predicted_rv = exp(predicted_log_rv)
```

A prediction floor of `1e-5` is applied to avoid near-zero volatility forecasts, which can cause unstable QLIKE scores.

## 6. Filtering Rule

The model filters out target windows with realised volatility at or below `1e-5`.

Reason:

- RMSPE divides by the true volatility.
- If true realised volatility is zero or extremely close to zero, a normal-sized prediction creates an enormous percentage error.
- These near-flat windows can dominate the average metric even though they are not representative of trader-facing volatility risk.

In the full feature table:

- total rows: 428,932
- stocks: 112
- rows filtered by `target_rv_480_600 <= 1e-5`: 27

This removes only a tiny fraction of the data while making RMSPE and QLIKE more stable.

## 7. Evaluation Framework

The model is evaluated using a random 80/20 holdout split and 5-fold cross-validation.

### 7.1 RMSPE

Root mean squared percentage error:

```text
RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))
```

This is useful because volatility levels differ across stocks. RMSPE measures relative forecasting error.

### 7.2 QLIKE

QLIKE evaluates volatility forecasts on variance scale:

```text
ratio = true_variance / predicted_variance
QLIKE = mean(ratio - log(ratio) - 1)
```

It is commonly used for volatility forecasting because it penalises poor variance forecasts, especially underprediction.

### 7.3 MSE

Mean squared error:

```text
MSE = mean((y_true - y_pred)^2)
```

MSE gives an absolute error measure on realised volatility scale.

## 8. Results

### 8.1 Holdout Metrics

| Split | Rows | Filtered Rows | Prediction Floor | RMSPE | QLIKE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| Train | 343,124 | 27 | 0.00001 | 0.8886 | 1.0503 | 4.5898e-07 |
| Test | 85,781 | 27 | 0.00001 | 0.7892 | 0.2451 | 4.8693e-07 |

The test RMSPE is approximately `0.789`, meaning the model's relative volatility forecast error is around 79% on the holdout set. This is not unusual for noisy high-frequency volatility prediction, especially using an interpretable linear model.

### 8.2 Model Coefficients

| Feature | Coefficient |
|---|---:|
| Intercept | -1.0328 |
| `log_rv_0_480` | 0.6454 |
| `log_rv_240_480` | 0.1479 |
| `log_rv_360_480` | 0.1356 |
| `volume_imbalance_mean_0_480` | -0.0134 |
| `log_spread_mean_0_480` | -0.0118 |
| `log_spread_max_0_480` | 0.0098 |
| `log_volume_sum_0_480` | -0.0027 |

The largest coefficient is `log_rv_0_480`, which means the full 8 minute realised volatility is the strongest predictor of final 2 minute volatility. The 4 minute and 2 minute realised volatility features also contribute, showing that recent volatility persistence matters.

The liquidity variables have smaller coefficients. They add context, but the model is primarily driven by realised volatility history.

### 8.3 Cross-Validation Metrics

| Fold | Rows | RMSPE | QLIKE | MSE |
|---:|---:|---:|---:|---:|
| 1 | 85,781 | 0.7892 | 0.2451 | 4.8693e-07 |
| 2 | 85,781 | 0.7867 | 0.2273 | 4.6127e-07 |
| 3 | 85,781 | 0.9428 | 0.2158 | 4.5436e-07 |
| 4 | 85,781 | 1.0772 | 3.5267 | 4.6142e-07 |
| 5 | 85,781 | 0.7011 | 0.2197 | 4.6047e-07 |

Most folds have similar RMSPE and QLIKE performance. Fold 4 has a much larger QLIKE score, caused by at least one case where the first 8 minutes looked quiet but the final 2 minutes became volatile. This is a meaningful limitation of HAR-RV: it captures volatility persistence better than sudden jumps.

## 9. Interpretation for Traders

The model can be explained to traders in simple terms:

> If a stock has been volatile over the first 8 minutes, especially across recent shorter windows, the model expects higher volatility in the next 2 minutes.

The model is useful because it is transparent:

- Traders can see the exact features used.
- Coefficients show which signals matter most.
- The strongest signal is historical realised volatility.
- Liquidity measures provide secondary context.

The output should be interpreted as a short-horizon volatility forecast, not a directional price forecast. It does not say whether the price will rise or fall; it estimates the size of future price movement.

## 10. Limitations

The main limitations are:

- `time_id` values are treated as independent buckets, so the model does not learn long sequential time dynamics across buckets.
- HAR-RV assumes volatility persistence, so it can miss sudden volatility jumps after a calm input window.
- The model is linear on log volatility, which improves interpretability but limits flexibility.
- The model uses order book snapshots only and does not include news, market-wide shocks, or cross-stock dependencies.
- RMSPE is unstable for near-zero realised volatility, so a small filtering threshold is required.

## 11. How to Run

To train from the raw stock files:

```bash
python3 har_rv_model.py
```

To rerun evaluation from an existing feature table:

```bash
python3 har_rv_model.py --features-in har_rv_features.csv
```

To test quickly on a smaller number of stocks:

```bash
python3 har_rv_model.py --max-files 5
```

To remove liquidity controls and run a pure HAR-RV model:

```bash
python3 har_rv_model.py --no-liquidity
```

## 12. Conclusion

The final model is an interpretable HAR-RV volatility predictor. It uses realised volatility from the first 8 minutes of each bucket, plus liquidity controls, to predict volatility over the final 2 minutes. The model is easy to explain to traders because its logic follows a direct market intuition: recent realised volatility is informative about near-future realised volatility.

The current holdout performance is reasonable for a simple interpretable high-frequency volatility model. The most important caveat is that sudden late-window jumps remain difficult to predict from historical volatility alone.
