# TCN Context + LSTM Volatility Pipeline

## 1. Input Features

### Local LSTM Branch (60 steps, 1-second resolution)
- **return** — clipped log-return between consecutive WAP values, bounded to ±0.05.
- **log_wap** — log ratio of WAP to the 60s window's opening price.

### Macro Context TCN Branch (48 steps, 10-second buckets)
Aggregated over a 480-second window to capture broader market regimes:
- **ret_bucket** — sum of 1-second log-returns within the 10-second bucket.
- **log_wap_bucket** — log ratio of the bucket's closing WAP to the 480s window's opening price.
- **rv_bucket** — realized volatility within the 10-second bucket ($\sqrt{\sum r^2}$).

---

## 2. Stock Profiling & Clustering
*Computed over training data only, used for K-Means grouping prior to model ingestion:*
- Realized volatility (RV)
- Mean return
- Skewness
- Kurtosis
- Maximum drawdown

---

## 3. Architecture

![Model Architecture Diagram](architecture.png)

### A. Macro Context Encoder (TCN)
Processes the 480s aggregated history to establish the volatility regime.
- **Input:** `(B·K, 3, 48)`
- **Stack:** 5 Dilated Residual Blocks (CausalConv1d + LayerNorm + GELU + Dropout p=0.1).
- **Kernel & Dilations:** Kernel=3, Dilations=[1, 2, 4, 8, 16].
- **Receptive Field:** 63 buckets (630 seconds), safely covering the 480s input.
- **Output:** Global average pooling over time yields a context representation of shape `(B·K, 64)`.

### B. Local Microstructure Encoder (LSTM)
Captures immediate, high-frequency price action.
- **Input:** `(B·K, 60, 2)`
- **Stack:** 3 LSTM layers × 256 hidden units (Dropout p=0.2).
- **Output:** Final hidden state `h_last` of shape `(B·K, 256)`.

### C. FiLM Fusion Layer
Feature-wise Linear Modulation dynamically scales and shifts the LSTM's micro-features based on the TCN's macro-regime context.
1. **Context Mapping:** The 64-dim TCN context is projected into scaling ($\gamma$) and shifting ($\beta$) vectors of size 256.
2. **Modulation:** $h_{cond} = \gamma \odot h_{last} + \beta$

### D. Prediction Heads
- **Main Head:** Linear(256, 64) → ReLU → Linear(64, 1). Predicts $\log(RV_{fut} / RV_{local})$.
- **Auxiliary Head:** Linear(64, 1) directly off the TCN context. Predicts $\log(RV_{480} / RV_{60})$.

---

## 4. Dual-Loss Optimization & Evaluation
- **Loss Function:** $L = L_{main} + \alpha \times MSE_{aux}$
  - $L_{main}$ is typically MSE evaluated on the main prediction.
  - Auxiliary loss ($\alpha=0.2$) forces the TCN to learn meaningful macro representations independent of the LSTM.
- **Evaluation:** Incorporates per-cluster Duan smearing correction $\hat{\delta}$ to adjust for log-scale bias when retransforming to raw RV.


### Pipeline complete — Test Set Summary

| Metric | No Smearing | Smearing |
| :--- | ---: | ---: |
| **QLIKE** | 0.013670 | 0.012947 |
| **RMSE** | 0.000397 | 0.000418 |
| **RMSPE** | 0.160707 | 0.172198 |
| **MAPE** | 0.116533 | 0.120704 |
| **R2** | 0.928134 | 0.920200 |