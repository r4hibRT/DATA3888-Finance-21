# Transformer Context + FiLM-Conditioned LSTM Volatility Pipeline (v4)

## 1. Input Features

### Local LSTM Branch (60 steps, 1-second resolution)
- **return** — clipped log-return between consecutive WAP values, bounded to ±0.05.
- **log_wap** — log ratio of WAP to the 60s window's opening price.

### Macro Context Transformer Branch (48 steps, 10-second buckets)
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

## 3. Architecture Overview

![Architecture Diagram](architecture.png)

### A. Macro Context Encoder (Transformer)
Replaces the previous TCN to leverage **full self-attention**, allowing every time bucket to directly attend to non-local regime signals without relying on fixed causal dilations.
- **Input:** `(B·K, 48, 3)`
- **Embedding:** Linear projection (3 → 64) + **Learned Positional Encoding**. A learned encoding is used instead of standard sinusoids because the 48-bucket sequence is short and fixed-length.
- **Transformer Stack:** 2 Encoder layers, 4 attention heads.
- **Pre-Norm Architecture:** Uses `norm_first=True` to stabilize gradients early in training, which is critical since the downstream FiLM layer initializes near identity.
- **Output:** Global average pooling produces a context vector of shape `(B·K, 64)`.

### B. Local Microstructure Encoder (LSTM)
Captures immediate, high-frequency price action.
- **Input:** `(B·K, 60, 2)`
- **Stack:** 3 LSTM layers × 256 hidden units (Dropout p=0.2).
- **Output:** Final hidden state `h_last` of shape `(B·K, 256)`.

### C. FiLM Fusion Layer (Feature-wise Linear Modulation)
Dynamically scales and shifts the LSTM's micro-features based on the Transformer's macro-regime context.
1. **Context Mapping:** The 64-dim Transformer context is projected into scaling ($\gamma$) and shifting ($\beta$) vectors of size 256.
2. **Modulation:** $h_{cond} = \gamma \odot h_{last} + \beta$

### D. Prediction Heads
- **Main Head:** Linear(256, 64) → ReLU → Linear(64, 1). Predicts $\log(RV_{fut} / RV_{local})$.
- **Auxiliary Head:** Linear(64, 1) directly off the Transformer context. Predicts $\log(RV_{480} / RV_{60})$.

---

## 4. Dual-Loss Optimization & Evaluation
- **Loss Function:** $L = L_{main} + \alpha \times MSE_{aux}$
  - $L_{main}$ is typically MSE evaluated on the main prediction.
  - Auxiliary loss ($\alpha=0.2$) forces the Transformer to learn meaningful macro representations (the RV ratio) independent of the LSTM gradient flow.
- **Evaluation:** Incorporates per-cluster **Duan smearing correction** $\hat{\delta}$ to adjust for exponential retransformation bias when mapping log-predictions back to raw RV.

### Pipeline complete — Test Set Summary (v4 Transformer)

| Metric | No Smearing | Smearing |
| :--- | ---: | ---: |
| **QLIKE** | 0.008297 | 0.008210 |
| **RMSE** | 0.000320 | 0.000328 |
| **RMSPE** | 0.130125 | 0.133402 |
| **MAPE** | 0.096225 | 0.097349 |
| **R2** | 0.951543 | 0.948991 |