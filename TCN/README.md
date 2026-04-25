# TCN Input Channels (2 per time step)

- **return** — clipped log-return between consecutive WAP values, bounded to ±0.05
- **log_wap** — log ratio of WAP to the opening price, capturing intraday drift

# Stock Profiling Features (5 per stock, used for clustering only)

These are computed over training data to group stocks via K-Means — they don't enter the model as inputs:

- Realized volatility
- Mean return
- Skewness
- Kurtosis
- Maximum drawdown

---

# Model Architecture

![Model Architecture Diagram](architecture.png)

## Input

- Shape: `(B, K, 480, 2)` — batch size B, K clusters, 480 time steps, 2 channels
- **Reshape**: merge B×K → batch dim and permute for 1D convolutions

This produces the input for the TCN Stack:
- TCN input: `(B·K, 2, 480)`

## TCN Stack (Encoder)

A multi-scale Temporal Convolutional Network composed of 8 residual blocks. Each block applies a causal dilated 1D convolution, Weight Normalization, ReLU activation, and Dropout, combined with a residual connection. 

| Layer | Kernel Size | Dilation | Channels | Dropout |
|-------|-------------|----------|----------|---------|
| CausalConv1dBlock 1 | 3 | 1 | 64 | p=0.2 |
| CausalConv1dBlock 2 | 3 | 2 | 64 | p=0.2 |
| CausalConv1dBlock 3 | 3 | 4 | 64 | p=0.2 |
| CausalConv1dBlock 4 | 3 | 8 | 64 | p=0.2 |
| CausalConv1dBlock 5 | 3 | 16 | 64 | p=0.2 |
| CausalConv1dBlock 6 | 3 | 32 | 64 | p=0.2 |
| CausalConv1dBlock 7 | 3 | 64 | 64 | p=0.2 |
| CausalConv1dBlock 8 | 3 | 128 | 64 | p=0.2 |

Output: `(B·K, 64, 480)` — sequence of hidden states across all 480 time steps.

## Global Average Pooling

- Averages the TCN output sequence across the time dimension (`dim=2`).
- Output: Global representation vector, shape `(B·K, 64)`

## Prediction

1. **Linear(64, 1)** — prediction head applied directly to the pooled TCN representation
2. **Reshape** → `(B, K)` — output is `log(RV_fut / RV_in)`
3. **Duan smearing correction** — bias adjustment for log-scale predictions applied during evaluation to reconstruct raw RV accurately

### Pipeline complete — Test Set Summary

| Metric | No Smearing | Smearing |
| :--- | ---: | ---: |
| **QLIKE** | 0.034754 | 0.034065 |
| **RMSE** | 0.000497 | 0.000530 |
| **RMSPE** | 0.702134 | 0.814740 |
| **MAPE** | 0.187019 | 0.202316 |
| **R2** | 0.896672 | 0.882369 |