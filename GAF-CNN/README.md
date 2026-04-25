# GAF/MTF Image Encoding + 2D-CNN Cluster Volatility Pipeline (v2)

This pipeline converts high-frequency limit order book (LOB) data into multi-channel images using Gramian Angular Fields (GAF) and Markov Transition Fields (MTF). It extracts spatial relationships from these images using a Multi-Scale Inception CNN, combining them with a scalar statistical branch to predict realized volatility.

## 1. Windowing Strategy
Each `time_id` contains exactly 600 raw seconds. These are divided into **3 non-overlapping samples** of 180 seconds each:
* **Sample 0:** Input `[0 : 60)`, Target `[60 : 180)`
* **Sample 1:** Input `[180 : 240)`, Target `[240 : 360)`
* **Sample 2:** Input `[360 : 420)`, Target `[420 : 540)`
* *Rows 540–599 are unused to maintain strict non-overlapping periods.*

---

## 2. Input Features & Preprocessing

### A. Raw Base Features (Computed over 60s Input Window)
For each second, 7 base features are computed:
1.  **`return`:** Clipped log-return bounded to ±0.05.
2.  **`log_wap`:** Intraday drift via $\log(WAP / open)$.
3.  **`rv_short`:** Realized volatility over a 5-step rolling window.
4.  **`rv_medium`:** Realized volatility over a 20-step rolling window.
5.  **`rv_long`:** Realized volatility over a 50-step rolling window.
6.  **`bpv`:** Bipower variation (5-step), measuring jump-robust volatility.
7.  **`jump`:** Derived jump component (`rv_short` - `bpv`).

### B. Stock Profiling & Clustering (Training Data Only)
Stocks are clustered using K-Means based on 5 summary statistics (RV, mean return, skewness, kurtosis, max drawdown) computed over the 60s input windows.

---

## 3. Data Modalities & Encoders

![CNN Architecture Diagram](cnn_architecture.png)

### A. Multi-Channel Image Branch (GAF/MTF)
For each cluster, the selected base features (default: `return`, `log_wap`, `rv_short`, `rv_medium`) are transformed into images.

1.  **Downsampling:** The 60-second mean series is downsampled to `GAF_SIZE` (default: 64) using Piecewise Aggregate Approximation (PAA).
2.  **Image Generation:**
    * **Gramian Angular Sum Field (GASF):** Encodes the series as a sum matrix in polar coordinates.
    * **Gramian Angular Difference Field (GADF):** Encodes the series as a difference matrix.
    * **Markov Transition Field (MTF):** *(Optional)* Encodes transition probabilities across quantiles.
3.  **Input Shape:** `(B, C, 64, 64)`. Channels `C` = `n_clusters` $\times$ `n_features` $\times$ (2 for GASF/GADF, or 3 if MTF is enabled).

### B. Spatial Extraction (Multi-Scale Inception CNN)
Extracts patterns from the stacked image tensor.
* **Stage 1-3:** Residual Inception Stages. Each block applies parallel convolutions (1x1, 3x3, 5x5, 7x7) concatenated together, followed by a residual connection and a strided 3x3 downsampling convolution.
* **Pooling:** Adaptive Average Pooling to `(4, 4)` spatial dimensions.
* **Output:** Flattened vector of size $256 \times 4 \times 4 = 4096$.

### C. Scalar Feature Branch
Captures summary statistics that might be lost during image conversion.
* **Features:** Extracts the last value, mean, and standard deviation for 5 base features (`rv_short`, `rv_medium`, `rv_long`, `bpv`, `jump`) per cluster.
* **Encoder:** An MLP maps these $5 \times 3 \times n\_clusters$ features to a hidden representation (default: 64 units).

---

## 4. Fusion and Prediction Head
* **Concatenation:** The CNN output vector (4096) and the Scalar MLP output (64) are concatenated.
* **Head:** `Linear(4160, 256) → ReLU → Dropout → Linear(256, 128) → ReLU → Dropout → Linear(128, n_clusters)`.
* **Target:** Predicts $\log(RV_{fut} / RV_{input})$ per cluster.
* **Evaluation:** Includes a robust (winsorized) Duan smearing factor during testing to correct for exponential retransformation bias.

### GAF-CNN Pipeline v2 — Evaluation Summary (With Smearing)

| Metric  | Validation | Test |
| :------ | ---------: | ---: |
| **MSE**   | 0.00000025 | 0.00000027 |
| **RMSE**  | 0.00050441 | 0.00051912 |
| **R²**    | 0.860822   | 0.898202   |
| **MAE**   | 0.00027547 | 0.00028925 |
| **RMSPE** | 0.443412   | 0.589552   |
| **MAPE**  | 0.165223   | 0.177148   |
| **QLIKE** | 0.034646   | 0.035295   |