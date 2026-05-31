# DATA3888 Finance Group 21

Volatility forecasting project using Optiver-style high-frequency limit order book data. The project compares classical and machine learning models for short-term realised volatility forecasting, then communicates the results through a Quarto report and an interactive Dash app.

## Project Summary

### Research question

Can a heterogeneous ensemble of volatility forecasting models systematically outperform individual classical benchmarks for short-term volatility forecasting using high-frequency order book data?

### Forecasting task

- Input window: first 8 minutes of each 10-minute `time_id`
- Target window: final 2 minutes of each 10-minute `time_id`
- Target variable: realised volatility over seconds `480-599`
- Dataset structure: per-second order book snapshots across 112 stocks

### Main models

- `GARCH`: classical volatility benchmark
- `HAR-X`: interpretable HAR baseline augmented with order book liquidity features
- `QLIKE-LGB`: HAR residual stacker using LightGBM and QLIKE-oriented correction
- `GNN`: spatio-temporal neural model with Mamba, GATv2, and a learned HAR blend

### Current checked-in results

From `outputs/evaluation/fold_metrics.json`:

| Model | Mean QLIKE | Mean RMSPE |
|---|---:|---:|
| HAR-X | 0.0511 | 0.5405 |
| QLIKE-LGB | 0.0450 | 0.5491 |
| GNN | 0.0725 | 0.4113 |
| GARCH | 0.0564 | 1.2637 |

Interpretation:

- `QLIKE-LGB` gives the best calibrated forecasts under QLIKE.
- `GNN` gives the lowest RMSPE.
- `HAR-X` is the main interpretable benchmark.
- `GARCH` is the classical baseline.

## Simplified Repo

The repo is easier to understand as four layers: data preparation, models, evaluation/reporting, and app delivery.

```text
DATA3888-Finance-21/
+-- Data preparation
|   +-- preprocess.ipynb          # raw order book CSVs -> processed folds
|   +-- eda.ipynb                 # exploratory analysis
|   +-- EDA_pictures/             # exported EDA figures
|
+-- Models
|   +-- HAR-X/                    # interpretable HAR-X benchmark
|   +-- GARCH/                    # classical GARCH benchmark
|   +-- HAR+LightGBM(QLIKE)/      # QLIKE-optimised residual stacker
|   +-- GNN/                      # spatio-temporal GNN model
|
+-- Evaluation
|   +-- diebold_mariano.ipynb     # statistical model comparison
|   +-- outputs/evaluation/       # metrics, DM tests, comparison plots
|
+-- Delivery
    +-- Report/                   # final Quarto report
    +-- DashApp/                  # interactive dashboard
    +-- README.md
```

### Key entry points

| Task | Start here |
|---|---|
| Create processed training folds | `preprocess.ipynb` |
| Explore the data | `eda.ipynb` |
| Run HAR-X baseline | `HAR-X/HAR.py` |
| Run GARCH benchmark | `GARCH/garch.ipynb` |
| Run GNN pipeline | `GNN/GNN_preprocess.py`, then `GNN/GNN.py` |
| Run QLIKE LightGBM stacker | `HAR+LightGBM(QLIKE)/qlike_hybrid.py` |
| Compare model significance | `diebold_mariano.ipynb` |
| Render final report | `Report/DATA3888_Report.qmd` |
| Launch dashboard | `DashApp/dashapp.py` |

### Expected external data folders

These folders are part of the runtime pipeline but are not committed to the repo because they are large generated data artifacts:

```text
individual_book_train/             raw Optiver-style stock CSVs
individual_book_train_denorm/      cleaned per-second stock CSVs
processed/                         fold-level parquet files used by models
```

## Step-By-Step Reproduction Order For Preprocessing

The preprocessing notebook is the foundation for the rest of the project. Run this first if you want to reproduce model training or rebuild downstream files.

### 1. Prepare raw data

Place raw stock CSV files in a folder like:

```text
individual_book_train/
+-- stock_0.csv
+-- stock_1.csv
+-- ...
```

Each file should contain the raw order book columns used in `preprocess.ipynb`, including:

- `time_id`
- `seconds_in_bucket`
- `bid_price1`, `bid_price2`
- `ask_price1`, `ask_price2`
- `bid_size1`, `bid_size2`
- `ask_size1`, `ask_size2`

### 2. Open and run `preprocess.ipynb`

The notebook creates the project base features:

- `wap`
- `bid_ask_spread`
- `total_volume`
- `price_spread`
- `depth_imbalance`

It also fills missing seconds within each `time_id`, then writes cleaned stock files to:

```text
individual_book_train_denorm/
```

### 3. Update the local data path inside `preprocess.ipynb`

The notebook currently contains a personal absolute path in the final `split_and_save(...)` call. Update `data_dir` so it points to your local `individual_book_train_denorm/` folder.

The call should conceptually look like:

```python
split_and_save(
    data_dir="individual_book_train_denorm",
    output_dir="processed",
)
```

### 4. Generate processed folds

After the notebook finishes, the expected output is:

```text
processed/
+-- full.parquet
+-- fold_0/train.parquet
+-- fold_0/test.parquet
+-- fold_1/train.parquet
+-- fold_1/test.parquet
+-- ...
+-- fold_4/train.parquet
+-- fold_4/test.parquet
```

These processed files are used by `HAR-X`, `GARCH`, `GNN`, `HAR+LightGBM(QLIKE)`, the dashboard build scripts, and the report.

## Report Render

The report is already rendered at:

```text
Report/DATA3888_Report.html
```

To regenerate it:

```powershell
cd Report
quarto render DATA3888_Report.qmd
```

### Report requirements

- Quarto installed on your machine
- Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`

### Report notes

- The report uses `Report/DATA3888_Report.qmd` as its source.
- Figures are loaded from `EDA_pictures/`, `DashApp/`, and `Report/`.
- On Windows, the current relative paths should work as-is.
- On case-sensitive systems, check the `EDA_pictures/` path casing in the `.qmd`.

## Dashboard And App Run Instructions

The Dash app is the easiest way to inspect model behaviour visually.

### 1. Install dashboard dependencies

```powershell
cd DashApp
python -m pip install -r requirements.txt
```

### 2. Run the app

```powershell
python dashapp.py
```

Then open the local address printed by Dash, usually:

```text
http://127.0.0.1:8050
```

### 3. Required dashboard data

The app expects:

```text
DashApp/dashboard_data.parquet
```

This file is already checked into the repo.

### 4. Optional: rebuild dashboard data

To rebuild the dashboard data from processed files and model predictions, update the hardcoded paths in:

- `DashApp/bucket_RV.py`
- `DashApp/merge.py`

Then run:

```powershell
cd DashApp
python bucket_RV.py
python merge.py
```

`bucket_RV.py` computes realised volatility buckets. `merge.py` combines bucket RVs with model prediction CSVs and writes `dashboard_data.parquet`.

## Dependency Guidance

There is currently no single root `requirements.txt`, so install dependencies by component.

### Core analysis and notebooks

```powershell
python -m pip install pandas numpy scipy matplotlib pyarrow scikit-learn joblib jupyter
```

### Report

Install Quarto separately from:

```text
https://quarto.org/
```

The report also uses the core Python analysis stack.

### Dashboard

```powershell
cd DashApp
python -m pip install -r requirements.txt
```

### GARCH

```powershell
python -m pip install arch
```

### HAR + LightGBM

```powershell
python -m pip install lightgbm optuna shap psutil
```

### GNN

```powershell
python -m pip install torch
python -m pip install torch-geometric
```

The GNN is the heaviest dependency stack and is best run in an environment with GPU support where possible.

## Notes On Hardcoded Paths

Several files still contain personal absolute paths. These must be updated before a full reproduction run on a new machine.

### Files to check

| File | What to update |
|---|---|
| `preprocess.ipynb` | raw and denormalized data directories |
| `GARCH/garch.ipynb` | `PROCESSED_DIR` if your `processed/` folder is elsewhere |
| `GNN/GNN_preprocess.py` | `DATA_DIR` |
| `GNN/GNN.py` | `DATA_DIR` |
| `HAR+LightGBM(QLIKE)/qlike_hybrid.py` | `DENORM_DIR`, `DATA_DIR`, `OUTPUT_DIR`, and derived output directories |
| `DashApp/bucket_RV.py` | `INPUT_PATH` |
| `DashApp/merge.py` | `BASE_DIR` and model prediction file paths |
| `diebold_mariano.ipynb` | model prediction directories and evaluation output directory |

### Most important path issue

The repo does not include the generated `processed/` folder. Most model scripts expect it to exist. Recreate it with `preprocess.ipynb` before rerunning model training.

### Recommended cleanup

For a cleaner future version of the repo:

1. move all paths into one shared config file
2. add a root `requirements.txt` or `environment.yml`
3. convert notebook-only stages into scripts where possible
4. add one orchestration script for preprocessing, evaluation, dashboard data, and report rendering
