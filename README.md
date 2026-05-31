# DATA3888 Finance Group 21

This project forecasts short-term realised volatility using high-frequency limit order book data. It compares classical volatility benchmarks, HAR-style models, a QLIKE-optimised LightGBM residual stacker, and a spatio-temporal GNN. The final outputs are a Quarto report and an interactive Dash app for inspecting model behaviour.

## Project Summary

### Research question

Can a heterogeneous ensemble of volatility forecasting models systematically outperform individual classical benchmarks for short-term volatility forecasting using high-frequency order book data?

### Forecasting setup

| Item | Description |
|---|---|
| Input window | First 8 minutes of each 10-minute `time_id`, seconds `0-479` |
| Target window | Final 2 minutes of each 10-minute `time_id`, seconds `480-599` |
| Target variable | Realised volatility over the target window |
| Data frequency | Per-second order book snapshots |
| Stock universe | 112 stocks |

### Models

| Model | Role |
|---|---|
| `GARCH` | Classical volatility benchmark |
| `HAR-X` | Interpretable HAR baseline with order book liquidity features |
| `QLIKE-LGB` | HAR residual stacker using LightGBM and QLIKE-oriented correction |
| `GNN` | Spatio-temporal neural model using Mamba, GATv2, and a learned HAR blend |

### Checked-in results

The current reported fold metrics are stored in `outputs/evaluation/fold_metrics.json`.

| Model | Mean QLIKE | Mean RMSPE |
|---|---:|---:|
| HAR-X | 0.0511 | 0.5405 |
| QLIKE-LGB | 0.0450 | 0.5491 |
| GNN | 0.0725 | 0.4113 |
| GARCH | 0.0564 | 1.2637 |

`QLIKE-LGB` gives the best calibrated forecasts under QLIKE. `GNN` gives the lowest RMSPE. `HAR-X` is the main interpretable benchmark, and `GARCH` is the classical baseline.

## Current Repo Tree

This tree mirrors the current repo structure. Repeated prediction files are grouped with `*`, and repeated GNN fold contents are collapsed to keep the README readable.

```text
DATA3888-Finance-21/
+-- README.md
+-- preprocess.ipynb
+-- eda.ipynb
+-- diebold_mariano.ipynb
+-- appendix_feature_definitions_tmp.html
+-- DashApp/
|   +-- dashapp.py
|   +-- dashboard_data.parquet
|   +-- bucket_RV.py
|   +-- merge.py
|   +-- requirements.txt
|   +-- dashapp.png
|   +-- model_ranking.png
|   +-- regime_heatmap.png
|   +-- asset/
|   |   +-- custom.css
|   +-- __pycache__/
|       +-- dashapp.cpython-314.pyc
+-- EDA_pictures/
|   +-- eda_cross_stock_variation.png
|   +-- eda_feature_distributions.png
|   +-- eda_intraday_patterns.png
|   +-- eda_per_stock_target.png
|   +-- eda_target_distribution.png
|   +-- stock interaction.png
+-- GARCH/
|   +-- garch.ipynb
|   +-- garch_predictions_fold*.csv
+-- GNN/
|   +-- GNN.py
|   +-- GNN_preprocess.py
|   +-- prediction/
|       +-- gnn_nested_cv_summary.json
|       +-- fold_0/
|       +-- fold_1/
|       +-- fold_2/
|       +-- fold_3/
|       +-- fold_4/
+-- HAR+LightGBM(QLIKE)/
|   +-- qlike_hybrid.py
|   +-- qlike_lgb_fold*_predictions.csv
|   +-- Results.png
+-- HAR-X/
|   +-- HAR.py
|   +-- har_rv_predictions.csv
|   +-- harx_fold*_predictions.csv
+-- outputs/
|   +-- evaluation/
|       +-- base_predictions.csv
|       +-- dm_tests_all_models.csv
|       +-- dm_tests_all_models_corrected.csv
|       +-- fold_metrics.json
|       +-- model_comparison_forest.png
|       +-- model_comparison_pooled_oof.png
+-- Report/
    +-- DATA3888_Report.qmd
    +-- DATA3888_Report.html
    +-- paper.css
    +-- references.bib
    +-- workflow diagram.png
```

### Key entry points

| Task | File |
|---|---|
| Create processed folds | `preprocess.ipynb` |
| Run EDA | `eda.ipynb` |
| Run HAR-X | `HAR-X/HAR.py` |
| Run GARCH | `GARCH/garch.ipynb` |
| Preprocess GNN data | `GNN/GNN_preprocess.py` |
| Train GNN | `GNN/GNN.py` |
| Run QLIKE LightGBM stacker | `HAR+LightGBM(QLIKE)/qlike_hybrid.py` |
| Run statistical model comparison | `diebold_mariano.ipynb` |
| Render report | `Report/DATA3888_Report.qmd` |
| Launch dashboard | `DashApp/dashapp.py` |

### External data folders

These folders are expected by the pipeline but are not committed to the repo.

```text
individual_book_train/             raw Optiver-style stock CSVs
individual_book_train_denorm/      cleaned per-second stock CSVs from preprocessing
processed/                         generated fold-level parquet files
```

The most important missing folder is `processed/`. Most model scripts expect it to exist before training or evaluation.

## Step-By-Step Reproduction Order For Preprocessing

Run preprocessing first if you want to reproduce model training, regenerate folds, or rebuild downstream artifacts.

### 1. Prepare raw data

Place the raw order book CSV files in:

```text
individual_book_train/
+-- stock_0.csv
+-- stock_1.csv
+-- ...
```

Each stock file should contain:

- `time_id`
- `seconds_in_bucket`
- `bid_price1`, `bid_price2`
- `ask_price1`, `ask_price2`
- `bid_size1`, `bid_size2`
- `ask_size1`, `ask_size2`

### 2. Run `preprocess.ipynb`

Open and run:

```text
preprocess.ipynb
```

The notebook creates the base per-second features:

- `wap`
- `bid_ask_spread`
- `total_volume`
- `price_spread`
- `depth_imbalance`

It also fills missing seconds within each `time_id` and writes cleaned stock files to `individual_book_train_denorm/`.

### 3. Update local paths in `preprocess.ipynb`

The notebook currently assumes local paths. Check both:

- the raw file glob: `individual_book_train\stock_*.csv`
- the `data_dir` argument inside the final `split_and_save(...)` call

The final split call should conceptually point to your cleaned data folder:

```python
split_and_save(
    data_dir="individual_book_train_denorm",
    output_dir="processed",
)
```

### 4. Confirm generated outputs

After preprocessing, the expected generated structure is:

```text
processed/
+-- full.parquet
+-- fold_0/
|   +-- train.parquet
|   +-- test.parquet
+-- fold_1/
+-- fold_2/
+-- fold_3/
+-- fold_4/
```

These processed files are used by `HAR-X`, `GARCH`, `GNN`, `HAR+LightGBM(QLIKE)`, the dashboard build scripts, and the report.

## Report Render

The rendered report is already checked in:

```text
Report/DATA3888_Report.html
```

To regenerate it:

```powershell
cd Report
quarto render DATA3888_Report.qmd
```

### Report requirements

- Quarto installed locally
- Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`

### Report notes

- Source file: `Report/DATA3888_Report.qmd`
- Bibliography: `Report/references.bib`
- Styling: `Report/paper.css`
- Figures are loaded from `EDA_pictures/`, `DashApp/`, and `Report/`
- On case-sensitive systems, check image path casing because the report currently references both `EDA_pictures` and `eda_pictures`

## Dashboard And App Run Instructions

The Dash app is the interactive project demo. It uses the checked-in dashboard dataset, so it can be run without retraining the models.

### 1. Install dashboard dependencies

```powershell
cd DashApp
python -m pip install -r requirements.txt
```

### 2. Run the app

```powershell
python dashapp.py
```

Open the local address printed by Dash, usually:

```text
http://127.0.0.1:8050
```

### 3. Required dashboard data

The app expects:

```text
DashApp/dashboard_data.parquet
```

This file is already checked into the repo.

### 4. Optional dashboard data rebuild

To rebuild `dashboard_data.parquet`, first update hardcoded paths in:

- `DashApp/bucket_RV.py`
- `DashApp/merge.py`

Then run:

```powershell
cd DashApp
python bucket_RV.py
python merge.py
```

`bucket_RV.py` computes realised-volatility buckets from processed data. `merge.py` combines bucket RVs with model prediction CSVs.

## Dependency Guidance

There is no single root `requirements.txt` yet, so install dependencies by component.

### Core analysis and notebooks

```powershell
python -m pip install pandas numpy scipy matplotlib pyarrow scikit-learn joblib jupyter
```

### Report

Install Quarto separately:

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

Several files still contain personal absolute paths. Update these before a full reproduction run on a new machine.

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

### Practical warning

The checked-in outputs make the report and dashboard easy to view, but a full rerun still requires local data path fixes and the generated `processed/` folder.
