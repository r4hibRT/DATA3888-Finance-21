"""
merge_predictions.py
====================
Merge bucket_rv.parquet with prediction CSVs from all models
into a single master parquet for the Dash app.

Input files:
  processed/bucket_rv.parquet          -- rv_b0..rv_b3, rv_target per (stock_id, time_id)
  processed/har_predictions.csv        -- stock_id, time_id, actual_rv, predicted_rv
  processed/lgbm_predictions.csv
  processed/xgb_predictions.csv
  processed/garch_predictions.csv
  processed/gnn_predictions.csv        -- optional

Output:
  processed/dashboard_data.parquet
  Columns:
    stock_id, time_id,
    rv_b0, rv_b1, rv_b2, rv_b3, rv_target,   <- from bucket_rv
    actual_rv,                                 <- from any model (all share same actual)
    pred_har, pred_lgbm, pred_xgb,
    pred_garch, pred_gnn,                      <- NaN if file not found
    regime                                     <- calm/normal/elevated/stressed

Run: python merge_predictions.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────

BASE_DIR    = Path("C:\\Users\\ngdo0466\\Downloads\\processed")
BUCKET_PATH = BASE_DIR / "bucket_rv.parquet"
OUTPUT_PATH = BASE_DIR / "dashboard_data.parquet"

# Map model name -> prediction file
MODEL_FILES = {
    "har":   BASE_DIR / "har_predictions.csv",
    "lgbm":  BASE_DIR / "lgbm_predictions.csv",
    "xgb":   BASE_DIR / "xgb_predictions.csv",
    "garch": BASE_DIR / "garch_predictions.csv",
    "gnn":   BASE_DIR / "gnn_predictions.csv",
}


def regime(rv):
    if rv < 0.001:  return "calm"
    if rv < 0.005:  return "normal"
    if rv < 0.02:   return "elevated"
    return "stressed"


def log(msg):
    print(f"[Merge] {msg}", flush=True)


def main():
    # ── Load bucket RV ────────────────────────────────────────────
    log(f"Loading bucket_rv ...")
    bucket = pd.read_parquet(BUCKET_PATH)
    bucket["stock_id"] = bucket["stock_id"].astype(np.int32)
    bucket["time_id"]  = bucket["time_id"].astype(np.int32)
    log(f"  {bucket.shape[0]:,} rows, columns: {list(bucket.columns)}")

    master = bucket.copy()
    actual_rv_loaded = False

    # ── Load and merge each model ─────────────────────────────────
    for model, path in MODEL_FILES.items():
        if not path.exists():
            log(f"  {path.name} not found — skipping {model}")
            master[f"pred_{model}"] = np.nan
            continue

        log(f"  Loading {path.name} ...")
        df = pd.read_csv(path)
        df["stock_id"] = df["stock_id"].astype(np.int32)
        df["time_id"]  = df["time_id"].astype(np.int32)

        # Pick up actual_rv from first available model
        if not actual_rv_loaded and "actual_rv" in df.columns:
            df_actual = df[["stock_id", "time_id", "actual_rv"]]
            master = master.merge(df_actual, on=["stock_id", "time_id"], how="inner")
            actual_rv_loaded = True
            log(f"    inner join on actual_rv: {len(master):,} rows remaining")

        # Merge predicted_rv with model-specific column name
        df_pred = df[["stock_id", "time_id", "predicted_rv"]].rename(
            columns={"predicted_rv": f"pred_{model}"})
        master = master.merge(df_pred, on=["stock_id", "time_id"], how="inner")
        log(f"    inner join pred_{model}: {len(master):,} rows remaining")
        log(f"    merged pred_{model}: {df_pred[f'pred_{model}'].notna().sum():,} values")

    if not actual_rv_loaded:
        log("  WARNING: no actual_rv found in any file — using rv_target from bucket_rv")
        master["actual_rv"] = master["rv_target"]

    # ── Regime label ──────────────────────────────────────────────
    master["regime"] = master["actual_rv"].apply(regime)

    # ── Reorder columns ───────────────────────────────────────────
    bucket_cols = ["rv_b0", "rv_b1", "rv_b2", "rv_b3", "rv_target"]
    pred_cols   = [f"pred_{m}" for m in MODEL_FILES]
    col_order   = (
        ["stock_id", "time_id"]
        + [c for c in bucket_cols if c in master.columns]
        + ["actual_rv"]
        + [c for c in pred_cols if c in master.columns]
        + ["regime"]
    )
    master = master[col_order]

    # ── Cast floats ───────────────────────────────────────────────
    float_cols = [c for c in master.columns if c not in ["stock_id", "time_id", "regime"]]
    master[float_cols] = master[float_cols].astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────
    master.to_parquet(OUTPUT_PATH, index=False, compression="snappy")

    log(f"\nSaved -> {OUTPUT_PATH}")
    log(f"Shape: {master.shape}")
    log(f"Columns: {list(master.columns)}")
    log(f"\nRegime distribution:")
    print(master["regime"].value_counts().to_string())
    log(f"\nMissing predictions per model:")
    for col in pred_cols:
        if col in master.columns:
            n_miss = master[col].isna().sum()
            pct = n_miss / len(master) * 100
            print(f"  {col:15s}: {n_miss:,} missing ({pct:.1f}%)")
    log(f"\nSample:")
    print(master.head(5).to_string(index=False))


if __name__ == "__main__":
    main()