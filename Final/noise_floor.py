"""
noise_floor.py
==============
Estimates per-stock microstructure noise variance (zeta) and the
minimum meaningful RV threshold using two methods from the literature:

Method 1 — Spread-based (Kurisu 2018, small noise framework):
    zeta_hat = (mean_spread / 2)^2  per second
    RV_noise_floor = sqrt(2 * zeta_hat * window_seconds)

Method 2 — RV-based (Li, Laeven, Vellekoop 2018, eq. 7-8):
    Var(U) ~ h[Y,Y]^(jn) = sum((Y_{i+jn} - Y_i)^2) / (2*(n-jn+1))
    At large jn, the cross-term vanishes, leaving only Var(U).
    This works at 1-second resolution where noise is ~i.i.d.

Both give you:
    - zeta_hat per stock (noise variance per second)
    - nsr (noise-to-signal ratio = zeta / RV)
    - rv_noise_floor (minimum meaningful RV)
    - quality flag for each (time_id, stock_id) pair

Reads:  processed_v2/train_features.parquet  (has mean_spread_b*)
        raw per-stock CSVs (optional, for RV-based method)
Writes: noise_floor_estimates.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_DIR = Path("processed_v2")
OUTPUT_DIR   = Path("models")

# ── Config ──────────────────────────────────────────────────────────
BUCKET_SIZE    = 100    # seconds per bucket
N_BUCKETS      = 3      # number of input buckets
WINDOW_SECONDS = BUCKET_SIZE * N_BUCKETS   # 300s input window

# SNR thresholds from Li et al. (2018) empirical findings
NSR_LIQUID     = 1e-3   # Citigroup-like liquid stocks
NSR_ILLIQUID   = 1e-2   # less liquid stocks

# Quality tiers
SNR_HIGH_THRESHOLD   = 10.0   # RV is 10x noise floor — reliable
SNR_MARGINAL_THRESHOLD = 2.0  # RV is 2x noise floor — use with caution


# ── Spread-based noise estimator (Kurisu 2018) ───────────────────────

def estimate_noise_spread(mean_spread: float, window_seconds: int = WINDOW_SECONDS) -> dict:
    """
    Estimate noise variance from bid-ask spread.

    Theory: observed WAP = true mid-price + bid-ask bounce
    Noise std per trade ~ half the spread (Roll 1984 model)
    So Var(U) per second ~ (spread/2)^2

    RV_noise_floor = sqrt(2 * Var(U) * T)
    where T = number of seconds in the window (from Kurisu eq 4).
    """
    noise_std    = mean_spread / 2.0
    zeta_hat     = noise_std ** 2          # Var(U) per second
    rv_floor     = np.sqrt(2 * zeta_hat * window_seconds)
    log_rv_floor = np.log(max(rv_floor, 1e-12))

    return {
        "zeta_spread" : zeta_hat,
        "rv_floor_spread" : rv_floor,
        "log_rv_floor_spread": log_rv_floor,
    }


# ── RV-based noise estimator (Li et al. 2018, Proposition 3.1) ───────

def estimate_noise_rv(log_returns: np.ndarray, jn: int = 5) -> dict:
    """
    Estimate Var(U) from lagged realized volatility (eq 7-8 of Li et al.).

    h[Y,Y]^(jn) = sum((Y_{i+jn} - Y_i)^2) / (2*(n - jn + 1))
               P→  Var(U) - gamma(jn)
               ≈  Var(U)  when jn is large enough that gamma(jn) ≈ 0

    At 1-second resolution, noise is ~i.i.d. (Li et al. Section 7.5.1),
    so gamma(jn) ≈ 0 for jn >= 3.  We use jn=5 as default.

    Args:
        log_returns: array of per-second log returns (length T)
        jn:         lag for the RV estimator (default 5 seconds)
    """
    n = len(log_returns)
    if n < jn + 2:
        return {"zeta_rv": np.nan, "rv_floor_rv": np.nan, "log_rv_floor_rv": np.nan}

    # Lagged squared differences: (Y_{i+jn} - Y_i)^2 = (sum of j returns)^2
    # Approximate via sum of returns over jn steps
    lagged_sums = np.array([
        log_returns[i:i+jn].sum()
        for i in range(n - jn)
    ])
    hYY = np.sum(lagged_sums ** 2) / (2 * (n - jn + 1))

    # hYY ≈ Var(U) when noise is i.i.d. and jn is large enough
    zeta_hat     = max(hYY, 0.0)
    rv_floor     = np.sqrt(2 * zeta_hat * WINDOW_SECONDS)
    log_rv_floor = np.log(max(rv_floor, 1e-12))

    return {
        "zeta_rv"        : zeta_hat,
        "rv_floor_rv"    : rv_floor,
        "log_rv_floor_rv": log_rv_floor,
    }


# ── Quality classification ───────────────────────────────────────────

def classify_quality(rv: float, rv_floor: float) -> str:
    """
    Classify a prediction's reliability based on signal-to-noise ratio.

    SNR = RV / RV_noise_floor
        > 10   : HIGH     — prediction is reliable
        2 to 10: MARGINAL — use with caution, widen confidence intervals
        < 2    : NOISE    — RV is noise-dominated, prediction unreliable
    """
    if rv_floor <= 0 or rv <= 0:
        return "UNKNOWN"
    snr = rv / rv_floor
    if snr >= SNR_HIGH_THRESHOLD:
        return "HIGH"
    elif snr >= SNR_MARGINAL_THRESHOLD:
        return "MARGINAL"
    else:
        return "NOISE"


# ── Per-stock aggregate noise estimates ──────────────────────────────

def compute_stock_noise_profiles(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-stock noise profile by averaging across all time_ids.

    Returns one row per stock with:
        mean_spread   — average bid-ask spread
        zeta_spread   — noise variance estimate (spread method)
        rv_floor      — minimum meaningful RV
        log_rv_floor  — log(rv_floor) — use as LOG_RV_FLOOR in model
    """
    spread_cols = [f"mean_spread_b{b}" for b in range(N_BUCKETS)]
    spread_cols = [c for c in spread_cols if c in features_df.columns]

    if not spread_cols:
        print("WARNING: no mean_spread columns found in features. Using global default.")
        return pd.DataFrame()

    rows = []
    for stock_id, grp in features_df.groupby("stock_id"):
        mean_spread = grp[spread_cols].values.mean()
        est         = estimate_noise_spread(mean_spread)
        rows.append({
            "stock_id"    : stock_id,
            "mean_spread" : mean_spread,
            **est,
        })

    return pd.DataFrame(rows).sort_values("stock_id").reset_index(drop=True)


# ── Per (time_id, stock_id) SNR ──────────────────────────────────────

def compute_snr_per_sample(
    features_df : pd.DataFrame,
    targets_df  : pd.DataFrame,
    stock_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SNR for every (time_id, stock_id) pair.

    Uses per-stock noise floor from stock_profiles and the actual
    target RV to determine if each sample is signal or noise dominated.
    """
    df = targets_df.merge(
        features_df[["time_id", "stock_id"] +
                    [f"mean_spread_b{b}" for b in range(N_BUCKETS)
                     if f"mean_spread_b{b}" in features_df.columns]],
        on=["time_id", "stock_id"], how="left"
    )

    # Use per-time_id spread if available, else fall back to stock profile
    spread_cols = [f"mean_spread_b{b}" for b in range(N_BUCKETS)
                   if f"mean_spread_b{b}" in df.columns]
    if spread_cols:
        df["mean_spread_sample"] = df[spread_cols].mean(axis=1)
    else:
        df = df.merge(
            stock_profiles[["stock_id", "mean_spread"]],
            on="stock_id", how="left"
        )
        df["mean_spread_sample"] = df["mean_spread"]

    # Noise floor per sample
    df["zeta_sample"]     = (df["mean_spread_sample"] / 2) ** 2
    df["rv_floor_sample"] = np.sqrt(2 * df["zeta_sample"] * WINDOW_SECONDS)

    # Actual RV from target
    df["rv_target"] = np.exp(df["log_rv_target"].clip(-20, 5))

    # SNR and quality
    df["snr"]     = df["rv_target"] / df["rv_floor_sample"].clip(lower=1e-12)
    df["quality"] = df.apply(
        lambda r: classify_quality(r["rv_target"], r["rv_floor_sample"]), axis=1
    )

    return df[["time_id", "stock_id", "log_rv_target", "rv_target",
               "mean_spread_sample", "zeta_sample", "rv_floor_sample",
               "snr", "quality"]]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Noise Floor Estimation")
    print(f"  Window : {WINDOW_SECONDS}s ({N_BUCKETS} x {BUCKET_SIZE}s buckets)")
    print(f"  Method : spread-based (Roll 1984 + Kurisu 2018)")
    print("=" * 60)

    print("Loading features and targets ...")
    train_feat = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_targ = pd.read_parquet(FEATURES_DIR / "train_targets.parquet")

    # ── Step 1: Per-stock noise profiles ──────────────────────────
    print("\nStep 1: Computing per-stock noise profiles ...")
    stock_profiles = compute_stock_noise_profiles(train_feat)

    if stock_profiles.empty:
        print("Could not compute stock profiles. Exiting.")
        return

    print(f"\nPer-stock noise floor summary:")
    print(f"  mean spread         : {stock_profiles['mean_spread'].mean():.6f}")
    print(f"  mean zeta (spread)  : {stock_profiles['zeta_spread'].mean():.2e}")
    print(f"  mean rv_floor       : {stock_profiles['rv_floor_spread'].mean():.6f}")
    print(f"  mean log(rv_floor)  : {stock_profiles['log_rv_floor_spread'].mean():.3f}")
    print(f"\n  Suggested LOG_RV_FLOOR for model.py:")
    suggested_floor = stock_profiles["log_rv_floor_spread"].quantile(0.75)
    print(f"  LOG_RV_FLOOR = {suggested_floor:.2f}  (75th percentile of per-stock floors)")
    print(f"  (current hardcoded value: -18.0  --  likely too low)")

    # ── Step 2: Per-sample SNR ────────────────────────────────────
    print("\nStep 2: Computing per-sample SNR ...")
    snr_df = compute_snr_per_sample(train_feat, train_targ, stock_profiles)

    # Exclude dead stocks (log_rv_target == 0 or very low)
    live = snr_df["log_rv_target"] > -18.0
    snr_live = snr_df[live]

    print(f"\nSNR distribution (live stocks only, n={len(snr_live):,}):")
    print(f"  SNR < 1   (noise dominated) : {(snr_live['snr'] < 1).mean()*100:.1f}%")
    print(f"  SNR 1-2   (marginal)        : {((snr_live['snr'] >= 1) & (snr_live['snr'] < 2)).mean()*100:.1f}%")
    print(f"  SNR 2-10  (marginal-high)   : {((snr_live['snr'] >= 2) & (snr_live['snr'] < 10)).mean()*100:.1f}%")
    print(f"  SNR > 10  (high quality)    : {(snr_live['snr'] >= 10).mean()*100:.1f}%")

    print(f"\nQuality breakdown:")
    print(snr_live["quality"].value_counts().to_string())

    print(f"\nSNR percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d}: {snr_live['snr'].quantile(p/100):.2f}")

    # ── Step 3: What LOG_RV_FLOOR should you use? ─────────────────
    print("\nStep 3: Recommended LOG_RV_FLOOR thresholds:")
    for pct in [50, 75, 90]:
        floor = stock_profiles["rv_floor_spread"].quantile(pct / 100)
        log_floor = np.log(floor)
        n_excluded = (snr_live["rv_floor_sample"] > snr_live["rv_target"]).mean() * 100
        print(f"  p{pct} noise floor: rv={floor:.2e}  log={log_floor:.2f}")

    print(f"\n  Currently you exclude {(snr_live['log_rv_target'] < -18).mean()*100:.1f}% of live rows")
    noise_dominated_pct = (snr_live["snr"] < 2).mean() * 100
    print(f"  Noise-dominated rows (SNR<2): {noise_dominated_pct:.1f}%")
    print(f"  These are samples the model CANNOT learn from reliably.")

    # ── Step 4: Save outputs ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_profiles.to_csv(OUTPUT_DIR / "stock_noise_profiles.csv", index=False)
    snr_df.to_csv(OUTPUT_DIR / "sample_snr.csv", index=False)

    print(f"\nSaved:")
    print(f"  -> {OUTPUT_DIR}/stock_noise_profiles.csv  (per-stock noise floor)")
    print(f"  -> {OUTPUT_DIR}/sample_snr.csv            (per-sample SNR + quality flag)")

    # ── Step 5: Suggest model config update ──────────────────────
    print("\n" + "=" * 60)
    print("Recommended model.py config updates:")
    rec_floor = stock_profiles["log_rv_floor_spread"].quantile(0.50)
    print(f"  LOG_RV_FLOOR = {rec_floor:.1f}   # median per-stock noise floor")
    print(f"  (replaces hardcoded -18.0 which is far below any real noise floor)")
    print(f"")
    print(f"  In RMSPELoss and masked_mse_loss:")
    print(f"    live = (target != 0.0) & (target > LOG_RV_FLOOR)")
    print(f"")
    print(f"  This will exclude {noise_dominated_pct:.0f}% of noise-dominated samples")
    print(f"  from the loss, reducing the signal your model trains on but")
    print(f"  improving the quality of each gradient update.")
    print("=" * 60)


if __name__ == "__main__":
    main()