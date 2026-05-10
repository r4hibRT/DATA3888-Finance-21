"""
test_stock_interaction.py
=========================
Standalone statistical tests for whether stock interactions matter
for realized volatility forecasting — completely independent of any model.

Three tests, each answering a different question:

  Test 1 — Cross-stock RV correlation matrix
    Do stocks with similar volatility tend to co-move?
    Are there clusters of stocks that spike together?
    → Pearson correlation of log(RV) across time_ids

  Test 2 — Granger causality
    Does stock A's past RV help predict stock B's future RV,
    beyond B's own past RV alone?
    → OLS F-test: restricted model (own lags only) vs
                  unrestricted model (own + neighbour lags)
    A significant F-stat means stock interactions are statistically
    informative for forecasting.

  Test 3 — Volatility spillover index (Diebold-Yilmaz)
    What fraction of each stock's forecast error variance is
    explained by shocks from other stocks?
    → VAR forecast error variance decomposition
    A high spillover index means cross-stock signals matter a lot.

  Test 4 — Permutation baseline
    Shuffle stock labels within each time_id and re-run correlation.
    The shuffled correlation is the null distribution — any real
    correlation above this level is genuine cross-stock co-movement.

Reads: processed_v2/train_features.parquet + train_targets.parquet
Outputs:
  - Console results
  - models/stock_interaction_results.json
  - models/rv_correlation_matrix.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
from scipy import stats
from scipy.stats import f as f_dist

EPS          = 1e-8
LOG_RV_FLOOR = -18.0
FEATURES_DIR = Path("processed_v2")
OUTPUT_DIR   = Path("models")

# Granger test config
GRANGER_LAGS    = 1     # how many time_id lags to use
N_STOCK_PAIRS   = 200   # number of random pairs to test (full N² is too slow)
GRANGER_ALPHA   = 0.05  # significance level

# Spillover config
VAR_LAGS        = 1     # VAR lag order
N_STOCKS_VAR    = 20    # subset of stocks for VAR (full 112 is slow)
HORIZON         = 10    # forecast horizon for variance decomposition

# Permutation test
N_PERMUTATIONS  = 100


# ── Load and pivot data ───────────────────────────────────────────────

def load_rv_panel(split: str = "train") -> pd.DataFrame:
    """
    Load realized volatility targets and pivot to wide format.

    Returns DataFrame with:
      index  : time_id (sorted)
      columns: stock_id
      values : log_rv_target (NaN for missing stocks)
    """
    targ = pd.read_parquet(FEATURES_DIR / f"{split}_targets.parquet")

    # Filter live rows
    live = (targ["log_rv_target"] != 0.0) & (targ["log_rv_target"] > LOG_RV_FLOOR)
    targ = targ[live]

    panel = targ.pivot(index="time_id", columns="stock_id", values="log_rv_target")
    panel = panel.sort_index()

    print(f"Panel shape: {panel.shape[0]} time_ids × {panel.shape[1]} stocks")
    print(f"Sparsity: {panel.isna().mean().mean()*100:.1f}% missing")
    return panel


# ── Test 1: Correlation matrix ────────────────────────────────────────

def test_correlation(panel: pd.DataFrame) -> dict:
    """
    Compute pairwise Pearson correlation of log(RV) across time_ids.
    Stocks that co-move in volatility should show high positive correlation.
    """
    print("\n" + "="*55)
    print("Test 1: Cross-Stock RV Correlation")
    print("="*55)

    # Use stocks with sufficient observations
    obs_count  = panel.notna().sum()
    good_stocks = obs_count[obs_count >= panel.shape[0] * 0.5].index
    sub = panel[good_stocks].dropna(how="all")

    corr_matrix = sub.corr(method="pearson")

    # Extract upper triangle (excluding diagonal)
    n = len(corr_matrix)
    upper = corr_matrix.values[np.triu_indices(n, k=1)]
    upper = upper[~np.isnan(upper)]

    print(f"Stocks analysed: {n}")
    print(f"Stock pairs    : {len(upper):,}")
    print(f"\nPairwise log(RV) correlation:")
    print(f"  Mean        : {upper.mean():.4f}")
    print(f"  Median      : {np.median(upper):.4f}")
    print(f"  Std         : {upper.std():.4f}")
    print(f"  % positive  : {(upper > 0).mean()*100:.1f}%")
    print(f"  % > 0.3     : {(upper > 0.3).mean()*100:.1f}%  (moderate correlation)")
    print(f"  % > 0.5     : {(upper > 0.5).mean()*100:.1f}%  (strong correlation)")

    # Interpretation
    if upper.mean() > 0.3:
        print("\n  FINDING: Strong average cross-stock RV correlation.")
        print("  Stocks move together — cross-stock signals should help forecasting.")
    elif upper.mean() > 0.1:
        print("\n  FINDING: Moderate average cross-stock RV correlation.")
        print("  Some co-movement — cross-stock signals may help on active days.")
    else:
        print("\n  FINDING: Weak average cross-stock RV correlation.")
        print("  Stocks are largely independent — cross-stock signals unlikely to help.")

    # Save correlation matrix
    corr_matrix.to_csv(OUTPUT_DIR / "rv_correlation_matrix.csv")
    print(f"\n  Correlation matrix saved -> models/rv_correlation_matrix.csv")

    return {
        "mean_corr"   : float(upper.mean()),
        "median_corr" : float(np.median(upper)),
        "pct_positive": float((upper > 0).mean()),
        "pct_gt_03"   : float((upper > 0.3).mean()),
        "pct_gt_05"   : float((upper > 0.5).mean()),
        "n_stocks"    : n,
        "n_pairs"     : len(upper),
    }


# ── Test 2: Granger causality ─────────────────────────────────────────

def granger_test_pair(
    y: np.ndarray,     # target stock log(RV) series
    x: np.ndarray,     # candidate stock log(RV) series
    lags: int = 1,
) -> float:
    """
    Test whether x Granger-causes y.

    Restricted model:   y_t = a + b1*y_{t-1} + e
    Unrestricted model: y_t = a + b1*y_{t-1} + c1*x_{t-1} + e

    Returns p-value of F-test. Low p-value → x helps predict y.
    """
    # Align and remove NaNs
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < lags + 10:
        return np.nan

    y_arr = df["y"].values
    x_arr = df["x"].values

    n = len(y_arr) - lags

    # Build lagged arrays
    Y     = y_arr[lags:]                    # dependent variable
    Y_lag = y_arr[:-lags].reshape(-1, 1)   # lagged Y
    X_lag = x_arr[:-lags].reshape(-1, 1)   # lagged X
    ones  = np.ones((n, 1))

    # Restricted: Y ~ 1 + Y_lag
    Z_r   = np.hstack([ones, Y_lag])
    beta_r = np.linalg.lstsq(Z_r, Y, rcond=None)[0]
    resid_r = Y - Z_r @ beta_r
    rss_r   = np.sum(resid_r**2)

    # Unrestricted: Y ~ 1 + Y_lag + X_lag
    Z_u   = np.hstack([ones, Y_lag, X_lag])
    beta_u = np.linalg.lstsq(Z_u, Y, rcond=None)[0]
    resid_u = Y - Z_u @ beta_u
    rss_u   = np.sum(resid_u**2)

    # F-statistic
    q  = lags       # number of restrictions
    k  = Z_u.shape[1]
    df1 = q
    df2 = n - k

    if df2 <= 0 or rss_u <= 0:
        return np.nan

    f_stat = ((rss_r - rss_u) / q) / (rss_u / df2)
    p_val  = 1 - f_dist.cdf(f_stat, df1, df2)
    return float(p_val)


def test_granger(panel: pd.DataFrame) -> dict:
    """
    Test Granger causality on random sample of stock pairs.
    """
    print("\n" + "="*55)
    print("Test 2: Granger Causality")
    print(f"  Lags: {GRANGER_LAGS} | Pairs tested: {N_STOCK_PAIRS} | α={GRANGER_ALPHA}")
    print("="*55)

    stocks = panel.columns.tolist()
    if len(stocks) < 2:
        print("Not enough stocks for Granger test.")
        return {}

    rng = np.random.default_rng(42)
    pairs = [(int(rng.choice(len(stocks))), int(rng.choice(len(stocks))))
             for _ in range(N_STOCK_PAIRS * 3)]
    pairs = [(a, b) for a, b in pairs if a != b][:N_STOCK_PAIRS]

    p_values = []
    for i, (a_idx, b_idx) in enumerate(pairs):
        s_a = stocks[a_idx]
        s_b = stocks[b_idx]
        y = panel[s_b].values   # target: stock B
        x = panel[s_a].values   # predictor: stock A
        p = granger_test_pair(y, x, lags=GRANGER_LAGS)
        if not np.isnan(p):
            p_values.append(p)

    p_values = np.array(p_values)
    pct_significant = float((p_values < GRANGER_ALPHA).mean())

    print(f"Valid pairs tested : {len(p_values)}")
    print(f"Significant pairs  : {(p_values < GRANGER_ALPHA).sum()} "
          f"({pct_significant*100:.1f}%)")
    print(f"Expected by chance : {GRANGER_ALPHA*100:.1f}%")
    print(f"Mean p-value       : {p_values.mean():.4f}")
    print(f"Median p-value     : {np.median(p_values):.4f}")

    # Binomial test: is significant% > alpha% by more than chance?
    binom_result = stats.binomtest(
        int((p_values < GRANGER_ALPHA).sum()),
        len(p_values),
        GRANGER_ALPHA,
        alternative="greater"
    )

    print(f"\nBinomial test p-value: {binom_result.pvalue:.4f}")
    if binom_result.pvalue < 0.05:
        print("  FINDING: Significantly more Granger-causal pairs than chance.")
        print("  Stock interactions are statistically informative for forecasting.")
    else:
        print("  FINDING: No significant excess of Granger-causal pairs.")
        print("  Cross-stock signals may not add reliable forecasting value.")

    return {
        "n_pairs_tested"   : len(p_values),
        "pct_significant"  : pct_significant,
        "expected_by_chance": GRANGER_ALPHA,
        "mean_pvalue"      : float(p_values.mean()),
        "median_pvalue"    : float(np.median(p_values)),
        "binom_pvalue"     : float(binom_result.pvalue),
        "interaction_significant": bool(binom_result.pvalue < 0.05),
    }


# ── Test 3: Permutation baseline ─────────────────────────────────────

def test_permutation(panel: pd.DataFrame) -> dict:
    """
    Shuffle stock labels within each time_id and measure how much
    the correlation structure degrades.

    Real correlation = correlation in observed data
    Null correlation = correlation after shuffling stock identities
                       within each time_id (destroys spatial structure,
                       preserves marginal distribution)

    If real >> null → spatial structure is genuinely informative.
    """
    print("\n" + "="*55)
    print("Test 3: Permutation Baseline")
    print(f"  Shuffles: {N_PERMUTATIONS}")
    print("="*55)

    obs_count   = panel.notna().sum()
    good_stocks = obs_count[obs_count >= panel.shape[0] * 0.5].index
    sub         = panel[good_stocks].dropna(how="all")

    # Real correlation
    corr_real = sub.corr().values
    n         = len(corr_real)
    upper_idx = np.triu_indices(n, k=1)
    real_upper = corr_real[upper_idx]
    real_upper = real_upper[~np.isnan(real_upper)]
    real_mean  = float(real_upper.mean())

    # Permuted correlations
    rng          = np.random.default_rng(42)
    perm_means   = []

    for _ in range(N_PERMUTATIONS):
        shuffled = sub.copy()
        # Shuffle stock columns within each time_id
        for tid in shuffled.index:
            row   = shuffled.loc[tid].values.copy()
            valid = ~np.isnan(row)
            row[valid] = rng.permutation(row[valid])
            shuffled.loc[tid] = row

        corr_perm  = shuffled.corr().values
        perm_upper = corr_perm[upper_idx]
        perm_upper = perm_upper[~np.isnan(perm_upper)]
        perm_means.append(float(perm_upper.mean()))

    perm_means  = np.array(perm_means)
    null_mean   = float(perm_means.mean())
    null_std    = float(perm_means.std())
    z_score     = (real_mean - null_mean) / (null_std + 1e-10)
    p_value     = float(stats.norm.sf(z_score))  # one-sided

    print(f"Real mean correlation     : {real_mean:.4f}")
    print(f"Null mean correlation     : {null_mean:.4f}  (±{null_std:.4f})")
    print(f"Excess correlation        : {real_mean - null_mean:.4f}")
    print(f"Z-score                   : {z_score:.2f}")
    print(f"P-value (one-sided)       : {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n  FINDING: Real correlation ({real_mean:.3f}) significantly exceeds")
        print(f"  null ({null_mean:.3f}). Stock spatial structure is genuine.")
        print(f"  Cross-stock signals carry real information beyond coincidence.")
    else:
        print(f"\n  FINDING: Real correlation not significantly above null.")
        print(f"  Observed co-movement may be a statistical artifact.")

    return {
        "real_mean_corr"  : real_mean,
        "null_mean_corr"  : null_mean,
        "null_std_corr"   : null_std,
        "excess_corr"     : real_mean - null_mean,
        "z_score"         : float(z_score),
        "p_value"         : p_value,
        "n_permutations"  : N_PERMUTATIONS,
        "spatial_significant": bool(p_value < 0.05),
    }


# ── Test 4: Same-time vs lagged correlation ───────────────────────────

def test_lead_lag(panel: pd.DataFrame) -> dict:
    """
    Compare same-time correlation vs lagged correlation.

    If same-time correlation >> lagged → stocks react simultaneously
    (market-wide factor, useful for nowcasting but not forecasting).

    If lagged correlation is significant → stock A today predicts
    stock B tomorrow (genuine predictive cross-stock signal).
    """
    print("\n" + "="*55)
    print("Test 4: Same-Time vs Lagged Correlation")
    print("="*55)

    obs_count   = panel.notna().sum()
    good_stocks = obs_count[obs_count >= panel.shape[0] * 0.7].index
    sub         = panel[good_stocks].dropna()

    if len(sub) < 10:
        print("Not enough complete rows for lead-lag test.")
        return {}

    n           = len(sub.columns)
    upper_idx   = np.triu_indices(n, k=1)

    # Same-time correlation
    corr_same   = sub.corr().values[upper_idx]
    corr_same   = corr_same[~np.isnan(corr_same)]

    # Lagged correlation: stock A at t predicts stock B at t+1
    sub_t       = sub.iloc[:-1].values
    sub_t1      = sub.iloc[1:].values
    lag_corrs   = []
    for i in range(n):
        for j in range(i+1, n):
            mask = ~(np.isnan(sub_t[:, i]) | np.isnan(sub_t1[:, j]))
            if mask.sum() < 10:
                continue
            r, _ = stats.pearsonr(sub_t[mask, i], sub_t1[mask, j])
            lag_corrs.append(r)
    lag_corrs = np.array(lag_corrs)

    print(f"Same-time correlation  : mean={corr_same.mean():.4f}  "
          f"std={corr_same.std():.4f}")
    print(f"Lagged correlation     : mean={lag_corrs.mean():.4f}  "
          f"std={lag_corrs.std():.4f}")
    print(f"Lag decay ratio        : {lag_corrs.mean()/max(corr_same.mean(),EPS):.3f}")
    print(f"  (1.0 = no decay, 0.0 = all contemporaneous)")

    pct_lag_sig = float((np.abs(lag_corrs) > 0.1).mean())
    print(f"Pairs with |lag corr| > 0.1: {pct_lag_sig*100:.1f}%")

    if lag_corrs.mean() > 0.05:
        print(f"\n  FINDING: Meaningful lagged correlation ({lag_corrs.mean():.3f}).")
        print(f"  Stock A's past RV predicts stock B's future RV.")
        print(f"  Cross-stock signals have genuine PREDICTIVE value.")
    else:
        print(f"\n  FINDING: Weak lagged correlation ({lag_corrs.mean():.3f}).")
        print(f"  Cross-stock co-movement is mostly contemporaneous.")
        print(f"  Useful for nowcasting but limited for forecasting.")

    return {
        "same_time_mean_corr": float(corr_same.mean()),
        "lagged_mean_corr"   : float(lag_corrs.mean()),
        "lag_decay_ratio"    : float(lag_corrs.mean() / max(corr_same.mean(), EPS)),
        "pct_lag_significant": pct_lag_sig,
        "predictive_value"   : bool(lag_corrs.mean() > 0.05),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Stock Interaction Importance Tests")
    print("  Independent of any model — pure statistics")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading RV panel ...")
    panel = load_rv_panel("train")

    results = {}

    results["correlation"] = test_correlation(panel)
    results["granger"]     = test_granger(panel)
    results["permutation"] = test_permutation(panel)
    results["lead_lag"]    = test_lead_lag(panel)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    c = results["correlation"]
    g = results["granger"]
    p = results["permutation"]
    l = results["lead_lag"]

    print(f"\n{'Test':<30} {'Finding':<15} {'Key Stat'}")
    print("-" * 55)
    print(f"{'1. Cross-stock correlation':<30} "
          f"{'STRONG' if c['mean_corr'] > 0.3 else 'WEAK':<15} "
          f"mean_corr={c['mean_corr']:.3f}")
    print(f"{'2. Granger causality':<30} "
          f"{'SIGNIFICANT' if g.get('interaction_significant') else 'NOT SIG':<15} "
          f"pct_sig={g.get('pct_significant', 0)*100:.1f}%")
    print(f"{'3. Permutation test':<30} "
          f"{'REAL' if p.get('spatial_significant') else 'RANDOM':<15} "
          f"z={p.get('z_score', 0):.2f}")
    print(f"{'4. Lagged correlation':<30} "
          f"{'PREDICTIVE' if l.get('predictive_value') else 'CONTEMPOR.':<15} "
          f"lag_corr={l.get('lagged_mean_corr', 0):.3f}")

    # Overall verdict
    n_positive = sum([
        c["mean_corr"] > 0.3,
        g.get("interaction_significant", False),
        p.get("spatial_significant", False),
        l.get("predictive_value", False),
    ])

    print(f"\nOverall: {n_positive}/4 tests support cross-stock interactions")
    if n_positive >= 3:
        print("VERDICT: Strong evidence — cross-stock signals are worth modelling.")
        print("  Your GNN graph and attention layers should help forecasting.")
    elif n_positive >= 2:
        print("VERDICT: Moderate evidence — cross-stock signals exist but are weak.")
        print("  GNN may help on high-volatility days but not overall.")
    else:
        print("VERDICT: Weak evidence — stocks behave mostly independently.")
        print("  Per-stock models (LGBM) likely sufficient for this dataset.")

    # Save results
    out_path = OUTPUT_DIR / "stock_interaction_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()