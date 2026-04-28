"""
cluster_explorer.py — Multi-Method Stock Clustering on Selected Features
=========================================================================

Usage (in notebook)
-------------------
  results = run_cluster_exploration(
      train_path    = "output/train.parquet",
      features_path = "output/selected_features.txt",
      output_dir    = "output/clustering/",
  )

Usage (CLI)
-----------
  python cluster_explorer.py \
      --train_path   output/train.parquet \
      --features_path output/selected_features.txt \
      --output_dir   output/clustering/

What it does
------------
  1. Loads training data + selected features
  2. Builds per-stock profiles (mean/std of each feature across time_ids)
  3. Tries 7 clustering methods × multiple k values
  4. Ranks all configurations by Silhouette Score
  5. Saves results, best labels, and a summary report
"""

import os
import json
import warnings
import time as _time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
    Birch,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

try:
    from kneed import KneeLocator
    _KNEED = True
except ImportError:
    _KNEED = False

try:
    import hdbscan as _hdbscan
    _HDBSCAN = True
except ImportError:
    _HDBSCAN = False


# ─────────────────────────────────────────────────────────────────────
# STOCK PROFILING
# ─────────────────────────────────────────────────────────────────────

def build_stock_profiles(df: pd.DataFrame,
                          feature_cols: list,
                          agg: str = "both") -> pd.DataFrame:
    """
    Aggregate per-stock profiles from the training data.

    Parameters
    ----------
    agg : "mean", "std", or "both"
        "both" creates mean_* and std_* for each feature → richer profiles.

    Returns
    -------
    profiles : DataFrame indexed by stock_id, columns = aggregated features
    """
    print("Building per-stock profiles …")
    
    
    valid_cols = [c for c in feature_cols if c in df.columns]
    exclude_for_clustering = [
    "volume_x_rv", "spread_x_rv", "spread_x_volume",
    "signed_volume", "rv_x_kurt", "spread_x_jump",
    "wap_first",       # absolute price level, not comparable across stocks
    ]

    cluster_features = [f for f in valid_cols if f not in exclude_for_clustering]
    # cluster_features = [f for f in valid_cols]

    print(f"  Using {len(cluster_features)} features for clustering ")

    grouped = df.groupby("stock_id")[cluster_features]

    if agg == "mean":
        profiles = grouped.mean()
    elif agg == "std":
        profiles = grouped.std().fillna(0)
    else:
        means = grouped.mean()
        stds  = grouped.std().fillna(0)
        means.columns = [f"mean_{c}" for c in means.columns]
        stds.columns  = [f"std_{c}" for c in stds.columns]
        profiles = pd.concat([means, stds], axis=1)

    profiles = profiles.replace([np.inf, -np.inf], np.nan)
    profiles = profiles.fillna(profiles.median())


    print(f"  Profiles: {profiles.shape[0]} stocks × {profiles.shape[1]} features")
    return profiles


# ─────────────────────────────────────────────────────────────────────
# CLUSTERING METHODS
# ─────────────────────────────────────────────────────────────────────

def _try_kmeans(X, k, seed):
    model = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=seed)
    return model.fit_predict(X)


def _try_minibatch_kmeans(X, k, seed):
    model = MiniBatchKMeans(n_clusters=k, n_init=10, max_iter=500,
                            batch_size=min(1024, len(X)), random_state=seed)
    return model.fit_predict(X)


def _try_agglomerative(X, k, linkage_type="ward"):
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_type)
    return model.fit_predict(X)


def _try_spectral(X, k, seed):
    if len(X) > 5000:
        return None  # too slow for large datasets
    model = SpectralClustering(n_clusters=k, affinity="rbf",
                                random_state=seed, n_init=10)
    return model.fit_predict(X)


def _try_gmm(X, k, seed):
    model = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=5, max_iter=300, random_state=seed)
    labels = model.fit_predict(X)
    return labels


def _try_birch(X, k):
    model = Birch(n_clusters=k, threshold=0.5)
    return model.fit_predict(X)


def _try_dbscan(X, eps, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return None
    return labels


def _try_hdbscan(X, min_cluster_size):
    if not _HDBSCAN:
        return None
    model = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              min_samples=5)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return None
    return labels


# ─────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────

def score_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics. Returns None if invalid."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        return None

    # For metrics, exclude noise points (label == -1) if any
    mask = labels >= 0
    if mask.sum() < 10:
        return None

    X_clean = X[mask]
    labels_clean = labels[mask]

    try:
        sil = silhouette_score(X_clean, labels_clean)
    except Exception:
        sil = -1.0

    try:
        ch = calinski_harabasz_score(X_clean, labels_clean)
    except Exception:
        ch = 0.0

    try:
        db = davies_bouldin_score(X_clean, labels_clean)
    except Exception:
        db = 999.0

    # Cluster size stats
    unique_labels, counts = np.unique(labels_clean, return_counts=True)
    sizes = dict(zip(unique_labels.tolist(), counts.tolist()))
    min_size = int(counts.min())
    max_size = int(counts.max())

    noise_frac = float((labels == -1).sum() / len(labels)) if -1 in labels else 0.0

    return {
        "silhouette":        round(float(sil), 4),
        "calinski_harabasz": round(float(ch), 2),
        "davies_bouldin":    round(float(db), 4),
        "n_clusters":        int(len(unique)),
        "min_cluster_size":  min_size,
        "max_cluster_size":  max_size,
        "noise_fraction":    round(noise_frac, 4),
        "cluster_sizes":     sizes,
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN EXPLORATION
# ─────────────────────────────────────────────────────────────────────

def run_cluster_exploration(
    train_path:    str,
    features_path: str,
    output_dir:    str   = "clustering_output",
    k_range:       list  = None,
    seed:          int   = 42,
    profile_agg:   str   = "both",
    scaler_type:   str   = "robust",
    pca_variance:  float = 0.95,
    run_pca:       bool  = True,
) -> pd.DataFrame:
    """
    Try multiple clustering methods × k values, rank by silhouette.

    Parameters
    ----------
    train_path    : path to train.parquet
    features_path : path to selected_features.txt
    output_dir    : where to save results
    k_range       : list of k values to try (default: 2–25)
    profile_agg   : "mean", "std", or "both" for stock profiles
    scaler_type   : "standard" or "robust"
    pca_variance  : cumulative variance for PCA (set run_pca=False to skip)
    """

    os.makedirs(output_dir, exist_ok=True)
    if k_range is None:
        k_range = list(range(2, 26))

    # ── Load data ────────────────────────────────────────────────────
    print("═" * 66)
    print("Multi-Method Stock Clustering Explorer")
    print("═" * 66)

    df = pd.read_parquet(train_path)
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    print(f"  Data: {df.shape} | Features: {len(feature_cols)}")

    # ── Build profiles ───────────────────────────────────────────────
    profiles = build_stock_profiles(df, feature_cols, agg=profile_agg)
    stock_ids = profiles.index.values

    # ── Scale ────────────────────────────────────────────────────────
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles.values)
    print(f"  Scaled: {scaler_type} → {X_scaled.shape}")

    # ── Optional PCA ─────────────────────────────────────────────────
    X_cluster = X_scaled
    pca_model = None
    if run_pca:
        pca_model = PCA(n_components=pca_variance, random_state=seed)
        X_cluster = pca_model.fit_transform(X_scaled)
        n_comp = X_cluster.shape[1]
        var_explained = pca_model.explained_variance_ratio_.sum()
        print(f"  PCA: {X_scaled.shape[1]} → {n_comp} components "
              f"({var_explained:.1%} variance)")

    # ── Run all methods ──────────────────────────────────────────────
    all_results = []

    print(f"\nRunning clustering methods (k ∈ {k_range[0]}–{k_range[-1]}) …\n")

    # 1. KMeans
    print("  [1/8] KMeans …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_kmeans(X_cluster, k, seed)
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "KMeans", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)

    # 2. MiniBatch KMeans
    print("  [2/8] MiniBatch KMeans …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_minibatch_kmeans(X_cluster, k, seed)
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "MiniBatchKMeans", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)

    # 3. Agglomerative (Ward)
    print("  [3/8] Agglomerative (Ward) …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_agglomerative(X_cluster, k, "ward")
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "Agglom_Ward", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)

    # 4. Agglomerative (Average)
    print("  [4/8] Agglomerative (Average) …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_agglomerative(X_cluster, k, "average")
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "Agglom_Average", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)

    # 5. Spectral
    print("  [5/8] Spectral …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_spectral(X_cluster, k, seed)
        if labels is not None:
            scores = score_clustering(X_cluster, labels)
            if scores:
                scores.update({"method": "Spectral", "k": k,
                               "time_s": round(_time.time() - t0, 2)})
                scores["labels"] = labels
                all_results.append(scores)

    # 6. GMM
    print("  [6/8] Gaussian Mixture …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_gmm(X_cluster, k, seed)
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "GMM", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)

    # 7. Birch
    print("  [7/8] Birch …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_birch(X_cluster, k)
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "Birch", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)

    # 8. DBSCAN / HDBSCAN (no k — auto-determined)
    print("  [8/8] DBSCAN + HDBSCAN …")
    # DBSCAN with multiple eps values
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_cluster)
    dists, _ = nn.kneighbors(X_cluster)
    mean_5nn = np.sort(dists[:, -1])
    eps_candidates = [
        np.percentile(mean_5nn, p)
        for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]
    ]
    for eps in eps_candidates:
        t0 = _time.time()
        labels = _try_dbscan(X_cluster, eps)
        if labels is not None:
            scores = score_clustering(X_cluster, labels)
            if scores:
                scores.update({"method": f"DBSCAN", "k": scores["n_clusters"],
                               "eps": round(eps, 4),
                               "time_s": round(_time.time() - t0, 2)})
                scores["labels"] = labels
                all_results.append(scores)

    # HDBSCAN
    if _HDBSCAN:
        for min_cs in [5, 10, 15, 20, 30, 50]:
            t0 = _time.time()
            labels = _try_hdbscan(X_cluster, min_cs)
            if labels is not None:
                scores = score_clustering(X_cluster, labels)
                if scores:
                    scores.update({
                        "method": "HDBSCAN", "k": scores["n_clusters"],
                        "min_cluster_size_param": min_cs,
                        "time_s": round(_time.time() - t0, 2),
                    })
                    scores["labels"] = labels
                    all_results.append(scores)

    # ── Rank results ─────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"Results: {len(all_results)} valid configurations")
    print(f"{'═' * 70}")

    # Build summary DataFrame (without labels column)
    summary_cols = ["method", "k", "silhouette", "calinski_harabasz",
                    "davies_bouldin", "n_clusters", "min_cluster_size",
                    "max_cluster_size", "noise_fraction", "time_s"]
    summary_rows = []
    for r in all_results:
        row = {c: r.get(c, None) for c in summary_cols}
        if "eps" in r:
            row["eps"] = r["eps"]
        if "min_cluster_size_param" in r:
            row["min_cluster_size_param"] = r["min_cluster_size_param"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("silhouette", ascending=False).reset_index(drop=True)
    summary_df.index.name = "rank"

    # Print top 20
    print(f"\nTop 20 by Silhouette Score:\n")
    display_cols = ["method", "k", "silhouette", "calinski_harabasz",
                    "davies_bouldin", "min_cluster_size", "max_cluster_size"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].head(20).to_string())

    # ── Best per method ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Best configuration per method:\n")
    best_per_method = summary_df.groupby("method").first().sort_values(
        "silhouette", ascending=False
    )
    print(best_per_method[["k", "silhouette", "calinski_harabasz",
                           "davies_bouldin"]].to_string())

    # ── Save best labels ─────────────────────────────────────────────
    # Find the overall best result
    best_idx = summary_df.iloc[0]
    best_method = best_idx["method"]
    best_k = int(best_idx["k"])
    best_sil = best_idx["silhouette"]

    # Find matching labels
    best_labels = None
    for r in all_results:
        if (r.get("method") == best_method and
            r.get("k") == best_k and
            abs(r.get("silhouette", -99) - best_sil) < 1e-6):
            best_labels = r["labels"]
            break

    if best_labels is not None:
        labels_df = pd.DataFrame({
            "stock_id":   stock_ids,
            "cluster_id": best_labels,
        })
        labels_path = os.path.join(output_dir, "best_cluster_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        print(f"\n  Best: {best_method} k={best_k} "
              f"(silhouette={best_sil:.4f})")
        print(f"  Saved labels → {labels_path}")

        # Cluster profile summary
        profiles_with_labels = profiles.copy()
        profiles_with_labels["cluster_id"] = best_labels
        cluster_summary = profiles_with_labels.groupby("cluster_id").agg(["mean", "count"])
        # Just show count per cluster
        cluster_counts = profiles_with_labels.groupby("cluster_id").size()
        print(f"\n  Cluster sizes:")
        for cid, count in cluster_counts.items():
            print(f"    Cluster {cid}: {count} stocks")

    # ── Save top 5 configs' labels ───────────────────────────────────
    for rank in range(min(5, len(summary_df))):
        row = summary_df.iloc[rank]
        method = row["method"]
        k = int(row["k"])
        sil = row["silhouette"]

        for r in all_results:
            if (r.get("method") == method and r.get("k") == k and
                abs(r.get("silhouette", -99) - sil) < 1e-6):
                ldf = pd.DataFrame({
                    "stock_id":   stock_ids,
                    "cluster_id": r["labels"],
                })
                fname = f"labels_rank{rank+1}_{method}_k{k}.csv"
                ldf.to_csv(os.path.join(output_dir, fname), index=False)
                break

    # ── Save summary ─────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "clustering_summary.csv")
    summary_df.to_csv(summary_path, index=True)
    print(f"\n  Saved summary → {summary_path}")

    # Save as JSON too (without labels)
    report = {
        "best_method":     best_method,
        "best_k":          best_k,
        "best_silhouette": float(best_sil),
        "n_configurations_tried": len(all_results),
        "k_range":         k_range,
        "profile_agg":     profile_agg,
        "scaler":          scaler_type,
        "pca":             run_pca,
        "pca_components":  int(X_cluster.shape[1]) if run_pca else None,
        "n_stocks":        len(stock_ids),
        "n_features":      len(feature_cols),
        "top_10": summary_df.head(10).drop(
            columns=["eps", "min_cluster_size_param"], errors="ignore"
        ).to_dict(orient="records"),
    }
    report_path = os.path.join(output_dir, "clustering_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved report → {report_path}")

    print(f"\n{'═' * 70}")
    print(f"Done. Best: {best_method} k={best_k} | "
          f"Silhouette={best_sil:.4f}")
    print(f"{'═' * 70}")

    return summary_df


def run_kmeans_exploration(train_path:    str,
    features_path: str,
    output_dir:    str   = "clustering_output",
    k_range:       list  = None,
    seed:          int   = 42,
    profile_agg:   str   = "both",
    scaler_type:   str   = "robust",
    pca_variance:  float = 0.95,
    run_pca:       bool  = True,) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    if k_range is None:
        k_range = list(range(2, 26))

    # ── Load data ────────────────────────────────────────────────────
    print("═" * 66)
    print("Multi-Method Stock Clustering Explorer")
    print("═" * 66)

    df = pd.read_parquet(train_path)
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    print(f"  Data: {df.shape} | Features: {len(feature_cols)}")

    # ── Build profiles ───────────────────────────────────────────────
    profiles = build_stock_profiles(df, feature_cols, agg=profile_agg)
    stock_ids = profiles.index.values

    # ── Scale ────────────────────────────────────────────────────────
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles.values)
    print(f"  Scaled: {scaler_type} → {X_scaled.shape}")

    # ── Optional PCA ─────────────────────────────────────────────────
    X_cluster = X_scaled
    pca_model = None
    if run_pca:
        pca_model = PCA(n_components=pca_variance, random_state=seed)
        X_cluster = pca_model.fit_transform(X_scaled)
        n_comp = X_cluster.shape[1]
        var_explained = pca_model.explained_variance_ratio_.sum()
        print(f"  PCA: {X_scaled.shape[1]} → {n_comp} components "
              f"({var_explained:.1%} variance)")

    # ── Run all methods ──────────────────────────────────────────────
    all_results = []
    print("  [1/8] KMeans …")
    for k in k_range:
        t0 = _time.time()
        labels = _try_kmeans(X_cluster, k, seed)
        scores = score_clustering(X_cluster, labels)
        if scores:
            scores.update({"method": "KMeans", "k": k,
                           "time_s": round(_time.time() - t0, 2)})
            scores["labels"] = labels
            all_results.append(scores)
    # ── Rank results ─────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"Results: {len(all_results)} valid configurations")
    print(f"{'═' * 70}")

    # Build summary DataFrame (without labels column)
    summary_cols = ["method", "k", "silhouette", "calinski_harabasz",
                    "davies_bouldin", "n_clusters", "min_cluster_size",
                    "max_cluster_size", "noise_fraction", "time_s"]
    summary_rows = []
    for r in all_results:
        row = {c: r.get(c, None) for c in summary_cols}
        if "eps" in r:
            row["eps"] = r["eps"]
        if "min_cluster_size_param" in r:
            row["min_cluster_size_param"] = r["min_cluster_size_param"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("silhouette", ascending=False).reset_index(drop=True)
    summary_df.index.name = "rank"

    # Print top 20
    print(f"\nTop 20 by Silhouette Score:\n")
    display_cols = ["method", "k", "silhouette", "calinski_harabasz",
                    "davies_bouldin", "min_cluster_size", "max_cluster_size"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].head(20).to_string())

    # ── Best per method ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Best configuration per method:\n")
    best_per_method = summary_df.groupby("method").first().sort_values(
        "silhouette", ascending=False
    )
    print(best_per_method[["k", "silhouette", "calinski_harabasz",
                           "davies_bouldin"]].to_string())

    # ── Save best labels ─────────────────────────────────────────────
    # Find the overall best result
    best_idx = summary_df.iloc[1]
    best_method = best_idx["method"]
    best_k = int(best_idx["k"])
    best_sil = best_idx["silhouette"]

    # Find matching labels
    best_labels = None
    for r in all_results:
        if (r.get("method") == best_method and
            r.get("k") == best_k and
            abs(r.get("silhouette", -99) - best_sil) < 1e-6):
            best_labels = r["labels"]
            break

    if best_labels is not None:
        labels_df = pd.DataFrame({
            "stock_id":   stock_ids,
            "cluster_id": best_labels,
        })
        labels_path = os.path.join(output_dir, "best_cluster_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        print(f"\n  Best: {best_method} k={best_k} "
              f"(silhouette={best_sil:.4f})")
        print(f"  Saved labels → {labels_path}")

        # Cluster profile summary
        profiles_with_labels = profiles.copy()
        profiles_with_labels["cluster_id"] = best_labels
        cluster_summary = profiles_with_labels.groupby("cluster_id").agg(["mean", "count"])
        # Just show count per cluster
        cluster_counts = profiles_with_labels.groupby("cluster_id").size()
        print(f"\n  Cluster sizes:")
        for cid, count in cluster_counts.items():
            print(f"    Cluster {cid}: {count} stocks")

    # ── Save top 5 configs' labels ───────────────────────────────────
    for rank in range(min(5, len(summary_df))):
        row = summary_df.iloc[rank]
        method = row["method"]
        k = int(row["k"])
        sil = row["silhouette"]

        for r in all_results:
            if (r.get("method") == method and r.get("k") == k and
                abs(r.get("silhouette", -99) - sil) < 1e-6):
                ldf = pd.DataFrame({
                    "stock_id":   stock_ids,
                    "cluster_id": r["labels"],
                })
                fname = f"labels_rank{rank+1}_{method}_k{k}.csv"
                ldf.to_csv(os.path.join(output_dir, fname), index=False)
                break

    # ── Save summary ─────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "clustering_summary.csv")
    summary_df.to_csv(summary_path, index=True)
    print(f"\n  Saved summary → {summary_path}")

    # Save as JSON too (without labels)
    report = {
        "best_method":     best_method,
        "best_k":          best_k,
        "best_silhouette": float(best_sil),
        "n_configurations_tried": len(all_results),
        "k_range":         k_range,
        "profile_agg":     profile_agg,
        "scaler":          scaler_type,
        "pca":             run_pca,
        "pca_components":  int(X_cluster.shape[1]) if run_pca else None,
        "n_stocks":        len(stock_ids),
        "n_features":      len(feature_cols),
        "top_10": summary_df.head(10).drop(
            columns=["eps", "min_cluster_size_param"], errors="ignore"
        ).to_dict(orient="records"),
    }
    report_path = os.path.join(output_dir, "clustering_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved report → {report_path}")

    print(f"\n{'═' * 70}")
    print(f"Done. Best: {best_method} k={best_k} | "
          f"Silhouette={best_sil:.4f}")
    print(f"{'═' * 70}")

    return summary_df
        
# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Set your paths here ──────────────────────────────────────────
    TRAIN_PATH    = "processed/train.parquet"
    FEATURES_PATH = "processed/clustering_feature.txt"
    OUTPUT_DIR    = "processed/clustering/"

    # summary = run_cluster_exploration(
    #     train_path    = TRAIN_PATH,
    #     features_path = FEATURES_PATH,
    #     output_dir    = OUTPUT_DIR,
    #     k_range       = list(range(2, 26)),
    #     seed          = 42,
    #     profile_agg   = "both",       # "mean", "std", or "both"
    #     scaler_type   = "robust",     # "robust" or "standard"
    #     pca_variance  = 0.95,         # cumulative variance threshold
    #     run_pca       = False,        # set to True to enable PCA dimensionality reduction
    # )

    summary=run_kmeans_exploration(
        train_path    = TRAIN_PATH,
        features_path = FEATURES_PATH,
        output_dir    = OUTPUT_DIR,
        k_range       = list(range(2, 26)),
        seed          = 42,
        profile_agg   = "both",       # "mean", "std", or "both"
        scaler_type   = "robust",     # "robust" or "standard"
        pca_variance  = 0.95,         # cumulative variance threshold
        run_pca       = False,        # set to True to enable PCA dimensionality reduction
    )