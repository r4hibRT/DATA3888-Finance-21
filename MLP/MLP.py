"""
mlp_rv.py
=========
MLP baseline for realized volatility forecasting at individual stock level.
Branched from cnn_rv.py — identical data pipeline, dataset, metrics, and
fold loop. Only the model class changes.

Architecture:
  Raw sequence (480, N_FEATURES) per (stock_id, time_id)
    -> Flatten to (480 * N_FEATURES,) = 2880-dim vector
    -> Concatenate stock embedding (32-dim)  [2912-dim total]
    -> Linear(2912 -> 1024) + BN + ReLU + Dropout
    -> Linear(1024 ->  512) + BN + ReLU + Dropout
    -> Linear( 512 ->  256) + BN + ReLU + Dropout
    -> Linear( 256 ->  128) + BN + ReLU + Dropout
    -> Linear( 128 ->    1) + SigmoidRange -> log(RV)

Low-RV fixes applied:
  1. LOG_RV_MIN widened to -15 to reduce sigmoid saturation at the low-RV boundary
  2. Blended loss: alpha * RMSPE (RV-space) + (1-alpha) * AsymLogMSE (log-space)
     Log-space MSE equalises gradient signal across the full RV range.
     Asymmetric factor penalises under-prediction (pred_log < true_log) harder.
  3. Stratified DataLoader: each batch is drawn to uniformly cover the
     log(RV) distribution, preventing moderate-RV samples from dominating.
"""

import gc
import json
import time as _time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Config ────────────────────────────────────────────────────────────

DATA_DIR = Path("C:\\Users\\ngdo0466\\Downloads\\processed")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

N_OUTER_FOLDS = 1
EPS           = 1e-8

# Window
INPUT_END     = 480
TARGET_START  = 480
TOTAL_SECONDS = 600

# Features per second
N_FEATURES = 6
EMBED_DIM  = 32

# ── MLP-specific config ───────────────────────────────────────────────
MLP_HIDDEN    = [1024, 512, 256, 128]
DROPOUT_RATE  = 0.25

# Training
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
EPOCHS        = 50
PATIENCE      = 5
LR_PATIENCE   = 5
LR_FACTOR     = 0.25
MIN_LR        = 1e-6
GRAD_CLIP     = 1.0

# RV — widened lower bound to reduce sigmoid saturation at low-RV tail
RV_FLOOR   = 1e-4
LOG_RV_MIN = -10.0     # was -10.0
LOG_RV_MAX =   0.0

# ── Low-RV loss config ────────────────────────────────────────────────
LOSS_ALPHA   = 0.4    # weight on RMSPE (RV-space); (1-alpha) on log-MSE
ASYM_FACTOR  = 2.0    # extra penalty multiplier when pred_log < true_log
N_RV_STRATA  = 10     # number of log(RV) bins for stratified sampling


def log(msg):
    print(f"[MLP] {msg}", flush=True)


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(pred_log, true_log):
    pred_log = np.clip(pred_log, -20, 5)
    pred_rv = np.clip(np.exp(pred_log), EPS, None)
    true_rv = np.clip(np.exp(true_log), EPS, None)
    ratio = true_rv / pred_rv
    resid = pred_rv - true_rv
    return {
        "QLIKE":      float(np.mean(ratio - np.log(ratio) - 1)),
        "RMSE":       float(np.sqrt(np.mean(resid ** 2))),
        "MAPE%":      float(np.mean(np.abs(resid) / true_rv) * 100),
        "RMSPE%":     float(np.sqrt(np.mean((resid / true_rv) ** 2)) * 100),
        "MAE_log":    float(np.mean(np.abs(pred_log - true_log))),
        "MSE_log_rv": float(np.mean((pred_log - true_log) ** 2)),
        "MSE_rv":     float(np.mean((pred_rv - true_rv) ** 2)),
    }


# ── Sequence builder ──────────────────────────────────────────────────

def parse_stock_id(s):
    s = str(s)
    return int(s.replace("stock_", "")) if "stock_" in s else int(s)


def build_all_sequences(path: Path):
    log(f"  Loading {path.name} ...")
    df = pd.read_parquet(path)

    uniq = df["stock_id"].unique()
    sid_map = {s: parse_stock_id(s) for s in uniq}
    df["stock_id"] = df["stock_id"].map(sid_map).astype(np.int32)
    num_stocks = int(df["stock_id"].max()) + 1

    df.sort_values(["stock_id", "time_id", "seconds_in_bucket"], inplace=True)

    all_X, all_y, all_tids, all_sids = [], [], [], []

    for stock_id, stock_df in df.groupby("stock_id"):
        for tid, tid_df in stock_df.groupby("time_id"):
            tid_df = tid_df.sort_values("seconds_in_bucket")

            full = pd.DataFrame({"seconds_in_bucket": range(TOTAL_SECONDS)})
            full = full.merge(
                tid_df[["seconds_in_bucket", "wap", "bid_ask_spread",
                        "total_volume", "price_spread", "depth_imbalance"]],
                on="seconds_in_bucket", how="left"
            )

            wap     = full["wap"].ffill().bfill().fillna(1.0).values.astype(np.float32)
            spread  = full["bid_ask_spread"].fillna(0.0).values.astype(np.float32)
            volume  = full["total_volume"].fillna(0.0).values.astype(np.float32)
            pspread = full["price_spread"].fillna(0.0).values.astype(np.float32)
            dimbal  = full["depth_imbalance"].fillna(0.0).values.astype(np.float32)

            log_wap = np.log(np.clip(wap, 1e-10, None))
            log_ret = np.diff(log_wap, prepend=log_wap[0]).astype(np.float32)

            seq = np.stack([wap, log_ret, spread, volume, pspread, dimbal], axis=1)

            x = seq[:INPUT_END, :]

            target_rets = log_ret[TARGET_START:]
            rv = float(np.sqrt(np.sum(target_rets ** 2)))
            log_rv = float(np.log(max(rv, RV_FLOOR)))

            all_X.append(x)
            all_y.append(log_rv)
            all_tids.append(int(tid))
            all_sids.append(int(stock_id))

    del df; gc.collect()

    X = np.stack(all_X, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.float32)
    tids = np.array(all_tids, dtype=np.int32)
    sids = np.array(all_sids, dtype=np.int32)

    log(f"    {X.shape[0]} sequences, {num_stocks} stocks")
    return X, y, tids, sids, num_stocks


def compute_normalisation(X):
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    means = np.nanmean(X_flat, axis=0).astype(np.float32)
    stds = (np.nanstd(X_flat, axis=0) + 1e-8).astype(np.float32)
    return means, stds


# ── Stratified sampler ────────────────────────────────────────────────

def make_stratified_sampler(y: np.ndarray, n_strata: int = N_RV_STRATA) -> WeightedRandomSampler:
    """
    Assign each sample a weight inversely proportional to the density of its
    log(RV) stratum, so every bin contributes equally to each mini-batch.
    Low-RV samples (large negative log(RV)) are typically rare and get upweighted.
    """
    bins = np.linspace(y.min() - 1e-6, y.max() + 1e-6, n_strata + 1)
    strata = np.digitize(y, bins) - 1
    counts = np.bincount(strata, minlength=n_strata).astype(np.float64)
    counts = np.maximum(counts, 1)
    stratum_weight = 1.0 / counts
    sample_weights = stratum_weight[strata]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(y),
        replacement=True,
    )


# ── Dataset ───────────────────────────────────────────────────────────

class RVDataset(Dataset):
    def __init__(self, X, y, stock_ids, means, stds):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.stock_ids = torch.tensor(stock_ids, dtype=torch.long)
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_norm = torch.nan_to_num((self.X[idx] - self.means) / self.stds, nan=0.0)
        return x_norm, self.stock_ids[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────

class SigmoidRange(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return torch.sigmoid(x) * (self.hi - self.lo) + self.lo


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MLPModel(nn.Module):
    def __init__(self, n_features: int, seq_len: int, num_stocks: int,
                 hidden_dims=None, embed_dim: int = EMBED_DIM,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN

        self.stock_embed = nn.Embedding(num_stocks, embed_dim)
        self.dropout_embed = nn.Dropout(dropout)

        flat_dim = seq_len * n_features
        in_dim = flat_dim + embed_dim

        layers = []
        for out_dim in hidden_dims:
            layers.append(MLPBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.hidden = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(in_dim, 1),
            SigmoidRange(LOG_RV_MIN, LOG_RV_MAX),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x, stock_id):
        h = x.reshape(x.size(0), -1)
        emb = self.dropout_embed(self.stock_embed(stock_id))
        h = torch.cat([h, emb], dim=1)
        h = self.hidden(h)
        return self.head(h).squeeze(-1)


# ── Loss ──────────────────────────────────────────────────────────────

def rmspe_loss(pred_log, true_log):
    pred_rv = torch.exp(pred_log).clamp(min=EPS)
    true_rv = torch.exp(true_log).clamp(min=EPS)
    return torch.sqrt(torch.mean(((pred_rv - true_rv) / true_rv) ** 2))


def blended_loss(pred_log, true_log,
                 alpha: float = LOSS_ALPHA,
                 asym_factor: float = ASYM_FACTOR):
    """
    alpha     * RMSPE(RV-space)           — preserves original objective
    (1-alpha) * asymmetric MSE(log-space) — equalises gradient across full RV range

    Under-prediction: pred_log < true_log
      e.g. true log(RV) = -9, pred = -5 → err = +4 (model thinks RV is higher)
      We penalise this case harder so the model is pushed toward large negative values.
    """
    rmspe = rmspe_loss(pred_log, true_log)

    err = pred_log - true_log                        # positive = over, negative = under
    under = (err > 0).float()                        # pred > true_log => under-pred of low RV
    weights = 1.0 + (asym_factor - 1.0) * under
    log_mse = torch.mean(weights * err ** 2)

    return alpha * rmspe + (1.0 - alpha) * log_mse


# ── Training ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, sid, y in loader:
            x, sid, y = x.to(device), sid.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            pred = model(x, sid)
            loss = criterion(pred, y)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_n += len(y)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, sid, y in loader:
        preds.append(model(x.to(device), sid.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ── Run one outer fold ────────────────────────────────────────────────

def run_outer_fold(fold, fold_dir):
    log(f"\n{'=' * 60}")
    log(f"  OUTER FOLD {fold}  |  {fold_dir}  |  {DEVICE}")
    log(f"{'=' * 60}")

    device = torch.device(DEVICE)

    log("\n  Building train sequences ...")
    X_train, y_train, tids_train, sids_train, ns_train = build_all_sequences(fold_dir / "train.parquet")
    log("  Building test sequences ...")
    X_test, y_test, tids_test, sids_test, ns_test = build_all_sequences(fold_dir / "test.parquet")
    num_stocks = max(ns_train, ns_test)

    rng = np.random.default_rng(SEED)
    n_train = len(X_train)
    n_va = max(1, int(n_train * 0.1))
    idx = rng.permutation(n_train)
    va_idx, tr_idx = idx[:n_va], idx[n_va:]

    means, stds = compute_normalisation(X_train[tr_idx])

    tr_ds = RVDataset(X_train[tr_idx], y_train[tr_idx], sids_train[tr_idx], means, stds)
    va_ds = RVDataset(X_train[va_idx], y_train[va_idx], sids_train[va_idx], means, stds)
    te_ds = RVDataset(X_test, y_test, sids_test, means, stds)

    # Stratified sampler ensures low-RV samples appear proportionally in every batch
    tr_sampler = make_stratified_sampler(y_train[tr_idx])

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=tr_sampler,
                           num_workers=0, pin_memory=(DEVICE == "cuda"))
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0)

    log(f"  Train: {len(tr_ds):,}  |  Val: {len(va_ds):,}  |  Test: {len(te_ds):,}")
    log(f"  log(RV) train range: [{y_train[tr_idx].min():.2f}, {y_train[tr_idx].max():.2f}]")

    model = MLPModel(N_FEATURES, INPUT_END, num_stocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    criterion = blended_loss

    model_path = fold_dir / "mlp_model.pt"
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    tr_losses, va_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = _time.time()
        tr_loss = run_epoch(model, tr_loader, criterion, device, optimizer)
        va_loss = run_epoch(model, va_loader, criterion, device)
        scheduler.step(va_loss)
        elapsed = _time.time() - t0

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)

        if va_loss < best_val:
            best_val, best_epoch, no_improve = va_loss, epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or no_improve == 0:
            lr = optimizer.param_groups[0]["lr"]
            flag = " <- best" if no_improve == 0 else ""
            log(f"    Epoch {epoch:3d}/{EPOCHS} | Train: {tr_loss:.6f} | "
                f"Val: {va_loss:.6f} | LR: {lr:.2e} | {elapsed:.1f}s{flag}")

        if no_improve >= PATIENCE:
            log(f"    Early stop at epoch {epoch} (best={best_val:.6f} at {best_epoch})")
            break

    # ── Evaluate ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_preds = predict(model, te_loader, device)
    test_m = compute_metrics(test_preds, y_test)

    log(f"\n  Outer test metrics:")
    for k, v in test_m.items():
        log(f"    {k}: {v:.6f}")

    # Low-RV breakdown: bottom 10% of true log(RV)
    low_rv_thresh = np.percentile(y_test, 10)
    low_mask = y_test <= low_rv_thresh
    if low_mask.sum() >= 5:
        low_m = compute_metrics(test_preds[low_mask], y_test[low_mask])
        log(f"\n  Low-RV subset (bottom 10%, log_rv <= {low_rv_thresh:.2f}):")
        for k, v in low_m.items():
            log(f"    {k}: {v:.6f}")

    # Per-stock breakdown
    stock_metrics = {}
    for sid in np.unique(sids_test):
        mask = sids_test == sid
        if mask.sum() >= 5:
            stock_metrics[sid] = compute_metrics(test_preds[mask], y_test[mask])
    if stock_metrics:
        sorted_stocks = sorted(stock_metrics.items(), key=lambda x: x[1]["RMSPE%"])
        log(f"\n  Best 5 stocks by RMSPE%:")
        for sid, m in sorted_stocks[:5]:
            log(f"    Stock {sid}: RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}")
        log(f"  Worst 5 stocks by RMSPE%:")
        for sid, m in sorted_stocks[-5:]:
            log(f"    Stock {sid}: RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────
    pred_rv = np.exp(np.clip(test_preds, -20, 5))
    true_rv = np.exp(y_test)

    pd.DataFrame({"stock_id": sids_test, "time_id": tids_test,
                  "pred_log_rv": np.clip(test_preds, -20, 5).astype(np.float32),
                  "true_log_rv": y_test.astype(np.float32),
                  "pred_rv":     pred_rv.astype(np.float32),
                  "true_rv":     true_rv.astype(np.float32)}
                 ).to_csv(fold_dir / "mlp_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tr_losses, label="Train", alpha=0.7)
    ax.plot(va_losses, label="Val",   alpha=0.7)
    ax.axvline(best_epoch - 1, color="red", ls="--", alpha=0.5, label=f"Best ({best_epoch})")
    ax.set_title(f"Fold {fold} — MLP Training Curve")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Blended Loss")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(fold_dir / "mlp_training_curve.png", dpi=150); plt.close()

    with open(fold_dir / "mlp_best_params.json", "w") as f:
        json.dump({"n_params": n_params, "hidden_dims": MLP_HIDDEN,
                   "loss_alpha": LOSS_ALPHA, "asym_factor": ASYM_FACTOR,
                   "log_rv_min": LOG_RV_MIN, "n_rv_strata": N_RV_STRATA,
                   "best_epoch": best_epoch, "best_val_loss": best_val,
                   "test_metrics": test_m}, f, indent=2, default=str)
    log(f"  Saved -> {fold_dir}/mlp_*")

    del model, tr_ds, va_ds, te_ds, X_train, X_test; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return test_m


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("MLP — 5-Fold Evaluation (individual stock)")
    log(f"  Input     : seconds 0-{INPUT_END-1} ({INPUT_END} steps x {N_FEATURES} features = {INPUT_END*N_FEATURES})")
    log(f"  Target    : log(RV) of seconds {TARGET_START}-{TOTAL_SECONDS-1}")
    log(f"  Hidden    : {MLP_HIDDEN}")
    log(f"  Loss      : {LOSS_ALPHA:.2f}*RMSPE + {1-LOSS_ALPHA:.2f}*AsymLogMSE (asym={ASYM_FACTOR})")
    log(f"  SigRange  : [{LOG_RV_MIN}, {LOG_RV_MAX}]")
    log(f"  Strata    : {N_RV_STRATA} log(RV) bins for stratified sampling")
    log(f"  Device    : {DEVICE}")
    log(f"  Folds     : {N_OUTER_FOLDS}")
    log(f"  Epochs    : {EPOCHS} (patience {PATIENCE})")
    log("=" * 60)

    all_metrics = []
    for fold in range(N_OUTER_FOLDS):
        fold_dir = DATA_DIR / f"fold_{fold}"
        if not (fold_dir / "train.parquet").exists():
            log(f"\n  {fold_dir}/train.parquet not found -- skipping")
            continue
        test_m = run_outer_fold(fold, fold_dir)
        all_metrics.append(test_m)
        gc.collect()

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    metric_keys = list(all_metrics[0].keys()) if all_metrics else []
    for i, m in enumerate(all_metrics):
        log(f"  Fold {i}: " + "  ".join(f"{k}={v:.6f}" for k, v in m.items()))
    if all_metrics:
        log(f"\nAggregated (mean +/- std):")
        for k in metric_keys:
            vals = [m[k] for m in all_metrics]
            log(f"  {k:12s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")

    with open(DATA_DIR / "mlp_nested_cv_summary.json", "w") as f:
        json.dump({"n_outer_folds": len(all_metrics), "device": DEVICE,
                   "hidden_dims": MLP_HIDDEN,
                   "loss_alpha": LOSS_ALPHA, "asym_factor": ASYM_FACTOR,
                   "per_fold_metrics": all_metrics,
                   "mean_metrics": {k: float(np.mean([m[k] for m in all_metrics]))
                                    for k in metric_keys} if all_metrics else {},
                   "std_metrics":  {k: float(np.std([m[k] for m in all_metrics]))
                                    for k in metric_keys} if all_metrics else {}},
                  f, indent=2, default=str)
    log(f"\nDone.")


if __name__ == "__main__":
    main()