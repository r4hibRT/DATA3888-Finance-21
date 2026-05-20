"""
gnn_rv.py
=========
Spatio-Temporal GNN for realized volatility forecasting.

Merged version:
  - HAR branch with learned blend weight (from better-performing version)
  - Deviation-from-mean global context (fixes mean collapse)
  - Per-epoch RMSPE/QLIKE printing
  - Combined RMSPE + QLIKE loss
  - Stratified sampling (oversamples low-RV time_ids)
  - Reads preprocessed .npz from preprocess_gnn.py
  - Per-tid spread penalty
  - 8 buckets of 60s (finer temporal resolution)

Pipeline:
  eda.ipynb -> processed/fold_{0..4}/{train,test}.parquet
  preprocess_gnn.py -> processed/fold_{0..4}/{train,test}_gnn.npz
  -> this script -> processed/fold_{0..4}/gnn_* outputs
"""

from __future__ import annotations

import gc
import json
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch_geometric.typing


try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Config ────────────────────────────────────────────────────────────

DATA_DIR        = Path("/content/drive/MyDrive/processed")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
N_OUTER_FOLDS   = 5
EPS             = 1e-8

# Window config
INPUT_END      = 480
TARGET_START   = 480
TOTAL_SECONDS  = 600
BUCKET_SIZE    = 60      # 8 buckets of 60s — finer temporal resolution
BUCKET_COUNT   = INPUT_END // BUCKET_SIZE   # 8
INPUT_DIM      = 7

# Per-bucket features
FEATURE_COLS = ["log_return", "realized_vol", "mean_wap", "mean_spread",
                "sum_volume", "mean_price_spread", "mean_depth_imbalance"]

# RV constants
RV_FLOOR     = 1e-4
LOG_RV_FLOOR = -10.0

# Direction switches
USE_LEARNED_GRAPH = True
USE_CROSS_ATTN    = True
USE_MAMBA         = True

# Fixed architecture
STOCK_EMBED_DIM       = 32
EDGE_MLP_HIDDEN       = 64
EDGE_TOPK             = 10
EDGE_TEMPERATURE      = 1.0
CROSS_ATTN_HEADS      = 4
NUM_CROSS_ATTN_LAYERS = 2
GAT_HEADS             = 4
NUM_GNN_LAYERS        = 2
MLP_DEPTH             = 3
MAMBA_D_STATE         = 16
MAMBA_D_CONV          = 2
MAMBA_EXPAND          = 2

# Loss weights
RMSPE_WEIGHT  = 0.5
QLIKE_WEIGHT  = 0.5
SPREAD_WEIGHT = 0.1

# Fixed hyperparameters
FIXED_HPARAMS = {
    "d_model":       128,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "gat_dropout":   0.1,
    "mlp_hidden":    128,
    "final_dropout": 0.2,
}

# Training
EPOCHS         = 50
PATIENCE       = 8
GRAD_CLIP      = 1.0
BATCH_SIZE     = 16


def log(msg):
    print(f"[GNN] {msg}", flush=True)


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


# ── Load preprocessed .npz ───────────────────────────────────────────

def load_gnn_npz(path: Path):
    data = np.load(path)
    X, y, time_ids = data["X"], data["y"], data["time_ids"]
    num_stocks = int(data["num_stocks"])
    X_dict = {int(time_ids[i]): X[i] for i in range(len(time_ids))}
    y_dict = {int(time_ids[i]): y[i] for i in range(len(time_ids))}
    log(f"    Loaded {path.name}: {len(X_dict)} time_ids, {num_stocks} stocks")
    return X_dict, y_dict, num_stocks


# ── Dataset ───────────────────────────────────────────────────────────

class StockDataset(Dataset):
    def __init__(self, X_dict, y_dict, num_stocks):
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.num_stocks = num_stocks
        self.time_ids = sorted(set(X_dict.keys()) & set(y_dict.keys()))

    def __len__(self):
        return len(self.time_ids)

    def __getitem__(self, idx):
        tid = self.time_ids[idx]
        X = torch.nan_to_num(torch.tensor(self.X_dict[tid], dtype=torch.float32), nan=0.0)
        y = torch.nan_to_num(torch.tensor(self.y_dict[tid], dtype=torch.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
        stock_ids = torch.arange(self.num_stocks, dtype=torch.long)
        return X, y, tid, stock_ids

    def get_sampling_weights(self):
        weights = np.empty(len(self.time_ids), dtype=np.float64)
        for i, tid in enumerate(self.time_ids):
            y = self.y_dict[tid]
            live = y > LOG_RV_FLOOR
            if live.sum() == 0:
                weights[i] = 1.0
            else:
                weights[i] = 1.0 / max(float(np.exp(y[live]).mean()), EPS)
        cap = np.percentile(weights, 99)
        weights = np.minimum(weights, cap)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch):
    X_list, y_list, tid_list, sid_list = zip(*batch)
    return (torch.stack(X_list), torch.stack(y_list),
            list(tid_list), torch.stack(sid_list))


# ── Model components ─────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dt_rank):
        super().__init__()
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, T, d = x.shape
        x_proj = self.x_proj(x)
        dt_rank = x_proj.shape[-1] - 2 * self.d_state
        dt_raw = x_proj[..., :dt_rank]
        B_mat = x_proj[..., dt_rank:dt_rank + self.d_state]
        C_mat = x_proj[..., dt_rank + self.d_state:]
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B_mat.unsqueeze(2)
        h = torch.zeros(B, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            ys.append((h * C_mat[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        d_inner = d_model * expand
        dt_rank = max(1, d_model // 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)
        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        d_in = xz.shape[-1] // 2
        x_b, z = xz[..., :d_in], xz[..., d_in:]
        x_b = self.conv1d(x_b.transpose(1, 2))[..., :x_b.shape[1]].transpose(1, 2)
        x_b = F.silu(x_b)
        y = self.ssm(x_b)
        return self.out_proj(y * F.silu(z)) + residual


class MambaTemporalEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND)
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm_out(h).mean(dim=1)


class TransformerTemporalEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, BUCKET_COUNT + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True,
                                               dim_feedforward=d_model * 4)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        BN = x.size(0)
        h = self.input_proj(x)
        cls = self.cls_token.expand(BN, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed[:, :BUCKET_COUNT + 1]
        return self.norm_out(self.encoder(h)[:, 0, :])


class LearnedGraphBuilder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, EDGE_MLP_HIDDEN), nn.GELU(),
            nn.Linear(EDGE_MLP_HIDDEN, EDGE_MLP_HIDDEN // 2), nn.GELU(),
            nn.Linear(EDGE_MLP_HIDDEN // 2, 1))

    def forward(self, h, live_mask, N, B, device):
        all_src, all_dst, all_attr = [], [], []
        for b in range(B):
            start = b * N
            live_ids = torch.where(live_mask[start:start + N])[0]
            n_live = live_ids.numel()
            if n_live < 2:
                continue
            h_live = h[start + live_ids]
            hi = h_live.unsqueeze(1).expand(-1, n_live, -1)
            hj = h_live.unsqueeze(0).expand(n_live, -1, -1)
            scores = self.edge_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)
            scores.fill_diagonal_(float("-inf"))
            k = min(EDGE_TOPK, n_live - 1)
            topk_scores, topk_idx = torch.topk(scores, k, dim=1)
            edge_weights = F.softmax(topk_scores / EDGE_TEMPERATURE, dim=-1)
            src_local = torch.arange(n_live, device=device).unsqueeze(1).expand(-1, k)
            all_src.append(start + live_ids[src_local.reshape(-1)])
            all_dst.append(start + live_ids[topk_idx.reshape(-1)])
            all_attr.append(edge_weights.reshape(-1))
        if not all_src:
            return (torch.zeros((2, 0), dtype=torch.long, device=device),
                    torch.zeros((0, 1), dtype=torch.float32, device=device))
        return (torch.stack([torch.cat(all_src), torch.cat(all_dst)]),
                torch.cat(all_attr).unsqueeze(-1))


class CrossStockAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, CROSS_ATTN_HEADS,
                                              dropout=0.1, batch_first=True),
                "ffn_norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(d_model * 4, d_model),
                                     nn.Dropout(0.1)),
            }) for _ in range(NUM_CROSS_ATTN_LAYERS)])

    def forward(self, h, live_mask):
        key_padding_mask = ~live_mask
        for layer in self.layers:
            h_norm = layer["norm"](h)
            attn_out, _ = layer["attn"](h_norm, h_norm, h_norm,
                                        key_padding_mask=key_padding_mask)
            h = h + attn_out
            h = h + layer["ffn"](layer["ffn_norm"](h))
            h = h * live_mask.unsqueeze(-1).float()
        return h


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ResidualMLPHead(nn.Module):
    def __init__(self, in_dim, hidden, depth, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden, dropout) for _ in range(depth)])
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


# ── Main model (HAR branch + deviation-from-mean context) ────────────

class SpatioTemporalGNN(nn.Module):
    def __init__(self, input_dim, num_stocks, d_model=128, gat_dropout=0.1,
                 mlp_hidden=128, final_dropout=0.2):
        super().__init__()
        self.num_stocks = num_stocks
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(input_dim)

        if USE_MAMBA:
            self.temporal_encoder = MambaTemporalEncoder(input_dim, d_model)
        else:
            self.temporal_encoder = TransformerTemporalEncoder(input_dim, d_model)

        self.stock_embed = nn.Embedding(num_stocks, STOCK_EMBED_DIM)
        self.embed_proj = nn.Linear(d_model + STOCK_EMBED_DIM, d_model)
        self.pre_gnn_norm = nn.LayerNorm(d_model)

        if USE_LEARNED_GRAPH:
            self.graph_builder = LearnedGraphBuilder(d_model)
        else:
            self.graph_builder = None

        if HAS_PYG:
            self.gat_layers = nn.ModuleList()
            self.gnn_norms = nn.ModuleList()
            for _ in range(NUM_GNN_LAYERS):
                self.gat_layers.append(GATv2Conv(d_model, d_model // GAT_HEADS,
                                                 heads=GAT_HEADS, concat=True,
                                                 dropout=gat_dropout, edge_dim=1))
                self.gnn_norms.append(nn.LayerNorm(d_model))
        else:
            self.gat_layers = nn.ModuleList()
            self.gnn_norms = nn.ModuleList()

        if USE_CROSS_ATTN:
            self.cross_attn = CrossStockAttention(d_model)
        else:
            self.cross_attn = None

        self.global_proj = nn.Linear(d_model * 2, d_model)
        self.head = ResidualMLPHead(d_model, mlp_hidden, MLP_DEPTH, final_dropout)

        # HAR branch: learned blend with GNN
        self.har_branch = nn.Linear(BUCKET_COUNT, 1, bias=True)
        self.har_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, stock_ids):
        B, N, T, F_dim = X.shape
        device = X.device
        X_flat = X.view(B * N, T, F_dim)
        live_mask = X_flat.abs().sum(dim=(1, 2)) > 0

        # 1. Input norm
        X_normed = torch.zeros_like(X_flat)
        if live_mask.any():
            live_x = X_flat[live_mask]
            L, T_, F_ = live_x.shape
            X_normed[live_mask] = self.input_norm(
                live_x.reshape(L * T_, F_)).reshape(L, T_, F_)

        # 2. Temporal encoding
        h = torch.zeros(B * N, self.d_model, device=device, dtype=X.dtype)
        if live_mask.any():
            h[live_mask] = self.temporal_encoder(X_normed[live_mask])

        # 3. Stock embedding
        s_emb = self.stock_embed(stock_ids.view(B * N))
        h = self.embed_proj(torch.cat([h, s_emb], dim=-1))
        h = h * live_mask.unsqueeze(-1).float()

        # 4. Pre-GNN norm
        if live_mask.any():
            h[live_mask] = self.pre_gnn_norm(h[live_mask])

        # 5. Graph message passing
        if USE_LEARNED_GRAPH and self.graph_builder is not None:
            edge_index, edge_attr = self.graph_builder(h, live_mask, N, B, device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = None

        if HAS_PYG and edge_index.numel() > 0 and len(self.gat_layers) > 0:
            for gat, norm in zip(self.gat_layers, self.gnn_norms):
                residual = h
                h_n = torch.zeros_like(h)
                if live_mask.any():
                    h_n[live_mask] = norm(h[live_mask])
                h = gat(h_n, edge_index, edge_attr=edge_attr)
                h = F.gelu(h) + residual
                h = h * live_mask.unsqueeze(-1).float()

        # 6. Cross-stock attention
        if USE_CROSS_ATTN and self.cross_attn is not None:
            h_s = h.view(B, N, self.d_model)
            live_s = live_mask.view(B, N)
            h = self.cross_attn(h_s, live_s).view(B * N, self.d_model)

        # 7. Global context: DEVIATION from cross-sectional mean
        h_s = h.view(B, N, self.d_model)
        live_s = live_mask.view(B, N).float().unsqueeze(-1)
        live_n = live_s.sum(dim=1, keepdim=True).clamp(min=1)
        mean_ctx = (h_s * live_s).sum(dim=1, keepdim=True) / live_n  # (B, 1, d)
        g_ctx = (h_s - mean_ctx.expand(-1, N, -1)) * live_s          # (B, N, d)
        h_aug = self.global_proj(torch.cat([h_s, g_ctx], dim=-1)) * live_s

        # 8. GNN head
        gnn_pred = self.head(h_aug).squeeze(-1)  # (B, N)

        # 9. HAR skip branch
        rv_buckets = X[:, :, :, 1]                            # realized_vol per bucket
        rv_log = torch.log(rv_buckets.clamp(min=RV_FLOOR))    # log scale
        har_pred = self.har_branch(rv_log).squeeze(-1)         # (B, N)

        # 10. Learned blend
        w = torch.sigmoid(self.har_weight)
        out = w * har_pred + (1 - w) * gnn_pred

        return out * live_mask.view(B, N).float()


# ── Loss (combined RMSPE + QLIKE + per-tid spread penalty) ───────────

class CombinedLoss(nn.Module):
    def __init__(self, rmspe_w=RMSPE_WEIGHT, qlike_w=QLIKE_WEIGHT,
                 spread_w=SPREAD_WEIGHT, eps=EPS):
        super().__init__()
        self.rmspe_w = rmspe_w
        self.qlike_w = qlike_w
        self.spread_w = spread_w
        self.eps = eps

    def forward(self, pred_log, true_log):
        B, N = pred_log.shape
        live = (true_log != 0.0) & (true_log > LOG_RV_FLOOR)
        if not live.any():
            return pred_log.sum() * 0.0

        p = pred_log[live].clamp(-20, 5)
        t = true_log[live]
        pred_rv = torch.exp(p).clamp(min=self.eps)
        true_rv = torch.exp(t).clamp(min=self.eps)

        # RMSPE
        rmspe = torch.sqrt(torch.mean(((pred_rv - true_rv) / true_rv) ** 2))

        # QLIKE
        ratio = true_rv / pred_rv
        qlike = torch.mean(ratio - torch.log(ratio) - 1)

        # Per-time_id spread penalty
        spread_penalties = []
        for b in range(B):
            live_b = live[b]
            if live_b.sum() < 3:
                continue
            p_b = pred_log[b, live_b].clamp(-20, 5)
            t_b = true_log[b, live_b]
            spread_penalties.append(F.relu(t_b.std() - p_b.std()) ** 2)

        spread_loss = torch.stack(spread_penalties).mean() if spread_penalties \
            else torch.tensor(0.0, device=pred_log.device)

        return self.rmspe_w * rmspe + self.qlike_w * qlike + self.spread_w * spread_loss


# ── Training ──────────────────────────────────────────────────────────

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_n = 0.0, 0
    all_pred, all_true = [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X_b, y_b, _, sid_b in loader:
            X_b, y_b, sid_b = X_b.to(device), y_b.to(device), sid_b.to(device)
            if is_train:
                optimizer.zero_grad()
            pred = model(X_b, sid_b)
            loss = loss_fn(pred, y_b)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item() * X_b.size(0)
            total_n += X_b.size(0)
            with torch.no_grad():
                live = (y_b != 0.0) & (y_b > LOG_RV_FLOOR)
                if live.any():
                    all_pred.append(pred[live].detach().cpu().numpy())
                    all_true.append(y_b[live].detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)

    if all_pred:
        p = np.clip(np.concatenate(all_pred), -20, 5)
        t = np.concatenate(all_true)
        pred_rv = np.exp(p).clip(EPS)
        true_rv = np.exp(t).clip(EPS)
        ratio = true_rv / pred_rv
        rmspe = float(np.sqrt(np.mean(((pred_rv - true_rv) / true_rv) ** 2)) * 100)
        qlike = float(np.mean(ratio - np.log(ratio) - 1))
    else:
        rmspe, qlike = 0.0, 0.0

    return avg_loss, rmspe, qlike


def train_model(num_stocks, train_ds, val_ds, hparams, device,
                verbose=False, save_path=None):
    model = SpatioTemporalGNN(
        INPUT_DIM, num_stocks,
        d_model=hparams["d_model"], gat_dropout=hparams["gat_dropout"],
        mlp_hidden=hparams["mlp_hidden"], final_dropout=hparams["final_dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    loss_fn = CombinedLoss()

    sampling_weights = train_ds.get_sampling_weights()
    sampler = WeightedRandomSampler(weights=sampling_weights,
                                    num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    best_val, best_epoch, no_improve = float("inf"), 0, 0

    if verbose:
        log(f"    {'Ep':>4s} | {'TrLoss':>8s} {'TrRMSPE%':>9s} {'TrQLIKE':>8s} | "
            f"{'VaLoss':>8s} {'VaRMSPE%':>9s} {'VaQLIKE':>8s} | {'LR':>8s} {'time':>5s}")
        log(f"    {'─'*4}─┼─{'─'*8}─{'─'*9}─{'─'*8}─┼─"
            f"{'─'*8}─{'─'*9}─{'─'*8}─┼─{'─'*8}─{'─'*5}")

    for epoch in range(1, EPOCHS + 1):
        t0 = _time.time()
        tr_loss, tr_rmspe, tr_qlike = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_rmspe, va_qlike = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(va_loss)
        elapsed = _time.time() - t0

        if va_loss < best_val:
            best_val, best_epoch, no_improve = va_loss, epoch, 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or no_improve == 0):
            lr_now = optimizer.param_groups[0]["lr"]
            flag = " *" if no_improve == 0 else ""
            log(f"    {epoch:4d} | {tr_loss:8.5f} {tr_rmspe:8.2f}% {tr_qlike:8.5f} | "
                f"{va_loss:8.5f} {va_rmspe:8.2f}% {va_qlike:8.5f} | "
                f"{lr_now:.1e} {elapsed:4.0f}s{flag}")

        if no_improve >= PATIENCE:
            if verbose:
                log(f"    Early stop at epoch {epoch} (best loss={best_val:.6f} at {best_epoch})")
            break

    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return best_val, model


# ── Run one outer fold ────────────────────────────────────────────────

def run_outer_fold(fold, fold_dir):
    log(f"\n{'=' * 60}")
    log(f"  OUTER FOLD {fold}  |  {fold_dir}  |  {DEVICE}")
    log(f"  Loss: {RMSPE_WEIGHT}*RMSPE + {QLIKE_WEIGHT}*QLIKE + {SPREAD_WEIGHT}*spread(per-tid)")
    log(f"  HAR branch: ON (learned blend)")
    log(f"  Global ctx: deviation from mean")
    log(f"{'=' * 60}")

    device = torch.device(DEVICE)

    log("\n  Loading preprocessed data ...")
    X_train, y_train, ns_train = load_gnn_npz(fold_dir / "train_gnn.npz")
    X_test, y_test, ns_test = load_gnn_npz(fold_dir / "test_gnn.npz")
    num_stocks = max(ns_train, ns_test)

    log("  Hparams:")
    for k, v in FIXED_HPARAMS.items():
        log(f"    {k}: {v}")

    # 90/10 early-stop holdout
    all_tids = sorted(X_train.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_tids)
    n_es = max(1, int(len(all_tids) * 0.1))
    es_tids = set(all_tids[:n_es])
    tr_tids = set(all_tids[n_es:])

    tr_ds = StockDataset({t: X_train[t] for t in tr_tids},
                         {t: y_train[t] for t in tr_tids}, num_stocks)
    es_ds = StockDataset({t: X_train[t] for t in es_tids},
                         {t: y_train[t] for t in es_tids}, num_stocks)

    log(f"  Train: {len(tr_ds)} | ES: {len(es_ds)} | Test: {len(X_test)}")

    model_path = fold_dir / "gnn_model.pt"
    _, final_model = train_model(num_stocks, tr_ds, es_ds, FIXED_HPARAMS, device,
                                  verbose=True, save_path=model_path)
    n_params = sum(p.numel() for p in final_model.parameters())
    har_w = torch.sigmoid(final_model.har_weight).item()
    log(f"  Model params: {n_params:,}")
    log(f"  HAR blend: {har_w:.3f} HAR + {1-har_w:.3f} GNN")

    # Evaluate
    test_ds = StockDataset(X_test, y_test, num_stocks)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    final_model.eval()
    all_pred, all_true, all_tids_out, all_live = [], [], [], []
    with torch.no_grad():
        for X_b, y_b, tid_b, sid_b in test_loader:
            X_b, sid_b = X_b.to(device), sid_b.to(device)
            live = X_b.abs().sum(dim=(2, 3)) > 0
            pred = final_model(X_b, sid_b)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y_b.numpy())
            all_tids_out.extend(tid_b)
            all_live.append(live.cpu().numpy())

    pred_log = np.concatenate(all_pred, axis=0)
    true_log = np.concatenate(all_true, axis=0)
    live_mask = np.concatenate(all_live, axis=0)

    test_m = compute_metrics(pred_log[live_mask], true_log[live_mask])
    log(f"\n  Outer test metrics:")
    for k, v in test_m.items():
        log(f"    {k}: {v:.6f}")

    # Spread diagnostic
    pred_std = float(np.std(pred_log[live_mask]))
    true_std = float(np.std(true_log[live_mask]))
    log(f"\n  Spread: pred_std={pred_std:.4f}  true_std={true_std:.4f}  "
        f"ratio={pred_std/true_std:.3f}")

    # Per-RV-tier
    pred_flat = pred_log[live_mask]
    true_flat = true_log[live_mask]
    true_rv_flat = np.exp(true_flat)
    log(f"\n  By RV tier:")
    for name, lo, hi in [("Very low (<0.001)", 0, 0.001),
                          ("Low (0.001-0.005)", 0.001, 0.005),
                          ("Mid (0.005-0.02)",  0.005, 0.02),
                          ("High (>=0.02)",     0.02, np.inf)]:
        mask = (true_rv_flat >= lo) & (true_rv_flat < hi)
        if mask.sum() < 10:
            continue
        m = compute_metrics(pred_flat[mask], true_flat[mask])
        log(f"    {name:25s} n={mask.sum():6,}: RMSPE%={m['RMSPE%']:.1f}  QLIKE={m['QLIKE']:.4f}")

    # Per-stock
    n_tids, n_stocks_out = pred_log.shape
    stock_metrics = {}
    for s in range(n_stocks_out):
        mask_s = live_mask[:, s]
        if mask_s.sum() >= 5:
            stock_metrics[s] = compute_metrics(pred_log[mask_s, s], true_log[mask_s, s])
    if stock_metrics:
        sorted_stocks = sorted(stock_metrics.items(), key=lambda x: x[1]["RMSPE%"])
        log(f"\n  Best 5 stocks:")
        for sid, m in sorted_stocks[:5]:
            log(f"    Stock {sid}: RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}")
        log(f"  Worst 5 stocks:")
        for sid, m in sorted_stocks[-5:]:
            log(f"    Stock {sid}: RMSPE%={m['RMSPE%']:.2f}  QLIKE={m['QLIKE']:.6f}")

    # Save
    rows = [{"time_id": all_tids_out[i], "stock_id": s,
             "pred_log_rv": float(np.clip(pred_log[i, s], -20, 5)),
             "true_log_rv": float(true_log[i, s]),
             "pred_rv": float(np.exp(np.clip(pred_log[i, s], -20, 5))),
             "true_rv": float(np.exp(true_log[i, s]))}
            for i in range(n_tids) for s in range(n_stocks_out) if live_mask[i, s]]
    pd.DataFrame(rows).to_csv(fold_dir / "gnn_predictions.csv", index=False)

    with open(fold_dir / "gnn_best_params.json", "w") as f:
        json.dump({"hparams": FIXED_HPARAMS, "n_params": n_params,
                   "har_blend_weight": har_w,
                   "spread_diagnostic": {"pred_std": pred_std, "true_std": true_std},
                   "loss": f"{RMSPE_WEIGHT}*RMSPE+{QLIKE_WEIGHT}*QLIKE+{SPREAD_WEIGHT}*spread",
                   "global_ctx": "deviation_from_mean",
                   "test_metrics": test_m}, f, indent=2, default=str)
    log(f"  Saved -> {fold_dir}/gnn_*")

    del final_model, tr_ds, es_ds, test_ds; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return test_m


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Spatio-Temporal GNN (merged: HAR + deviation ctx + per-epoch metrics)")
    log(f"  Input      : .npz ({BUCKET_COUNT} buckets x {INPUT_DIM} features)")
    log(f"  Bucket     : {BUCKET_SIZE}s ({BUCKET_COUNT} buckets)")
    log(f"  Target     : log(RV) of seconds {TARGET_START}-{TOTAL_SECONDS-1}")
    log(f"  Device     : {DEVICE}")
    log(f"  Mamba/Graph/Attn: {USE_MAMBA}/{USE_LEARNED_GRAPH}/{USE_CROSS_ATTN}")
    log(f"  HAR branch : ON (learned blend)")
    log(f"  Global ctx : deviation from mean")
    log(f"  Loss       : {RMSPE_WEIGHT}*RMSPE + {QLIKE_WEIGHT}*QLIKE + {SPREAD_WEIGHT}*spread")
    log(f"  Sampling   : stratified (inverse RV)")
    log(f"  Folds      : {N_OUTER_FOLDS}")
    log(f"  PyG        : {HAS_PYG}")
    log("=" * 60)

    all_metrics = []
    for fold in range(N_OUTER_FOLDS):
        fold_dir = DATA_DIR / f"fold_{fold}"
        npz_path = fold_dir / "train_gnn.npz"
        if not npz_path.exists():
            log(f"\n  {npz_path} not found -- run preprocess_gnn.py first")
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

    with open(DATA_DIR / "gnn_nested_cv_summary.json", "w") as f:
        json.dump({"n_outer_folds": len(all_metrics), "hparams": FIXED_HPARAMS,
                   "bucket_size": BUCKET_SIZE, "bucket_count": BUCKET_COUNT,
                   "loss": f"{RMSPE_WEIGHT}*RMSPE+{QLIKE_WEIGHT}*QLIKE+{SPREAD_WEIGHT}*spread",
                   "har_branch": True, "global_ctx": "deviation_from_mean",
                   "device": DEVICE, "per_fold_metrics": all_metrics,
                   "mean_metrics": {k: float(np.mean([m[k] for m in all_metrics]))
                                    for k in metric_keys} if all_metrics else {},
                   "std_metrics": {k: float(np.std([m[k] for m in all_metrics]))
                                   for k in metric_keys} if all_metrics else {}},
                  f, indent=2, default=str)
    log(f"\nDone.")

if __name__ == "__main__":
    main()