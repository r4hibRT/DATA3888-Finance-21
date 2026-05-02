"""
model.py  —  Spatio-Temporal GNN v5
====================================
Three architectural upgrades over v4, each togglable via config flags:

  Direction 1  USE_LEARNED_GRAPH    NRI-style learned dynamic graph
               Instead of a hand-crafted top-K correlation graph, a small
               edge MLP scores every stock pair from their temporal embeddings.
               The graph is fully data-driven and changes every time_id.

  Direction 2  USE_CROSS_ATTN       Cross-stock joint attention block
               Replaces (or stacks after) GATv2 with a standard multi-head
               self-attention layer where stocks are the sequence.
               For 112 stocks this is 112² = 12 544 pairs — trivially cheap.

  Direction 3  USE_MAMBA            Mamba-style SSM as temporal encoder
               Pure-PyTorch selective state-space model (no mamba-ssm package).
               O(T) complexity vs O(T²) for Transformer.  Captures long-range
               dependencies in the 3-bucket sequence efficiently.

Preprocess compatibility:
  Expects output of preprocess.py v2 (100s buckets):
    SEQ_LEN   = 3      (3 buckets)
    INPUT_DIM = 4      (log_return, realized_vol, mean_spread, total_volume)

All three directions can be combined freely:
  USE_LEARNED_GRAPH=True  + USE_CROSS_ATTN=True  + USE_MAMBA=True
  → full stack: Mamba encoder → learned graph → GATv2 → cross-stock attention
"""

from __future__ import annotations

import pickle
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv


# ====================================================================
# CONFIGURATION
# ====================================================================

FEATURES_DIR = "/content/drive/MyDrive/processed_v2"
MODEL_DIR    = "models"
CACHE_FILE   = "graph_cache_v5.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data ---
SEQ_LEN   = 3      # buckets (3 × 100s)
INPUT_DIM = 4      # features per bucket

FEATURE_COLS = ["log_return", "realized_vol", "mean_spread", "total_volume"]

# --- Direction switches ---
USE_LEARNED_GRAPH = True    # Direction 1: NRI-style edge MLP
USE_CROSS_ATTN    = True    # Direction 2: cross-stock attention
USE_MAMBA         = True    # Direction 3: SSM temporal encoder

# --- Temporal encoder (used when USE_MAMBA=False) ---
TEMPORAL_ENCODER = "transformer"   # "cnn" | "transformer"

# --- Model dimensions ---
D_MODEL       = 128
CNN_CHANNELS  = 64
CNN_KERNELS   = [1, 2, 3]    # small kernels — only 3 buckets

# Transformer options
NHEAD        = 4
NUM_TF_LAYERS = 2
TF_DROPOUT   = 0.1

# Mamba SSM options
MAMBA_D_STATE  = 16    # SSM state dimension
MAMBA_D_CONV   = 2     # local conv width (≤ SEQ_LEN)
MAMBA_EXPAND   = 2     # inner dim multiplier

# --- Stock embeddings ---
STOCK_EMBED_DIM = 32

# --- Direction 1: Learned graph ---
EDGE_MLP_HIDDEN  = 64     # hidden dim of the edge scoring MLP
EDGE_TOPK        = 10     # keep top-K edges per node after scoring
EDGE_TEMPERATURE = 1.0    # softmax temperature for edge weights

# --- Direction 2: Cross-stock attention ---
CROSS_ATTN_HEADS   = 4
CROSS_ATTN_DROPOUT = 0.1
NUM_CROSS_ATTN_LAYERS = 2

# --- Graph attention (GATv2, used when USE_LEARNED_GRAPH or static graph) ---
GAT_HEADS      = 4
GAT_DROPOUT    = 0.1
NUM_GNN_LAYERS = 2
TOP_K          = 10       # for static fallback graph
USE_EDGE_ATTR  = True

# --- MLP head ---
MLP_HIDDEN    = 128
MLP_DEPTH     = 3
FINAL_DROPOUT = 0.2

# --- Training ---
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
EPOCHS        = 50
PATIENCE      = 8
LOSS_FN       = "rmspe"    # "mse" | "rmspe"

EPS         = 1e-8
RV_FLOOR    = 1e-8     # minimum meaningful RV — matches preprocess.py
LOG_RV_FLOOR = -18.0   # log(RV_FLOOR) ~ -18.42, used to detect dead stocks
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ====================================================================
# DIRECTION 3 — MAMBA SSM TEMPORAL ENCODER (pure PyTorch)
# ====================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core — the inner loop of Mamba.

    For each input token x_t the model computes input-dependent
    state-transition matrices (A, B, C, dt), allowing it to selectively
    remember or forget information.  This is the key difference from
    a fixed-transition SSM like S4.

    Complexity: O(T × d_state) — linear in sequence length.
    """

    def __init__(self, d_model: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent projections
        self.x_proj  = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # SSM parameters (log-space A for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, d_model)
        Returns:
            y : (B, T, d_model)
        """
        B, T, d = x.shape
        d_state  = self.d_state

        # Project to get dt, B_mat, C_mat
        x_proj  = self.x_proj(x)                             # (B, T, dt_rank + 2*d_state)
        dt_rank = x_proj.shape[-1] - 2 * d_state
        dt_raw  = x_proj[..., :dt_rank]
        B_mat   = x_proj[..., dt_rank : dt_rank + d_state]   # (B, T, d_state)
        C_mat   = x_proj[..., dt_rank + d_state :]            # (B, T, d_state)

        # Softplus dt — must be positive
        dt = F.softplus(self.dt_proj(dt_raw))                 # (B, T, d_model)

        # Discretise A: A_bar = exp(-exp(A_log) * dt)
        A     = -torch.exp(self.A_log)                        # (d_model, d_state)
        A_bar = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )                                                     # (B, T, d_model, d_state)

        # B_bar = dt * B  (simplified zero-order hold)
        B_bar = dt.unsqueeze(-1) * B_mat.unsqueeze(2)        # (B, T, d_model, d_state)

        # Sequential scan over T
        h = torch.zeros(B, d, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)     # (B, d_model)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                            # (B, T, d_model)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    """
    Full Mamba block:
      LayerNorm → split into x and z branches →
      depthwise conv on x → SelectiveSSM → SiLU gate with z → project out
      + residual

    Mirrors the original Mamba architecture without requiring the
    mamba-ssm CUDA kernel — uses pure PyTorch instead.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        d_inner  = d_model * expand
        dt_rank  = max(1, d_model // 16)

        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal depthwise conv over time
        self.conv1d  = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        self.ssm     = SelectiveSSM(d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        residual = x
        x = self.norm(x)

        # Split into two branches
        xz   = self.in_proj(x)                     # (B, T, 2*d_inner)
        d_in = xz.shape[-1] // 2
        x_b, z = xz[..., :d_in], xz[..., d_in:]   # each (B, T, d_inner)

        # Causal conv on x branch
        x_b = x_b.transpose(1, 2)                  # (B, d_inner, T)
        x_b = self.conv1d(x_b)[..., :x_b.shape[-1]]  # trim causal padding
        x_b = x_b.transpose(1, 2)                  # (B, T, d_inner)
        x_b = F.silu(x_b)

        # SSM
        y = self.ssm(x_b)                          # (B, T, d_inner)

        # Gating with z branch
        y = y * F.silu(z)

        return self.out_proj(y) + residual          # (B, T, d_model)


class MambaTemporalEncoder(nn.Module):
    """
    Mamba SSM encoder for the bucket sequence.

    Projects INPUT_DIM → D_MODEL, runs N_MAMBA_LAYERS Mamba blocks,
    then mean-pools over the T buckets to get a fixed-size embedding.

    With SEQ_LEN=3, this is extremely fast — the SSM scan is 3 steps.
    The real payoff comes if you later increase bucket count.
    """

    N_LAYERS = 2

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        self.layers = nn.ModuleList([
            MambaBlock(D_MODEL, MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND)
            for _ in range(self.N_LAYERS)
        ])
        self.norm_out = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (BN, T, F) → (BN, D_MODEL)"""
        h = self.input_proj(x)       # (BN, T, D_MODEL)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h)
        return h.mean(dim=1)         # mean pool over buckets → (BN, D_MODEL)


# ====================================================================
# STANDARD TEMPORAL ENCODERS (fallback when USE_MAMBA=False)
# ====================================================================

class CNNTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        C = CNN_CHANNELS
        self.branches = nn.ModuleList([
            nn.Conv1d(input_dim, C, kernel_size=k, padding=k - 1)
            for k in CNN_KERNELS
        ])
        self.norm = nn.BatchNorm1d(C * len(CNN_KERNELS))
        self.proj = nn.Linear(C * len(CNN_KERNELS), D_MODEL)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        outs = [F.gelu(conv(xt)[..., :SEQ_LEN]) for conv in self.branches]
        h = torch.cat(outs, dim=1)
        h = self.norm(h)
        return self.proj(self.pool(h).squeeze(-1))


class TransformerTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        self.pos_embed  = nn.Parameter(torch.randn(1, SEQ_LEN + 1, D_MODEL) * 0.02)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, D_MODEL))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD,
            dropout=TF_DROPOUT, batch_first=True,
            dim_feedforward=D_MODEL * 4,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=NUM_TF_LAYERS)
        self.norm_out = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BN = x.size(0)
        h  = self.input_proj(x)
        cls = self.cls_token.expand(BN, -1, -1)
        h   = torch.cat([cls, h], dim=1) + self.pos_embed[:, :SEQ_LEN + 1]
        h   = self.encoder(h)
        return self.norm_out(h[:, 0, :])


def build_temporal_encoder(input_dim: int) -> nn.Module:
    if USE_MAMBA:
        return MambaTemporalEncoder(input_dim)
    if TEMPORAL_ENCODER == "cnn":
        return CNNTemporalEncoder(input_dim)
    if TEMPORAL_ENCODER == "transformer":
        return TransformerTemporalEncoder(input_dim)
    raise ValueError(f"Unknown TEMPORAL_ENCODER: {TEMPORAL_ENCODER!r}")


# ====================================================================
# DIRECTION 1 — LEARNED GRAPH (NRI-style edge MLP)
# ====================================================================

class LearnedGraphBuilder(nn.Module):
    """
    NRI-style dynamic graph constructor.

    Given node embeddings h of shape (B*N, D_MODEL), scores every directed
    pair (i→j) with a small edge MLP operating on [h_i || h_j].

    The top-K edges per node are kept.  Edge weights are softmax-normalised
    scores — these act as attention-like edge attributes for GATv2.

    This replaces the static rv_long similarity graph with a fully
    data-driven graph that changes every time_id.
    """

    def __init__(self) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(D_MODEL * 2, EDGE_MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(EDGE_MLP_HIDDEN, EDGE_MLP_HIDDEN // 2),
            nn.GELU(),
            nn.Linear(EDGE_MLP_HIDDEN // 2, 1),
        )

    def forward(
        self,
        h:         torch.Tensor,    # (B*N, D_MODEL)
        live_mask: torch.Tensor,    # (B*N,) bool
        N:         int,             # stocks per sample
        B:         int,             # batch size
        device:    torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_index : (2, E)  long — global flat indices into B*N nodes
            edge_attr  : (E, 1)  float — normalised edge scores
        """
        all_src, all_dst, all_attr = [], [], []

        for b in range(B):
            start = b * N
            end   = start + N

            # Live nodes for this sample
            live_local = live_mask[start:end]                # (N,)
            live_ids   = torch.where(live_local)[0]          # local indices
            n_live     = live_ids.numel()

            if n_live < 2:
                continue

            h_live = h[start + live_ids]                     # (n_live, D_MODEL)

            # Build all pairs: [h_i || h_j] for i≠j
            hi = h_live.unsqueeze(1).expand(-1, n_live, -1)  # (n_live, n_live, D)
            hj = h_live.unsqueeze(0).expand(n_live, -1, -1)  # (n_live, n_live, D)
            pairs = torch.cat([hi, hj], dim=-1)              # (n_live, n_live, 2D)

            # Score all pairs
            scores = self.edge_mlp(pairs).squeeze(-1)        # (n_live, n_live)
            scores.fill_diagonal_(float("-inf"))              # no self-loops

            # Top-K per source node
            k = min(EDGE_TOPK, n_live - 1)
            topk_scores, topk_idx = torch.topk(scores, k, dim=1)  # (n_live, k)

            # Normalise with softmax (temperature scaling)
            edge_weights = F.softmax(topk_scores / EDGE_TEMPERATURE, dim=-1)

            # Build edge_index in local-live space → remap to global B*N space
            src_local = torch.arange(n_live, device=device).unsqueeze(1).expand(-1, k)
            dst_local = topk_idx                             # (n_live, k)

            src_global = start + live_ids[src_local.reshape(-1)]
            dst_global = start + live_ids[dst_local.reshape(-1)]

            all_src.append(src_global)
            all_dst.append(dst_global)
            all_attr.append(edge_weights.reshape(-1))

        if not all_src:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0, 1), dtype=torch.float32, device=device),
            )

        edge_index = torch.stack([
            torch.cat(all_src),
            torch.cat(all_dst),
        ])
        edge_attr = torch.cat(all_attr).unsqueeze(-1)        # (E, 1)
        return edge_index, edge_attr


# ====================================================================
# STATIC GRAPH CACHE (fallback when USE_LEARNED_GRAPH=False)
# ====================================================================

class GraphCache:
    """
    Precomputes a static top-K graph per time_id based on rv_long similarity.
    Used only when USE_LEARNED_GRAPH=False.
    """

    _META_KEY = "__meta__"

    def __init__(self, features_df: pd.DataFrame, num_stocks: int) -> None:
        self._meta = {"top_k": TOP_K, "use_edge_attr": USE_EDGE_ATTR, "version": "v5"}
        self._num_stocks = num_stocks
        self._cache: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._build_or_load(features_df)

    def _build_or_load(self, features_df: pd.DataFrame) -> None:
        cache_path = Path(CACHE_FILE)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    saved = pickle.load(f)
                stored_meta = saved.pop(self._META_KEY, None)
                if stored_meta == self._meta:
                    self._cache = saved
                    print(f"Graph cache loaded ({len(self._cache)} time_ids)")
                    return
                print("Cache config mismatch — rebuilding.")
            except Exception as exc:
                print(f"Cache load failed ({exc}) — rebuilding.")
        self._build(features_df)
        self._save()

    def _build(self, features_df: pd.DataFrame) -> None:
        print("Building static graph cache ...")
        device = torch.device("cpu")

        # Use realized_vol_b2 (last bucket) as the similarity signal
        sim_col = "realized_vol_b2" if "realized_vol_b2" in features_df.columns else None

        for i, (tid, group) in enumerate(features_df.groupby("time_id")):
            group = group.set_index("stock_id").reindex(range(self._num_stocks))

            if sim_col:
                rv_vec = torch.tensor(
                    group[sim_col].fillna(0.0).values.astype(np.float32)
                )
            else:
                rv_vec = torch.zeros(self._num_stocks)

            valid_mask    = rv_vec > 0
            valid_ids     = torch.where(valid_mask)[0]
            rv_valid      = rv_vec[valid_mask]
            N_valid       = rv_valid.size(0)

            if N_valid < 2:
                self._cache[int(tid)] = (
                    torch.zeros((2, 0), dtype=torch.long),
                    None,
                )
                continue

            k    = min(TOP_K, N_valid - 1)
            diff = torch.abs(rv_valid.unsqueeze(0) - rv_valid.unsqueeze(1))
            sim  = torch.exp(-diff)
            sim.fill_diagonal_(0.0)

            _, topk_idx = torch.topk(sim, k, dim=1)
            src = torch.arange(N_valid).unsqueeze(1).expand(-1, k).reshape(-1)
            dst = topk_idx.reshape(-1)
            ei  = torch.stack([valid_ids[src], valid_ids[dst]])
            ea  = sim[src, dst].unsqueeze(-1) if USE_EDGE_ATTR else None

            self._cache[int(tid)] = (ei, ea)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1} time_ids processed")

        print(f"Static graph cache built: {len(self._cache)} time_ids")

    def _save(self) -> None:
        payload = {self._META_KEY: self._meta, **self._cache}
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(payload, f)
        print(f"Graph cache saved -> {CACHE_FILE}")

    def get(self, tid: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if int(tid) not in self._cache:
            return torch.zeros((2, 0), dtype=torch.long), None
        return self._cache[int(tid)]


# ====================================================================
# DIRECTION 2 — CROSS-STOCK ATTENTION BLOCK
# ====================================================================

class CrossStockAttention(nn.Module):
    """
    Multi-head self-attention where the sequence dimension is stocks (N),
    not time.  Each stock attends to all other stocks simultaneously,
    learning a dense global dependency structure with no graph topology needed.

    For N=112 stocks this is 112² ≈ 12K pairs — negligible cost.

    Pre-norm architecture with residual connection for training stability.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm" : nn.LayerNorm(D_MODEL),
                "attn" : nn.MultiheadAttention(
                    embed_dim=D_MODEL,
                    num_heads=CROSS_ATTN_HEADS,
                    dropout=CROSS_ATTN_DROPOUT,
                    batch_first=True,
                ),
                "ffn_norm": nn.LayerNorm(D_MODEL),
                "ffn"     : nn.Sequential(
                    nn.Linear(D_MODEL, D_MODEL * 4),
                    nn.GELU(),
                    nn.Dropout(CROSS_ATTN_DROPOUT),
                    nn.Linear(D_MODEL * 4, D_MODEL),
                    nn.Dropout(CROSS_ATTN_DROPOUT),
                ),
            })
            for _ in range(NUM_CROSS_ATTN_LAYERS)
        ])

    def forward(
        self,
        h:         torch.Tensor,    # (B, N, D_MODEL)
        live_mask: torch.Tensor,    # (B, N) bool — dead stocks masked from attention
    ) -> torch.Tensor:
        """Returns (B, N, D_MODEL)."""
        # Build key padding mask: True means IGNORE (PyTorch convention)
        key_padding_mask = ~live_mask    # (B, N) — True for dead stocks

        for layer in self.layers:
            # Pre-norm attention
            h_norm = layer["norm"](h)
            attn_out, _ = layer["attn"](
                h_norm, h_norm, h_norm,
                key_padding_mask=key_padding_mask,
            )
            h = h + attn_out

            # Pre-norm FFN
            h = h + layer["ffn"](layer["ffn_norm"](h))

            # Zero out dead stocks after each layer
            h = h * live_mask.unsqueeze(-1).float()

        return h


# ====================================================================
# RESIDUAL MLP HEAD
# ====================================================================

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResidualMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden, dropout) for _ in range(depth)
        ])
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


# ====================================================================
# MAIN MODEL — SpatioTemporalGNN v5
# ====================================================================

class SpatioTemporalGNN(nn.Module):
    """
    Full forward pass:

    1.  Input LayerNorm on raw sequence
    2.  Temporal encoder:
          USE_MAMBA=True   → MambaTemporalEncoder  (Direction 3)
          USE_MAMBA=False  → CNN or Transformer
        → (B*N, D_MODEL)

    3.  Stock embedding concat + projection → (B*N, D_MODEL)

    4.  Pre-GNN LayerNorm

    5.  Graph message passing:
          USE_LEARNED_GRAPH=True  → LearnedGraphBuilder (Direction 1)
          USE_LEARNED_GRAPH=False → static cached graph + GATv2

    6.  Cross-stock attention (Direction 2, if USE_CROSS_ATTN=True)

    7.  Global mean readout concat → (B, N, D_MODEL)

    8.  HAR skip branch (parallel):
          realized_vol_b0/b1/b2 → Linear(SEQ_LEN, 1) → har_pred (B, N)

    9.  Blend: sigmoid(har_weight) * har_pred + (1-w) * gnn_pred
        har_weight is a learned scalar — starts at 0.5, converges to
        whatever split minimises the loss. If HAR dominates, w→1.
        If GNN adds value, w settles below 1.
    """

    def __init__(self, input_dim: int, num_stocks: int) -> None:
        super().__init__()
        self.num_stocks = num_stocks

        # 1. Input norm
        self.input_norm = nn.LayerNorm(input_dim)

        # 2. Temporal encoder
        self.temporal_encoder = build_temporal_encoder(input_dim)

        # 3. Stock embedding
        self.stock_embed = nn.Embedding(num_stocks, STOCK_EMBED_DIM)
        self.embed_proj  = nn.Linear(D_MODEL + STOCK_EMBED_DIM, D_MODEL)

        # 4. Pre-GNN norm
        self.pre_gnn_norm = nn.LayerNorm(D_MODEL)

        # 5. Graph component
        if USE_LEARNED_GRAPH:
            self.graph_builder = LearnedGraphBuilder()
        else:
            self.graph_builder = None

        self.gat_layers: nn.ModuleList = nn.ModuleList()
        self.gnn_norms:  nn.ModuleList = nn.ModuleList()
        for _ in range(NUM_GNN_LAYERS):
            self.gat_layers.append(GATv2Conv(
                in_channels=D_MODEL,
                out_channels=D_MODEL // GAT_HEADS,
                heads=GAT_HEADS,
                concat=True,
                dropout=GAT_DROPOUT,
                edge_dim=1,
            ))
            self.gnn_norms.append(nn.LayerNorm(D_MODEL))

        # 6. Cross-stock attention (Direction 2)
        if USE_CROSS_ATTN:
            self.cross_attn = CrossStockAttention()
        else:
            self.cross_attn = None

        # 7. Global readout
        self.global_proj = nn.Linear(D_MODEL * 2, D_MODEL)

        # 8. Head
        self.head = ResidualMLPHead(D_MODEL, MLP_HIDDEN, MLP_DEPTH, FINAL_DROPOUT)

        # HAR skip branch — linear model over realized_vol buckets.
        # Feature index 1 = realized_vol (matches FEATURE_COLS order).
        # Maps (SEQ_LEN,) RV values directly to a scalar log(RV) prediction.
        # The GNN head then learns the residual on top of this HAR solution,
        # which is far easier than learning the full mapping from scratch.
        self.har_branch = nn.Linear(SEQ_LEN, 1, bias=True)
        self.har_weight = nn.Parameter(torch.tensor(0.5))  # learnable blend

    def forward(
        self,
        X:          torch.Tensor,                       # (B, N, T, F)
        edge_index: Optional[torch.Tensor],             # (2, E) or None
        edge_attr:  Optional[torch.Tensor],             # (E, 1) or None
        stock_ids:  torch.Tensor,                       # (B, N)
    ) -> torch.Tensor:
        B, N, T, F_dim = X.shape
        device = X.device

        X_flat    = X.view(B * N, T, F_dim)
        live_mask = X_flat.abs().sum(dim=(1, 2)) > 0   # (B*N,)

        # 1. Input norm (live nodes only)
        X_normed = torch.zeros_like(X_flat)
        if live_mask.any():
            live_x = X_flat[live_mask]
            L, T_, F_ = live_x.shape
            normed = self.input_norm(live_x.reshape(L * T_, F_)).reshape(L, T_, F_)
            X_normed[live_mask] = normed

        # 2. Temporal encoding
        h = torch.zeros(B * N, D_MODEL, device=device, dtype=X.dtype)
        if live_mask.any():
            h[live_mask] = self.temporal_encoder(X_normed[live_mask])

        # 3. Stock embedding concat
        s_emb = self.stock_embed(stock_ids.view(B * N))
        h = self.embed_proj(torch.cat([h, s_emb], dim=-1))
        h = h * live_mask.unsqueeze(-1).float()

        # 4. Pre-GNN norm
        if live_mask.any():
            h[live_mask] = self.pre_gnn_norm(h[live_mask])

        # 5. Graph message passing
        if USE_LEARNED_GRAPH:
            # Direction 1: build graph dynamically from current embeddings
            edge_index, edge_attr = self.graph_builder(h, live_mask, N, B, device)
        # else: use the static edge_index / edge_attr passed in from collate_fn

        if edge_index is not None and edge_index.numel() > 0:
            for gat, norm in zip(self.gat_layers, self.gnn_norms):
                residual = h
                h_n = torch.zeros_like(h)
                if live_mask.any():
                    h_n[live_mask] = norm(h[live_mask])
                h = gat(h_n, edge_index, edge_attr=edge_attr)
                h = F.gelu(h)
                h = F.dropout(h, p=GAT_DROPOUT, training=self.training)
                h = h + residual
                h = h * live_mask.unsqueeze(-1).float()

        # 6. Cross-stock attention (Direction 2)
        if USE_CROSS_ATTN and self.cross_attn is not None:
            h_s    = h.view(B, N, D_MODEL)
            live_s = live_mask.view(B, N)
            h_s    = self.cross_attn(h_s, live_s)
            h      = h_s.view(B * N, D_MODEL)

        # 7. Global mean readout
        h_s    = h.view(B, N, D_MODEL)
        live_s = live_mask.view(B, N).float().unsqueeze(-1)
        live_n = live_s.sum(dim=1, keepdim=True).clamp(min=1)
        g_ctx  = (h_s * live_s).sum(dim=1, keepdim=True) / live_n
        g_ctx  = g_ctx.expand(-1, N, -1)
        h_aug  = self.global_proj(torch.cat([h_s, g_ctx], dim=-1))
        h_aug  = h_aug * live_s

        # HAR skip branch — log(realized_vol) → log(RV_target)
        # Feature index 1 = realized_vol (raw, always >= 0).
        # Must log-transform to match the log-log HAR specification:
        # both predictors and target are in log space (~-18 to 0).
        # Using raw RV (~1e-3) against log(RV) target (~-8) would give
        # the linear layer huge coefficients and produce out-of-range preds.
        # Clamp to RV_FLOOR before log to avoid log(0) = -inf for dead buckets.
        rv_buckets = X[:, :, :, 1]                              # (B, N, T) raw RV
        rv_log     = torch.log(rv_buckets.clamp(min=RV_FLOOR))  # (B, N, T) log scale
        har_pred   = self.har_branch(rv_log).squeeze(-1)        # (B, N)

        # 8. Head — learns residual on top of HAR prediction
        out = self.head(h_aug).squeeze(-1)                 # (B, N)

        # Blend: sigmoid-gated so har_weight stays in (0, 1)
        w   = torch.sigmoid(self.har_weight)
        out = w * har_pred + (1 - w) * out

        out = out * live_mask.view(B, N).float()
        return out


# ====================================================================
# DATASET
# ====================================================================

class StockDataset(Dataset):
    """
    One sample = one time_id.
    Features shape: (num_stocks, SEQ_LEN, INPUT_DIM)
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df:  pd.DataFrame,
        graph_cache: Optional[GraphCache],
        num_stocks:  int,
    ) -> None:
        self.graph_cache = graph_cache
        self.num_stocks  = num_stocks

        print("  Pivoting features into sequence tensors ...")
        self._X: Dict[int, np.ndarray] = {}
        self._y: Dict[int, np.ndarray] = {}

        # features_df columns: time_id, stock_id, log_return_b0, ..., total_volume_b2
        bucket_cols = [f"{feat}_b{b}" for b in range(SEQ_LEN) for feat in FEATURE_COLS]
        # Only keep cols that exist
        bucket_cols = [c for c in bucket_cols if c in features_df.columns]

        for tid, grp in features_df.groupby("time_id"):
            arr = np.zeros((num_stocks, SEQ_LEN, INPUT_DIM), dtype=np.float32)
            for _, row in grp.iterrows():
                sid = int(row["stock_id"])
                if sid >= num_stocks:
                    continue
                for b in range(SEQ_LEN):
                    for fi, feat in enumerate(FEATURE_COLS):
                        col = f"{feat}_b{b}"
                        if col in row.index:
                            arr[sid, b, fi] = float(row[col]) if pd.notna(row[col]) else 0.0
            self._X[int(tid)] = arr

        for tid, grp in targets_df.groupby("time_id"):
            grp = grp.set_index("stock_id").reindex(range(num_stocks))
            self._y[int(tid)] = grp["log_rv_target"].fillna(0.0).to_numpy(dtype=np.float32)

        self.time_ids: List[int] = sorted(set(self._X.keys()) & set(self._y.keys()))
        print(f"  Dataset ready: {len(self.time_ids)} time_ids")

    def __len__(self) -> int:
        return len(self.time_ids)

    def __getitem__(self, idx: int):
        tid = self.time_ids[idx]

        X = torch.tensor(self._X[tid], dtype=torch.float32)
        X = torch.nan_to_num(X, nan=0.0)

        y = torch.tensor(self._y[tid], dtype=torch.float32)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if self.graph_cache is not None:
            edge_index, edge_attr = self.graph_cache.get(tid)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = None

        stock_ids = torch.arange(self.num_stocks, dtype=torch.long)
        return X, y, edge_index, edge_attr, tid, stock_ids


# ====================================================================
# COLLATE
# ====================================================================

def collate_fn(batch):
    X_list, y_list, ei_list, ea_list, tid_list, sid_list = zip(*batch)
    N = X_list[0].size(0)

    X_batch   = torch.stack(X_list)
    y_batch   = torch.stack(y_list)
    sid_batch = torch.stack(sid_list)

    ei_parts, ea_parts = [], []
    for i, (ei, ea) in enumerate(zip(ei_list, ea_list)):
        if ei.numel() == 0:
            continue
        ei_parts.append(ei + i * N)
        if ea is not None:
            ea_parts.append(ea)

    edge_index_batch = (
        torch.cat(ei_parts, dim=1) if ei_parts
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr_batch = torch.cat(ea_parts, dim=0) if ea_parts else None

    return X_batch, y_batch, edge_index_batch, edge_attr_batch, list(tid_list), sid_batch


# ====================================================================
# LOSSES
# ====================================================================

class RMSPELoss(nn.Module):
    """
    RMSPE on RV scale over live stocks only.

    Live = target is not 0.0 AND not at the RV_FLOOR sentinel
    (log(1e-8) = -18.42).  Both represent dead/missing stocks and must
    be excluded — using only true_log != 0.0 misses the floor case,
    causing exp(-18) ~ 0 in the denominator and exploding train loss.

    Predictions are clamped to [-20, 5] in log space before exp() to
    prevent inf/nan from an untrained model in early epochs.
    """

    LOG_RV_FLOOR = LOG_RV_FLOOR   # from config — log(1e-8) ~ -18.42
    LOG_PRED_MIN = -20.0
    LOG_PRED_MAX =   5.0

    def __init__(self, eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_log: torch.Tensor, true_log: torch.Tensor) -> torch.Tensor:
        # Exclude dead stocks: target == 0 or at RV_FLOOR
        live = (true_log != 0.0) & (true_log > self.LOG_RV_FLOOR)
        if not live.any():
            return pred_log.sum() * 0.0

        p = pred_log[live].clamp(self.LOG_PRED_MIN, self.LOG_PRED_MAX)
        t = true_log[live]

        pred_rv = torch.exp(p)
        true_rv = torch.exp(t).clamp(min=self.eps)
        return torch.sqrt(torch.mean(((pred_rv - true_rv) / true_rv) ** 2))


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on log-RV scale, excluding dead stocks (target == 0 or at RV_FLOOR)."""
    live = (target != 0.0) & (target > LOG_RV_FLOOR)
    if not live.any():
        return pred.sum() * 0.0
    p = pred[live].clamp(-20.0, 5.0)
    return F.mse_loss(p, target[live])


# ====================================================================
# TRAIN / EVAL
# ====================================================================

def run_epoch(
    model:     SpatioTemporalGNN,
    loader:    DataLoader,
    loss_fn,
    device:    torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_n = 0.0, 0

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for X_b, y_b, ei_b, ea_b, _, sid_b in loader:
            X_b   = X_b.to(device)
            y_b   = y_b.to(device)
            ei_b  = ei_b.to(device)
            ea_b  = ea_b.to(device) if ea_b is not None else None
            sid_b = sid_b.to(device)

            if is_train:
                optimizer.zero_grad()

            pred = model(X_b, ei_b, ea_b, sid_b)
            loss = loss_fn(pred, y_b)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * X_b.size(0)
            total_n    += X_b.size(0)

    return total_loss / max(total_n, 1)


# ====================================================================
# METRICS
# ====================================================================

def compute_metrics(pred_log: np.ndarray, true_log: np.ndarray) -> Dict[str, float]:
    pred_rv = np.clip(np.exp(pred_log), EPS, None)
    true_rv = np.clip(np.exp(true_log), EPS, None)
    resid   = pred_rv - true_rv
    rmse    = float(np.sqrt(np.mean(resid ** 2)))
    mape    = float(np.mean(np.abs(resid) / true_rv) * 100)
    rmspe   = float(np.sqrt(np.mean((resid / true_rv) ** 2)) * 100)
    ratio   = true_rv / pred_rv
    qlike   = float(np.mean(ratio - np.log(ratio) - 1))
    return {"QLIKE": qlike, "RMSE": rmse, "MAPE%": mape, "RMSPE%": rmspe}


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:
    print("=" * 65)
    print("SpatioTemporalGNN v5")
    print(f"  Device          : {DEVICE}")
    print(f"  USE_MAMBA       : {USE_MAMBA}  (Direction 3 — SSM encoder)")
    print(f"  USE_LEARNED_GRAPH: {USE_LEARNED_GRAPH}  (Direction 1 — NRI graph)")
    print(f"  USE_CROSS_ATTN  : {USE_CROSS_ATTN}  (Direction 2 — stock attention)")
    print(f"  SEQ_LEN / INPUT_DIM: {SEQ_LEN} / {INPUT_DIM}")
    print(f"  D_MODEL         : {D_MODEL}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load parquet data
    # ------------------------------------------------------------------
    data_dir = Path(FEATURES_DIR)
    print("Loading parquet files ...")
    train_feat = pd.read_parquet(data_dir / "train_features.parquet", engine="pyarrow")
    train_targ = pd.read_parquet(data_dir / "train_targets.parquet",  engine="pyarrow")
    val_feat   = pd.read_parquet(data_dir / "val_features.parquet",   engine="pyarrow")
    val_targ   = pd.read_parquet(data_dir / "val_targets.parquet",    engine="pyarrow")
    test_feat  = pd.read_parquet(data_dir / "test_features.parquet",  engine="pyarrow")
    test_targ  = pd.read_parquet(data_dir / "test_targets.parquet",   engine="pyarrow")

    all_stocks = (
        set(train_feat["stock_id"].unique()) |
        set(val_feat["stock_id"].unique()) |
        set(test_feat["stock_id"].unique())
    )
    num_stocks = max(all_stocks) + 1
    print(f"num_stocks = {num_stocks}")

    # ------------------------------------------------------------------
    # 2. Graph cache (static, only needed when USE_LEARNED_GRAPH=False)
    # ------------------------------------------------------------------
    if not USE_LEARNED_GRAPH:
        all_feat    = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
        graph_cache = GraphCache(all_feat, num_stocks)
        del all_feat
    else:
        graph_cache = None
        print("Learned graph active — skipping static graph cache.")

    # ------------------------------------------------------------------
    # 3. Datasets
    # ------------------------------------------------------------------
    print("Building train dataset ...")
    train_ds = StockDataset(train_feat, train_targ, graph_cache, num_stocks)
    print("Building val dataset ...")
    val_ds   = StockDataset(val_feat,   val_targ,   graph_cache, num_stocks)
    print("Building test dataset ...")
    test_ds  = StockDataset(test_feat,  test_targ,  graph_cache, num_stocks)

    print(
        f"Time IDs — train: {len(train_ds)} | "
        f"val: {len(val_ds)} | test: {len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ------------------------------------------------------------------
    # 4. Model, optimiser, scheduler, loss
    # ------------------------------------------------------------------
    device = torch.device(DEVICE)
    model  = SpatioTemporalGNN(input_dim=INPUT_DIM, num_stocks=num_stocks).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    if LOSS_FN == "rmspe":
        loss_fn = RMSPELoss()
    elif LOSS_FN == "mse":
        loss_fn = masked_mse_loss
    else:
        raise ValueError(f"Unknown LOSS_FN: {LOSS_FN!r}")

    print(f"Loss: {LOSS_FN.upper()} (dead stocks masked)")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    model_path = Path(MODEL_DIR) / "best_model_v5.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0         = _time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss   = run_epoch(model, val_loader,   loss_fn, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train {LOSS_FN.upper()}: {train_loss:.6f} | "
            f"Val   {LOSS_FN.upper()}: {val_loss:.6f} | "
            f"LR: {lr:.2e} | {_time.time() - t0:.1f}s"
        )

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print("  -> best model saved")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Best val {LOSS_FN.upper()}: {best_val:.6f}")

    # ------------------------------------------------------------------
    # 6. Test evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_pred, all_true, all_tids, all_live = [], [], [], []

    with torch.no_grad():
        for X_b, y_b, ei_b, ea_b, tid_b, sid_b in test_loader:
            X_b   = X_b.to(device)
            ei_b  = ei_b.to(device)
            ea_b  = ea_b.to(device) if ea_b is not None else None
            sid_b = sid_b.to(device)

            live = X_b.abs().sum(dim=(2, 3)) > 0
            pred = model(X_b, ei_b, ea_b, sid_b)

            all_pred.append(pred.cpu().numpy())
            all_true.append(y_b.numpy())
            all_tids.extend(tid_b)
            all_live.append(live.cpu().numpy())

    pred_log  = np.concatenate(all_pred, axis=0)
    true_log  = np.concatenate(all_true, axis=0)
    live_mask = np.concatenate(all_live, axis=0)

    metrics = compute_metrics(pred_log[live_mask], true_log[live_mask])
    print("\nTest metrics (live nodes only):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # ------------------------------------------------------------------
    # 7. Export predictions
    # ------------------------------------------------------------------
    n_tids, n_stocks = pred_log.shape
    rows = [
        {
            "time_id"    : all_tids[i],
            "stock_id"   : s,
            "pred_log_rv": pred_log[i, s],
            "true_log_rv": true_log[i, s],
            "pred_rv"    : float(np.exp(pred_log[i, s])),
            "true_rv"    : float(np.exp(true_log[i, s])),
        }
        for i in range(n_tids)
        for s in range(n_stocks)
        if live_mask[i, s]
    ]
    out_path = Path(MODEL_DIR) / "predictions_v5.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Predictions -> {out_path} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()