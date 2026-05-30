"""
dashapp.py  —  Volatility Regime Monitor  (v3)
===============================================
Three-tab layout:
  Tab 1 – MONITOR    : heatmap + summary cards + risk table + track record
  Tab 2 – RV BUCKETS : previous-3-bucket RV viewer for a selected stock × snapshot
  Tab 3 – LEADERBOARD: global model ranking table across all regimes + snapshots

Reads: dashboard_data.parquet
Columns: stock_id, time_id, rv_b0..rv_b3, rv_target,
         actual_rv, pred_har, pred_lgbm, pred_garch, pred_gnn, regime

Run:
  pip install dash dash-bootstrap-components plotly pandas pyarrow
  python dashapp.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, ctx, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# ── Load data ──────────────────────────────────────────────────────────────────

DATA_PATH = Path("dashboard_data.parquet")
df = pd.read_parquet(DATA_PATH)

STOCKS   = sorted(df["stock_id"].unique())
TIME_IDS = sorted(df["time_id"].unique())
EPS      = 1e-8

MODEL_COLS = {
    "HAR-RV":   "pred_har_base",   # baseline HAR (Fold 0)
    "HAR-X":    "pred_har",        # extended HAR variant
    "LightGBM": "pred_lgbm",
    "GARCH":    "pred_garch",
    "GNN":      "pred_gnn",
}
AVAILABLE_MODELS = {k: v for k, v in MODEL_COLS.items()
                    if v in df.columns and df[v].notna().any()}

RV_BUCKET_COLS = [c for c in ["rv_b0", "rv_b1", "rv_b2", "rv_b3"] if c in df.columns]

MODEL_COLORS = {
    "HAR-RV":   "#f97316",   # orange — baseline HAR
    "HAR-X":    "#5b8dee",   # blue   — extended HAR
    "LightGBM": "#3ecf8e",
    "GARCH":    "#ff6b6b",
    "GNN":      "#c084fc",
}
REGIME_COLORS = {"calm": "#3ecf8e", "normal": "#5b8dee",
                 "elevated": "#f59e0b", "stressed": "#ef4444"}
REGIME_BADGE  = {"calm": "success", "normal": "info",
                 "elevated": "warning", "stressed": "danger"}

# ── Design tokens ──────────────────────────────────────────────────────────────

BG_PAGE  = "#060a10"
BG_CARD  = "#0c1220"
BG_CHART = "#08101c"
BG_TAB   = "#0a1018"
BORDER   = "#182035"
BORDER_HI= "#2a3a5c"
TEXT_DIM = "#4a6080"
TEXT_MID = "#8099b8"
TEXT_HI  = "#dce8f5"
ACCENT   = "#3b7ef8"
ACCENT2  = "#0f3060"

PLOT_TMPL = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CHART,
    font=dict(family="'JetBrains Mono', monospace", color=TEXT_MID, size=11),
    xaxis=dict(gridcolor="#0e1e30", zerolinecolor="#182035", linecolor=BORDER,
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#0e1e30", zerolinecolor="#182035", linecolor=BORDER,
               tickfont=dict(size=10)),
    margin=dict(l=52, r=16, t=36, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
))

# ── Helpers ────────────────────────────────────────────────────────────────────

def rmspe(true, pred):
    mask = pred.notna() & true.notna()
    if mask.sum() == 0:
        return np.nan
    t = true[mask].clip(lower=EPS)
    p = pred[mask].clip(lower=EPS)
    return float(np.sqrt(np.mean(((p - t) / t) ** 2)) * 100)

def bias_pct(true, pred):
    mask = pred.notna() & true.notna()
    if mask.sum() == 0:
        return np.nan
    t = true[mask].clip(lower=EPS)
    p = pred[mask].clip(lower=EPS)
    return float(((p - t) / t).mean() * 100)

def hit_rate(true, pred):
    mask = pred.notna() & true.notna()
    if mask.sum() < 2:
        return np.nan
    d_true = np.sign(true[mask].values[1:] - true[mask].values[:-1])
    d_pred = np.sign(pred[mask].values[1:] - pred[mask].values[:-1])
    return float((d_true == d_pred).mean() * 100)

def qlike(true, pred):
    """QLIKE loss: mean(σ²/h − log(σ²/h) − 1). Lower is better."""
    mask = pred.notna() & true.notna()
    if mask.sum() == 0:
        return np.nan
    t = true[mask].clip(lower=EPS)
    h = pred[mask].clip(lower=EPS)
    return float(np.mean(t / h - np.log(t / h) - 1))

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def card(children, style=None, glow=None):
    base = {
        "backgroundColor": BG_CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "18px",
        "position": "relative",
    }
    if glow:
        base["boxShadow"] = f"0 0 28px 0 {glow}22"
        base["borderColor"] = glow + "44"
    if style:
        base.update(style)
    return html.Div(children, style=base)

def dim(text, size="11px"):
    return html.Span(text, style={"color": TEXT_DIM, "fontSize": size,
                                  "letterSpacing": "0.03em"})

def hi(text, size="22px", color=TEXT_HI, mono=True):
    return html.Span(text, style={
        "color": color, "fontSize": size, "fontWeight": "600",
        **({"fontFamily": "'JetBrains Mono', monospace"} if mono else {})
    })

def label(text):
    return html.Div(text, style={
        "color": TEXT_DIM, "fontSize": "11px", "letterSpacing": "1px",
        "fontWeight": "500", "marginBottom": "6px", "textTransform": "uppercase",
    })

def section_title(text):
    return html.Div(text, style={
        "fontFamily": "'Syne', sans-serif", "fontSize": "12px", "fontWeight": "700",
        "color": TEXT_MID, "letterSpacing": "1.5px", "marginBottom": "12px",
        "textTransform": "uppercase",
    })

def metric_chip(lbl, val, accent=BORDER, unit=""):
    return html.Div([
        html.Div(lbl, style={"color": TEXT_DIM, "fontSize": "11px",
                             "letterSpacing": "0.8px", "marginBottom": "5px"}),
        html.Div([
            html.Span(val, style={"color": TEXT_HI, "fontSize": "19px",
                                  "fontWeight": "600",
                                  "fontFamily": "'JetBrains Mono', monospace"}),
            html.Span(f" {unit}", style={"color": TEXT_DIM, "fontSize": "11px"})
            if unit else None,
        ]),
    ], style={
        "backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
        "borderTop": f"2px solid {accent}", "borderRadius": "8px",
        "padding": "12px 14px", "flex": "1",
    })

# ── App ────────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    assets_folder="asset",
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&"
        "family=Syne:wght@400;600;800&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
app.title = "Vol Regime Monitor"
server = app.server  # expose Flask server for Gunicorn

# ══════════════════════════════════════════════════════════════════════════════
# Shared Controls
# ══════════════════════════════════════════════════════════════════════════════

HEADER = html.Div([
    html.Div([
        html.Div([
            html.Div("VOL REGIME MONITOR", style={
                "fontFamily": "'Syne', sans-serif", "fontWeight": "800",
                "fontSize": "17px", "color": TEXT_HI, "letterSpacing": "4px",
            }),
            html.Div(
                f"{len(STOCKS)} stocks · {len(TIME_IDS):,} snapshots · "
                f"{len(AVAILABLE_MODELS)} models loaded",
                style={"color": TEXT_MID, "fontSize": "12px", "marginTop": "3px"}
            ),
        ]),
        html.Div([
            html.Div("MODEL", style={"color": TEXT_DIM, "fontSize": "11px",
                                     "letterSpacing": "1px", "marginBottom": "7px"}),
            dbc.RadioItems(
                id="model-sel",
                options=[{"label": html.Span([
                    html.Span("▮ ", style={"color": MODEL_COLORS.get(m, TEXT_MID)}), m
                ]), "value": m} for m in AVAILABLE_MODELS],
                value=list(AVAILABLE_MODELS.keys())[0],
                inline=True,
                style={"display": "flex", "gap": "20px"},
                className="model-radio",
            ),
        ]),
    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "16px 28px", "borderBottom": f"1px solid {BORDER}",
        "backgroundColor": BG_CARD,
    }),

], style={"position": "sticky", "top": "0", "zIndex": "100"})

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — MONITOR
# ══════════════════════════════════════════════════════════════════════════════

TAB_MONITOR = html.Div([

    # ── Snapshot bar (Monitor tab only) ───────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("SNAPSHOT", style={"color": TEXT_DIM, "fontSize": "11px",
                                        "letterSpacing": "1px", "marginBottom": "8px"}),
            dcc.Dropdown(
                id="tid-dd",
                options=[{"label": f"Snapshot {t}", "value": i}
                         for i, t in enumerate(TIME_IDS)],
                value=0, clearable=False,
                style={"width": "220px", "backgroundColor": BG_CARD,
                       "border": f"1px solid {BORDER}", "color": TEXT_HI},
                className="snapshot-dd",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),
        html.Div(id="snapshot-badge", style={"whiteSpace": "nowrap"}),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "24px",
        "padding": "10px 28px", "borderBottom": f"1px solid {BORDER}",
        "backgroundColor": "#08111a",
    }),

    html.Div([

        # ── Left ──────────────────────────────────────────────────────────────
        html.Div([
            html.Div(id="summary-cards", style={
                "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "10px", "marginBottom": "14px",
            }),

            card([
                html.Div([
                    section_title("Portfolio Risk Map"),
                    html.Div("ranked by predicted RV · colour = regime",
                             style={"color": TEXT_MID, "fontSize": "12px",
                                    "marginTop": "-8px", "marginBottom": "10px"}),
                ]),
                dcc.Graph(id="heatmap", config={"displayModeBar": False},
                          style={"height": "200px"}),
            ], style={"marginBottom": "14px"}),

            card([
                html.Div([
                    html.Div([
                        section_title("Risk Table"),
                        html.Div("click ★ to watchlist · click row to inspect",
                                 style={"color": TEXT_MID, "fontSize": "12px",
                                        "marginTop": "-8px", "marginBottom": "10px"}),
                    ]),
                    dbc.Button("Export CSV", id="export-btn", size="sm", outline=True,
                               color="secondary",
                               style={"fontSize": "9px", "letterSpacing": "1px",
                                      "padding": "4px 12px"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "flex-start"}),
                html.Div(id="risk-table",
                         style={"overflowY": "auto", "maxHeight": "340px"}),
            ]),
        ], style={"flex": "1 1 0", "minWidth": "0", "display": "flex",
                  "flexDirection": "column"}),

        # ── Right ─────────────────────────────────────────────────────────────
        html.Div([
            card([
                html.Div([
                    section_title("Stock Track Record"),
                    dbc.Checkbox(
                        id="compare-models-toggle", label="Compare all models",
                        value=False,
                        style={"fontSize": "10px", "color": TEXT_DIM},
                        className="compare-toggle",
                    ),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center"}),
                html.Div([
                    html.Div([
                        label("STOCK"),
                        dcc.Dropdown(
                            id="stock-sel",
                            options=[{"label": f"Stock {s}", "value": s} for s in STOCKS],
                            value=STOCKS[0], clearable=False,
                            style={"backgroundColor": BG_CARD,
                                   "border": f"1px solid {BORDER}",
                                   "color": TEXT_HI, "width": "145px"},
                            className="snapshot-dd",
                        ),
                    ]),
                    html.Div(id="stock-stats", style={"marginLeft": "auto"}),
                ], style={"display": "flex", "alignItems": "flex-end",
                          "gap": "16px", "marginBottom": "12px"}),
                dcc.Graph(id="track-scatter", config={"displayModeBar": False},
                          style={"height": "280px"}),
                dcc.Graph(id="error-dist", config={"displayModeBar": False},
                          style={"height": "175px"}),
            ], style={"marginBottom": "14px"}),

            card([
                section_title("Regime Distribution"),
                dcc.Graph(id="regime-bars", config={"displayModeBar": False},
                          style={"height": "155px"}),
            ]),
        ], style={"width": "430px", "flexShrink": "0", "display": "flex",
                  "flexDirection": "column"}),
    ], style={"display": "flex", "gap": "14px", "padding": "14px 28px",
              "flex": "1", "minHeight": "0", "overflowY": "auto"}),
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — RV BUCKETS
# (removed: metric chips row, regime trajectory chart)
# ══════════════════════════════════════════════════════════════════════════════

TAB_BUCKETS = html.Div([
    html.Div([

        # Controls row
        card([
            html.Div([
                html.Div([
                    label("SELECT STOCK"),
                    dcc.Dropdown(
                        id="bkt-stock-sel",
                        options=[{"label": f"Stock {s}", "value": s} for s in STOCKS],
                        value=STOCKS[0], clearable=False,
                        style={"backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
                               "color": TEXT_HI, "width": "160px"},
                        className="snapshot-dd",
                    ),
                ]),
                html.Div([
                    label("SELECT SNAPSHOT"),
                    dcc.Dropdown(
                        id="bkt-tid-dd",
                        options=[{"label": f"Snapshot {t}", "value": t}
                                 for t in TIME_IDS],
                        value=TIME_IDS[0], clearable=False,
                        style={"backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
                               "color": TEXT_HI, "width": "180px"},
                        className="snapshot-dd",
                    ),
                ]),
                html.Div([
                    label("CHART TYPE"),
                    dbc.RadioItems(
                        id="bkt-chart-type",
                        options=[
                            {"label": "Bar", "value": "bar"},
                            {"label": "Line", "value": "line"},
                        ],
                        value="line", inline=True,
                        style={"display": "flex", "gap": "14px"},
                        className="model-radio",
                    ),
                ]),
                html.Div([
                    label("SYNC WITH MAIN"),
                    dbc.Checkbox(
                        id="bkt-sync-toggle",
                        label="Use main snapshot",
                        value=True,
                        style={"fontSize": "10px", "color": TEXT_DIM},
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "flex-end",
                      "gap": "28px", "flexWrap": "wrap"}),
        ], style={"marginBottom": "14px"}),

        # Charts row
        html.Div([
            # Main bucket bar / line chart — PRIMARY focus
            html.Div([
                card([
                    html.Div([
                        html.Div("RV BUCKET HISTORY", style={
                            "fontFamily": "'Syne', sans-serif", "fontSize": "14px",
                            "fontWeight": "800", "color": TEXT_HI,
                            "letterSpacing": "3px", "marginBottom": "4px",
                        }),
                        html.Div(
                            "rv_b0 = most recent · rv_b3 = oldest · rv_target = forecast target · pred = model",
                            style={"color": TEXT_MID, "fontSize": "12px", "marginBottom": "10px"},
                        ),
                    ]),
                    dcc.Graph(id="bkt-main-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ], glow=ACCENT),
            ], style={"flex": "2", "minWidth": "0"}),

            # Cross-stock distribution
            html.Div([
                card([
                    section_title("Cross-Stock RV Profile"),
                    html.Div(
                        "selected stock vs portfolio distribution at this snapshot",
                        style={"color": TEXT_MID, "fontSize": "12px",
                               "marginTop": "-8px", "marginBottom": "10px"}
                    ),
                    dcc.Graph(id="bkt-cross-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ]),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "14px"}),

        # ── Regime Leaderboard (prominent, full-width) ────────────────────────
        card([
            html.Div([
                html.Div("MODEL LEADERBOARD BY REGIME", style={
                    "fontFamily": "'Syne', sans-serif", "fontSize": "13px",
                    "fontWeight": "800", "color": TEXT_HI,
                    "letterSpacing": "3px", "marginBottom": "3px",
                }),
                html.Div("QLIKE  ·  lower is better  ·  updates with stock",
                         style={"color": TEXT_MID, "fontSize": "12px",
                                "letterSpacing": "0.5px"}),
            ], style={"marginBottom": "14px"}),
            html.Div(id="regime-leaderboard"),
        ], glow=ACCENT),

    ], style={"padding": "14px 28px", "overflowY": "auto"}),
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — LEADERBOARD
# (SCOPE replaced with STOCK filter)
# ══════════════════════════════════════════════════════════════════════════════

TAB_LEADERBOARD = html.Div([
    html.Div([

        # Filter bar
        card([
            html.Div([
                html.Div([
                    label("REGIME FILTER"),
                    dbc.Checklist(
                        id="lb-regime-filter",
                        options=[{"label": r.capitalize(), "value": r}
                                 for r in ["calm", "normal", "elevated", "stressed"]],
                        value=["calm", "normal", "elevated", "stressed"],
                        inline=True,
                        style={"display": "flex", "gap": "14px"},
                        className="model-radio",
                    ),
                ]),
                html.Div([
                    label("METRIC"),
                    dbc.RadioItems(
                        id="lb-metric",
                        options=[
                            {"label": "QLIKE",    "value": "qlike"},
                            {"label": "RMSPE %",  "value": "rmspe"},
                            {"label": "Bias %",   "value": "bias"},
                            {"label": "Hit Rate", "value": "hit"},
                            {"label": "Corr",     "value": "corr"},
                        ],
                        value="qlike", inline=True,
                        style={"display": "flex", "gap": "14px"},
                        className="model-radio",
                    ),
                ]),
                # SCOPE replaced with STOCK filter
                html.Div([
                    label("STOCK FILTER"),
                    dcc.Dropdown(
                        id="lb-stock-filter",
                        options=[{"label": f"S{s:03d}", "value": s} for s in STOCKS],
                        value=[],
                        multi=True,
                        placeholder="All stocks",
                        clearable=True,
                        style={"backgroundColor": BG_CARD,
                               "border": f"1px solid {BORDER}",
                               "color": TEXT_HI, "minWidth": "220px",
                               "maxWidth": "400px"},
                        className="snapshot-dd",
                    ),
                ]),
            ], style={"display": "flex", "gap": "36px", "flexWrap": "wrap",
                      "alignItems": "flex-end"}),
        ], style={"marginBottom": "14px"}),

        # Winner banner + score chips
        html.Div(id="lb-winner-banner", style={"marginBottom": "14px"}),

        # Two panels side-by-side
        html.Div([
            html.Div([
                card([
                    section_title("Model Rankings"),
                    html.Div(id="lb-table"),
                ]),
            ], style={"flex": "1", "minWidth": "0"}),

            html.Div([
                card([
                    section_title("Score Comparison"),
                    dcc.Graph(id="lb-bar-chart", config={"displayModeBar": False},
                              style={"height": "340px"}),
                ]),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "14px"}),

        # Per-snapshot QLIKE line chart (dynamic to stock filter)
        card([
            html.Div(id="lb-time-title", style={
                "fontFamily": "'Syne', sans-serif", "fontSize": "11px", "fontWeight": "700",
                "color": TEXT_DIM, "letterSpacing": "2.5px", "marginBottom": "12px",
                "textTransform": "uppercase",
            }),
            html.Div(id="lb-time-subtitle",
                     style={"color": TEXT_DIM, "fontSize": "10px",
                            "marginTop": "-8px", "marginBottom": "10px"}),
            dcc.Loading(
                dcc.Graph(id="lb-time-chart", config={"displayModeBar": False},
                          style={"height": "240px"}),
                type="circle", color=ACCENT,
            ),
        ]),

    ], style={"padding": "14px 28px", "overflowY": "auto"}),
])

# ══════════════════════════════════════════════════════════════════════════════
# Root layout
# ══════════════════════════════════════════════════════════════════════════════

TAB_STYLE = {
    "fontFamily": "'Syne', sans-serif",
    "fontSize": "11px",
    "letterSpacing": "2px",
    "fontWeight": "600",
    "color": TEXT_DIM,
    "backgroundColor": BG_TAB,
    "border": "none",
    "padding": "12px 24px",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": TEXT_HI,
    "borderBottom": f"2px solid {ACCENT}",
    "backgroundColor": BG_CARD,
}

app.layout = html.Div([
    dcc.Store(id="tid-index", data=0),
    dcc.Store(id="watchlist-store", data=[], storage_type="local"),
    dcc.Download(id="download-csv"),

    HEADER,

    dcc.Tabs(
        id="main-tabs",
        value="tab-buckets",
        children=[
            dcc.Tab(label="◎  RV BUCKETS",  value="tab-buckets",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="◈  MONITOR",     value="tab-monitor",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="◆  LEADERBOARD", value="tab-leaderboard",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
        ],
        style={"borderBottom": f"1px solid {BORDER}",
               "backgroundColor": BG_TAB},
        colors={"border": BORDER, "primary": ACCENT,
                "background": BG_TAB},
    ),

    html.Div(id="tab-content",
             style={"flex": "1", "overflowY": "auto", "backgroundColor": BG_PAGE}),

], style={"backgroundColor": BG_PAGE, "minHeight": "100vh",
          "display": "flex", "flexDirection": "column",
          "fontFamily": "'JetBrains Mono', monospace"})


# ══════════════════════════════════════════════════════════════════════════════
# Common Callbacks
# ══════════════════════════════════════════════════════════════════════════════

@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-monitor":
        return TAB_MONITOR
    if tab == "tab-buckets":
        return TAB_BUCKETS
    return TAB_LEADERBOARD


@callback(Output("tid-index", "data"),
          Input("tid-dd", "value"))
def dd_to_store(dd_val):
    return dd_val if dd_val is not None else 0


@callback(Output("snapshot-badge", "children"), Input("tid-index", "data"))
def update_badge(idx):
    tid  = TIME_IDS[idx]
    snap = df[df["time_id"] == tid]
    n_stressed = (snap["regime"] == "stressed").sum()
    n_elevated = (snap["regime"] == "elevated").sum()
    color = "#ef4444" if n_stressed > 5 else "#f59e0b" if n_elevated > 10 else "#3ecf8e"
    label_txt = "HIGH STRESS" if n_stressed > 5 else "ELEVATED" if n_elevated > 10 else "CALM"
    return html.Div([
        html.Div(f"time_id {tid}", style={"color": TEXT_DIM, "fontSize": "10px",
                                          "textAlign": "right", "marginBottom": "3px"}),
        html.Div(label_txt, style={
            "color": color, "fontSize": "11px", "fontWeight": "600",
            "letterSpacing": "2px", "border": f"1px solid {color}44",
            "padding": "3px 12px", "borderRadius": "4px",
            "backgroundColor": color + "11",
        }),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 Callbacks — MONITOR
# All per-model error metrics now use QLIKE instead of RMSPE
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("summary-cards", "children"),
    Input("tid-index", "data"), Input("model-sel", "value"),
)
def update_summary(idx, model):
    tid  = TIME_IDS[idx]
    snap = df[df["time_id"] == tid]
    col  = AVAILABLE_MODELS.get(model, "")

    mean_rv    = snap["actual_rv"].mean()
    pct_elev   = ((snap["regime"].isin(["elevated", "stressed"])).sum()
                  / max(len(snap), 1)) * 100
    mean_pred  = snap[col].mean() if col and col in snap.columns else np.nan
    # ── Changed from RMSPE to QLIKE ──
    model_ql   = (qlike(snap["actual_rv"], snap[col])
                  if col and col in snap.columns else np.nan)

    def stat_card(lbl, val, unit="", color=TEXT_HI, accent=BORDER):
        return html.Div([
            html.Div(lbl, style={"color": TEXT_DIM, "fontSize": "11px",
                                 "letterSpacing": "0.8px", "marginBottom": "6px"}),
            html.Div([
                html.Span(val, style={"color": color, "fontSize": "20px",
                                      "fontWeight": "600",
                                      "fontFamily": "'JetBrains Mono', monospace"}),
                html.Span(f" {unit}",
                          style={"color": TEXT_DIM, "fontSize": "10px"}) if unit else None,
            ]),
        ], style={
            "backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
            "borderTop": f"2px solid {accent}", "borderRadius": "8px",
            "padding": "12px 14px",
        })

    sc = "#ef4444" if pct_elev > 20 else "#f59e0b" if pct_elev > 10 else "#3ecf8e"
    return [
        stat_card("MEAN ACTUAL RV",
                  f"{mean_rv:.5f}" if not np.isnan(mean_rv) else "—",
                  "", TEXT_HI, "#5b8dee"),
        stat_card("MEAN PRED RV",
                  f"{mean_pred:.5f}" if not np.isnan(mean_pred) else "—",
                  "", TEXT_HI, MODEL_COLORS.get(model, BORDER)),
        stat_card("% ELEVATED+",
                  f"{pct_elev:.0f}", "%", sc, sc),
        # ── Changed from SNAPSHOT RMSPE to SNAPSHOT QLIKE ──
        stat_card("SNAPSHOT QLIKE",
                  f"{model_ql:.4f}" if not np.isnan(model_ql) else "—",
                  "", TEXT_HI, BORDER),
    ]


@callback(
    Output("heatmap", "figure"),
    Input("tid-index", "data"), Input("model-sel", "value"),
)
def update_heatmap(idx, model):
    tid  = TIME_IDS[idx]
    snap = df[df["time_id"] == tid].copy()
    col  = AVAILABLE_MODELS.get(model, "")

    if col and col in snap.columns:
        snap  = snap.sort_values(col, ascending=False)
        z_vals = snap[col].fillna(0).values
    else:
        snap  = snap.sort_values("actual_rv", ascending=False)
        z_vals = snap["actual_rv"].values

    regime_map = {"calm": 0, "normal": 1, "elevated": 2, "stressed": 3}
    r_vals = snap["regime"].map(regime_map).fillna(1).values

    n = len(snap)
    n_cols = 14
    n_rows = int(np.ceil(n / n_cols))
    pad = n_rows * n_cols - n

    z_pad   = np.concatenate([r_vals,                   np.full(pad, np.nan)]).reshape(n_rows, n_cols)
    rv_pad  = np.concatenate([z_vals,                   np.full(pad, np.nan)]).reshape(n_rows, n_cols)
    sid_pad = np.concatenate([snap["stock_id"].values,  np.full(pad, -1, int)]).reshape(n_rows, n_cols)

    hover = [
        [f"Stock {sid_pad[r,c]}<br>Pred RV: {rv_pad[r,c]:.5f}<br>"
         f"Regime: {snap['regime'].values[r*n_cols+c] if r*n_cols+c < n else ''}"
         if sid_pad[r, c] >= 0 else ""
         for c in range(n_cols)]
        for r in range(n_rows)
    ]

    fig = go.Figure(go.Heatmap(
        z=z_pad,
        colorscale=[[0.0, REGIME_COLORS["calm"]], [0.33, REGIME_COLORS["normal"]],
                    [0.66, REGIME_COLORS["elevated"]], [1.0, REGIME_COLORS["stressed"]]],
        zmin=0, zmax=3,
        hovertext=hover, hoverinfo="text",
        showscale=False, xgap=2, ygap=2,
    ))
    fig.update_layout(
        template=PLOT_TMPL, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
    )
    return fig


@callback(
    Output("risk-table", "children"),
    Input("tid-index", "data"), Input("model-sel", "value"),
    Input("watchlist-store", "data"),
)
def update_risk_table(idx, model, watchlist):
    tid      = TIME_IDS[idx]
    snap     = df[df["time_id"] == tid].copy()
    col      = AVAILABLE_MODELS.get(model, "")
    watchlist = watchlist or []

    snap["_pred"] = snap[col] if col and col in snap.columns else np.nan
    snap["_err"]  = snap["_pred"] - snap["actual_rv"]
    # per-row QLIKE (pointwise)
    def _ql_pt(t, h):
        t, h = max(float(t), EPS), max(float(h), EPS)
        return t / h - np.log(t / h) - 1
    snap["_ql"] = snap.apply(
        lambda r: _ql_pt(r["actual_rv"], r["_pred"])
        if not (np.isnan(r["actual_rv"]) or np.isnan(r["_pred"])) else np.nan,
        axis=1,
    )
    snap["_wtch"] = snap["stock_id"].isin(watchlist)
    snap = snap.sort_values(["_wtch", "_pred"],
                            ascending=[False, False], na_position="last")

    th   = {"color": TEXT_DIM, "fontSize": "11px", "letterSpacing": "0.5px",
            "padding": "6px 10px", "borderBottom": f"1px solid {BORDER}",
            "fontWeight": "500", "textAlign": "right"}
    th_l = {**th, "textAlign": "left"}

    header = html.Thead(html.Tr([
        html.Th("",        style={**th_l, "width": "28px"}),
        html.Th("STOCK",   style=th_l),
        html.Th("REGIME",  style=th_l),
        html.Th("PRED RV", style=th),
        html.Th("TRUE RV", style=th),
        html.Th("ERROR",   style=th),
        html.Th("QLIKE",   style=th),
    ]))

    rows = []
    for _, row in snap.iterrows():
        regime  = row.get("regime", "normal")
        err     = row["_err"]
        ql      = row["_ql"]
        is_w    = row["stock_id"] in watchlist
        ec      = ("#ef4444" if abs(err) > 0.005 else
                   "#f59e0b" if abs(err) > 0.002 else TEXT_MID) if not np.isnan(err) else TEXT_MID
        ql_col  = ("#ef4444" if ql > 0.05 else
                   "#f59e0b" if ql > 0.01 else TEXT_MID) if not np.isnan(ql) else TEXT_MID
        err_str = (f"{'+' if err >= 0 else ''}{err:.5f}"
                   if not np.isnan(err) else "—")
        ql_str  = f"{ql:.4f}" if not np.isnan(ql) else "—"
        pred_str= f"{row['_pred']:.5f}" if not np.isnan(row['_pred']) else "—"

        td   = {"fontSize": "11px", "padding": "5px 10px",
                "borderBottom": f"1px solid {BORDER}",
                "fontFamily": "'JetBrains Mono', monospace", "textAlign": "right"}
        td_l = {**td, "textAlign": "left"}

        star = html.Button(
            "★" if is_w else "☆",
            id={"type": "star-btn", "index": int(row["stock_id"])},
            style={"background": "none", "border": "none", "cursor": "pointer",
                   "color": "#f59e0b" if is_w else TEXT_DIM,
                   "fontSize": "14px", "padding": "0"},
            n_clicks=0,
        )
        rows.append(html.Tr([
            html.Td(star, style={**td_l, "width": "28px", "paddingRight": "0"}),
            html.Td(f"S{row['stock_id']:03d}",
                    style={**td_l, "color": TEXT_HI, "fontWeight": "500"}),
            html.Td(dbc.Badge(regime, color=REGIME_BADGE.get(regime, "secondary"),
                              style={"fontSize": "11px"}), style=td_l),
            html.Td(pred_str, style={**td, "color": TEXT_HI}),
            html.Td(f"{row['actual_rv']:.5f}", style=td),
            html.Td(err_str, style={**td, "color": ec}),
            html.Td(ql_str,  style={**td, "color": ql_col}),
        ], style={"backgroundColor": "#111c30" if is_w else "transparent"}))

    return dbc.Table([header, html.Tbody(rows)],
                     bordered=False, hover=True, responsive=True, size="sm",
                     style={"backgroundColor": BG_CARD, "color": TEXT_MID,
                            "marginBottom": "0", "fontSize": "11px"})


@callback(
    Output("track-scatter", "figure"),
    Output("stock-stats", "children"),
    Input("stock-sel", "value"), Input("model-sel", "value"),
    Input("compare-models-toggle", "value"),
)
def update_track_scatter(stock_id, model, compare_all):
    sdf    = df[df["stock_id"] == stock_id].copy()
    col    = AVAILABLE_MODELS.get(model, "")
    rv_max = sdf["actual_rv"].quantile(0.99)
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=[0, rv_max], y=[0, rv_max], mode="lines",
                             line=dict(dash="dot", color=BORDER, width=1),
                             showlegend=False, hoverinfo="skip"))

    if compare_all:
        stats_rows = []
        for m_name, m_col in AVAILABLE_MODELS.items():
            if m_col not in sdf.columns:
                continue
            mask = sdf[m_col].notna()
            mc   = MODEL_COLORS.get(m_name, TEXT_MID)
            fig.add_trace(go.Scatter(
                x=sdf.loc[mask, m_col], y=sdf.loc[mask, "actual_rv"],
                mode="markers",
                marker=dict(size=5 if m_name == model else 4, color=mc,
                            opacity=0.8 if m_name == model else 0.5,
                            line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
                hovertemplate=(f"<b>{m_name}</b><br>Snapshot: %{{customdata}}<br>"
                               "Pred: %{x:.5f}<br>Actual: %{y:.5f}<extra></extra>"),
                customdata=sdf.loc[mask, "time_id"], name=m_name,
            ))
            # ── Changed from RMSPE to QLIKE ──
            ql   = qlike(sdf["actual_rv"], sdf[m_col])
            corr = np.corrcoef(sdf.loc[mask, m_col], sdf.loc[mask, "actual_rv"])[0, 1]
            is_sel = m_name == model
            stats_rows.append(html.Div([
                html.Span("● ", style={"color": mc, "fontSize": "10px"}),
                html.Span(f"{m_name}: ",
                          style={"color": TEXT_HI if is_sel else TEXT_DIM,
                                 "fontSize": "10px",
                                 "fontWeight": "500" if is_sel else "400"}),
                html.Span(f"QL={ql:.4f} ",
                          style={"color": TEXT_HI if is_sel else TEXT_MID,
                                 "fontSize": "10px"}),
                html.Span(f"(r={corr:.2f})",
                          style={"color": TEXT_DIM, "fontSize": "11px"}),
            ], style={"marginBottom": "2px"}))
        stats = html.Div(stats_rows, style={"textAlign": "right"})
        fig.update_layout(legend=dict(orientation="h", y=-0.18, font=dict(size=9),
                                      bgcolor="rgba(0,0,0,0)"))
    else:
        if col and col in sdf.columns:
            mask      = sdf[col].notna()
            colors_pt = [REGIME_COLORS.get(r, TEXT_MID) for r in sdf.loc[mask, "regime"]]
            fig.add_trace(go.Scatter(
                x=sdf.loc[mask, col], y=sdf.loc[mask, "actual_rv"],
                mode="markers",
                marker=dict(size=5, color=colors_pt, opacity=0.75,
                            line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
                hovertemplate=("Snapshot: %{customdata}<br>"
                               "Pred: %{x:.5f}<br>Actual: %{y:.5f}<extra></extra>"),
                customdata=sdf.loc[mask, "time_id"], showlegend=False,
            ))
            # ── Changed from RMSPE to QLIKE ──
            ql   = qlike(sdf["actual_rv"], sdf[col])
            corr = np.corrcoef(sdf.loc[mask, col], sdf.loc[mask, "actual_rv"])[0, 1]
            stats = html.Div([
                html.Div([dim("QLIKE "), hi(f"{ql:.4f}", "14px")],
                         style={"marginBottom": "2px"}),
                html.Div([dim("CORR  "), hi(f"{corr:.3f}", "14px")]),
            ], style={"textAlign": "right"})
        else:
            stats = html.Div()
        for reg, rc in REGIME_COLORS.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(size=7, color=rc), name=reg))
        fig.update_layout(legend=dict(orientation="h", y=-0.2, font=dict(size=9)))

    fig.update_layout(
        template=PLOT_TMPL,
        title=dict(text=f"Stock {stock_id} — pred vs actual across all snapshots",
                   font=dict(size=11, color=TEXT_DIM), x=0),
        xaxis=dict(title=f"{model} predicted RV", range=[0, rv_max]),
        yaxis=dict(title="Actual RV", range=[0, rv_max]),
        margin=dict(l=52, r=8, t=32, b=48),
    )
    return fig, stats


@callback(Output("error-dist", "figure"),
          Input("stock-sel", "value"), Input("model-sel", "value"))
def update_error_dist(stock_id, model):
    sdf   = df[df["stock_id"] == stock_id].copy()
    col   = AVAILABLE_MODELS.get(model, "")
    color = MODEL_COLORS.get(model, TEXT_MID)
    fig   = go.Figure()
    if col and col in sdf.columns:
        resid = (sdf[col] - sdf["actual_rv"]).dropna()
        fig.add_trace(go.Histogram(x=resid, nbinsx=40,
                                   marker_color=color, opacity=0.8))
        fig.add_vline(x=0, line_dash="dot", line_color=TEXT_DIM, line_width=1)
    fig.update_layout(
        template=PLOT_TMPL,
        title=dict(text="Residual distribution  (pred − actual)",
                   font=dict(size=10, color=TEXT_DIM), x=0),
        xaxis=dict(title="Residual"),
        yaxis=dict(title="Count"),
        showlegend=False,
        margin=dict(l=52, r=8, t=28, b=36), bargap=0.05,
    )
    return fig


@callback(Output("regime-bars", "figure"), Input("tid-index", "data"))
def update_regime_bars(idx):
    tid    = TIME_IDS[idx]
    snap   = df[df["time_id"] == tid]
    counts = snap["regime"].value_counts().reindex(
        ["calm", "normal", "elevated", "stressed"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[REGIME_COLORS[r] for r in counts.index],
        text=counts.values, textposition="outside",
        textfont=dict(size=10, color=TEXT_MID),
    ))
    fig.update_layout(
        template=PLOT_TMPL,
        title=dict(text="Regime breakdown · this snapshot",
                   font=dict(size=10, color=TEXT_DIM), x=0),
        yaxis=dict(title="# stocks", range=[0, max(counts.values) * 1.25]),
        showlegend=False, margin=dict(l=40, r=8, t=28, b=28),
    )
    return fig


@callback(Output("watchlist-store", "data"),
          Input({"type": "star-btn", "index": ALL}, "n_clicks"),
          State("watchlist-store", "data"), prevent_initial_call=True)
def toggle_watchlist(n_clicks_list, watchlist):
    watchlist = list(watchlist or [])
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict) and triggered.get("type") == "star-btn":
        sid = triggered["index"]
        if sid in watchlist:
            watchlist.remove(sid)
        else:
            watchlist.append(sid)
    return watchlist


@callback(Output("download-csv", "data"),
          Input("export-btn", "n_clicks"),
          State("tid-index", "data"), State("model-sel", "value"),
          prevent_initial_call=True)
def export_csv(n_clicks, idx, model):
    # Guard: only fire on an actual button click, not on component mount
    if not n_clicks:
        raise PreventUpdate
    tid  = TIME_IDS[idx]
    snap = df[df["time_id"] == tid].copy()
    col  = AVAILABLE_MODELS.get(model, "")
    snap["pred_rv"] = snap[col] if col and col in snap.columns else np.nan
    snap["error"]   = snap["pred_rv"] - snap["actual_rv"]
    snap["rpe_pct"] = (snap["error"].abs() / snap["actual_rv"].clip(lower=EPS)) * 100
    out = snap[["stock_id", "regime", "pred_rv", "actual_rv", "error", "rpe_pct"]]\
            .sort_values("pred_rv", ascending=False)
    return dcc.send_data_frame(out.to_csv, f"risk_table_{tid}_{model}.csv", index=False)


@callback(Output("regime-leaderboard", "children"),
          Input("bkt-stock-sel", "value"))
def update_regime_leaderboard(stock_id):
    # Stock selected → all snapshots for that stock, grouped by regime
    # No stock      → aggregate over the full dataset
    if stock_id is not None:
        data = df[df["stock_id"] == stock_id]
        context = f"Stock {stock_id:03d}  ·  all snapshots"
    else:
        data = df
        context = "All stocks  ·  all snapshots"

    context_div = html.Div(
        f"Viewing: {context}",
        style={"color": ACCENT, "fontSize": "11px", "letterSpacing": "0.5px",
               "marginBottom": "12px", "fontWeight": "500"},
    )

    th   = {"color": TEXT_DIM, "fontSize": "11px", "letterSpacing": "0.5px",
            "padding": "8px 10px", "borderBottom": f"1px solid {BORDER}",
            "fontWeight": "500", "textAlign": "center"}
    th_l = {**th, "textAlign": "left"}

    header = html.Thead(html.Tr([
        html.Th("REGIME", style=th_l),
        *[html.Th(m, style={**th, "color": MODEL_COLORS.get(m, TEXT_DIM)})
          for m in AVAILABLE_MODELS],
        html.Th("BEST", style=th),
    ]))

    rows = []
    for regime in ["calm", "normal", "elevated", "stressed"]:
        rdf = data[data["regime"] == regime]
        if len(rdf) == 0:
            continue
        scores = {m: qlike(rdf["actual_rv"], rdf[c])
                  for m, c in AVAILABLE_MODELS.items() if c in rdf.columns}
        if not scores:
            continue
        best = min(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else np.inf)
        td   = {"fontSize": "12px", "padding": "7px 10px",
                "borderBottom": f"1px solid {BORDER}",
                "fontFamily": "'JetBrains Mono', monospace", "textAlign": "center"}
        td_l = {**td, "textAlign": "left"}
        cells = [html.Td(dbc.Badge(regime, color=REGIME_BADGE.get(regime, "secondary"),
                                   style={"fontSize": "11px"}), style=td_l)]
        for m in AVAILABLE_MODELS:
            s = scores.get(m)
            cells.append(html.Td(
                f"{s:.4f}" if s is not None and not np.isnan(s) else "—",
                style={**td,
                       "color": "#3ecf8e" if m == best else TEXT_MID,
                       "fontWeight": "600" if m == best else "400"}
            ))
        cells.append(html.Td(
            html.Span(best, style={"color": MODEL_COLORS.get(best, TEXT_HI),
                                   "fontWeight": "700", "fontSize": "11px"}),
            style=td,
        ))
        rows.append(html.Tr(cells))

    if not rows:
        return [context_div,
                html.Div("No data for this selection",
                         style={"color": TEXT_MID, "fontSize": "12px"})]
    table = dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True,
                      responsive=True, size="sm",
                      style={"backgroundColor": BG_CARD, "color": TEXT_MID,
                             "marginBottom": "0", "fontSize": "12px"})
    return [context_div, table]


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 Callbacks — RV BUCKETS
# (removed: bkt-metric-chips callback, bkt-trajectory callback)
# (added: model-sel input to bkt-main-chart to overlay predicted value)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bkt-tid-dd", "value"),
    Input("bkt-sync-toggle", "value"),
    Input("tid-index", "data"),
    State("bkt-tid-dd", "value"),
)
def sync_bucket_tid(sync_on, main_idx, current_val):
    if sync_on:
        return TIME_IDS[main_idx]
    return current_val


@callback(
    Output("bkt-main-chart", "figure"),
    Input("bkt-stock-sel", "value"),
    Input("bkt-tid-dd", "value"),
    Input("bkt-chart-type", "value"),
    Input("model-sel", "value"),          # ← added to overlay predicted value
)
def update_bkt_main(stock_id, tid, chart_type, model):
    row = df[(df["stock_id"] == stock_id) & (df["time_id"] == tid)]

    bucket_labels    = ["rv_b3\n(t−4)", "rv_b2\n(t−3)", "rv_b1\n(t−2)", "rv_b0\n(t−1)"]
    bucket_cols_rev  = list(reversed(RV_BUCKET_COLS))   # oldest → newest
    bucket_colors    = ["#c084fc", "#f59e0b", "#3ecf8e", "#5b8dee"]

    fig = go.Figure()

    if len(row) == 0:
        fig.update_layout(template=PLOT_TMPL,
                          title=dict(text="No data", font=dict(size=10, color=TEXT_DIM)))
        return fig

    row_data  = row.iloc[0]
    bkt_vals  = [row_data.get(c, np.nan) for c in bucket_cols_rev]
    actual_rv = row_data.get("actual_rv", np.nan)
    rv_target = row_data.get("rv_target", np.nan)

    # ── Predicted value for selected model ──
    pred_col  = AVAILABLE_MODELS.get(model, "")
    pred_val  = row_data.get(pred_col, np.nan) if pred_col else np.nan
    pred_color = MODEL_COLORS.get(model, "#ffffff")

    if chart_type == "bar":
        for i, (lbl, val, col_) in enumerate(zip(bucket_labels, bkt_vals, bucket_colors)):
            fig.add_trace(go.Bar(
                x=[lbl], y=[val if not np.isnan(val) else 0],
                marker_color=col_,
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                name=lbl.replace("\n", " "),
                text=[f"{val:.5f}" if not np.isnan(val) else "—"],
                textposition="outside",
                textfont=dict(size=10, color=TEXT_MID),
                showlegend=True,
            ))
        # Predicted value — same bar format, model colour
        if not np.isnan(pred_val):
            fig.add_trace(go.Bar(
                x=[f"{model}\n(pred)"],
                y=[pred_val],
                marker_color=pred_color,
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                name=f"{model} pred",
                text=[f"{pred_val:.5f}"],
                textposition="outside",
                textfont=dict(size=10, color=pred_color),
                showlegend=True,
            ))
    else:
        clean_labels = ["t−4", "t−3", "t−2", "t−1"]
        fig.add_trace(go.Scatter(
            x=clean_labels, y=bkt_vals, mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8, color=bucket_colors,
                        line=dict(width=1.5, color="rgba(0,0,0,0.4)")),
            name="RV buckets",
            hovertemplate="%{x}: %{y:.5f}<extra></extra>",
        ))
        # Predicted value — dashed extension from t−1 to pred, star marker at end
        if not np.isnan(pred_val) and not np.isnan(bkt_vals[-1]):
            fig.add_trace(go.Scatter(
                x=["t−1", "pred"],
                y=[bkt_vals[-1], pred_val],
                mode="lines+markers",
                line=dict(color=pred_color, width=2, dash="dash"),
                marker=dict(size=[0, 12], color=pred_color,
                            symbol=["circle", "star"],
                            line=dict(width=1.5, color="rgba(0,0,0,0.4)")),
                name=f"{model} pred",
                hovertemplate="pred: %{y:.5f}<extra></extra>",
            ))

    # ── Reference lines: actual RV and target RV ──
    if not np.isnan(actual_rv):
        fig.add_hline(y=actual_rv, line_dash="dot",
                      line_color="#ef4444", line_width=1.5,
                      annotation_text=f"Actual: {actual_rv:.5f}",
                      annotation_font=dict(size=10, color="#ef4444"),
                      annotation_position="right")
    if not np.isnan(rv_target):
        fig.add_hline(y=rv_target, line_dash="dash",
                      line_color="#f59e0b", line_width=1.5,
                      annotation_text=f"Target: {rv_target:.5f}",
                      annotation_font=dict(size=10, color="#f59e0b"),
                      annotation_position="right")

    fig.update_layout(
        template=PLOT_TMPL,
        title=dict(
            text=f"Stock {stock_id} · Snapshot {tid} — Previous RV Buckets",
            font=dict(size=11, color=TEXT_DIM), x=0,
        ),
        xaxis=dict(title="Lag"),
        yaxis=dict(title="Realised Volatility"),
        barmode="group",
        legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        margin=dict(l=52, r=120, t=32, b=60),
    )
    return fig


@callback(
    Output("bkt-cross-chart", "figure"),
    Input("bkt-stock-sel", "value"),
    Input("bkt-tid-dd", "value"),
)
def update_bkt_cross(stock_id, tid):
    snap = df[df["time_id"] == tid]
    fig  = go.Figure()

    if len(RV_BUCKET_COLS) == 0:
        return fig

    bucket_colors = {"rv_b0": "#5b8dee", "rv_b1": "#3ecf8e",
                     "rv_b2": "#f59e0b", "rv_b3": "#c084fc"}
    labels_map = {"rv_b0": "t−1", "rv_b1": "t−2", "rv_b2": "t−3", "rv_b3": "t−4"}

    for bc in RV_BUCKET_COLS:
        if bc not in snap.columns:
            continue
        fig.add_trace(go.Box(
            y=snap[bc].dropna(), name=labels_map.get(bc, bc),
            marker_color=bucket_colors.get(bc, TEXT_MID),
            line_color=bucket_colors.get(bc, TEXT_MID),
            fillcolor=hex_to_rgba(bucket_colors.get(bc, "#888888"), 0.2),
            boxpoints=False, showlegend=True,
        ))

    row = snap[snap["stock_id"] == stock_id]
    if len(row) > 0:
        row = row.iloc[0]
        for bc in RV_BUCKET_COLS:
            val = row.get(bc, np.nan)
            if not np.isnan(val):
                fig.add_trace(go.Scatter(
                    x=[labels_map.get(bc, bc)], y=[val],
                    mode="markers",
                    marker=dict(size=10, color="#ffffff", symbol="diamond",
                                line=dict(width=2, color=bucket_colors.get(bc, TEXT_MID))),
                    name=f"S{stock_id:03d}",
                    showlegend=bc == RV_BUCKET_COLS[0],
                ))

    fig.update_layout(
        template=PLOT_TMPL,
        title=dict(text="Portfolio distribution per bucket",
                   font=dict(size=10, color=TEXT_DIM), x=0),
        xaxis=dict(title="Lag"),
        yaxis=dict(title="RV"),
        legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        margin=dict(l=52, r=8, t=32, b=56),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 Callbacks — LEADERBOARD
# SCOPE replaced with stock filter; time chart is now dynamic to stock filter
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(subset, metric):
    scores = {}
    for m_name, m_col in AVAILABLE_MODELS.items():
        if m_col not in subset.columns:
            continue
        true, pred = subset["actual_rv"], subset[m_col]
        if metric == "qlike":
            scores[m_name] = qlike(true, pred)
        elif metric == "rmspe":
            scores[m_name] = rmspe(true, pred)
        elif metric == "bias":
            scores[m_name] = bias_pct(true, pred)
        elif metric == "hit":
            scores[m_name] = hit_rate(true, pred)
        elif metric == "corr":
            mask = pred.notna() & true.notna()
            if mask.sum() < 2:
                scores[m_name] = np.nan
            else:
                scores[m_name] = float(np.corrcoef(pred[mask], true[mask])[0, 1])
    return scores


def metric_is_lower_better(metric):
    return metric in ("qlike", "rmspe")

def metric_is_abs_better(metric):
    """True when the best score is the one closest to zero (e.g. bias)."""
    return metric == "bias"


@callback(
    Output("lb-winner-banner", "children"),
    Output("lb-table", "children"),
    Output("lb-bar-chart", "figure"),
    Input("lb-regime-filter", "value"),
    Input("lb-metric", "value"),
    Input("lb-stock-filter", "value"),
)
def update_leaderboard_tab(regimes, metric, selected_stocks):
    METRIC_LABELS = {
        "qlike": ("QLIKE",       "",  "lower is better"),
        "rmspe": ("RMSPE",       "%", "lower is better"),
        "bias":  ("Bias",        "%", "closer to 0 is better"),
        "hit":   ("Hit Rate",    "%", "higher is better"),
        "corr":  ("Correlation", "",  "higher is better"),
    }

    # ── Filter ────────────────────────────────────────────────────────────────
    subset = df.copy()
    if regimes:
        subset = subset[subset["regime"].isin(regimes)]
    stock_label = "All stocks"
    if selected_stocks:
        subset = subset[subset["stock_id"].isin(selected_stocks)]
        stock_label = (f"S{selected_stocks[0]:03d}" if len(selected_stocks) == 1
                       else f"{len(selected_stocks)} stocks selected")

    scores = compute_scores(subset, metric)
    if not scores:
        empty = html.Div("No data available",
                         style={"color": TEXT_DIM, "fontSize": "11px", "padding": "20px"})
        return empty, empty, go.Figure(), go.Figure()

    lower_better = metric_is_lower_better(metric)
    abs_better   = metric_is_abs_better(metric)
    if abs_better:
        ranked = sorted(scores.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else np.inf)
    elif lower_better:
        ranked = sorted(scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else np.inf)
    else:
        ranked = sorted(scores.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -np.inf)

    best_model, best_score = ranked[0]
    best_color = MODEL_COLORS.get(best_model, TEXT_HI)
    m_label, m_unit, m_note = METRIC_LABELS.get(metric, (metric, "", ""))

    # ── Winner banner ─────────────────────────────────────────────────────────
    banner = html.Div([
        html.Div([
            html.Div("BEST MODEL", style={"color": TEXT_DIM, "fontSize": "11px",
                                          "letterSpacing": "1px", "marginBottom": "6px"}),
            html.Div(best_model, style={
                "color": best_color, "fontSize": "28px", "fontWeight": "800",
                "fontFamily": "'Syne', sans-serif", "letterSpacing": "2px",
            }),
            html.Div(f"{best_score:.3f}{m_unit}  ·  {m_label}  ·  {stock_label}",
                     style={"color": TEXT_MID, "fontSize": "12px", "marginTop": "4px"}),
        ], style={
            "backgroundColor": BG_CARD, "border": f"1px solid {best_color}33",
            "borderLeft": f"4px solid {best_color}",
            "borderRadius": "10px", "padding": "16px 20px", "flex": "1",
            "boxShadow": f"0 0 40px 0 {best_color}18",
        }),
        html.Div([
            html.Div([
                html.Div([
                    html.Span("▮ ", style={"color": MODEL_COLORS.get(m, TEXT_MID),
                                           "fontSize": "12px"}),
                    html.Span(m, style={"color": TEXT_HI, "fontSize": "12px",
                                        "fontWeight": "500"}),
                ], style={"marginBottom": "2px"}),
                html.Div(f"{s:.3f}{m_unit}" if not np.isnan(s) else "—",
                         style={"color": MODEL_COLORS.get(m, TEXT_MID),
                                "fontSize": "14px", "fontWeight": "600",
                                "fontFamily": "'JetBrains Mono', monospace"}),
            ], style={
                "backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
                "borderTop": f"2px solid {MODEL_COLORS.get(m, BORDER)}",
                "borderRadius": "8px", "padding": "10px 14px", "flex": "1",
            })
            for m, s in ranked
        ], style={"display": "flex", "gap": "10px", "flex": "3"}),
    ], style={"display": "flex", "gap": "14px", "alignItems": "stretch"})

    # ── Rankings table ────────────────────────────────────────────────────────
    th   = {"color": TEXT_DIM, "fontSize": "11px", "letterSpacing": "0.5px",
            "padding": "8px 12px", "borderBottom": f"1px solid {BORDER}",
            "fontWeight": "500", "textAlign": "center"}
    th_l = {**th, "textAlign": "left"}

    header = html.Thead(html.Tr([
        html.Th("RANK",   style=th),
        html.Th("MODEL",  style=th_l),
        html.Th(m_label,  style=th),
        html.Th("vs BEST", style=th),
        html.Th("QLIKE",  style=th) if metric != "qlike"  else None,
        html.Th("RMSPE",  style=th) if metric != "rmspe"  else None,
        html.Th("CORR",   style=th) if metric != "corr"   else None,
        html.Th("BIAS",   style=th) if metric != "bias"   else None,
    ]))

    full_stats = {}
    for m, c in AVAILABLE_MODELS.items():
        if c in subset.columns:
            mask = subset[c].notna()
            full_stats[m] = {
                "qlike": qlike(subset["actual_rv"], subset[c]),
                "rmspe": rmspe(subset["actual_rv"], subset[c]),
                "corr":  (float(np.corrcoef(subset.loc[mask, c],
                                             subset.loc[mask, "actual_rv"])[0, 1])
                          if mask.sum() > 1 else np.nan),
                "bias":  bias_pct(subset["actual_rv"], subset[c]),
            }

    rows = []
    for rank, (m, s) in enumerate(ranked, start=1):
        mc      = MODEL_COLORS.get(m, TEXT_MID)
        is_best = rank == 1
        if abs_better:
            # distance from zero — positive means further away (worse)
            vs_best = abs(s) - abs(best_score)
        elif lower_better:
            vs_best = s - best_score
        else:
            vs_best = best_score - s
        vs_str  = (f"{'+' if vs_best >= 0 else ''}{vs_best:.4f}{m_unit}"
                   if vs_best != 0 else "—")
        vs_col  = TEXT_DIM if vs_best == 0 else "#ef4444" if vs_best > 0 else "#3ecf8e"

        td   = {"fontSize": "11px", "padding": "7px 12px",
                "borderBottom": f"1px solid {BORDER}",
                "fontFamily": "'JetBrains Mono', monospace", "textAlign": "center"}
        td_l = {**td, "textAlign": "left"}

        st = full_stats.get(m, {})
        row_cells = [
            html.Td(html.Span(f"#{rank}", style={
                "color": (["#f0c040", "#aaaaaa", "#cd7f32"] + [TEXT_DIM] * 10)[rank - 1],
                "fontWeight": "700", "fontSize": "13px",
            }), style=td),
            html.Td([
                html.Span("▮ ", style={"color": mc}),
                html.Span(m, style={"color": TEXT_HI if is_best else TEXT_MID,
                                    "fontWeight": "600" if is_best else "400"}),
            ], style=td_l),
            html.Td(f"{s:.4f}{m_unit}" if not np.isnan(s) else "—",
                    style={**td, "color": mc, "fontWeight": "600"}),
            html.Td(vs_str, style={**td, "color": vs_col}),
        ]
        if metric != "qlike":
            v = st.get("qlike")
            row_cells.append(html.Td(f"{v:.4f}" if v is not None and not np.isnan(v) else "—", style=td))
        if metric != "rmspe":
            v = st.get("rmspe")
            row_cells.append(html.Td(f"{v:.2f}%" if v is not None and not np.isnan(v) else "—", style=td))
        if metric != "corr":
            v = st.get("corr")
            row_cells.append(html.Td(f"{v:.3f}" if v is not None and not np.isnan(v) else "—", style=td))
        if metric != "bias":
            v = st.get("bias")
            row_cells.append(html.Td(f"{v:.2f}%" if v is not None and not np.isnan(v) else "—", style=td))

        rows.append(html.Tr(row_cells,
                            style={"backgroundColor": ACCENT2 if is_best
                                   else "transparent"}))

    table = dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True,
                      responsive=True, size="sm",
                      style={"backgroundColor": BG_CARD, "color": TEXT_MID,
                             "marginBottom": "0"})

    # ── Bar chart ─────────────────────────────────────────────────────────────
    models_ord = [m for m, _ in ranked]
    scores_ord = [s for _, s in ranked]
    colors_ord = [MODEL_COLORS.get(m, TEXT_MID) for m in models_ord]

    fig_bar = go.Figure(go.Bar(
        x=models_ord, y=scores_ord,
        marker_color=colors_ord,
        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
        text=[f"{s:.3f}{m_unit}" for s in scores_ord],
        textposition="outside", textfont=dict(size=10, color=TEXT_MID),
    ))
    fig_bar.update_layout(
        template=PLOT_TMPL,
        title=dict(text=f"{m_label} by model  ·  {m_note}",
                   font=dict(size=10, color=TEXT_DIM), x=0),
        yaxis=dict(title=m_label), showlegend=False,
        margin=dict(l=52, r=8, t=32, b=36),
    )

    return banner, table, fig_bar


@callback(
    Output("lb-time-chart", "figure"),
    Output("lb-time-title", "children"),
    Output("lb-time-subtitle", "children"),
    Input("lb-regime-filter", "value"),
    Input("lb-metric", "value"),
    Input("lb-stock-filter", "value"),
)
def update_leaderboard_timechart(regimes, metric, selected_stocks):
    METRIC_LABELS = {
        "qlike": ("QLIKE",       "",  "lower is better"),
        "rmspe": ("RMSPE",       "%", "lower is better"),
        "bias":  ("Bias",        "%", "closer to 0 is better"),
        "hit":   ("Hit Rate",    "%", "higher is better"),
        "corr":  ("Correlation", "",  "higher is better"),
    }
    m_label, m_unit, m_note = METRIC_LABELS.get(metric, (metric, "", ""))

    # Filter ONCE outside the model loop — no copy needed
    ts_data = df
    if regimes:
        ts_data = ts_data[ts_data["regime"].isin(regimes)]
    if selected_stocks:
        ts_data = ts_data[ts_data["stock_id"].isin(selected_stocks)]

    stock_label = ("All stocks" if not selected_stocks else
                   f"S{selected_stocks[0]:03d}" if len(selected_stocks) == 1
                   else f"{len(selected_stocks)} stocks selected")

    # Determine best model for line emphasis (fast aggregate — no per-time_id needed)
    scores = compute_scores(ts_data, metric)
    lower_better = metric_is_lower_better(metric)
    abs_better   = metric_is_abs_better(metric)
    if scores:
        if abs_better:
            best_model = min(scores,
                             key=lambda k: abs(scores[k]) if not np.isnan(scores[k]) else np.inf)
        else:
            best_model = (min if lower_better else max)(
                scores,
                key=lambda k: scores[k] if not np.isnan(scores[k])
                else (np.inf if lower_better else -np.inf),
            )
    else:
        best_model = None

    fig_time = go.Figure()
    for m, c in AVAILABLE_MODELS.items():
        if c not in df.columns:
            continue

        d = ts_data[["time_id", "actual_rv", c]].dropna(subset=["actual_rv", c])
        if d.empty:
            continue

        # Vectorized per-time_id computation — replaces slow groupby.apply for
        # qlike / rmspe / bias (most-used metrics).  hit / corr fall back to apply.
        if metric == "qlike":
            t = d["actual_rv"].clip(lower=EPS)
            h = d[c].clip(lower=EPS)
            d = d.assign(_s=t / h - np.log(t / h) - 1)
            ts_scores = d.groupby("time_id")["_s"].mean().reset_index(name="score")
        elif metric == "rmspe":
            t = d["actual_rv"].clip(lower=EPS)
            h = d[c].clip(lower=EPS)
            d = d.assign(_s=((h - t) / t) ** 2)
            raw = d.groupby("time_id")["_s"].mean()
            ts_scores = (np.sqrt(raw) * 100).reset_index(name="score")
        elif metric == "bias":
            t = d["actual_rv"].clip(lower=EPS)
            h = d[c].clip(lower=EPS)
            d = d.assign(_s=(h - t) / t)
            ts_scores = (d.groupby("time_id")["_s"].mean() * 100).reset_index(name="score")
        elif metric == "corr":
            ts_scores = (ts_data[["time_id", "actual_rv", c]]
                         .dropna(subset=["actual_rv", c])
                         .groupby("time_id")
                         .apply(lambda g: g[c].corr(g["actual_rv"]))
                         .reset_index(name="score"))
        else:  # hit rate — requires consecutive differences, keep apply
            ts_scores = (ts_data.groupby("time_id")
                         .apply(lambda g, _c=c: hit_rate(g["actual_rv"], g[_c]))
                         .reset_index(name="score"))

        mc = MODEL_COLORS.get(m, TEXT_MID)
        fig_time.add_trace(go.Scatter(
            x=ts_scores["time_id"], y=ts_scores["score"],
            mode="lines",
            line=dict(color=mc, width=2 if m == best_model else 1.2),
            opacity=1.0 if m == best_model else 0.65,
            name=m,
            hovertemplate=f"<b>{m}</b><br>Snapshot: %{{x}}<br>{m_label}: %{{y:.4f}}<extra></extra>",
        ))

    fig_time.update_layout(
        template=PLOT_TMPL,
        title=dict(text=f"{m_label} over snapshots · {stock_label}",
                   font=dict(size=10, color=TEXT_DIM), x=0),
        xaxis=dict(title="Snapshot (time_id)"),
        yaxis=dict(title=m_label),
        legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        margin=dict(l=52, r=8, t=32, b=56),
    )

    return (
        fig_time,
        f"{m_label} over Snapshots",
        f"how each model performed across time · {m_note} · {stock_label}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loaded {len(df):,} rows | {len(STOCKS)} stocks | {len(TIME_IDS):,} snapshots")
    print(f"Models: {list(AVAILABLE_MODELS.keys())}")
    print(f"RV Buckets: {RV_BUCKET_COLS}")
    print("Starting → http://127.0.0.1:8050")
    app.run(debug=False, port=8050)