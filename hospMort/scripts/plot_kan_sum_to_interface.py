"""
plot_kan_sum_to_interface.py
-----------------------------
Shows how the KAN's three per-hidden-edge splines are summed (Σ) to
produce the single feature-contribution curve shown in the study interface.

Layout per feature row:
  [H1 edge spline]
  [H2 edge spline]  →  Σ  →  [Interface spline (shown to participant)]
  [H3 edge spline]

B&W, square panels, same style as kan_patient_8259_top3.py.
"""

import os, sys, json, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR   = os.path.dirname(os.path.abspath(__file__))
INTERFACE_DIR = os.path.join(SCRIPTS_DIR, "..", "..", "interface")
MODEL_DIR     = os.path.join(INTERFACE_DIR, "model")
PLOT_DIR      = os.path.join(SCRIPTS_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
sys.path.insert(0, SCRIPTS_DIR)

from kan import MultKAN

EDGE_ID   = 8259
FEAT_KEYS = ["Respiratory Rate", "DBP (mmHg)", "PTT (s)"]
N_HID     = 3
N_GRID    = 80
PANEL_IN      = 0.30   # per-edge small panel (square inches)
SUM_PANEL_IN  = 0.52   # interface sum panel (square inches)

# ── load model & patient ──────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
    cfg = json.load(f)
shape = cfg["kan_shape"]
if isinstance(shape[0], list):
    shape = [s[0] for s in shape]

model = MultKAN(width=shape, grid=cfg["grid"], k=cfg["k"], seed=42)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "kan_weights.pt"),
                                  map_location="cpu"))
model.eval()

with open(os.path.join(MODEL_DIR, "original_df.pkl"), "rb") as f:
    DF = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_labels.json")) as f:
    FEATURE_NAMES = json.load(f)

_FEAT_MIN   = np.array([float(DF[f].min()) for f in FEATURE_NAMES])
_FEAT_MAX   = np.array([float(DF[f].max()) for f in FEATURE_NAMES])
_FEAT_RANGE = np.where(_FEAT_MAX > _FEAT_MIN, _FEAT_MAX - _FEAT_MIN, 1.0)
_GI = FEATURE_NAMES.index("Gender (Male)")

def scale_all(raw):
    s = (raw - _FEAT_MIN) / _FEAT_RANGE
    s[:, _GI] = raw[:, _GI]
    return s.astype(np.float32)

feat_idx   = [FEATURE_NAMES.index(k) for k in FEAT_KEYS]
edge_pos   = int(np.where(DF.index == EDGE_ID)[0][0])
scaled     = scale_all(DF[FEATURE_NAMES].values.astype(float))
row_scaled = scaled[edge_pos]
row_raw    = DF.loc[EDGE_ID, FEATURE_NAMES].values.astype(float)

with torch.no_grad():
    logits = model(torch.tensor(scaled, dtype=torch.float32))
    probs  = torch.softmax(logits, dim=1)[edge_pos].numpy()

pre0 = model.spline_postacts[0].detach().numpy()
feat_contrib    = pre0[edge_pos].sum(axis=0)
_contrib_scale  = 100.0 / (float(np.abs(feat_contrib).sum()) or 1.0)

def fmt_val(feat_name, raw):
    return f"{raw:.0f}" if feat_name == "Respiratory Rate" else f"{raw:.1f}"

def feat_label_block(r):
    name = FEAT_KEYS[r]
    raw  = float(row_raw[feat_idx[r]])
    pct  = float(feat_contrib[feat_idx[r]]) * _contrib_scale
    sign = "+" if pct >= 0 else ""
    return f"{name}\n{fmt_val(name, raw)}  ({sign}{pct:.1f}%)"

def norm_to_raw(feat_i, x_norm):
    return _FEAT_MIN[feat_i] + x_norm * _FEAT_RANGE[feat_i]

def eval_layer0_edge(feat_i, hid_j):
    x_norm = np.linspace(0.0, 1.0, N_GRID)
    batch  = np.tile(row_scaled, (N_GRID, 1))
    batch[:, feat_i] = x_norm
    with torch.no_grad():
        model(torch.tensor(batch, dtype=torch.float32))
    y = model.spline_postacts[0][:, hid_j, feat_i].detach().numpy()
    return norm_to_raw(feat_i, x_norm), y

def eval_sum_spline(feat_i):
    x_norm = np.linspace(0.0, 1.0, N_GRID)
    batch  = np.tile(row_scaled, (N_GRID, 1))
    batch[:, feat_i] = x_norm
    with torch.no_grad():
        model(torch.tensor(batch, dtype=torch.float32))
    y = model.spline_postacts[0][:, :, feat_i].sum(dim=1).detach().numpy()
    return norm_to_raw(feat_i, x_norm), y

# ── figure & axes ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 9), dpi=300)
fig.patch.set_facecolor("white")
ax  = fig.add_axes([0.04, 0.05, 0.94, 0.90])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def d2f(xd, yd):
    return fig.transFigure.inverted().transform(ax.transData.transform((xd, yd)))

def f2d(xf, yf):
    return ax.transData.inverted().transform(fig.transFigure.transform((xf, yf)))

def panel_x_edge(xc, yc, size_in, side="right"):
    hw  = (size_in / fig.get_figwidth()) / 2
    cx_f, cy_f = d2f(xc, yc)
    xf  = cx_f + hw if side == "right" else cx_f - hw
    return float(f2d(xf, cy_f)[0])

def panel_y_edge(xc, yc, size_in, side="top"):
    hh  = (size_in / fig.get_figheight()) / 2
    cx_f, cy_f = d2f(xc, yc)
    yf  = cy_f + hh if side == "top" else cy_f - hh
    return float(f2d(cx_f, yf)[1])

def add_square_spline(xc, yc, xv, yv, curve_lw=1.0, curve_alpha=1.0,
                      size_in=PANEL_IN, pat_x=None, show_zero=False):
    cx, cy = d2f(xc, yc)
    w = size_in / fig.get_figwidth()
    h = size_in / fig.get_figheight()
    inset = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    xpad  = (xv.max() - xv.min()) * 0.04 or 0.01
    ypad  = (yv.max() - yv.min()) * 0.15 or 0.01
    y_lo  = min(float(yv.min()) - ypad, 0.0)
    y_hi  = max(float(yv.max()) + ypad, 0.0)
    if y_hi - y_lo < 1e-9:
        y_hi = y_lo + 0.01
    inset.set_xlim(xv.min() - xpad, xv.max() + xpad)
    inset.set_ylim(y_lo, y_hi)
    inset.axhline(0, color="#999999", lw=0.55, ls="--", alpha=0.85, zorder=1)
    inset.plot(xv, yv, color="black", lw=curve_lw, alpha=curve_alpha, zorder=2)
    if pat_x is not None:
        py = float(np.interp(pat_x, xv, yv))
        inset.axvline(pat_x, color="#666666", lw=0.6, ls="--", alpha=0.7, zorder=3)
        inset.plot(pat_x, py, "o", ms=3.5, color="black", zorder=4)
    inset.set_xticks([])
    if show_zero:
        inset.set_yticks([0])
        inset.set_yticklabels(["0"], fontsize=6, color="#444444")
        inset.tick_params(axis="y", length=2, pad=1)
    else:
        inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("black")
    inset.set_facecolor("white")

# ── layout positions (data coords, ax xlim/ylim = 0..1) ──────────────────────
X_NAME  = 0.22    # feature label right-edge
X_EDGE  = 0.32    # per-edge panel column centre
X_SIGMA = 0.535   # Σ symbol / convergence point
X_SUM   = 0.75    # interface sum panel centre

Y_ROW = np.array([0.78, 0.50, 0.22])
Y_OFF = np.array([0.095, 0.000, -0.095])   # H1 / H2 / H3 offsets within row

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.995,
        f"Patient #{EDGE_ID}   ·   P(died) = {probs[1]*100:.1f}%",
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#111111", transform=ax.transAxes)

# ── column headers (y=0.925 leaves clear gap below title at 0.995) ────────────
ax.text(X_EDGE,  0.925, "per-edge contributions (H1, H2, H3)", ha="center",
        fontsize=10, fontweight="bold", color="#333333")
ax.text(X_SIGMA, 0.925, "Σ", ha="center", fontsize=13,
        fontweight="bold", color="#333333")
ax.text(X_SUM,   0.925, "shown in interface", ha="center",
        fontsize=10, fontweight="bold", color="#333333")

# ── one row per feature ───────────────────────────────────────────────────────
for r, feat_i in enumerate(feat_idx):
    y_row = Y_ROW[r]
    pat_x = float(row_raw[feat_i])

    # Feature label
    ax.text(X_NAME, y_row, feat_label_block(r),
            ha="right", va="center", fontsize=9.5, color="black",
            linespacing=1.25)

    # Per-edge spline panels (H1, H2, H3) stacked vertically
    for j in range(N_HID):
        y = y_row + Y_OFF[j]

        # H-label to the left of panel
        ax.text(panel_x_edge(X_EDGE, y, PANEL_IN, "left") - 0.005,
                y, f"H{j+1}",
                ha="right", va="center", fontsize=8,
                fontweight="bold", color="#444444")

        x_raw, y_act = eval_layer0_edge(feat_i, j)
        add_square_spline(X_EDGE, y, x_raw, y_act,
                          curve_lw=0.9, curve_alpha=0.85,
                          size_in=PANEL_IN, pat_x=pat_x)

        # "+" between consecutive panels
        if j < N_HID - 1:
            mid_y = (y_row + Y_OFF[j] + y_row + Y_OFF[j + 1]) / 2.0
            ax.text(X_EDGE, mid_y, "+",
                    ha="center", va="center", fontsize=11,
                    color="#555555", fontweight="bold")

        # Converging line from right edge of panel to Σ point
        x_from = panel_x_edge(X_EDGE, y, PANEL_IN, "right")
        ax.plot([x_from, X_SIGMA - 0.016], [y, y_row],
                color="#888888", lw=0.65, ls="-", alpha=0.6,
                solid_capstyle="round", zorder=1)

    # Σ symbol at convergence point
    ax.text(X_SIGMA, y_row, "Σ",
            ha="center", va="center", fontsize=20,
            fontweight="bold", color="#111111", zorder=5)

    # Arrow from Σ to interface panel
    x_arr_start = X_SIGMA + 0.018
    x_arr_end   = panel_x_edge(X_SUM, y_row, SUM_PANEL_IN, "left") - 0.004
    ax.annotate("",
                xy=(x_arr_end, y_row),
                xytext=(x_arr_start, y_row),
                arrowprops=dict(arrowstyle="-|>", color="#333333",
                                lw=0.9, mutation_scale=9),
                annotation_clip=False, zorder=4)

    # Interface sum spline
    x_raw, y_sum = eval_sum_spline(feat_i)
    add_square_spline(X_SUM, y_row, x_raw, y_sum,
                      curve_lw=1.4, curve_alpha=1.0,
                      size_in=SUM_PANEL_IN, pat_x=pat_x, show_zero=True)

out_path = os.path.join(PLOT_DIR, "kan_sum_to_interface.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
