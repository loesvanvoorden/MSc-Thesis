"""
plot_kan_annotated.py
---------------------
Horizontal KAN diagram (Tara-style layout, full architecture):
  - Each of 29 features: 3 spline panels (one per hidden node)
  - Each of 3 hidden nodes: 2 spline panels (one per output class)
Black edges; thickness = mean |spline activation|.
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(SCRIPTS_DIR, "results", "models")
PLOT_DIR    = os.path.join(SCRIPTS_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
sys.path.insert(0, SCRIPTS_DIR)

from kan import MultKAN
from preprocess import load_and_preprocess

with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
    cfg = json.load(f)

model = MultKAN(width=cfg["kan_shape"], grid=cfg["grid"], k=cfg["k"], seed=42)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "kan_weights.pt"),
                                  map_location="cpu"))
model.eval()

prep = load_and_preprocess(
    data_path=os.path.join(SCRIPTS_DIR, "../data/MIMIC-IV.csv"),
    resample_strategy="none",
)
x_test      = prep["dataset"]["test_input"]
feat_labels = prep["feature_labels"]

with torch.no_grad():
    _ = model(x_test)

pre0  = model.spline_preacts[0].detach().numpy()   # (N, 3, 29)
post0 = model.spline_postacts[0].detach().numpy()
pre1  = model.spline_preacts[1].detach().numpy()   # (N, 2, 3)
post1 = model.spline_postacts[1].detach().numpy()

def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)

score0_n = norm01(np.abs(post0).mean(axis=0))   # (3, 29)
score1_n = norm01(np.abs(post1).mean(axis=0))   # (2, 3)

N_IN  = len(feat_labels)
N_HID = 3
# KAN softmax output: index 0 = Survived, index 1 = Died (see train_kan classification_report)
OUT_LABELS = ["Survived (0)", "Died (1)"]

# ── layout ────────────────────────────────────────────────────────────────────
X_DOT  = 0.08
X_PIN  = np.array([0.12, 0.165, 0.21])
X_HID  = 0.36
X_POUT = 0.58   # single column: per hidden node, Survived/Died stacked vertically
X_OUT  = 0.86

y_in  = np.linspace(1.0, 0.0, N_IN)
y_hid = np.array([0.68, 0.50, 0.32])
y_out = np.array([0.62, 0.38])

PIN_SIZE  = 0.022   # square input spline panels
POUT_SIZE = 0.040   # square output spline panels
STACK_GAP = 0.046   # vertical offset between Survived / Died panels per hidden node

fig = plt.figure(figsize=(19, 17), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.24, 0.04, 0.72, 0.94])
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.02)
ax.axis("off")

def d2f(xd, yd):
    return fig.transFigure.inverted().transform(ax.transData.transform((xd, yd)))

def smooth_curve(xv, yv, n=50):
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    _, idx = np.unique(xv, return_index=True)
    xv, yv = xv[idx], yv[idx]
    if len(xv) < 4:
        return xv, yv
    xs = np.linspace(xv[0], xv[-1], n)
    spl = make_interp_spline(xv, yv, k=min(3, len(xv) - 1))
    return xs, spl(xs)

def add_spline_inset(xc, yc, w, h, xv, yv, lw=0.8, alpha=1.0):
    ll = d2f(xc - w / 2, yc - h / 2)
    ur = d2f(xc + w / 2, yc + h / 2)
    fw, fh = ur[0] - ll[0], ur[1] - ll[1]
    if fw <= 0 or fh <= 0:
        return
    inset = fig.add_axes([ll[0], ll[1], fw, fh])
    xs, ys = smooth_curve(xv, yv)
    inset.plot(xs, ys, color="black", lw=lw, alpha=alpha)
    inset.set_xticks([])
    inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("black")
    inset.set_facecolor("white")

def clamp_alpha(a):
    return float(np.clip(a, 0.04, 1.0))

# column headers
for j, xc in enumerate(X_PIN):
    ax.text(xc, 1.015, f"→H{j+1}", ha="center", va="bottom",
            fontsize=11, color="#444444", fontweight="bold")
# ── input: label · dot · 3 spline panels ────────────────────────────────────
for i, fname in enumerate(feat_labels):
    y = y_in[i]
    ax.text(X_DOT - 0.010, y, fname, ha="right", va="center",
            fontsize=10.5, color="black")
    ax.plot(X_DOT, y, "o", ms=3.2, color="black", zorder=5)
    ax.plot([X_DOT, X_PIN[0] - PIN_SIZE / 2], [y, y], color="black", lw=0.35, zorder=2)

    for j in range(N_HID):
        s = float(score0_n[j, i])
        xc = X_PIN[j]
        add_spline_inset(xc, y, PIN_SIZE, PIN_SIZE,
                         pre0[:, j, i], post0[:, j, i],
                         lw=0.4 + s * 1.2, alpha=clamp_alpha(0.25 + s * 0.65))
        ax.plot([xc + PIN_SIZE / 2, X_HID], [y, y_hid[j]],
                color="black", lw=0.12 + s * 1.6,
                alpha=clamp_alpha(0.05 + s * 0.55), zorder=1)

# ── hidden nodes ──────────────────────────────────────────────────────────────
for j in range(N_HID):
    ax.plot(X_HID, y_hid[j], "o", ms=9, color="black", zorder=6)
    ax.text(X_HID - 0.022, y_hid[j], f"H{j+1}",
            ha="right", va="center", fontsize=8, color="black", fontweight="bold")

# ── hidden → output: per hidden node, Survived / Died squares stacked vertically ─
for j in range(N_HID):
    yh = y_hid[j]
    panel_y = [yh + STACK_GAP / 2, yh - STACK_GAP / 2]   # k=0 Survived on top

    for k in range(2):
        s = float(score1_n[k, j])
        ym = panel_y[k]
        yo = y_out[k]

        ax.plot([X_HID, X_POUT - POUT_SIZE / 2], [yh, ym],
                color="black", lw=0.2 + s * 2.5,
                alpha=clamp_alpha(0.08 + s * 0.60), zorder=2)

        add_spline_inset(X_POUT, ym, POUT_SIZE, POUT_SIZE,
                         pre1[:, k, j], post1[:, k, j],
                         lw=0.5 + s * 1.8, alpha=clamp_alpha(0.30 + s * 0.60))

        ax.plot([X_POUT + POUT_SIZE / 2, X_OUT], [ym, yo],
                color="black", lw=0.3 + s * 3.2,
                alpha=clamp_alpha(0.15 + s * 0.75), zorder=2)

# ── output nodes ──────────────────────────────────────────────────────────────
for k, lbl in enumerate(OUT_LABELS):
    ax.plot(X_OUT, y_out[k], "o", ms=10, color="black", zorder=6)
    ax.text(X_OUT + 0.02, y_out[k], lbl,
            ha="left", va="center", fontsize=9, color="black", fontweight="bold")

out_path = os.path.join(PLOT_DIR, "kan_hosp_mort_annotated.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
