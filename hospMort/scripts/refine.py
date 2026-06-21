"""
refine.py
---------
Refines the class-weighted KAN by:
  1. Sweeping the died-class loss weight (1× … 7×)
  2. For every trained model, finding the optimal classification threshold
     using the ROC curve (Youden's J: max sensitivity + specificity)
  3. Printing a comparison table and saving results/metrics/refine.csv

Best settings from sweep.py:  nodes=1, grid=3, k=3, 50 epochs

Usage:
    cd ThesisLoes/hospMort/scripts
    python refine.py
"""

import os, sys, csv, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
KAN_DIR     = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", "..", "code", "KAN"))
sys.path.insert(0, KAN_DIR)

from pykan.kan import MultKAN
from preprocess import load_and_preprocess

# ------------------------------------------------------------------ #
# Config (best from sweep)                                             #
# ------------------------------------------------------------------ #
SEED       = 42
NUM_EPOCHS = 50
KAN_SHAPE  = [54, 1, 2]
GRID       = 3
K          = 3

# Weight ratios to try: died gets W× more loss than survived
WEIGHT_RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.1]

METRICS_DIR = "results/metrics"
PLOT_DIR    = "results/plots"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,    exist_ok=True)


def seed_everything(seed=SEED):
    import random; random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    if preds.sum() == 0:
        return None
    cm         = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold":   round(threshold, 3),
        "accuracy":    accuracy_score(labels, preds) * 100,
        "f1_macro":    f1_score(labels, preds, average="macro",  zero_division=0) * 100,
        "f1_pos":      f1_score(labels, preds, pos_label=1,      zero_division=0) * 100,
        "recall_pos":  recall_score(labels, preds, pos_label=1,  zero_division=0) * 100,
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0) * 100,
        "fpr": fp / (fp + tn) * 100 if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) * 100 if (fn + tp) > 0 else 0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(labels, probs):
    """Youden's J: threshold that maximises sensitivity + specificity."""
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ------------------------------------------------------------------ #
# Load data once                                                       #
# ------------------------------------------------------------------ #
print("Loading data (no resampling — using loss weights instead)...")
prep          = load_and_preprocess(resample_strategy="none")
dataset       = prep["dataset"]
base_weights  = prep["class_weights"]   # [w_survived, w_died] balanced
labels_np     = dataset["test_label"].cpu().numpy()
print(f"  Train: {len(dataset['train_input'])}  Test: {len(dataset['test_input'])}\n")

# ------------------------------------------------------------------ #
# Header                                                               #
# ------------------------------------------------------------------ #
W = 7
header = (
    f"{'ratio':>6}  {'thresh':>6}  "
    f"{'acc%':>{W}} {'f1mac%':>{W}} {'f1+%':>{W}} "
    f"{'rec+%':>{W}} {'prec+%':>{W}} {'auroc%':>{W}} "
    f"{'fpr%':>{W}} {'fnr%':>{W}}  {'t(s)':>5}"
)
divider = "-" * len(header)
print(header)
print(divider)

csv_rows   = []
roc_curves = {}   # for combined plot

for ratio in WEIGHT_RATIOS:
    seed_everything()

    # Build weighted loss
    w_survived = 1.0
    w_died     = ratio
    loss_weights = torch.tensor([w_survived, w_died], dtype=torch.float32)
    loss_fn      = torch.nn.CrossEntropyLoss(weight=loss_weights)

    # Train
    model = MultKAN(width=KAN_SHAPE, grid=GRID, k=K, seed=SEED)
    t0    = time.time()
    model.fit(dataset, opt="LBFGS", steps=NUM_EPOCHS, loss_fn=loss_fn)
    elapsed = time.time() - t0

    # Get probabilities
    model.eval()
    with torch.no_grad():
        logits = model(dataset["test_input"])
    probs_np = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    auroc = roc_auc_score(labels_np, probs_np) * 100
    fpr_arr, tpr_arr, _ = roc_curve(labels_np, probs_np)
    roc_curves[ratio]   = (fpr_arr, tpr_arr, auroc)

    # Default threshold (0.5)
    m_default = metrics_at_threshold(labels_np, probs_np, 0.5)

    # Youden-optimal threshold
    best_thr  = find_best_threshold(labels_np, probs_np)
    m_optimal = metrics_at_threshold(labels_np, probs_np, best_thr)

    # Print default row
    def row_str(m, auroc_val, t):
        return (
            f"{ratio:>6.1f}  {m['threshold']:>6.2f}  "
            f"{m['accuracy']:>{W}.2f} {m['f1_macro']:>{W}.2f} {m['f1_pos']:>{W}.2f} "
            f"{m['recall_pos']:>{W}.2f} {m['precision_pos']:>{W}.2f} {auroc_val:>{W}.2f} "
            f"{m['fpr']:>{W}.2f} {m['fnr']:>{W}.2f}  {t:>5.1f}"
        )

    print(row_str(m_default, auroc, elapsed) + "  ← thr=0.5")
    print(row_str(m_optimal, auroc, elapsed) + "  ← Youden")
    print()

    for m, label in [(m_default, "0.5"), (m_optimal, "youden")]:
        csv_rows.append({
            "weight_ratio": ratio,
            "threshold_type": label,
            **m,
            "auroc": round(auroc, 2),
            "train_time_s": round(elapsed, 1),
        })

print(divider)

# ------------------------------------------------------------------ #
# Combined ROC curve plot                                              #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
for ratio, (fpr_arr, tpr_arr, auroc) in roc_curves.items():
    ax.plot(fpr_arr, tpr_arr, label=f"weight {ratio:.1f}×  (AUROC {auroc:.1f}%)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC curves — KAN hospital mortality\n(varying died-class loss weight)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(PLOT_DIR, "refine_roc.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"\nROC plot saved to {roc_path}")

# ------------------------------------------------------------------ #
# Summary: best per metric (Youden rows only)                         #
# ------------------------------------------------------------------ #
youden_rows = [r for r in csv_rows if r["threshold_type"] == "youden"]
print("\n--- Best Youden-threshold result per metric ---")
for metric, col, direction in [
    ("F1+ (died)",     "f1_pos",         "max"),
    ("Recall+ (died)", "recall_pos",      "max"),
    ("Precision+",     "precision_pos",   "max"),
    ("AUROC",          "auroc",           "max"),
    ("FNR (↓)",        "fnr",             "min"),
    ("FPR (↓)",        "fpr",             "min"),
]:
    fn   = max if direction == "max" else min
    best = fn(youden_rows, key=lambda r: r[col])
    print(
        f"  {metric:<18} weight={best['weight_ratio']:.1f}×"
        f"  thr={best['threshold']:.2f}"
        f"  F1+={best['f1_pos']:.1f}%"
        f"  rec={best['recall_pos']:.1f}%"
        f"  prec={best['precision_pos']:.1f}%"
        f"  fpr={best['fpr']:.1f}%"
        f"  fnr={best['fnr']:.1f}%"
    )

# ------------------------------------------------------------------ #
# Save CSV                                                             #
# ------------------------------------------------------------------ #
csv_path = os.path.join(METRICS_DIR, "refine.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nFull results saved to {csv_path}")
