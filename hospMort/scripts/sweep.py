"""
sweep.py
--------
Grid search over KAN hyperparameters for hospital mortality prediction.

Parameters explored:
  hidden_nodes : number of nodes in the single hidden layer
  epochs       : number of L-BFGS training steps
  lr           : L-BFGS learning rate

Grid and spline order are fixed at the values found optimal in the earlier
architecture sweep (grid=3, k=3).
Results are printed as a table and saved to results/metrics/sweep.csv.

Usage:
    cd hospMort/scripts
    python sweep.py
"""

import os
import sys
import csv
import time
import itertools

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from kan import MultKAN                      # noqa: E402
from preprocess import load_and_preprocess   # noqa: E402

# ------------------------------------------------------------------ #
# Sweep configuration                                                  #
# ------------------------------------------------------------------ #
N_FEATURES   = 29       # feature set used in train_kan.py
SEED         = 42
STRATEGY     = "none"   # class-weighted loss; no resampling
GRID         = 3        # fixed: optimal from architecture sweep
K            = 3        # fixed: cubic B-splines

HIDDEN_NODES  = [1, 2, 3, 4, 5]
EPOCH_VALUES  = [5, 10, 20]
LR_VALUES     = [1.0, 0.1, 0.01, 0.001, 0.0001]   # includes small LRs

METRICS_DIR  = "results/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #
def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model: MultKAN, dataset: dict) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(dataset["test_input"])
    preds  = logits.argmax(1).cpu().numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    labels = dataset["test_label"].cpu().numpy()

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  accuracy_score(labels, preds) * 100,
        "f1_macro":  f1_score(labels, preds, average="macro", zero_division=0) * 100,
        "f1_pos":    f1_score(labels, preds, pos_label=1, zero_division=0) * 100,
        "recall_pos":recall_score(labels, preds, pos_label=1, zero_division=0) * 100,
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0) * 100,
        "auroc":     roc_auc_score(labels, probs) * 100,
        "fpr":       fp / (fp + tn) * 100 if (fp + tn) > 0 else 0,
        "fnr":       fn / (fn + tp) * 100 if (fn + tp) > 0 else 0,
    }


def fmt(val: float) -> str:
    return f"{val:6.2f}"


# ------------------------------------------------------------------ #
# Load data once — same split for every run                           #
# ------------------------------------------------------------------ #
print("Loading data (once)...")
prep     = load_and_preprocess(resample_strategy=STRATEGY)
dataset  = prep["dataset"]
print(f"  Train: {len(dataset['train_input'])}  Test: {len(dataset['test_input'])}\n")

# ------------------------------------------------------------------ #
# Header                                                               #
# ------------------------------------------------------------------ #
COL = 8
header = (
    f"{'nodes':>5} {'epochs':>6} {'lr':>7}  "
    f"{'acc%':>{COL}} {'f1mac%':>{COL}} {'f1+%':>{COL}} "
    f"{'rec+%':>{COL}} {'prec+%':>{COL}} {'auroc%':>{COL}} "
    f"{'fpr%':>{COL}} {'fnr%':>{COL}}  {'t(s)':>6}"
)
divider = "-" * len(header)
print(header)
print(divider)

# class-weighted loss (same ratio as train_kan.py)
loss_fn = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 6.0], dtype=torch.float32)
)

# ------------------------------------------------------------------ #
# Sweep                                                                #
# ------------------------------------------------------------------ #
csv_rows = []
combinations = list(itertools.product(HIDDEN_NODES, EPOCH_VALUES, LR_VALUES))

for hidden, epochs, lr in combinations:
    seed_everything()
    shape = [N_FEATURES, hidden, 2]
    model = MultKAN(width=shape, grid=GRID, k=K, seed=SEED)

    t0 = time.time()
    model.fit(
        dataset,
        opt="LBFGS",
        steps=epochs,
        lr=lr,
        loss_fn=loss_fn,
    )
    elapsed = time.time() - t0

    m = evaluate(model, dataset)

    row = (
        f"{hidden:>5} {epochs:>6} {lr:>7}  "
        f"{fmt(m['accuracy'])} {fmt(m['f1_macro'])} {fmt(m['f1_pos'])} "
        f"{fmt(m['recall_pos'])} {fmt(m['precision_pos'])} {fmt(m['auroc'])} "
        f"{fmt(m['fpr'])} {fmt(m['fnr'])}  {elapsed:>6.1f}"
    )
    print(row)

    csv_rows.append({
        "hidden_nodes": hidden, "epochs": epochs, "lr": lr,
        "train_time_s": round(elapsed, 1),
        **{key: round(val, 2) for key, val in m.items()},
    })

print(divider)

# ------------------------------------------------------------------ #
# Best by key metrics                                                  #
# ------------------------------------------------------------------ #
best_f1    = max(csv_rows, key=lambda r: r["f1_pos"])
best_auroc = max(csv_rows, key=lambda r: r["auroc"])
best_fnr   = min(csv_rows, key=lambda r: r["fnr"])

print(f"\nBest F1 (died) : nodes={best_f1['hidden_nodes']} epochs={best_f1['epochs']} lr={best_f1['lr']}"
      f"  →  F1+={best_f1['f1_pos']:.2f}%  AUROC={best_f1['auroc']:.2f}%  FNR={best_f1['fnr']:.2f}%")
print(f"Best AUROC     : nodes={best_auroc['hidden_nodes']} epochs={best_auroc['epochs']} lr={best_auroc['lr']}"
      f"  →  AUROC={best_auroc['auroc']:.2f}%  F1+={best_auroc['f1_pos']:.2f}%")
print(f"Lowest FNR     : nodes={best_fnr['hidden_nodes']} epochs={best_fnr['epochs']} lr={best_fnr['lr']}"
      f"  →  FNR={best_fnr['fnr']:.2f}%  F1+={best_fnr['f1_pos']:.2f}%")

# ------------------------------------------------------------------ #
# Save CSV                                                             #
# ------------------------------------------------------------------ #
csv_path = os.path.join(METRICS_DIR, "sweep.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nFull results saved to {csv_path}")
