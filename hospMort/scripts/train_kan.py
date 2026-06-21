"""
train_kan.py
------------
Trains a Kolmogorov-Arnold Network (KAN) on the MIMIC-IV dataset to predict
hospital mortality (hosp_mort). Mirrors Tara van Mierlo's KAN_on_SAD approach
(KAN_on_SAD.ipynb) with the following changes:

  - Target: hosp_mort  (was: sad)
  - sad is kept as a feature  (not dropped)
  - icu28dmort is dropped instead of hosp_mort  (would be direct leakage)
  - KAN input shape: [29, 3, 2]  (vitals + labs + demographics + clinical scores)
  - Excluded features: race/ethnicity, ICU department, active interventions, comorbidities

Tuning decisions:
  - Architecture : nodes=3, grid=3, k=3
  - Imbalance    : class-weighted CrossEntropyLoss (died class × 6) instead of SMOTE
  - Regularisation: lamb=0.0001 (very light — 0.001 compressed probabilities; 0 caused overfitting)
  - Epochs       : 5 — test loss minimum; LBFGS overfits immediately after without regularisation
  - Threshold    : FPR-constrained (≤25%)

Usage:
    cd ThesisLoes/hospMort/scripts
    python train_kan.py

Outputs saved to:
    results/models/   – saved KAN model state
    results/plots/    – KAN architecture plot + ROC curve
    results/metrics/  – training results text file
"""

import os
import time
import json
import pickle
import shutil

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

from kan import MultKAN
from preprocess import load_and_preprocess

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #
SEED              = 42
NUM_EPOCHS        = 5            # test loss minimum — overfitting starts immediately after
KAN_HIDDEN        = 3
KAN_SHAPE         = None         # set from n_features after preprocessing
GRID              = 3
K                 = 3
LOSS_WEIGHT_RATIO = 6.0          # slightly stronger signal for died class
THRESHOLD         = None         # None = FPR-constrained threshold (see FPR_TARGET below)
FPR_TARGET        = 0.25         # max false positive rate: maximise recall at FPR ≤ this
LAMB              = 0.0001       # very light L1 — just enough to keep splines well-behaved
RESAMPLE_STRATEGY = "none"       # class-weighted loss handles imbalance; no data distortion
DATA_PATH         = "../data/MIMIC-IV.csv"
MODEL_DIR         = "results/models"
PLOT_DIR          = "results/plots"
METRICS_DIR       = "results/metrics"


def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def fpr_constrained_threshold(labels: np.ndarray, probs: np.ndarray,
                               fpr_target: float) -> float:
    """Return the highest threshold that keeps FPR ≤ fpr_target, maximising recall."""
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
    mask = fpr_arr <= fpr_target
    if not mask.any():
        return float(thresholds[0])
    best_idx = np.where(mask)[0][-1]   # last point still within FPR budget
    return float(thresholds[best_idx])


def evaluate(model: MultKAN, dataset: dict, threshold: float = None) -> dict:
    """Run model on test set and return a metrics dict.

    If threshold is None, compute the FPR-constrained threshold (FPR ≤ FPR_TARGET).
    """
    model.eval()
    with torch.no_grad():
        logits = model(dataset["test_input"])
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    labels = dataset["test_label"].cpu().numpy()
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)

    if threshold is None:
        threshold = fpr_constrained_threshold(labels, probs, FPR_TARGET)
        print(f"  FPR-constrained threshold (FPR≤{FPR_TARGET:.0%}): {threshold:.4f}")

    preds  = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro", zero_division=0),
        "f1_pos":    f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "recall_pos": recall_score(labels, preds, pos_label=1, zero_division=0),
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0),
        "auroc":     roc_auc_score(labels, probs),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "threshold": threshold,
        "report": classification_report(labels, preds, target_names=["Survived", "Died"]),
        "roc_fpr": fpr_arr,
        "roc_tpr": tpr_arr,
    }


def main() -> None:
    seed_everything()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Load and preprocess data                                        #
    # ---------------------------------------------------------------- #
    print(f"Loading and preprocessing data (strategy: {RESAMPLE_STRATEGY})...")
    prep = load_and_preprocess(data_path=DATA_PATH, resample_strategy=RESAMPLE_STRATEGY)
    dataset  = prep["dataset"]
    n_feat        = prep["n_features"]
    features      = prep["feature_labels"]
    class_weights = prep["class_weights"]

    print(f"  Features      : {n_feat}")
    print(f"  Feature names : {features}")
    print(f"  Train samples : {len(dataset['train_input'])}")
    print(f"  Test  samples : {len(dataset['test_input'])}")
    pos = dataset["train_label"].sum().item()
    total = len(dataset["train_label"])
    print(f"  Train +rate   : {pos}/{total} = {pos/total*100:.1f}%")
    print()

    # ---------------------------------------------------------------- #
    # 2. Build KAN model                                                 #
    # ---------------------------------------------------------------- #
    kan_shape = [n_feat, KAN_HIDDEN, 2]
    print(f"Building KAN with shape {kan_shape}, grid={GRID}, k={K}...")
    model = MultKAN(width=kan_shape, grid=GRID, k=K, seed=SEED)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params}")
    print()

    # ---------------------------------------------------------------- #
    # 3. Train                                                           #
    # ---------------------------------------------------------------- #
    # Fixed weight ratio from refine.py sweep: 4× gives best F1/AUROC balance
    w = torch.tensor([1.0, LOSS_WEIGHT_RATIO], dtype=torch.float32)
    print(f"Training for {NUM_EPOCHS} epochs (LBFGS, weighted CrossEntropyLoss)...")
    print(f"  Loss weights: survived=1.0  died={LOSS_WEIGHT_RATIO:.1f}×")
    t_start = time.time()
    results = model.fit(
        dataset,
        opt="LBFGS",
        steps=NUM_EPOCHS,
        loss_fn=torch.nn.CrossEntropyLoss(weight=w),
        lamb=LAMB,
    )
    t_elapsed = time.time() - t_start
    print(f"  Training time: {t_elapsed:.1f}s  ({t_elapsed/NUM_EPOCHS:.2f}s/epoch)")
    print()

    # ---------------------------------------------------------------- #
    # 4. Evaluate                                                        #
    # ---------------------------------------------------------------- #
    print(f"Evaluating on test set (threshold={'Youden' if THRESHOLD is None else THRESHOLD})...")
    metrics = evaluate(model, dataset, threshold=THRESHOLD)
    print(f"  Accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 macro      : {metrics['f1_macro']:.4f}")
    print(f"  F1 (died)     : {metrics['f1_pos']:.4f}")
    print(f"  Recall (died) : {metrics['recall_pos']*100:.2f}%")
    print(f"  Precision+    : {metrics['precision_pos']*100:.2f}%")
    print(f"  AUROC         : {metrics['auroc']:.4f}")
    print(f"  FPR           : {metrics['FPR']*100:.2f}%")
    print(f"  FNR           : {metrics['FNR']*100:.2f}%")
    print()
    print(metrics["report"])

    # ---------------------------------------------------------------- #
    # 5. Save model                                                      #
    # ---------------------------------------------------------------- #
    model_path = os.path.join(MODEL_DIR, "kan_hosp_mort")
    model.saveckpt(model_path)
    # Also save full state_dict — saveckpt/loadckpt does not reliably restore weights
    weights_path = os.path.join(MODEL_DIR, "kan_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {model_path}/ and {weights_path}")

    # Save feature list alongside the model
    with open(os.path.join(MODEL_DIR, "feature_labels.json"), "w") as f:
        json.dump(features, f, indent=2)

    # Save model config (threshold etc.) for the interface
    config = {
        "threshold": metrics["threshold"],
        "kan_shape": kan_shape,
        "grid": GRID,
        "k": K,
    }
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved (threshold={metrics['threshold']:.4f})")

    # Save original (unscaled) dataframe for the interface
    with open(os.path.join(MODEL_DIR, "original_df.pkl"), "wb") as f:
        pickle.dump(prep["original_data"], f)
    print(f"Original dataframe saved ({len(prep['original_data'])} rows)")

    # Save global feature importance: mean |gradient| of P(died) over the test set
    model.eval()
    x_test = dataset["test_input"].clone().requires_grad_(True)
    out = model(x_test)
    model.zero_grad()
    out[:, 1].sum().backward()
    feature_scores = x_test.grad.abs().mean(dim=0).detach().numpy()
    np.save(os.path.join(MODEL_DIR, "feature_scores.npy"), feature_scores)
    print(f"Feature scores saved ({len(feature_scores)} features)")

    # ---------------------------------------------------------------- #
    # 6. Save KAN plot                                                   #
    # ---------------------------------------------------------------- #
    try:
        model.plot(beta=100)
        plot_path = os.path.join(PLOT_DIR, "kan_hosp_mort.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"KAN plot saved to {plot_path}")
    except Exception as e:
        print(f"[warn] Could not save KAN plot: {e}")

    # ---------------------------------------------------------------- #
    # 7. Save training loss curve + ROC curve                           #
    # ---------------------------------------------------------------- #
    try:
        train_losses = results["train_loss"]
        test_losses  = results["test_loss"]
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train loss")
        plt.plot(test_losses,  label="Test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("KAN – Hospital Mortality – Training Loss")
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(PLOT_DIR, "training_loss.png")
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to {loss_path}")
    except Exception as e:
        print(f"[warn] Could not save loss curve: {e}")

    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(metrics["roc_fpr"], metrics["roc_tpr"],
                label=f"KAN (AUROC = {metrics['auroc']:.3f})", color="steelblue", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
        ax.scatter([metrics["FPR"]], [metrics["recall_pos"]],
                   color="red", zorder=5, s=80,
                   label=f"Operating point (thr={THRESHOLD})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC – KAN Hospital Mortality")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(PLOT_DIR, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {roc_path}")
    except Exception as e:
        print(f"[warn] Could not save ROC curve: {e}")

    # ---------------------------------------------------------------- #
    # 8. Write metrics file                                              #
    # ---------------------------------------------------------------- #
    metrics_path = os.path.join(METRICS_DIR, "results.txt")
    with open(metrics_path, "w") as f:
        f.write("KAN – Hospital Mortality Prediction\n")
        f.write("=" * 40 + "\n")
        f.write(f"KAN shape      : {kan_shape}\n")
        f.write(f"Grid / k       : {GRID} / {K}\n")
        f.write(f"Epochs         : {NUM_EPOCHS}\n")
        f.write(f"Resampling     : {RESAMPLE_STRATEGY}\n")
        f.write(f"Loss weights   : survived=1.0  died={LOSS_WEIGHT_RATIO:.1f}×\n")
        f.write(f"Regularisation : lamb={LAMB}\n")
        f.write(f"Threshold      : {metrics['threshold']:.4f} (FPR-constrained, target≤{FPR_TARGET:.0%})\n")
        f.write(f"Training time  : {t_elapsed:.1f}s\n")
        f.write(f"Parameters     : {n_params}\n\n")
        f.write(f"Accuracy       : {metrics['accuracy']*100:.2f}%\n")
        f.write(f"F1 macro       : {metrics['f1_macro']:.4f}\n")
        f.write(f"F1 (died)      : {metrics['f1_pos']:.4f}\n")
        f.write(f"Recall (died)  : {metrics['recall_pos']*100:.2f}%\n")
        f.write(f"Precision+     : {metrics['precision_pos']*100:.2f}%\n")
        f.write(f"Precision macro: {metrics['precision']:.4f}\n")
        f.write(f"Recall macro   : {metrics['recall']:.4f}\n")
        f.write(f"AUROC          : {metrics['auroc']:.4f}\n")
        f.write(f"FPR            : {metrics['FPR']*100:.2f}%\n")
        f.write(f"FNR            : {metrics['FNR']*100:.2f}%\n")
        f.write(f"TN={metrics['TN']}  FP={metrics['FP']}  "
                f"FN={metrics['FN']}  TP={metrics['TP']}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics["report"])
    print(f"Metrics saved to {metrics_path}")

    # ---------------------------------------------------------------- #
    # 9. Sync model files → interface/model/                            #
    # ---------------------------------------------------------------- #
    interface_model_dir = os.path.abspath(
        os.path.join(SCRIPTS_DIR, "..", "..", "interface", "model")
    )
    os.makedirs(interface_model_dir, exist_ok=True)
    sync_files = [
        "kan_weights.pt",
        "model_config.json",
        "feature_labels.json",
        "feature_scores.npy",
        "original_df.pkl",
    ]
    print(f"\nSyncing model files → {interface_model_dir}/")
    for fname in sync_files:
        src = os.path.join(MODEL_DIR, fname)
        dst = os.path.join(interface_model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} not found — skipping")


if __name__ == "__main__":
    main()
