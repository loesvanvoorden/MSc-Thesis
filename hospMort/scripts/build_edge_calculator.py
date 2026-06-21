"""
Regenerate edge_patient_calculator.html from the trained model in interface/model/.

Embeds fresh patient probabilities (29-feature KAN) and min/max/median stats.
Run after train_kan.py:

    cd hospMort/scripts
    python build_edge_calculator.py
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kan import MultKAN

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
MODEL_DIR = REPO_ROOT / "interface" / "model"
TEMPLATE = REPO_ROOT / "interface" / "static" / "tools" / "edge_patient_calculator.html"
OUT_PATHS = [
    TEMPLATE,
    SCRIPTS_DIR / "results" / "edge_patient_calculator.html",
]

CURRENT_EDGE_ID = 8259

CANDIDATE_IDS = [8259, 1617, 8680, 11685, 13552, 14405]
CANDIDATE_NOTES = {
    8259: (
        "MAP 74, HR 74, SpO2 94, SOFA 2, GCS 14. Geen interventies. "
        "Enige opvallende bevinding: aPTT 91.6s (~3× normaal). Verder rustig beeld. "
        "[Huidig geselecteerd — P≈60%]"
    ),
    1617: (
        "MAP 89, HR 74, SpO2 100, SOFA 2, GCS 15. Geen interventies. Zeer rustig beeld. "
        "(P≈63%)"
    ),
    8680: (
        "MAP 94, HR 114, SpO2 95, SOFA 2, GCS 15. Geen interventies. "
        "Stille bevinding: Hb 6.8 g/dL (ernstige anemie). (P≈66%)"
    ),
    5164: (
        "MAP 85, HR 67, SpO2 96, SOFA 2, GCS 15. Geen interventies. "
        "Meerdere stille metabole bevindingen: INR 2.6, creatinine 2.5, anion gap 21, glucose 275. "
        "(P≈70%; veel comorbiditeit / hoge leeftijd)"
    ),
    11046: (
        "MAP 68, HR 83, SpO2 98, SOFA 2, GCS 15. Geen interventies. "
        "Stille bevindingen: aPTT 83.3s + Ca 8.1 (licht verlaagd) + Hb 8.5. "
        "(P≈47% met huidig model)"
    ),
    14405: (
        "MAP 95, HR 108, SpO2 100, SOFA 2, GCS 15. Geen interventies. "
        "Lichte tachycardie. Verborgen: aPTT 150s (5× normaal) en glucose 346."
    ),
    12789: (
        "MAP 95, HR 62, SpO2 92, SOFA 2, GCS 15. Geen interventies. Hoge leeftijd. "
        "Stille bevindingen: aPTT 135s + Ca 6.2 (ernstige hypocalciemie)."
    ),
}

# Slider / distance-search column order (matches model feature set + display layout).
DISPLAY_FEATURES = [
    "Age",
    "Weight (kg)",
    "Temperature (°C)",
    "Heart Rate (bpm)",
    "Respiratory Rate",
    "SpO2",
    "SBP (mmHg)",
    "DBP (mmHg)",
    "MBP (mmHg)",
    "WBC (K/uL)",
    "Hemoglobin (g/dL)",
    "Platelet (k/uL)",
    "BUN (mg/dL)",
    "Creatinine (mg/dL)",
    "Glucose (mg/dL)",
    "Sodium (mEq/L)",
    "Chloride (mEq/L)",
    "Potassium (mEq/L)",
    "Magnesium (mg/dL)",
    "Calcium (mg/dL)",
    "Phosphate (mg/dL)",
    "INR",
    "Prothrombin Time (s)",
    "PTT (s)",
    "Bicarbonate (mEq/L)",
    "Anion Gap (mEq/L)",
    "Gender (Male)",
    "GCS",
    "SOFA Score",
]


def _load_model_and_probs():
    with open(MODEL_DIR / "model_config.json") as f:
        cfg = json.load(f)
    shape = cfg.get("kan_shape", [29, 3, 2])
    if isinstance(shape[0], list):
        shape = [s[0] for s in shape]

    model = MultKAN(width=shape, grid=cfg.get("grid", 3), k=cfg.get("k", 3), seed=42)
    model.load_state_dict(torch.load(MODEL_DIR / "kan_weights.pt", map_location="cpu"))
    model.eval()

    with open(MODEL_DIR / "feature_labels.json") as f:
        model_features = json.load(f)
    with open(MODEL_DIR / "original_df.pkl", "rb") as f:
        df = pickle.load(f)

    feat_min = np.array([df[f].min() for f in model_features], dtype=float)
    feat_max = np.array([df[f].max() for f in model_features], dtype=float)
    feat_range = np.where(feat_max > feat_min, feat_max - feat_min, 1.0)
    x_raw = df[model_features].values.astype(float)
    x_scaled = (x_raw - feat_min) / feat_range
    if "Gender (Male)" in model_features:
        gi = model_features.index("Gender (Male)")
        x_scaled[:, gi] = x_raw[:, gi]

    with torch.no_grad():
        probs = F.softmax(model(torch.tensor(x_scaled, dtype=torch.float32)), dim=1)[:, 1].numpy()

    return df, probs, cfg.get("threshold", 0.5)


def _build_data(df, probs) -> dict:
    feat_min = [float(df[f].min()) for f in DISPLAY_FEATURES]
    feat_max = [float(df[f].max()) for f in DISPLAY_FEATURES]
    medians = {f: float(df[f].median()) for f in DISPLAY_FEATURES}

    patients = []
    for pos, pid in enumerate(df.index):
        row = df.loc[pid]
        patients.append(
            {
                "id": int(pid),
                "p": round(float(probs[pos] * 100), 2),
                "died": int(row["Hospital Mortality"]),
                "vals": [float(row[f]) for f in DISPLAY_FEATURES],
            }
        )

    return {
        "features": DISPLAY_FEATURES,
        "feat_min": feat_min,
        "feat_max": feat_max,
        "medians": medians,
        "patients": patients,
    }


def _patch_html(html: str, data: dict) -> str:
    data_blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html, n = re.subn(
        r"const DATA = \{.*?\};",
        f"const DATA = {data_blob};",
        html,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("Could not replace DATA blob in calculator template")

    html = re.sub(
        r"const CANDIDATE_IDS = \[.*?\];",
        f"const CANDIDATE_IDS = {json.dumps(CANDIDATE_IDS)};",
        html,
        count=1,
    )
    notes_js = json.dumps(CANDIDATE_NOTES, ensure_ascii=False, indent=2)
    notes_js = notes_js.replace("\n", "\n  ")
    html = re.sub(
        r"const CANDIDATE_NOTES = \{.*?\};",
        f"const CANDIDATE_NOTES = {notes_js};",
        html,
        count=1,
        flags=re.DOTALL,
    )
    return html


def main() -> None:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    df, probs, threshold = _load_model_and_probs()
    data = _build_data(df, probs)

    edge_pos = df.index.get_loc(CURRENT_EDGE_ID)
    edge_prob = probs[edge_pos]
    print(f"Edge patient #{CURRENT_EDGE_ID}: P(died)={edge_prob*100:.2f}%  threshold={threshold:.4f}")
    print(f"Embedded {len(data['patients'])} patients, {len(data['features'])} display features")

    html = TEMPLATE.read_text(encoding="utf-8")
    html = _patch_html(html, data)

    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
