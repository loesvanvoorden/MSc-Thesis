"""
Co-12 Continuity: optional slight input perturbation for LLM + chart context.

Set environment variable PERTURBATION=1 (or true/yes/on) to enable.
When enabled, perturbed inputs are run through a batch-1 KAN forward pass;
stap 3 then shows the updated probability, dynamically re-selected top-6+/6−
panels, Bijdrage %, patient markers, and LLM context. Steps 1–2 profile
chips and displayed baseline probability stay at the original #8259 case.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np

PERTURBATION = os.environ.get("PERTURBATION", "").strip().lower() in {
    "1", "true", "yes", "on",
}

# Model column name → perturbed raw value (same units as original_df).
# Base: patient #8259. Keys not listed are left unchanged when perturbation is off;
# when on, every model input below is set explicitly.
EDGE_PERTURBATION_MAP: Dict[str, float] = {
    # Primary continuity probes (thesis §3.4)
    "PTT (s)": 85.0,              # aPTT: 91.6 → 85 s
    "Heart Rate (bpm)": 70.0,     # hartslag: 74 → 70 /min
    "GCS": 13.0,                  # GCS: 14 → 13
    # Minor perturbations on remaining model inputs
    "Temperature (°C)": 36.4,
    "Respiratory Rate": 28.0,
    "SpO2": 95.0,
    "SBP (mmHg)": 105.0,
    "DBP (mmHg)": 68.0,
    "MBP (mmHg)": 73.0,
    "WBC (K/uL)": 13.0,
    "Hemoglobin (g/dL)": 12.8,
    "Platelet (k/uL)": 210.0,
    "BUN (mg/dL)": 21.5,
    "Creatinine (mg/dL)": 0.82,
    "Glucose (mg/dL)": 87.0,
    "Sodium (mEq/L)": 138.0,
    "Chloride (mEq/L)": 102.0,
    "Potassium (mEq/L)": 5.3,
    "Magnesium (mg/dL)": 2.25,
    "Calcium (mg/dL)": 8.95,
    "Phosphate (mg/dL)": 4.4,
    "INR": 1.18,
    "Prothrombin Time (s)": 13.7,
    "Bicarbonate (mEq/L)": 25.0,
    "Anion Gap (mEq/L)": 16.0,
    "Age": 68.0,
    "Weight (kg)": 74.5,
    "Gender (Male)": 1.0,
    "SOFA Score": 2.0,
}


def apply_perturbation(
    row_vals_raw: np.ndarray,
    feature_names: Sequence[str],
    *,
    enabled: bool = PERTURBATION,
) -> np.ndarray:
    """Return a copy of row_vals_raw with EDGE_PERTURBATION_MAP applied."""
    out = np.array(row_vals_raw, dtype=float, copy=True)
    if not enabled:
        return out
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for feat, val in EDGE_PERTURBATION_MAP.items():
        idx = name_to_idx.get(feat)
        if idx is not None:
            out[idx] = float(val)
    return out
