"""
preprocess.py
-------------
Loads and preprocesses the MIMIC-IV dataset for hospital mortality (hosp_mort)
prediction. Adapted from Tara van Mierlo's KAN_on_SAD.ipynb.

Key differences from the SAD setup:
  - Target is `hosp_mort` instead of `sad`
  - `icu28dmort` is dropped (another mortality outcome — leakage)
  - `hosp_mort` is NOT dropped from the dataset (it is the target)
  - Feature count: 29 (vitals + labs + demographics + clinical scores only)
  - Excluded: race/ethnicity, ICU department, active interventions, comorbidities

Included features (29 total):
  - Demographics : age, weight, gender
  - Vitals       : temperature, heart_rate, resp_rate, spo2, sbp, dbp, mbp
  - Labs         : wbc, hemoglobin, platelet, bun, cr, glu, Na, Cl, K, Mg, Ca, P,
                   inr, pt, ptt, bicarbonate, aniongap
  - Clinical scores: gcs, sofa_score
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = "../data/MIMIC-IV.csv"
TARGET_COL = "Hospital Mortality"

# Columns to drop because they belong to excluded feature groups
COLS_EXCLUDED = [
    # Active interventions
    "vent", "crrt", "vaso", "seda",
    # Comorbidities
    "ami", "ckd", "copd", "hyperte", "dm", "sad", "aki", "stroke",
    # Race/ethnicity and ICU department are dropped directly (not one-hot encoded)
    "race", "first_careunit",
]

# Indices into the processed feature matrix (order: [continuous | binary | ordinal]).
#
# After all drops, columns in x are:
#   0: age,  1: weight,  2: gender,  3: temperature,  4: heart_rate,
#   5: resp_rate,  6: spo2,  7: sbp,  8: dbp,  9: mbp,
#  10: wbc, 11: hemoglobin, 12: platelet, 13: bun, 14: cr,
#  15: glu, 16: Na, 17: Cl, 18: K, 19: Mg, 20: Ca, 21: P,
#  22: inr, 23: pt, 24: ptt, 25: bicarbonate, 26: aniongap,
#  27: gcs,  28: sofa_score
#
# Continuous (26 features, indices 0-26, skipping gender at 2):
CONTINUOUS_INDICES = [
    0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
]

# Ordinal (2 features):
#   gcs (27), sofa_score (28)
ORDINAL_INDICES = [27, 28]

# Binary (1 feature):
#   gender (2)
BINARY_INDICES = [2]

COLUMN_RENAMES = {
    "age":          "Age",
    "weight":       "Weight (kg)",
    "gender":       "Gender (Male)",
    "temperature":  "Temperature (°C)",
    "heart_rate":   "Heart Rate (bpm)",
    "resp_rate":    "Respiratory Rate",
    "spo2":         "SpO2",
    "sbp":          "SBP (mmHg)",
    "dbp":          "DBP (mmHg)",
    "mbp":          "MBP (mmHg)",
    "wbc":          "WBC (K/uL)",
    "hemoglobin":   "Hemoglobin (g/dL)",
    "platelet":     "Platelet (k/uL)",
    "bun":          "BUN (mg/dL)",
    "cr":           "Creatinine (mg/dL)",
    "glu":          "Glucose (mg/dL)",
    "Na":           "Sodium (mEq/L)",
    "Cl":           "Chloride (mEq/L)",
    "K":            "Potassium (mEq/L)",
    "Mg":           "Magnesium (mg/dL)",
    "Ca":           "Calcium (mg/dL)",
    "P":            "Phosphate (mg/dL)",
    "inr":          "INR",
    "pt":           "Prothrombin Time (s)",
    "ptt":          "PTT (s)",
    "bicarbonate":  "Bicarbonate (mEq/L)",
    "aniongap":     "Anion Gap (mEq/L)",
    "gcs":          "GCS",
    "sofa_score":   "SOFA Score",
    "hosp_mort":    TARGET_COL,
}


def load_and_preprocess(
    data_path: str = DATA_PATH,
    test_size: float = 0.3,
    random_state: int = 42,
    batch_size: int = 512,
    resample_strategy: str = "combined",
) -> dict:
    """
    Load the MIMIC-IV CSV, preprocess it for hosp_mort prediction, and
    return tensors + metadata ready for KAN training.

    Resampling is applied to the training split only. The test set is always
    left untouched so evaluation reflects the real-world distribution.
    Features: 29 (vitals, labs, demographics, clinical scores).

    resample_strategy options:
      'none'        – no resampling; use class_weights in the loss instead
      'smote'       – SMOTE only: oversample minority to 1:1
      'undersample' – RandomUnderSampler only: downsample majority to 1:1
      'combined'    – SMOTE minority up to 1:2, then undersample majority
                      down to 1:1. Keeps more data than pure undersample
                      while avoiding the extreme synthetic inflation of pure SMOTE.

    Returns a dict with keys:
        dataset         : dict with train_input/label, test_input/label tensors
        train_loader    : DataLoader for batch training
        test_loader     : DataLoader for evaluation
        original_data   : original (unscaled) DataFrame in feature order
        feature_labels  : list of feature column names in processed order
        n_features      : int, number of input features (29)
        class_weights   : torch.Tensor [w_survived, w_died] computed from the
                          raw (pre-resampling) training split — pass to
                          CrossEntropyLoss(weight=...) when strategy='none'
    """
    data = pd.read_csv(data_path)

    # ------------------------------------------------------------------ #
    # Drop post-hoc / leakage columns                                      #
    # ------------------------------------------------------------------ #
    data = data.drop([
        "icu28dmort",   # another mortality outcome — direct leakage
        "stay_id",      # patient identifier
        "icustay",      # post-hoc: length of ICU stay
        "hospstay",     # post-hoc: length of hospital stay
        "deliriumtime", # post-hoc: time of delirium onset
        "sepsistime",   # post-hoc: time of sepsis onset
    ], axis=1).dropna()

    # ------------------------------------------------------------------ #
    # Save display-only columns BEFORE dropping them                       #
    # These are not used for prediction but shown in the patient profile.  #
    # ------------------------------------------------------------------ #
    # One-hot encode race and careunit for display
    race_dummies     = pd.get_dummies(data["race"]).rename(columns={
        "AISAN": "Race: Asian", "BLACK": "Race: Black",
        "HISPANIC": "Race: Hispanic", "OTHER": "Race: Other",
        "WHITE": "Race: White", "unknown": "Race: Unknown",
    })
    careunit_dummies = pd.get_dummies(data["first_careunit"]).rename(columns={
        "CCU": "ICU: CCU", "CVICU": "ICU: CVICU", "MICU": "ICU: MICU",
        "MICU/SICU": "ICU: MICU/SICU", "NICU": "ICU: NICU",
        "SICU": "ICU: SICU", "TSICU": "ICU: TSICU",
    })
    display_cols = pd.concat([
        data[["vent", "crrt", "vaso", "seda",
              "ami", "ckd", "copd", "hyperte", "dm", "sad", "aki", "stroke"]]
            .rename(columns={
                "vent": "Mechanical Ventilation", "crrt": "CRRT",
                "vaso": "Vasopressor", "seda": "Sedation",
                "ami": "AMI", "ckd": "CKD", "copd": "COPD",
                "hyperte": "Hypertension", "dm": "Diabetes",
                "sad": "SAD", "aki": "AKI", "stroke": "Stroke",
            }),
        race_dummies,
        careunit_dummies,
    ], axis=1)

    # ------------------------------------------------------------------ #
    # Drop excluded feature groups from the model input                    #
    # ------------------------------------------------------------------ #
    data = data.drop(COLS_EXCLUDED, axis=1)

    # ------------------------------------------------------------------ #
    # Rename columns to readable English names                            #
    # ------------------------------------------------------------------ #
    data = data.rename(columns=COLUMN_RENAMES)

    # ------------------------------------------------------------------ #
    # Separate features and target                                         #
    # ------------------------------------------------------------------ #
    x = data.drop(TARGET_COL, axis=1).values
    y = data[TARGET_COL].values

    # ------------------------------------------------------------------ #
    # Scale / encode each feature type                                     #
    # ------------------------------------------------------------------ #
    continuous_data = x[:, CONTINUOUS_INDICES].astype(np.float32)
    scaler = MinMaxScaler()
    normalized_continuous = scaler.fit_transform(continuous_data)

    ordinal_data = x[:, ORDINAL_INDICES]
    ordinal_scaler = MinMaxScaler()
    normalized_ordinal = ordinal_scaler.fit_transform(ordinal_data)

    binary_data = x[:, BINARY_INDICES]
    label_encoders = [LabelEncoder() for _ in BINARY_INDICES]
    encoded_binary = np.array(
        [le.fit_transform(binary_data[:, i]) for i, le in enumerate(label_encoders)]
    ).T.astype(np.int64)

    processed = np.hstack((normalized_continuous, encoded_binary, normalized_ordinal))

    # ------------------------------------------------------------------ #
    # Train / test split                                                   #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        processed, y, np.arange(len(data)),
        random_state=random_state,
        test_size=test_size,
        stratify=y,
    )

    # ------------------------------------------------------------------ #
    # Class weights — computed from raw training split (before resampling) #
    # Use with CrossEntropyLoss(weight=class_weights) as an alternative   #
    # to data resampling.                                                  #
    # ------------------------------------------------------------------ #
    raw_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_train
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    counts = np.bincount(y_train.astype(int))
    print(f"  Class weights: survived={raw_weights[0]:.3f}  died={raw_weights[1]:.3f}"
          f"  (ratio 1:{raw_weights[1]/raw_weights[0]:.1f})")

    # ------------------------------------------------------------------ #
    # Resampling — training set only                                       #
    # ------------------------------------------------------------------ #
    before = np.bincount(y_train.astype(int))

    if resample_strategy == "smote":
        sampler = SMOTE(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    elif resample_strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    elif resample_strategy == "combined":
        # Step 1: SMOTE minority up to half the majority count (1:2 ratio)
        n_majority = before[0]
        smote_target = n_majority // 2
        smote = SMOTE(
            sampling_strategy={1: smote_target},
            random_state=random_state,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # Step 2: RandomUnderSampler brings majority down to match minority (1:1)
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    elif resample_strategy != "none":
        raise ValueError(
            f"Unknown resample_strategy '{resample_strategy}'. "
            "Choose from: 'none', 'smote', 'undersample', 'combined'."
        )

    after = np.bincount(y_train.astype(int))
    print(f"  Resampling ({resample_strategy}): {before} → {after}")

    # Keep original data in the same column order as the processed matrix
    feature_cols = data.drop(TARGET_COL, axis=1).columns.tolist()
    continuous_labels = [feature_cols[i] for i in CONTINUOUS_INDICES]
    binary_labels     = [feature_cols[i] for i in BINARY_INDICES]
    ordinal_labels    = [feature_cols[i] for i in ORDINAL_INDICES]
    ordered_cols      = continuous_labels + binary_labels + ordinal_labels

    original_data = data[[*ordered_cols, TARGET_COL]].copy()
    # Attach display-only columns (not used for prediction).
    # Reset index on display_cols so it aligns with data after dropna().
    display_cols.index = data.index
    original_data = original_data.join(display_cols)

    # ------------------------------------------------------------------ #
    # Convert to tensors                                                   #
    # ------------------------------------------------------------------ #
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=len(X_test_t), shuffle=False
    )

    dataset = {
        "train_input": X_train_t,
        "train_label": y_train_t,
        "test_input":  X_test_t,
        "test_label":  y_test_t,
    }

    return {
        "dataset":        dataset,
        "train_loader":   train_loader,
        "test_loader":    test_loader,
        "original_data":  original_data,
        "feature_labels": ordered_cols,
        "n_features":     processed.shape[1],
        "class_weights":  class_weights,
    }


if __name__ == "__main__":
    for strategy in ("none", "undersample", "smote", "combined"):
        print(f"\n--- strategy: {strategy} ---")
        result = load_and_preprocess(resample_strategy=strategy)
        pos   = result['dataset']['train_label'].sum().item()
        total = len(result['dataset']['train_label'])
        print(f"  Train size : {total}  (+rate {pos/total*100:.1f}%)")
        print(f"  Test  size : {len(result['dataset']['test_input'])}")
