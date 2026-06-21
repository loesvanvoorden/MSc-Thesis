"""
MortalityAlert – Flask interface (3-stap studieopzet)

Groep A (XAI):  Stap 1 → Stap 2 → Stap 3 (profiel + voorspelling + XAI)
Groep B (LLM):  Stap 1 → Stap 2 → Stap 3 (profiel + voorspelling + XAI + LLM)

Elke stap: klinische risico-inschatting + open tekstveld.
Randpatiënt (~65 % voorspeld mortaliteitsrisico) voor beide groepen.
"""

import os, io, base64, re
from pathlib import Path

from dotenv import load_dotenv

# Load interface/.env for local dev (OPENAI_API_KEY, SECRET_KEY, …). Not used if vars already set.
load_dotenv(Path(__file__).resolve().parent / ".env")

import json
import pickle
import sqlite3
import csv
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request, session, jsonify, send_file
from werkzeug.middleware.proxy_fix import ProxyFix

import torch
import torch.nn.functional as F
from kan import MultKAN

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mortalityalert-studie-2025")
# Reverse proxies: honour X-Forwarded-* so request.scheme/host stay correct.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1
)

# ── Persistent storage ────────────────────────────────────────────────────────
DATA_DIR       = os.environ.get("DATA_DIR",
                    os.path.join(os.path.dirname(__file__), "data"))
DB_PATH        = os.path.join(DATA_DIR, "responses.db")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "2055036")

# Randpatient calculator HTML (manager-only). Bundled copy lives under static/tools/.
# Override path via EDGE_PATIENT_CALCULATOR_HTML if needed.
_INTERFACE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _INTERFACE_DIR.parent
_BUNDLED_EDGE_CALC = _INTERFACE_DIR / "static" / "tools" / "edge_patient_calculator.html"
_REPO_EDGE_CALC = (
    _REPO_ROOT / "hospMort" / "scripts" / "results" / "edge_patient_calculator.html"
)


def _default_edge_calculator_path() -> Path:
    if _BUNDLED_EDGE_CALC.is_file():
        return _BUNDLED_EDGE_CALC
    return _REPO_EDGE_CALC


EDGE_PATIENT_CALCULATOR_HTML = Path(
    os.environ.get(
        "EDGE_PATIENT_CALCULATOR_HTML",
        str(_default_edge_calculator_path()),
    )
).expanduser()


def _init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS deelnemers (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT,
                groep          TEXT,
                functie        TEXT,
                stap1_risico   TEXT, stap1_notities TEXT, stap1_handelen TEXT,
                stap2_risico   TEXT, stap2_notities TEXT, stap2_handelen TEXT,
                stap3_risico   TEXT, stap3_notities TEXT, stap3_handelen TEXT,
                stap3_llm_chats TEXT,
                cas1 TEXT, cas2 TEXT, cas3 TEXT, cas4 TEXT, cas5 TEXT
            )
        """)
        # Add CAS columns to pre-existing tables (safe to call multiple times)
        for col in ["cas1", "cas2", "cas3", "cas4", "cas5"]:
            try:
                con.execute(f"ALTER TABLE deelnemers ADD COLUMN {col} TEXT")
            except Exception:
                pass
        try:
            con.execute("ALTER TABLE deelnemers ADD COLUMN stap3_llm_chats TEXT")
        except Exception:
            pass
        con.commit()


_init_db()


def _assign_group() -> bool:
    """Balanced A/B assignment. Returns True = Group B (LLM on)."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT groep, COUNT(*) FROM deelnemers GROUP BY groep"
        ).fetchall()
    counts = {r[0]: r[1] for r in rows}
    n_a = counts.get("A", 0)
    n_b = counts.get("B", 0)
    return n_b < n_a   # assign to the smaller group; ties go to A


_MAX_LL_CHAT_JSON_BYTES = 2 * 1024 * 1024


def _sanitize_llm_chats_json(raw: str) -> str:
    """
    Validate JSON from stap 3 hidden field: per-graph chat transcripts (Groep B).
    Drops embedded image data; keeps attachment titles/mimes only.
    """
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if len(raw.encode("utf-8")) > _MAX_LL_CHAT_JSON_BYTES:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    out: dict = {}
    for gid in ("3",):
        arr = data.get(gid)
        if arr is None:
            continue
        if not isinstance(arr, list):
            continue
        clean_msgs = []
        for m in arr:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content")
            if not isinstance(content, str):
                content = str(content or "")
            if len(content) > 800_000:
                content = content[:800_000]
            entry = {"role": role, "content": content}
            att = m.get("attachments")
            if isinstance(att, list) and att:
                safe_att = []
                for a in att[:25]:
                    if isinstance(a, dict):
                        safe_att.append(
                            {
                                "title": str(a.get("title", ""))[:500],
                                "mime": str(a.get("mime", ""))[:120],
                            }
                        )
                if safe_att:
                    entry["attachments"] = safe_att
            clean_msgs.append(entry)
        out[gid] = clean_msgs
    return json.dumps(out, ensure_ascii=False)


_LL_PANEL_LABELS = {
    "3": "Toelichting — modeluitleg",
}


def _llm_chats_panels_for_display(raw_json: str) -> list:
    """Parse stap3_llm_chats into UI-friendly panels (Groep B)."""
    if not raw_json or not isinstance(raw_json, str):
        return []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return [
            {
                "panel_key": "?",
                "panel_label": "LLM-gesprek",
                "turns": [],
                "parse_error": True,
            }
        ]
    if not isinstance(data, dict):
        return []
    out = []
    for gid in ("3",):
        msgs = data.get(gid)
        if not isinstance(msgs, list):
            continue
        turns = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            att = m.get("attachments")
            if isinstance(att, list) and att:
                titles = [
                    str(a.get("title", "")).strip()
                    for a in att
                    if isinstance(a, dict) and str(a.get("title", "")).strip()
                ]
                if titles:
                    content = (
                        content
                        + ("\n\n" if content else "")
                        + "[Bijgevoegde figuur(en): "
                        + "; ".join(titles[:8])
                        + "]"
                    )
                else:
                    content = content + ("\n\n" if content else "") + "[Bijlage bij dit antwoord]"
            turns.append({"role": role, "content": content})
        label = _LL_PANEL_LABELS.get(gid, f"Paneel {gid}")
        out.append({"panel_key": gid, "panel_label": label, "turns": turns})
    return out


def _participant_display_dict(row: sqlite3.Row) -> dict:
    """sqlite Row → dict + parsed LLM panels for templates."""
    d = dict(row)
    panels = _llm_chats_panels_for_display(d.get("stap3_llm_chats") or "")
    d["_llm_panels"] = panels
    d["_llm_has_turns"] = any(len(x.get("turns") or []) > 0 for x in panels)
    d["_llm_parse_error"] = any(x.get("parse_error") for x in panels)
    return d


def _save_response():
    """Write completed session to the database (called once on /klaar)."""
    ts    = datetime.now(timezone.utc).isoformat()
    groep = "B" if session.get("llm_aan") else "A"
    s1    = session.get("stap1", {})
    s2    = session.get("stap2", {})
    s3    = session.get("stap3", {})
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO deelnemers
              (timestamp, groep, functie,
               stap1_risico, stap1_notities, stap1_handelen,
               stap2_risico, stap2_notities, stap2_handelen,
               stap3_risico, stap3_notities, stap3_handelen,
               stap3_llm_chats,
               cas1, cas2, cas3, cas4, cas5)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, groep, session.get("functie", ""),
              s1.get("risico", ""), s1.get("notities", ""), s1.get("handelen", ""),
              s2.get("risico", ""), s2.get("notities", ""), s2.get("handelen", ""),
              s3.get("risico", ""), s3.get("notities", ""), s3.get("handelen", ""),
              s3.get("llm_chats_json", ""),
              s3.get("cas1", ""), s3.get("cas2", ""), s3.get("cas3", ""),
              s3.get("cas4", ""), s3.get("cas5", "")))
        con.commit()

# ══════════════════════════════════════════════════════════════════════════════
#  Model & data laden (eenmalig bij opstarten)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "model")
)


def _load_model():
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    shape, grid, k = [29, 3, 2], 3, 3   # defaults — overridden by saved config
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
            shape = cfg.get("kan_shape", shape)
            grid  = cfg.get("grid",  grid)
            k     = cfg.get("k",     k)
    if isinstance(shape[0], list):   # pykan saves shape as [[29,0],[3,0],[2,0]]
        shape = [s[0] for s in shape]
    m = MultKAN(width=shape, grid=grid, k=k, seed=42)
    # Use state_dict — saveckpt/loadckpt does not reliably restore trained weights
    weights_path = os.path.join(MODEL_DIR, "kan_weights.pt")
    state = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    scores = np.load(os.path.join(MODEL_DIR, "feature_scores.npy"))
    return m, scores


def _load_data():
    with open(os.path.join(MODEL_DIR, "original_df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_labels.json")) as f:
        feature_labels = json.load(f)
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    threshold = 0.5
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            threshold = json.load(f).get("threshold", 0.5)
    return df, feature_labels, threshold


print("▶ Model laden…")
MODEL, FEATURE_SCORES = _load_model()
DF, FEATURE_NAMES, THRESHOLD = _load_data()
print(f"✓ Klaar  (drempelwaarde = {THRESHOLD:.3f})")

TARGET_COL = "Hospital Mortality"

# ── Per-feature min/max for scaling raw → [0, 1] ─────────────────────────────
# DF holds raw (unscaled) values; the KAN was trained on 0–1 scaled inputs.
# Binary model features stay as-is; all others are min-max scaled.
_FEAT_MIN   = np.array([float(DF[f].min()) for f in FEATURE_NAMES])
_FEAT_MAX   = np.array([float(DF[f].max()) for f in FEATURE_NAMES])
_FEAT_RANGE = np.where(_FEAT_MAX > _FEAT_MIN, _FEAT_MAX - _FEAT_MIN, 1.0)
_BINARY_COLS = {f for f in FEATURE_NAMES if f in {"Gender (Male)"}}

# Hidden from participant UI, charts, and LLM toelichting (still in model inputs).
_DEMO_HIDDEN_FEATURES = frozenset({"Age", "Weight (kg)", "Gender (Male)"})


def _is_hidden_feature(feat_name: str) -> bool:
    return feat_name in _DEMO_HIDDEN_FEATURES


def _visible_feature_indices() -> list:
    return [i for i, f in enumerate(FEATURE_NAMES) if not _is_hidden_feature(f)]


def _scale_features(raw: np.ndarray) -> np.ndarray:
    """Min-max scale a 1-D or 2-D raw feature array to [0, 1]."""
    scaled = (raw - _FEAT_MIN) / _FEAT_RANGE
    # Restore binary columns unchanged
    for feat in _BINARY_COLS:
        if feat in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(feat)
            if raw.ndim == 1:
                scaled[idx] = raw[idx]
            else:
                scaled[:, idx] = raw[:, idx]
    return scaled.astype(np.float32)


# Voorspellingen voor alle patiënten + spline cache vullen
_X_RAW    = DF[FEATURE_NAMES].values.astype(float)
_X_SCALED = _scale_features(_X_RAW)
_X = torch.tensor(_X_SCALED, dtype=torch.float32)
with torch.no_grad():
    _OUT = MODEL(_X)   # also populates MODEL.spline_preacts / spline_postacts
PROBS_ALL = F.softmax(_OUT, dim=1).numpy()   # kolom 1 = P(died)

# Extract and cache the layer-0 spline curves.
# spline_preacts[0]  shape: [N, n_hidden, n_features]  — input x (same for all hidden nodes)
# spline_postacts[0] shape: [N, n_hidden, n_features]  — φ(x) per hidden node
# We sum postacts across hidden nodes to get the total contribution of each feature.
SPLINE_X = MODEL.spline_preacts[0][:, 0, :].detach().numpy()          # [N, n_features]
SPLINE_Y = MODEL.spline_postacts[0].sum(dim=1).detach().numpy()        # [N, n_features]

# ── Randpatiënt: vastgepinde studiepatiënt (zie build_edge_calculator.py) ──
EDGE_ID   = 8259
_edge_pos = int(np.where(DF.index == EDGE_ID)[0][0])
EDGE_PROB = float(PROBS_ALL[_edge_pos, 1])
print(f"✓ Randpatiënt: #{EDGE_ID}  |  P(died) = {EDGE_PROB:.1%}")

from perturbation import PERTURBATION, EDGE_PERTURBATION_MAP, apply_perturbation

# Klinische context op stap 1 (vóór vitale parameters en laboratoriumwaarden)
EDGE_CASUS_TEKST = (
    "Het betreft een 68-jarige man met in de voorgeschiedenis een hemiplegie met "
    "decubitus graad 3, status na pneumonectomie in verband met longcarcinoom "
    "(curatief behandeld), diastolische dysfunctie, atriumfibrilleren en hypertensie. "
    "Hij presenteerde zich vier dagen geleden via de spoedeisende hulp in verband met "
    "een pneumonie en werd opgenomen op de afdeling hematologie/oncologie. In verband "
    "met een verhoogde work of breathing in combinatie met zorgzwaarte is patiënt op "
    "dag 4 opgenomen op de IC voor observatie. Vier uur na IC-opname werd sepsis "
    "vastgesteld. De antibiotica is gecontinueerd.\n\n"
    "U ziet de vitale parameters en laboratoriumwaarden 8 uur na IC-opname."
)

# ── Klinische uitleg per variabele (key = model column name) ───────────────
FEATURE_UITLEG = {
    "Age":                    "Leeftijd van de patiënt in jaren.",
    "Weight (kg)":            "Lichaamsgewicht in kilogram.",
    "Temperature (°C)":       "Lichaamstemperatuur.",
    "Heart Rate (bpm)":       "Hartslagen per minuut.",
    "Respiratory Rate":       "Ademhalingen per minuut.",
    "SpO2":                   "Perifere zuurstofverzadiging via pulsoximetrie.",
    "SBP (mmHg)":             "Bovendruk bij bloeddrukmeting.",
    "DBP (mmHg)":             "Onderdruk bij bloeddrukmeting.",
    "MBP (mmHg)":             "Gemiddelde arteriële bloeddruk.",
    "WBC (K/uL)":             "Witte bloedcellen; maat voor afweerreactie.",
    "Hemoglobin (g/dL)":      "Zuurstoftransporterend eiwit in rode bloedcellen.",
    "Platelet (k/uL)":        "Bloedplaatjes; betrokken bij bloedstolling.",
    "BUN (mg/dL)":            "Stikstofhoudende afvalstof; maat voor nierfunctie.",
    "Creatinine (mg/dL)":     "Afvalstof van spiermetabolisme; maat voor glomerulaire filtratiesnelheid.",
    "Glucose (mg/dL)":        "Bloedsuikerconcentratie.",
    "Sodium (mEq/L)":         "Natriumconcentratie; regulatie van vochtbalans.",
    "Chloride (mEq/L)":       "Chlorideconcentratie in bloed.",
    "Potassium (mEq/L)":      "Kaliumconcentratie; essentieel voor hart- en spierfunctie.",
    "Magnesium (mg/dL)":      "Magnesiumconcentratie; betrokken bij spier- en zenuwfunctie.",
    "Calcium (mg/dL)":        "Totale calciumconcentratie; botopbouw en spierwerking.",
    "Phosphate (mg/dL)":      "Fosfaatconcentratie; energiemetabolisme en botmineralisatie.",
    "INR":                    "International Normalized Ratio: gestandaardiseerde maat voor bloedstollingstijd.",
    "Prothrombin Time (s)":   "Protrombinetijd: stollingsduur via de extrinsieke route.",
    "PTT (s)":                "Geactiveerde partiële tromboplastinetijd: stolling via de intrinsieke route.",
    "Bicarbonate (mEq/L)":    "Bicarbonaat in bloed; maat voor metabole zuur-base balans.",
    "Anion Gap (mEq/L)":      "Verschil tussen gemeten kationen en anionen; verhoogd bij metabole acidose.",
    "GCS":                    "Glasgow Coma Scale: bewustzijnsniveau (3 = diep coma, 15 = volledig alert).",
    "SOFA Score":             "Sequential Organ Failure Assessment: samengestelde score voor orgaanfalen (0–24).",
    "Gender (Male)":          "Of de patiënt mannelijk is. Binaire variabele (ja / nee).",
    "Mechanical Ventilation": "Of de patiënt mechanisch beademd wordt (invasief of non-invasief). Binaire variabele.",
    "CRRT":                   "Continue niervervangingstherapie bij acuut nierfalen. Binaire variabele.",
    "Vasopressor":            "Of de patiënt vaatvernauwende medicatie ontvangt om de bloeddruk op peil te houden. Binaire variabele.",
    "Sedation":               "Of de patiënt kalmerende of verdovende medicatie ontvangt. Binaire variabele.",
    "AMI":                    "Acuut myocardinfarct (hartaanval) in de voorgeschiedenis. Binaire variabele.",
    "CKD":                    "Chronische nierziekte in de voorgeschiedenis. Binaire variabele.",
    "COPD":                   "Chronische obstructieve longziekte in de voorgeschiedenis. Binaire variabele.",
    "Hypertension":           "Hoge bloeddruk in de voorgeschiedenis. Binaire variabele.",
    "Diabetes":               "Suikerziekte (type 1 of 2) in de voorgeschiedenis. Binaire variabele.",
    "SAD":                    "Sepsis-geassocieerd delier: acuut optredende cognitieve stoornis bij sepsis. Binaire variabele.",
    "AKI":                    "Plotselinge verslechtering van nierfunctie. Binaire variabele.",
    "Stroke":                 "Hersenbloeding of herseninfarct in de voorgeschiedenis. Binaire variabele.",
    "ICU: CCU":               "Coronary Care Unit – gespecialiseerde bewaking voor hartpatiënten.",
    "ICU: CVICU":             "Cardiovascular ICU – intensive care na cardiovasculaire chirurgie.",
    "ICU: MICU":              "Medical ICU – medische intensive care voor niet-chirurgische patiënten.",
    "ICU: MICU/SICU":         "Medical/Surgical ICU – gecombineerde medische en chirurgische intensive care.",
    "ICU: NICU":              "Neuroscience ICU (NICU) – intensive care voor neurologische en neurochirurgische aandoeningen. Let op: in MIMIC-IV staat NICU voor Neuroscience ICU, niet Neonatale ICU.",
    "ICU: SICU":              "Surgical ICU – postoperatieve intensive care na grote chirurgische ingrepen.",
    "ICU: TSICU":             "Trauma/Surgical ICU – intensive care voor traumapatiënten en na spoedchirurgie.",
    "Race: Asian":            "Patiënt heeft een Aziatische achtergrond.",
    "Race: Black":            "Patiënt heeft een Afrikaanse achtergrond.",
    "Race: Hispanic":         "Patiënt heeft een Latijns-Amerikaanse achtergrond.",
    "Race: Other":            "Patiënt heeft een andere of niet-gespecificeerde achtergrond.",
    "Race: White":            "Patiënt heeft een Europese of westerse achtergrond.",
    "Race: Unknown":          "Achtergrond van de patiënt is niet geregistreerd of onbekend.",
}

# ── Referentiewaarden voor volwassen IC-patiënten ────────────────────────────
# Labwaarden: OLVG Lab referentiewaarden (volwassenen), omgezet naar MIMIC-IV eenheden.
# Vitale waarden: standaard klinische normen.
# Tuple: (laag, hoog) — None betekent geen grens aan die kant.
_FEATURE_NORMAL_RANGES: dict[str, tuple] = {
    # elektrolyten / zuur-base (OLVG, volwassenen >1 jaar / >15 jaar)
    "Potassium (mEq/L)":      (3.5,  5.0),
    "Sodium (mEq/L)":         (135., 147.),
    "Chloride (mEq/L)":       (96.,  109.),
    "Bicarbonate (mEq/L)":    (22.,  29.),
    "Anion Gap (mEq/L)":      (9.,   14.),
    # overige biochemie — omgezet van OLVG mmol/l → mg/dL of µmol/l → mg/dL
    "Magnesium (mg/dL)":      (1.7,  2.4),   # OLVG 0.70–1.00 mmol/l × 2.43
    "Glucose (mg/dL)":        (72.,  110.),   # OLVG 4.0–6.1 mmol/l × 18 (nuchter)
    "BUN (mg/dL)":            (6.,   20.),    # OLVG ureum 2.1–7.1 mmol/l × 2.80
    "Creatinine (mg/dL)":     (0.5,  1.2),   # OLVG 45–104 µmol/l ÷ 88.4, gecombineerd M/V
    # hemostase (OLVG)
    "PTT (s)":                (23.,  30.),    # OLVG aPTT 23.3–30.1 s
    "Prothrombin Time (s)":   (10.,  12.2),   # OLVG PT 10.0–12.2 s
    "INR":                    (0.9,  1.1),
    # hematologie (OLVG, >15 jaar)
    "Hemoglobin (g/dL)":      (12.0, 17.7),   # OLVG 7.5–11.0 mmol/l × 1.611, gecombineerd M/V
    "WBC (K/uL)":             (4.0,  10.0),   # OLVG leukocyten >15 jaar
    "Platelet (k/uL)":        (150., 400.),   # OLVG trombocyten >15 jaar
    # vitale parameters — standaard klinisch
    "Respiratory Rate":       (12.,  20.),
    "Heart Rate (bpm)":       (60.,  100.),
    "Temperature (°C)":       (36.1, 37.2),
    "SBP (mmHg)":             (90.,  140.),
    "DBP (mmHg)":             (60.,  90.),
    "MBP (mmHg)":             (70.,  100.),
    "SpO2":                   (95.,  None),   # ≥95 %
    # klinische scores
    "GCS":                    (13.,  15.),    # ≥13 = mild/normaal bewustzijn
    "SOFA Score":             (None, 2.),     # ≤2 = normaal orgaanfunctie
}

# ── Weergave: één bron voor profiel, glossary en grafieken ─────────────────
# conv: raw model value → clinician-facing number; unit shown in UI / chart axes
FEATURE_DISPLAY = {
    "Age":                    {"label": "Leeftijd", "unit": "jaar", "conv": lambda v: v, "dec": 0},
    "Weight (kg)":            {"label": "Gewicht", "unit": "kg", "conv": lambda v: v, "dec": 0},
    "Heart Rate (bpm)":       {"label": "Hartslag", "unit": "/min", "conv": lambda v: v, "dec": 0},
    "Respiratory Rate":       {"label": "Ademfrequentie", "unit": "/min", "conv": lambda v: v, "dec": 0},
    "Temperature (°C)":       {"label": "Temperatuur", "unit": "°C", "conv": lambda v: v, "dec": 1},
    "SpO2":                   {"label": "SpO₂", "unit": "%", "conv": lambda v: v, "dec": 0},
    "SBP (mmHg)":             {"label": "Systolische BD", "unit": "mmHg", "conv": lambda v: v, "dec": 0},
    "DBP (mmHg)":             {"label": "Diastolische BD", "unit": "mmHg", "conv": lambda v: v, "dec": 0},
    "MBP (mmHg)":             {"label": "MAP", "unit": "mmHg", "conv": lambda v: v, "dec": 0},
    "WBC (K/uL)":             {"label": "Leukocyten", "unit": "×10⁹/L", "conv": lambda v: v, "dec": 1},
    "Hemoglobin (g/dL)":      {"label": "Hemoglobine", "unit": "mmol/L", "conv": lambda v: v * 0.6206, "dec": 1},
    "Platelet (k/uL)":        {"label": "Trombocyten", "unit": "×10⁹/L", "conv": lambda v: v, "dec": 0},
    "Creatinine (mg/dL)":     {"label": "Creatinine", "unit": "µmol/L", "conv": lambda v: v * 88.4, "dec": 0},
    "BUN (mg/dL)":            {"label": "Ureum", "unit": "mmol/L", "conv": lambda v: v * 0.357, "dec": 1},
    "Glucose (mg/dL)":        {"label": "Glucose", "unit": "mmol/L", "conv": lambda v: v * 0.0555, "dec": 1},
    "Sodium (mEq/L)":         {"label": "Natrium", "unit": "mmol/L", "conv": lambda v: v, "dec": 0},
    "Chloride (mEq/L)":       {"label": "Chloride", "unit": "mmol/L", "conv": lambda v: v, "dec": 0},
    "Potassium (mEq/L)":      {"label": "Kalium", "unit": "mmol/L", "conv": lambda v: v, "dec": 1},
    "Bicarbonate (mEq/L)":    {"label": "Bicarbonaat", "unit": "mmol/L", "conv": lambda v: v, "dec": 0},
    "Anion Gap (mEq/L)":      {"label": "Anion gap", "unit": "mmol/L", "conv": lambda v: v, "dec": 0},
    "Magnesium (mg/dL)":      {"label": "Magnesium", "unit": "mmol/L", "conv": lambda v: v / 2.431, "dec": 2},
    "Calcium (mg/dL)":        {"label": "Calcium", "unit": "mmol/L", "conv": lambda v: v / 4.0, "dec": 2},
    "Phosphate (mg/dL)":      {"label": "Fosfaat", "unit": "mmol/L", "conv": lambda v: v / 3.1, "dec": 2},
    "INR":                    {"label": "INR", "unit": "", "conv": lambda v: v, "dec": 2},
    "Prothrombin Time (s)":   {"label": "PT", "unit": "s", "conv": lambda v: v, "dec": 1},
    "PTT (s)":                {"label": "aPTT", "unit": "s", "conv": lambda v: v, "dec": 0},
    "GCS":                    {"label": "GCS", "unit": "", "conv": lambda v: v, "dec": 0},
    "SOFA Score":             {"label": "SOFA-score", "unit": "", "conv": lambda v: v, "dec": 0},
    "Gender (Male)":          {"label": "Geslacht (man)", "unit": "", "conv": lambda v: v, "dec": 0},
    "Mechanical Ventilation": {"label": "Mechanische beademing", "unit": "", "conv": lambda v: v, "dec": 0},
    "CRRT":                   {"label": "CRRT", "unit": "", "conv": lambda v: v, "dec": 0},
    "Vasopressor":            {"label": "Vasopressoren", "unit": "", "conv": lambda v: v, "dec": 0},
    "Sedation":               {"label": "Sedatie", "unit": "", "conv": lambda v: v, "dec": 0},
    "AMI":                    {"label": "Acuut myocardinfarct", "unit": "", "conv": lambda v: v, "dec": 0},
    "CKD":                    {"label": "Chronische nierziekte", "unit": "", "conv": lambda v: v, "dec": 0},
    "COPD":                   {"label": "COPD", "unit": "", "conv": lambda v: v, "dec": 0},
    "Hypertension":           {"label": "Hypertensie", "unit": "", "conv": lambda v: v, "dec": 0},
    "Diabetes":               {"label": "Diabetes", "unit": "", "conv": lambda v: v, "dec": 0},
    "SAD":                    {"label": "SAD (delier)", "unit": "", "conv": lambda v: v, "dec": 0},
    "AKI":                    {"label": "Acuut nierfalen", "unit": "", "conv": lambda v: v, "dec": 0},
    "Stroke":                 {"label": "Beroerte", "unit": "", "conv": lambda v: v, "dec": 0},
    "Race: Asian":            {"label": "Afkomst: Aziatisch", "unit": "", "conv": lambda v: v, "dec": 0},
    "Race: Black":            {"label": "Afkomst: Afrikaans", "unit": "", "conv": lambda v: v, "dec": 0},
    "Race: Hispanic":         {"label": "Afkomst: Latijns-Amerikaans", "unit": "", "conv": lambda v: v, "dec": 0},
    "Race: Other":            {"label": "Afkomst: Anders", "unit": "", "conv": lambda v: v, "dec": 0},
    "Race: White":            {"label": "Afkomst: Europees/Westers", "unit": "", "conv": lambda v: v, "dec": 0},
    "Race: Unknown":          {"label": "Afkomst: Onbekend", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: CCU":               {"label": "ICU Type: CCU", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: CVICU":             {"label": "ICU Type: CVICU", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: MICU":              {"label": "ICU Type: MICU", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: MICU/SICU":         {"label": "ICU Type: MICU/SICU", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: NICU":              {"label": "ICU Type: NICU", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: SICU":              {"label": "ICU Type: SICU", "unit": "", "conv": lambda v: v, "dec": 0},
    "ICU: TSICU":             {"label": "ICU Type: TSICU", "unit": "", "conv": lambda v: v, "dec": 0},
}

# ── Nederlandse weergavenamen (English feature name → Dutch display) ───────
NL_LABEL = {k: v["label"] for k, v in FEATURE_DISPLAY.items()}

# ══════════════════════════════════════════════════════════════════════════════
#  Hulpfuncties
# ══════════════════════════════════════════════════════════════════════════════

def _get_edge_row():
    pos = DF.index.get_loc(EDGE_ID)
    return DF.iloc[pos], pos


def _derive_meta(row):
    geslacht = "Man" if row.get("Gender (Male)", 0) == 1 else "Vrouw"
    icu_map = {
        "ICU: CCU":      "CCU",        "ICU: CVICU":    "CVICU",
        "ICU: MICU":     "MICU",       "ICU: MICU/SICU":"MICU/SICU",
        "ICU: NICU":     "Neuro ICU",  "ICU: SICU":     "SICU",
        "ICU: TSICU":    "TSICU",
    }
    icu = next((v for k, v in icu_map.items() if row.get(k, 0) == 1), "Onbekend")
    eth_map = {
        "Race: Asian":    "Aziatisch",
        "Race: Black":    "Afrikaans",
        "Race: Hispanic": "Latijns-Amerikaans",
        "Race: Other":    "Anders",
        "Race: White":    "Europees/Westers",
        "Race: Unknown":  "Onbekend",
    }
    eth = next((v for k, v in eth_map.items() if row.get(k, 0) == 1), "Onbekend")
    return geslacht, icu, eth


def _fmt_chip_num(v: float, decimals: int = 0) -> str:
    """Format a numeric chip value for display."""
    if decimals == 0:
        return f"{round(v):.0f}"
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


def _is_binary(feat_name: str) -> bool:
    if feat_name not in DF.columns:
        return False
    return set(DF[feat_name].dropna().unique()).issubset({0, 1, 0.0, 1.0})


def _display_spec(feat_name: str) -> dict:
    return FEATURE_DISPLAY.get(feat_name, {
        "label": NL_LABEL.get(feat_name, feat_name),
        "unit": "",
        "conv": lambda v: v,
        "dec": 1,
    })


def _display_label(feat_name: str) -> str:
    return _display_spec(feat_name)["label"]


def _raw_to_display(feat_name: str, raw: float) -> float:
    return float(_display_spec(feat_name)["conv"](float(raw)))


def _format_feature_value(feat_name: str, raw: float) -> str:
    """Clinician-facing value string — same rules as patient profile chips."""
    if _is_binary(feat_name):
        return "Aanwezig" if int(round(raw)) == 1 else "Afwezig"
    spec = _display_spec(feat_name)
    disp = float(spec["conv"](float(raw)))
    num = _fmt_chip_num(disp, decimals=spec.get("dec", 1))
    unit = spec.get("unit", "")
    return f"{num} {unit}".strip() if unit else num


def _fmt_val(raw: float, feat_name: str) -> str:
    return _format_feature_value(feat_name, raw)


def _chart_axis_label(feat_name: str) -> str:
    """X-axis title for spline charts (label + unit when applicable)."""
    spec = _display_spec(feat_name)
    label, unit = spec["label"], spec.get("unit", "")
    return f"{label} ({unit})" if unit else label


def _vitals(row):
    """Vitale functies voor de deelnemersinterface."""
    out = []
    for col in (
        "Heart Rate (bpm)",
        "Respiratory Rate",
        "Temperature (°C)",
        "SpO2",
    ):
        if col not in row.index:
            continue
        spec = _display_spec(col)
        disp = float(spec["conv"](float(row[col])))
        out.append({
            "label": spec["label"],
            "value": _fmt_chip_num(disp, decimals=spec.get("dec", 0)),
            "unit": spec.get("unit", ""),
        })

    if "SBP (mmHg)" in row.index and "DBP (mmHg)" in row.index:
        sbp = float(row["SBP (mmHg)"])
        dbp = float(row["DBP (mmHg)"])
        out.append({
            "label": "Bloeddruk",
            "value": f"{_fmt_chip_num(sbp)}/{_fmt_chip_num(dbp)}",
            "unit": "mmHg",
        })

    if "MBP (mmHg)" in row.index:
        spec = _display_spec("MBP (mmHg)")
        mbp = float(spec["conv"](float(row["MBP (mmHg)"])))
        out.append({
            "label": spec["label"],
            "value": _fmt_chip_num(mbp, decimals=spec.get("dec", 0)),
            "unit": spec.get("unit", ""),
        })

    return out


def _labs(row):
    """Laboratory values in SI / European units for clinician display."""
    lab_cols = (
        "WBC (K/uL)", "Hemoglobin (g/dL)", "Platelet (k/uL)",
        "Creatinine (mg/dL)", "BUN (mg/dL)", "Sodium (mEq/L)", "Chloride (mEq/L)",
        "Potassium (mEq/L)", "Bicarbonate (mEq/L)", "Anion Gap (mEq/L)",
        "Glucose (mg/dL)", "Magnesium (mg/dL)", "Calcium (mg/dL)", "Phosphate (mg/dL)",
        "INR", "Prothrombin Time (s)", "PTT (s)",
    )
    out = []
    for col in lab_cols:
        if col not in row.index:
            continue
        spec = _display_spec(col)
        disp = float(spec["conv"](float(row[col])))
        out.append({
            "label": spec["label"],
            "value": _fmt_chip_num(disp, decimals=spec.get("dec", 1)),
            "unit": spec.get("unit", ""),
        })
    return out


def _interventies(row):
    mapping = {
        "Mechanische beademing": "Mechanical Ventilation",
        "CRRT":                  "CRRT",
        "Vasopressoren":         "Vasopressor",
        "Sedatie":               "Sedation",
    }
    return [{"label": k, "actief": int(row.get(v, 0)) == 1}
            for k, v in mapping.items()]


def _comorbiditeiten(row):
    mapping = {
        "Acuut myocardinfarct":  "AMI",
        "Chronische nierziekte": "CKD",
        "COPD":                  "COPD",
        "Hypertensie":           "Hypertension",
        "Diabetes":              "Diabetes",
        "Acuut nierfalen":       "AKI",
        "Beroerte":              "Stroke",
        "SAD (delier)":          "SAD",
    }
    return [{"label": k, "actief": int(row.get(v, 0)) == 1}
            for k, v in mapping.items()]


def _buf_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _patient_gradients(row_values: np.ndarray) -> np.ndarray:
    """Bereken gradiënt van P(died) t.o.v. elke inputfeature (eenmalig)."""
    x = torch.tensor(row_values.reshape(1, -1), dtype=torch.float32,
                     requires_grad=True)
    out = MODEL(x)
    MODEL.zero_grad()
    out[0, 1].backward()
    return x.grad.abs().squeeze().detach().numpy()


def _feature_glossary(importance: np.ndarray, top_n: int,
                      row_vals: np.ndarray = None,
                      feat_indices: list = None) -> list:
    """
    Geef de top-N features als lijst van dicts met label, uitleg,
    en optioneel de patiëntwaarde.
    If feat_indices is provided, those indices are used directly (in order)
    instead of selecting by importance.
    """
    top_idx   = feat_indices if feat_indices is not None \
                else np.argsort(importance)[::-1][:top_n]
    result    = []

    for i in top_idx:
        feat_name = FEATURE_NAMES[i]
        label     = _display_label(feat_name)

        is_binary = _is_binary(feat_name)

        waarde_label = actief = None
        if row_vals is not None:
            raw = float(row_vals[i])
            if is_binary:
                actief       = raw == 1.0
                waarde_label = "Aanwezig" if actief else "Afwezig"
            else:
                waarde_label = _format_feature_value(feat_name, raw)

        result.append({
            "label":     label,
            "uitleg":    FEATURE_UITLEG.get(feat_name, "Geen uitleg beschikbaar."),
            "waarde":    waarde_label,
            "is_binary": is_binary,
            "actief":    actief,
        })

    return result


def _draw_spline_panel(
    ax,
    feat_i: int,
    row_values: np.ndarray,
    *,
    row_raw: np.ndarray = None,
    show_color_legend: bool = True,
    contribution_pct: float = None,
) -> None:
    """Plot one KAN spline panel (binary bar pair or continuous curve) on ``ax``."""
    from matplotlib.patches import Patch

    WARM   = "#ef4444"
    COOL   = "#3b82f6"
    WARM_A = "#fecaca"
    COOL_A = "#bfdbfe"
    PAT_C  = "#7c3aed"

    feat_name = FEATURE_NAMES[feat_i]
    label = _display_label(feat_name)
    patient_val = float(row_values[feat_i])

    ax.set_facecolor("white")

    x_norm = SPLINE_X[:, feat_i]
    y_vals = SPLINE_Y[:, feat_i]
    sort_i = np.argsort(x_norm)
    x_norm = x_norm[sort_i]
    y_vals = y_vals[sort_i]

    is_binary = _is_binary(feat_name)

    if is_binary:
        y_absent = float(np.interp(0.0, x_norm, y_vals))
        y_present = float(np.interp(1.0, x_norm, y_vals))
        patient_has = int(round(patient_val)) == 1

        bar_vals = [y_absent, y_present]
        bar_labels = ["Afwezig", "Aanwezig"]
        bar_colors = [WARM if v >= 0 else COOL for v in bar_vals]
        edge_widths = [
            3.5 if (not patient_has) else 1.0,
            3.5 if patient_has else 1.0,
        ]
        edge_cols = [
            PAT_C if (not patient_has) else "#e2e8f0",
            PAT_C if patient_has else "#e2e8f0",
        ]

        bars = ax.bar(
            bar_labels,
            bar_vals,
            color=bar_colors,
            alpha=0.85,
            width=0.5,
            edgecolor=edge_cols,
            linewidth=edge_widths,
            zorder=3,
        )

        ax.axhline(0, color="#94a3b8", linewidth=1.0, linestyle="--", zorder=2)

        for bar, val in zip(bars, bar_vals):
            mid_y = val / 2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mid_y,
                f"{val:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="700",
                color="white",
                zorder=4,
            )

        px = 1 if patient_has else 0
        py = bar_vals[px]
        label_txt = (
            "Deze patiënt\n(aanwezig)" if patient_has else "Deze patiënt\n(afwezig)"
        )
        y_range = max(abs(y_absent), abs(y_present)) or 0.01
        ax.annotate(
            label_txt,
            xy=(px, py),
            xytext=(px + (0.35 if px == 0 else -0.35), py + y_range * 0.5),
            fontsize=8,
            color=PAT_C,
            fontweight="bold",
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=PAT_C,
                linewidth=1,
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="->", color=PAT_C, lw=1.2),
            zorder=6,
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7, zorder=0)

    else:
        if feat_name in DF.columns:
            x_min = float(DF[feat_name].min())
            x_max = float(DF[feat_name].max())
        else:
            x_min, x_max = 0.0, 1.0
        x_orig = x_min + x_norm * (x_max - x_min)
        x_plot = np.array([_raw_to_display(feat_name, float(x)) for x in x_orig])

        ax.fill_between(
            x_plot, y_vals, 0, where=(y_vals >= 0), color=WARM_A, alpha=0.7
        )
        ax.fill_between(
            x_plot, y_vals, 0, where=(y_vals < 0), color=COOL_A, alpha=0.7
        )

        for k in range(len(x_plot) - 1):
            mid = (y_vals[k] + y_vals[k + 1]) / 2
            ax.plot(
                x_plot[k : k + 2],
                y_vals[k : k + 2],
                color=WARM if mid >= 0 else COOL,
                linewidth=1.8,
            )

        ax.axhline(0, color="#94a3b8", linewidth=0.9, linestyle="--", zorder=1)

        if row_raw is not None:
            pat_orig = float(row_raw[feat_i])
        else:
            pat_orig = x_min + patient_val * (x_max - x_min)
        pat_plot = _raw_to_display(feat_name, pat_orig)
        pat_norm = (pat_orig - x_min) / (x_max - x_min) if x_max > x_min else patient_val
        pat_y = float(np.interp(pat_norm, x_norm, y_vals))
        direction = (
            "↑ meer naar hogere voorspelde kans"
            if pat_y >= 0
            else "↓ meer naar lagere voorspelde kans"
        )

        ax.axvline(pat_plot, color=PAT_C, linewidth=1.6, linestyle="--", alpha=0.85, zorder=3)
        ax.scatter(
            [pat_plot],
            [pat_y],
            s=80,
            color=PAT_C,
            edgecolors="white",
            linewidth=1.5,
            zorder=5,
        )

        x_disp_min = float(x_plot.min())
        x_disp_max = float(x_plot.max())
        x_range = x_disp_max - x_disp_min
        y_span = max(float(y_vals.max() - y_vals.min()), 1e-9)
        y_lo, y_hi = float(y_vals.min()), float(y_vals.max())
        y_mid = (y_lo + y_hi) / 2

        x_rel = (pat_plot - x_disp_min) / (x_range or 1.0)
        if x_rel < 0.33:
            off_x = x_range * 0.14
        elif x_rel > 0.67:
            off_x = -x_range * 0.14
        else:
            off_x = x_range * (0.12 if x_rel < 0.55 else -0.12)

        if pat_y >= y_mid:
            off_y = -y_span * 0.30
            va = "top"
        else:
            off_y = y_span * 0.26
            va = "bottom"

        txt_x = pat_plot + off_x
        txt_y = pat_y + off_y

        ax_u = (txt_x - x_disp_min) / (x_range or 1.0)
        ay_u = (txt_y - y_lo) / ((y_hi - y_lo) or 1.0)
        if show_color_legend and ax_u < 0.45 and ay_u > 0.62:
            off_y = -y_span * 0.34
            va = "top"
            txt_y = pat_y + off_y

        pat_value_txt = _format_feature_value(feat_name, pat_orig)
        ax.annotate(
            f"Deze patiënt\n{pat_value_txt}",
            xy=(pat_plot, pat_y),
            xytext=(txt_x, txt_y),
            fontsize=7.5,
            color=PAT_C,
            fontweight="bold",
            ha="center",
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=PAT_C,
                linewidth=1,
                alpha=0.92,
            ),
            arrowprops=dict(arrowstyle="->", color=PAT_C, lw=1.2),
            zorder=8,
        )

        ax.set_xlabel(_chart_axis_label(feat_name), fontsize=9, color="#374151")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7, zorder=0)

        if show_color_legend:
            ax.legend(
                handles=[
                    Patch(facecolor=WARM_A, edgecolor=WARM, label="Meer naar hogere voorspelde kans"),
                    Patch(facecolor=COOL_A, edgecolor=COOL, label="Meer naar lagere voorspelde kans"),
                ],
                fontsize=7,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                framealpha=0.92,
                edgecolor="#e2e8f0",
                borderpad=0.5,
            )

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(labelsize=8, colors="#64748b")
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("")

    if contribution_pct is not None:
        sign = "+" if contribution_pct >= 0 else ""
        ax.set_title(
            f"{label}\nBijdrage: {sign}{contribution_pct:.1f}%",
            fontsize=10,
            fontweight="bold",
            pad=10,
            color="#1e293b",
        )
    else:
        ax.set_title(label, fontsize=10.5, fontweight="bold", pad=8, color="#1e293b")



def _chart3_feature_indices(contributions: np.ndarray, n_each: int = 6) -> list:
    """Top ``n_each`` highest + lowest signed contributions among visible features."""
    visible = _visible_feature_indices()
    pos = sorted(visible, key=lambda i: contributions[i], reverse=True)[:n_each]
    neg = sorted(visible, key=lambda i: contributions[i])[:n_each]
    return pos + neg


def _contribution_pct_map(contributions: np.ndarray, feat_indices: list) -> dict:
    total_abs = float(np.abs(contributions).sum()) or 1.0
    scale = 100.0 / total_abs
    return {i: float(contributions[i]) * scale for i in feat_indices}


def _feature_context_chart_b64(row_values: np.ndarray,
                                importance: np.ndarray, top_n=3,
                                feat_indices: list = None,
                                row_raw: np.ndarray = None,
                                contributions: np.ndarray = None) -> str:
    """
    KAN spline activation plots for the top-N features (Grafiek 2).

    Shows the actual learned spline function φ(x) from the KAN's first layer
    for each feature.  The y-axis is the spline's activation output:
      > 0  (warm/red)  → pushes prediction toward "gestorven"
      < 0  (cool/blue) → pushes prediction toward "overleefd"

    The patient's current feature value is marked with a vertical line and dot.
    """
    top_idx   = feat_indices if feat_indices is not None \
                else list(np.argsort(importance)[::-1][:top_n])

    pct_map = {}
    if contributions is not None:
        pct_map = _contribution_pct_map(contributions, top_idx)

    n     = len(top_idx)
    ncols = 6
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes_grid = plt.subplots(nrows, ncols,
                                  figsize=(3.5 * ncols, 3.2 * nrows))
    fig.patch.set_facecolor("white")
    axes = np.array(axes_grid).flatten().tolist()
    # hide unused subplot slots
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, feat_i in zip(axes, top_idx):
        _draw_spline_panel(
            ax, feat_i, row_values,
            row_raw=row_raw,
            show_color_legend=False,
            contribution_pct=pct_map.get(feat_i),
        )

    # Reserve band at top for shared legend and left for shared y-label.
    fig.tight_layout(
        pad=1.1,
        h_pad=1.25,
        w_pad=1.1,
        rect=[0.04, 0.02, 0.98, 0.94],
    )
    fig.text(0.01, 0.5, "Risicosignaal",
             va="center", ha="center", rotation="vertical",
             fontsize=9.5, color="#374151")

    from matplotlib.patches import Patch

    _w, _c = "#ef4444", "#3b82f6"
    _wa, _ca = "#fecaca", "#bfdbfe"
    fig.legend(
        handles=[
            Patch(facecolor=_wa, edgecolor=_w, label="Meer naar hogere voorspelde kans"),
            Patch(facecolor=_ca, edgecolor=_c, label="Meer naar lagere voorspelde kans"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.022, 1.0),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=9.8,
        handlelength=2.6,
        handleheight=1.45,
        borderpad=0.65,
        columnspacing=1.35,
        labelspacing=0.45,
        framealpha=0.96,
        edgecolor="#e2e8f0",
    )

    return _buf_to_b64(fig)





def _edge_row_vals_raw() -> np.ndarray:
    row, _ = _get_edge_row()
    return np.array(row[FEATURE_NAMES].values, dtype=float)


def _kan_forward_single(row_vals_raw: np.ndarray) -> tuple[float, np.ndarray]:
    """Batch-1 KAN forward pass: P(died) and per-feature signed contributions."""
    xs = _scale_features(np.asarray(row_vals_raw, dtype=float))
    x = torch.tensor(xs.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        out = MODEL(x)
    prob = float(F.softmax(out, dim=1)[0, 1].item())
    contrib = MODEL.spline_postacts[0].sum(dim=1).squeeze().detach().numpy()
    return prob, contrib.astype(float)


def _effective_edge_row_raw() -> np.ndarray:
    return apply_perturbation(_edge_row_vals_raw(), FEATURE_NAMES)


_STAP3_CHART_BUNDLE = None


def _get_stap3_chart_bundle():
    """Lazy-cache gradients + base64 plots for stap 3 (avoids repeated torch + matplotlib work)."""
    global _STAP3_CHART_BUNDLE
    if _STAP3_CHART_BUNDLE is not None:
        return _STAP3_CHART_BUNDLE

    row_vals_raw = _effective_edge_row_raw()
    row_vals_scaled = _scale_features(row_vals_raw)
    pat_grad = _patient_gradients(row_vals_scaled)
    if PERTURBATION:
        stap3_prob, waterfall_signed = _kan_forward_single(row_vals_raw)
    else:
        stap3_prob = EDGE_PROB
        waterfall_signed = SPLINE_Y[_edge_pos, :].astype(float)
    chart3_indices = _chart3_feature_indices(waterfall_signed, n_each=6)

    _STAP3_CHART_BUNDLE = {
        "row_vals_raw": row_vals_raw,
        "pat_grad": pat_grad,
        "waterfall_signed": waterfall_signed,
        "chart3_indices": chart3_indices,
        "prob": stap3_prob,
        "context_chart_b64": _feature_context_chart_b64(
            row_vals_scaled, pat_grad, feat_indices=chart3_indices,
            row_raw=row_vals_raw, contributions=waterfall_signed),
    }
    return _STAP3_CHART_BUNDLE


# ══════════════════════════════════════════════════════════════════════════════
#  LLM data extraction  (packages the same numbers the charts are drawn from)
# ══════════════════════════════════════════════════════════════════════════════

# Clinical-ish tails on the *training* cohort — used only to ground “sparse extremes” talk.
# Each rule: (threshold, Dutch label for clinicians, "high" = count >= thr, "low" = count <= thr).
def _normal_range_str(feat_name: str) -> str:
    """Reference range string in the same display units as the patient value.

    Applies the same conv/dec from FEATURE_DISPLAY so units always match
    what the LLM sees in 'patientwaarde'. E.g. Magnesium stored as mg/dL
    but displayed as mmol/L → reference is also rendered in mmol/L.
    """
    rng = _FEATURE_NORMAL_RANGES.get(feat_name)
    if rng is None:
        return ""
    lo, hi = rng
    spec = FEATURE_DISPLAY.get(feat_name, {})
    unit = spec.get("unit", "")
    conv = spec.get("conv", lambda v: v)
    dec  = spec.get("dec", 1)

    def _fmt(v):
        converted = float(conv(float(v)))
        return f"{converted:.{dec}f}"

    if lo is None:
        return f"≤{_fmt(hi)} {unit}".strip()
    if hi is None:
        return f"≥{_fmt(lo)} {unit}".strip()
    return f"{_fmt(lo)}–{_fmt(hi)} {unit}".strip()


def _reference_status_str(feat_name: str, raw: float) -> str:
    """Whether the patient value is binnen / onder / boven the given reference range."""
    rng = _FEATURE_NORMAL_RANGES.get(feat_name)
    if rng is None:
        return ""
    lo, hi = rng
    spec = FEATURE_DISPLAY.get(feat_name, {})
    conv = spec.get("conv", lambda v: v)
    disp = float(conv(float(raw)))
    if lo is None and hi is not None:
        return "boven referentie" if disp > float(conv(float(hi))) else "binnen referentie"
    if hi is None and lo is not None:
        return "onder referentie" if disp < float(conv(float(lo))) else "binnen referentie"
    if lo is not None and hi is not None:
        lo_d = float(conv(float(lo)))
        hi_d = float(conv(float(hi)))
        if disp < lo_d:
            return "onder referentie"
        if disp > hi_d:
            return "boven referentie"
        return "binnen referentie"
    return ""


def _graph3_clinical_summary_nl(
    feat_indices: list,
    contributions: np.ndarray,
    row_vals_raw: np.ndarray,
    prob: float,
) -> str:
    """
    Compact Dutch summary for the LLM: model outcome mix and clinical read risks —
    not a chart legend.
    """
    total_abs = float(np.abs(contributions).sum()) or 1.0
    scale = 100.0 / total_abs
    pos = [
        (i, float(contributions[i]) * scale)
        for i in feat_indices
        if contributions[i] > 0
    ]
    neg = [
        (i, float(contributions[i]) * scale)
        for i in feat_indices
        if contributions[i] < 0
    ]
    pos.sort(key=lambda t: t[1], reverse=True)
    neg.sort(key=lambda t: t[1])

    lines = [
        f"Hoofdlijn: voorspelde kans intrahospitaal overlijden ≈ {prob * 100:.1f}% "
        "(modeloutput, geen klinische waarheid).",
        "Sterkste bijdragen in de twaalf detailpanelen op deze pagina:",
    ]
    for i, pct in pos[:4]:
        nl = _display_label(FEATURE_NAMES[i])
        pv = _format_feature_value(FEATURE_NAMES[i], float(row_vals_raw[i]))
        lines.append(f"  → hogere voorspelde kans (model): {nl} ({pct:+.1f}%), patiënt {pv}")
    for i, pct in neg[:4]:
        nl = _display_label(FEATURE_NAMES[i])
        pv = _format_feature_value(FEATURE_NAMES[i], float(row_vals_raw[i]))
        lines.append(f"  → lagere voorspelde kans (model): {nl} ({pct:+.1f}%), patiënt {pv}")

    prob_pct = prob * 100
    lines.append(
        f"Eén kop «Wat betekent dit…». Geen plaat beschrijven (geen rood/blauw/paars, "
        f"geen %-bijdragen). (1) max 2 zinnen: netto {prob_pct:.1f}% + balans; "
        "(2) verplicht 5–6 zinnen correlatie vs causaliteit + geen behandeladvies; "
        "(3) 2–4 zinnen: meest opvallend patroon — eerst [boven/onder referentie] + "
        "(lagere kans), anders [binnen referentie] + (hogere kans); sla [binnen referentie] + "
        "(lagere kans) over. Anders één zin isolated-panel. Lege regels tussen alinea's."
    )
    return "\n".join(lines)


def _graph4_llm_data(row_vals_raw: np.ndarray, feat_indices: list = None) -> list:
    """Grafiek 3 – spline panels: top-6 positive + top-6 negative contributors."""
    bundle = _get_stap3_chart_bundle()
    contributions = bundle["waterfall_signed"]
    if feat_indices is None:
        feat_indices = bundle["chart3_indices"]
    row_scaled = _scale_features(row_vals_raw)
    labels_nl = [NL_LABEL.get(f, f) for f in FEATURE_NAMES]
    total_abs = float(np.abs(contributions).sum()) or 1.0
    scale = 100.0 / total_abs

    features = []
    for i in feat_indices:
        feat_name = FEATURE_NAMES[i]
        patient_val_scaled = float(row_scaled[i])

        x_s = SPLINE_X[:, i]
        y_s = SPLINE_Y[:, i]
        ord_i = np.argsort(x_s)
        x_s, y_s = x_s[ord_i], y_s[ord_i]

        activation = float(np.interp(patient_val_scaled, x_s, y_s))
        contrib_pct = float(contributions[i]) * scale
        pat_raw = float(row_vals_raw[i])

        features.append({
            "name":             labels_nl[i],
            "patient_value":    _fmt_val(pat_raw, feat_name),
            "activation":       activation,
            "direction":        (
                "hogere kans" if activation > 0
                else "lagere kans" if activation < 0
                else "neutraal"
            ),
            "contribution_pct": contrib_pct,
            "normal_range":     _normal_range_str(feat_name),
            "reference_status": _reference_status_str(feat_name, pat_raw),
        })
    return features


def _full_patient_kan_table(row_vals_raw: np.ndarray, pat_grad: np.ndarray) -> str:
    """
    Compact master table of every feature's model outputs for the edge patient —
    same quantities underlying graphs 1–3. Lets the LLM answer questions about
    any feature (e.g. creatinine's row) without hallucinating.
    """
    bundle = _get_stap3_chart_bundle()
    contrib = bundle["waterfall_signed"].astype(float)
    total_abs = float(np.abs(contrib).sum()) or 1.0
    scale = 100.0 / total_abs
    bundle = _get_stap3_chart_bundle()
    g3_idx = bundle["chart3_indices"]
    g3_set = set(g3_idx)
    g3_names = [NL_LABEL.get(FEATURE_NAMES[j], FEATURE_NAMES[j]) for j in g3_idx]

    lines = [
        "VOLLEDIG MODELSIGNAAL PER KENMERK (IC-afdelingstype-indicatoren weggelaten; "
        "verder dezelfde bron als de drie figuren op stap 3)",
        "Legenda — patientwaarde: zoals gemeten in de studie-data. gevoeligheid (gevoeligheidsoverzicht): "
        "hoe sterk een kleine verschuiving van deze waarde het modelscore kan beïnvloeden "
        "(hoger = sterker). interne scorebijdrage: hoe dit kenmerk nu meetelt in het "
        "samengevoegde risicoscore-signaal van het model (zelfde als de staafwaarde vóór "
        "cumulatie in het staafdiagram). pct: dit aandeel als percentage van de som van alle absolute "
        "bijdragen (+ betekent dat deze bijdrage in het trainingsmateriaal samenging met hogere "
        "voorspelde kans zoals het model die ziet; − met lagere voorspelde kans). Geen causale "
        "betekenis.",
        "",
        "Kenmerk | patientwaarde | gevoeligheid | interne_scorebijdrage | pct",
        "-" * 78,
    ]
    for i, fname in enumerate(FEATURE_NAMES):
        if fname.startswith("ICU:") or _is_hidden_feature(fname):
            continue
        nl = NL_LABEL.get(fname, fname)
        pv = _fmt_val(float(row_vals_raw[i]), fname)
        sens = float(pat_grad[i])
        act = float(contrib[i])
        pct = act * scale
        note = " [eigen detailpaneel in het detailpaneel-overzicht]" if i in g3_set else ""
        lines.append(f"{nl} | {pv} | {sens:.5g} | {act:+.5g} | {pct:+.2f}%{note}")

    lines.append("")
    lines.append(
        "Kenmerken die in het detailpaneel-overzicht als apart subpaneel staan: "
        + ", ".join(g3_names)
        + ". (Twaalf panelen: zes met positieve en zes met negatieve bijdrage.) Andere kenmerken hebben wel interne_scorebijdrage en pct in deze tabel, maar "
        "geen apart subpaneel in de bijgevoegde afbeelding."
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  OpenAI client (lazy-initialised on first API call)
# ══════════════════════════════════════════════════════════════════════════════

_oai_client = None


def _get_oai_client():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        from prompts import OPENAI_API_KEY

        key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("OPENAI_API", "").strip()
            or (OPENAI_API_KEY or "").strip()
        )
        if not key:
            raise RuntimeError(
                "Geen OpenAI-sleutel: zet OPENAI_API_KEY of OPENAI_API in de omgeving."
            )
        _oai_client = OpenAI(api_key=key)
    return _oai_client


def _normalize_assistant_content(content) -> str:
    """Turn ChatCompletion message.content (str or list of parts) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            else:
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
        return "".join(parts).strip()
    return str(content).strip()


def _call_llm(messages: list, max_tokens: int = 450) -> str:
    from prompts import OPENAI_MODEL

    client = _get_oai_client()
    model = (OPENAI_MODEL or "").strip()
    params: dict = {"model": model, "messages": messages}
    lm = model.lower()
    if lm.startswith(("o1", "o3", "o4")):
        # Reasoning models charge hidden "reasoning" tokens against this budget; a small
        # limit often yields message.content == "" (see OpenAI community reports).
        floor = int(os.environ.get("OPENAI_REASONING_TOKEN_FLOOR", "8192"))
        params["max_completion_tokens"] = max(max_tokens, floor)
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "low").strip()
        if effort:
            params["reasoning_effort"] = effort
    else:
        params["max_tokens"] = max_tokens
        params["temperature"] = 0.4

    try:
        resp = client.chat.completions.create(**params)
    except Exception as exc:
        from openai import BadRequestError

        if isinstance(exc, BadRequestError) and params.pop(
            "reasoning_effort", None
        ) is not None:
            resp = client.chat.completions.create(**params)
        else:
            raise
    choice = resp.choices[0]
    msg = choice.message
    content = _normalize_assistant_content(msg.content)
    if not content:
        refusal = getattr(msg, "refusal", None)
        if isinstance(refusal, str) and refusal.strip():
            content = refusal.strip()

    used = getattr(resp.usage, "completion_tokens", None) if resp.usage else None
    details = getattr(resp.usage, "completion_tokens_details", None) if resp.usage else None
    reasoning_toks = getattr(details, "reasoning_tokens", None) if details else None
    print(
        f"[LLM] model={OPENAI_MODEL}  finish={choice.finish_reason}"
        f"  tokens={used}  reasoning_tokens={reasoning_toks}"
        f"  preview={repr(content[:100])}"
    )

    if not content:
        raise RuntimeError(
            "Het model gaf geen zichtbare tekst terug (vaak door te lage max_completion_tokens "
            "bij redeneermodellen). Verhoog OPENAI_REASONING_TOKEN_FLOOR of kies een ander model."
        )

    return content


def _llm_chart_b64_for_graph(graph_id: int, bundle: dict) -> str:
    """PNG base64 aligned with stap 3 (detail panels only)."""
    if graph_id != 3:
        raise ValueError("LLM toelichting is alleen beschikbaar voor het detailpaneel")
    return bundle["context_chart_b64"]



def _build_graph_context(graph_id: int) -> str:
    """Ingevulde intro voor het detailpaneel (graph_id moet 3 zijn)."""
    from prompts import format_graph_intro

    if graph_id != 3:
        raise ValueError(f"LLM-context alleen voor detailpaneel (3), niet {graph_id}")

    bundle = _get_stap3_chart_bundle()
    row_vals = bundle["row_vals_raw"]
    pat_grad = bundle["pat_grad"]
    feat_idx = bundle["chart3_indices"]
    features = _graph4_llm_data(row_vals, feat_indices=feat_idx)
    clinical_summary = _graph3_clinical_summary_nl(
        feat_idx, bundle["waterfall_signed"], row_vals, bundle["prob"]
    )
    intro = format_graph_intro(
        graph_id=3,
        prob=bundle["prob"] * 100,
        features=features,
        clinical_summary=clinical_summary,
    )
    return intro + "\n\n---\n\n" + _full_patient_kan_table(row_vals, pat_grad)


def _base_ctx(stap: int):
    """Context die elke stap-template nodig heeft."""
    row, _ = _get_edge_row()
    geslacht, icu, etniciteit = _derive_meta(row)
    return {
        "patient_id":      EDGE_ID,
        "leeftijd":        int(row["Age"]),
        "geslacht":        geslacht,
        "gewicht":         int(round(float(row["Weight (kg)"]))),
        "icu":             icu,
        "etniciteit":      etniciteit,
        "sofa":            int(row["SOFA Score"]),
        "gcs":             int(row["GCS"]),
        "casus_tekst":     EDGE_CASUS_TEKST,
        "vitals":          _vitals(row),
        "labs":            _labs(row),
        "interventies":    _interventies(row),
        "comorbiditeiten": _comorbiditeiten(row),
        "llm_aan":         session.get("llm_aan", False),
        "stap":            stap,
        "prob_mort":       round(EDGE_PROB * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Context processor (sidebar)
# ══════════════════════════════════════════════════════════════════════════════

@app.context_processor
def inject_globals():
    row, _ = _get_edge_row()
    _, icu, _ = _derive_meta(row)
    return {
        "llm_aan":           session.get("llm_aan", False),
        "groep_toegewezen":  session.get("groep_toegewezen", False),
        "patient_id":        EDGE_ID,
        "icu":               icu,
        "huidige_stap":      session.get("huidige_stap", 0),
        "prev_answers":      {},
        "manager_preview":   session.get("manager_preview"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

# Keys cleared when opening the landing page or resetting a participant session.
# Manager-/admin-login keys are never removed here.
_PARTICIPANT_SESSION_KEYS = (
    "groep_toegewezen",
    "llm_aan",
    "functie",
    "huidige_stap",
    "stap1",
    "stap2",
    "stap3",
    "respons_opgeslagen",
)


def _redirect_if_participant_finished():
    """After a real participant submits, block restarting the study in this browser session."""
    if session.get("participant_finished"):
        return redirect(url_for("index"))
    return None


@app.route("/")
def index():
    """
    Welkomstpagina voor deelnemers: wis studie-sessie, maar niet manager-/admin-login.
    Eén afgeronde deelname per sessie: daarna alleen bedankpagina zonder opnieuw starten.
    """
    if session.get("participant_finished"):
        return render_template("index_afgerond.html")
    for k in _PARTICIPANT_SESSION_KEYS:
        session.pop(k, None)
    session.pop("forced_groep", None)
    session.pop("manager_preview", None)
    return render_template("index.html")


@app.route("/profiel", methods=["GET"])
def profiel():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    return render_template("profiel.html")


@app.route("/profiel", methods=["POST"])
def profiel_submit():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    functie = request.form.get("functie", "").strip()
    if not functie:
        return render_template("profiel.html",
                               error="Selecteer uw functie om verder te gaan.")
    forced = session.pop("forced_groep", None)
    if forced == "B":
        llm_aan = True
    elif forced == "A":
        llm_aan = False
    else:
        llm_aan = _assign_group()
    session["functie"]          = functie
    session["llm_aan"]          = llm_aan
    session["groep_toegewezen"] = True
    session["huidige_stap"]     = 1
    return redirect(url_for("stap1"))


# ── Stap 1: Patiëntprofiel ────────────────────────────────────────────────

@app.route("/stap/1", methods=["GET"])
def stap1():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    if not session.get("groep_toegewezen"):
        return redirect(url_for("profiel"))
    session["huidige_stap"] = 1
    ctx = _base_ctx(stap=1)
    ctx["prev_answers"] = session.get("stap1", {})
    return render_template("stap1.html", **ctx)


@app.route("/stap/1", methods=["POST"])
def stap1_submit():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    session["stap1"] = {
        "risico":   request.form.get("risico", ""),
        "notities": request.form.get("notities", "").strip(),
        "handelen": request.form.get("handelen", "").strip(),
    }
    session["huidige_stap"] = 2
    return redirect(url_for("stap2"))


# ── Stap 2: + Modelvoorspelling ───────────────────────────────────────────

@app.route("/stap/2", methods=["GET"])
def stap2():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    if not session.get("stap1"):
        return redirect(url_for("stap1"))
    session["huidige_stap"] = 2
    ctx = _base_ctx(stap=2)
    ctx["hoog_risico"] = EDGE_PROB >= THRESHOLD
    ctx["prev_answers"] = session.get("stap2", {})
    return render_template("stap2.html", **ctx)


@app.route("/stap/2", methods=["POST"])
def stap2_submit():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    session["stap2"] = {
        "risico":   request.form.get("risico", ""),
        "notities": request.form.get("notities", "").strip(),
        "handelen": request.form.get("handelen", "").strip(),
    }
    session["huidige_stap"] = 3
    return redirect(url_for("stap3"))


# ── Stap 3: + XAI (+ LLM voor Groep B) ───────────────────────────────────

@app.route("/stap/3", methods=["GET"])
def stap3():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    if not session.get("stap2"):
        return redirect(url_for("stap2"))
    session["huidige_stap"] = 3

    bundle = _get_stap3_chart_bundle()
    row_vals_raw = bundle["row_vals_raw"]
    pat_grad = bundle["pat_grad"]
    waterfall_signed = bundle["waterfall_signed"]
    chart3_indices = bundle["chart3_indices"]

    ctx = _base_ctx(stap=3)
    ctx["prob_mort"] = round(bundle["prob"] * 100, 1)
    ctx["hoog_risico"] = bundle["prob"] >= THRESHOLD
    ctx["context_chart_b64"] = bundle["context_chart_b64"]
    ctx["glossary"] = _feature_glossary(
        waterfall_signed, top_n=len(chart3_indices), row_vals=row_vals_raw,
                                                feat_indices=chart3_indices)
    ctx["prev_answers"] = session.get("stap3", {})
    return render_template("stap3.html", **ctx)


@app.route("/stap/3", methods=["POST"])
def stap3_submit():
    blocked = _redirect_if_participant_finished()
    if blocked:
        return blocked
    llm_raw = request.form.get("llm_chats_json", "")
    session["stap3"] = {
        "risico":   request.form.get("risico", ""),
        "notities": request.form.get("notities", "").strip(),
        "handelen": request.form.get("handelen", "").strip(),
        "cas1":     request.form.get("cas1", ""),
        "cas2":     request.form.get("cas2", ""),
        "cas3":     request.form.get("cas3", ""),
        "cas4":     request.form.get("cas4", ""),
        "cas5":     request.form.get("cas5", ""),
        "llm_chats_json": _sanitize_llm_chats_json(llm_raw)
        if session.get("llm_aan")
        else "",
    }
    session["huidige_stap"] = 4
    return redirect(url_for("klaar"))


# ── Klaar ──────────────────────────────────────────────────────────────────

@app.route("/klaar")
def klaar():
    if session.get("participant_finished"):
        return redirect(url_for("index"))
    if not session.get("stap3"):
        return redirect(url_for("index"))

    snap_stap1 = dict(session.get("stap1", {}))
    snap_stap2 = dict(session.get("stap2", {}))
    snap_stap3 = dict(session.get("stap3", {}))
    snap_llm_aan = session.get("llm_aan", False)

    if not session.get("respons_opgeslagen") and not session.get("manager_preview"):
        _save_response()
        session["respons_opgeslagen"] = True

    # Real participants: one submission per browser session — drop payload after thank-you page.
    if not session.get("manager_preview"):
        for k in _PARTICIPANT_SESSION_KEYS:
            session.pop(k, None)
        session["participant_finished"] = True

    return render_template(
        "klaar.html",
        stap1=snap_stap1,
        stap2=snap_stap2,
        stap3=snap_stap3,
        llm_aan=snap_llm_aan,
        prob_mort=round(EDGE_PROB * 100, 1),
    )


# ── Admin ──────────────────────────────────────────────────────────────────────

def _manager_db_data():
    """Return (rows_a, rows_b, stats) for the manager dashboard."""
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        all_rows = con.execute(
            "SELECT * FROM deelnemers ORDER BY id DESC"
        ).fetchall()
    participants = [_participant_display_dict(r) for r in all_rows]
    rows_a = [p for p in participants if p.get("groep") == "A"]
    rows_b = [p for p in participants if p.get("groep") == "B"]

    def _risk_dist(rows, col):
        dist = {}
        for r in rows:
            v = r.get(col) or "—"
            dist[v] = dist.get(v, 0) + 1
        return dist

    stats = {
        "n_total": len(all_rows),
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "a": {
            "s1": _risk_dist(rows_a, "stap1_risico"),
            "s2": _risk_dist(rows_a, "stap2_risico"),
            "s3": _risk_dist(rows_a, "stap3_risico"),
        },
        "b": {
            "s1": _risk_dist(rows_b, "stap1_risico"),
            "s2": _risk_dist(rows_b, "stap2_risico"),
            "s3": _risk_dist(rows_b, "stap3_risico"),
        },
    }
    return rows_a, rows_b, stats


@app.route("/manager", methods=["GET", "POST"])
def manager():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["manager"] = True
            return redirect(url_for("manager"))
        return render_template("manager.html", logged_in=False,
                               error="Onjuist wachtwoord.")
    if not session.get("manager"):
        return render_template("manager.html", logged_in=False)
    # Dashboard home: end preview mode so the yellow banner does not stay on this page.
    session.pop("manager_preview", None)
    rows_a, rows_b, stats = _manager_db_data()
    tab = request.args.get("tab", "groep-a")
    return render_template("manager.html", logged_in=True,
                           forced_groep=session.get("forced_groep"),
                           rows_a=rows_a, rows_b=rows_b, stats=stats,
                           active_tab=tab)


@app.route("/manager/randpatient-calculator")
def manager_randpatient_calculator():
    """
    Standalone KAN edge-patient explorer (from hospMort pipeline output).
    Requires manager session — share /manager plus password, then open this link.
    """
    if not session.get("manager"):
        return redirect(url_for("manager"))
    path = EDGE_PATIENT_CALCULATOR_HTML.resolve()
    if not path.is_file():
        back = url_for("manager")
        return (
            "<!DOCTYPE html><html lang='nl'><head><meta charset='utf-8'><title>Calculator ontbreekt</title></head>"
            "<body style='font-family:system-ui,sans-serif;padding:2rem;max-width:42rem'>"
            "<h2>Randpatient-calculator niet gevonden</h2>"
            "<p>Verwacht bestand:</p><pre style='background:#f1f5f9;padding:12px;border-radius:8px'>"
            f"{path}</pre>"
            "<p>Zet <code>EDGE_PATIENT_CALCULATOR_HTML</code> in de omgeving naar het juiste pad, "
            "of voeg het bestand toe aan <code>interface/static/tools/edge_patient_calculator.html</code> "
            "(standaard in deze deployment).</p>"
            f"<p><a href='{back}'>Terug naar manager</a></p></body></html>",
            404,
        )
    return send_file(path, mimetype="text/html; charset=utf-8", max_age=0)


@app.route("/manager/preview/<groep>")
def manager_preview(groep):
    if not session.get("manager"):
        return redirect(url_for("manager"))
    if groep not in ("A", "B"):
        return redirect(url_for("manager"))
    for k in _PARTICIPANT_SESSION_KEYS:
        session.pop(k, None)
    session.pop("participant_finished", None)
    session["manager_preview"] = groep
    session["forced_groep"] = groep
    # Toon de echte welkomstpagina zonder via "/" te gaan (die zou preview-keys wissen).
    return render_template("index.html")


@app.route("/manager/groep/<groep>")
def manager_set_groep(groep):
    if not session.get("manager"):
        return redirect(url_for("manager"))
    if groep in ("A", "B"):
        session["forced_groep"] = groep
    return redirect(url_for("manager"))


@app.route("/manager/wis-groep")
def manager_wis_groep():
    if not session.get("manager"):
        return redirect(url_for("manager"))
    session.pop("forced_groep", None)
    return redirect(url_for("manager"))


@app.route("/manager/uitloggen")
def manager_uitloggen():
    session.pop("manager", None)
    session.pop("forced_groep", None)
    session.pop("manager_preview", None)
    return redirect(url_for("manager"))


def _antwoorden_csv_response():
    """Alle kolommen uit ``deelnemers`` als CSV (incl. notities, CAS, stap3_llm_chats)."""
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM deelnemers ORDER BY id").fetchall()
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])
    else:
        buf.write("geen data\n")
    buf.seek(0)
    return app.response_class(
        buf.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment;filename=antwoorden.csv"},
    )


@app.route("/manager/download")
def manager_download():
    if not session.get("manager"):
        return redirect(url_for("manager"))
    return _antwoorden_csv_response()


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if pw == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin"))
        return render_template("admin.html", logged_in=False,
                               error="Onjuist wachtwoord.")
    if not session.get("admin"):
        return render_template("admin.html", logged_in=False)
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows_raw = con.execute(
            "SELECT * FROM deelnemers ORDER BY id DESC"
        ).fetchall()
    rows = [_participant_display_dict(r) for r in rows_raw]
    n_a = sum(1 for r in rows if r.get("groep") == "A")
    n_b = sum(1 for r in rows if r.get("groep") == "B")
    return render_template("admin.html", logged_in=True,
                           rows=rows, n_a=n_a, n_b=n_b)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin"))


@app.route("/admin/download")
def admin_download():
    if not session.get("admin"):
        return redirect(url_for("admin"))
    return _antwoorden_csv_response()


# ── LLM API routes ────────────────────────────────────────────────────────────

@app.route("/api/llm/<int:graph_id>/intro", methods=["GET"])
def llm_intro(graph_id: int):
    """
    Return the opening LLM message for a graph's conversation box.

    Rebuilds graph context from live patient data plus the same PNG chart shown
    in stap 3 (multimodal message). The client receives the assistant opening only.
    """
    if graph_id != 3:
        return jsonify({"error": "Onbekende grafiek"}), 400

    bundle = _get_stap3_chart_bundle()

    from prompts import SYSTEM_PROMPT, build_intro_messages

    context_msg = _build_graph_context(graph_id)
    chart_b64 = _llm_chart_b64_for_graph(graph_id, bundle)
    messages = build_intro_messages(SYSTEM_PROMPT, context_msg, chart_b64)

    try:
        content = _call_llm(messages, max_tokens=420)
    except Exception as exc:
        print(f"[LLM ERROR] graph {graph_id}: {exc}")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"content": content})




# Pre-render stap 3 charts once at import so the first participant request is fast.
_get_stap3_chart_bundle()

if PERTURBATION:
    _b = _get_stap3_chart_bundle()
    print(
        f"⚠ PERTURBATION aan — KAN forward pass op licht afwijkende inputs; "
        f"stap 3/LLM P(died) ≈ {_b['prob']:.1%} (baseline {EDGE_PROB:.1%}); "
        f"{len(_b['chart3_indices'])} panelen dynamisch herkozen"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
