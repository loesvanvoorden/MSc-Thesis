"""
Co-12 measurements for Table tab:co12_measurement (thesis §3.4).

Outputs raw facts as JSON — no aggregates, no pass/fail, no markdown summary.
Feature mentions: gazetteer lookup over Dutch labels from the feature-row record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

HEADING = "Wat betekent dit voor deze patiënt — volgens het model?"
HEADING_ALT = HEADING.replace("—", "-")

EXTRA_ALIASES: dict[str, list[str]] = {
    "Respiratory Rate": ["ademfrequentie", "ademhalingen"],
    "DBP (mmHg)": ["diastolische bd", "diastolische bloeddruk", "onderdruk"],
    "Potassium (mEq/L)": ["kalium"],
    "Bicarbonate (mEq/L)": ["bicarbonaat", "bicarbonaatgehalte"],
    "SpO2": ["spo₂", "spo2", "zuurstofsaturatie", "zuurstofverzadiging"],
    "Platelet (k/uL)": ["trombocyten", "trombocytenaantal", "bloedplaatjes"],
    "Chloride (mEq/L)": ["chloride", "chloridewaarde"],
    "GCS": ["gcs", "gcs-score", "gcs-waarden"],
    "PTT (s)": ["aptt", "a-ptt", "partiële tromboplastinetijd"],
    "Magnesium (mg/dL)": ["magnesium", "magnesiumwaarden"],
    "Heart Rate (bpm)": ["hartslag", "hartslagen"],
    "Anion Gap (mEq/L)": ["anion gap"],
}

JARGON_BLOCKLIST = ["spline", "splines", "activatie", "gradient", "KAN"]
CHAT_MARKERS = ["stel een vraag", "vraag gerust", "stel gerust", r"\bchat\b"]
TABLE_MARKERS = [
    "volledig modelsignaal per kenmerk",
    "patientwaarde |",
    "interne_scorebijdrage",
]

UP = re.compile(r"\b(omhoog|hogere|verhoogd|verhoogde|opvoeren|drijven)\b", re.I)
DOWN = re.compile(r"\b(omlaag|lagere|verminderen|verlagen|reduceren|bijstellen)\b", re.I)

NOT_AUTOMATED = {
    "Contrastivity": {
        "mode": "design_constraint",
        "procedure": "Not measured; counterfactual reasoning excluded in prompts.py",
    },
    "Context": {
        "mode": "expert_review",
        "procedure": "Domain-expert evaluation prior to deployment (not scripted)",
    },
    "Coherence": {
        "mode": "user_study",
        "procedure": "CAS after Step 3; Group A vs Group B (not in pre-study files)",
    },
    "Controllability": {
        "mode": "design_constraint",
        "procedure": "Not measured; chat removed, zero by design",
    },
}


def build_gazetteer(features: Sequence[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in features:
        key = f.get("_feat_name") or f.get("name", "")
        if not key:
            continue
        terms = {f.get("name", "").lower()}
        for a in EXTRA_ALIASES.get(key, []):
            terms.add(a.lower())
        out[key] = sorted(terms, key=len, reverse=True)
    return out


def _patterns(gazetteer: dict[str, list[str]]) -> list[tuple[str, re.Pattern]]:
    return [
        (k, re.compile(r"\b(" + "|".join(re.escape(t) for t in ts) + r")\b", re.I))
        for k, ts in gazetteer.items()
        if ts
    ]


def extract_entities(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for key, pat in patterns:
        if pat.search(lower):
            found.append(key)
    return found


def _normalise(text: str) -> str:
    return text.replace("\u00a0", " ").strip()


def parse_multi_file(path: Path) -> list[tuple[str, str]]:
    raw = _normalise(path.read_text(encoding="utf-8"))
    parts = re.split(r"(?m)^(?:Antwoord|Response)\s+(\d+)\s*$", raw)
    if len(parts) > 1:
        return [(f"run_{parts[i]}", parts[i + 1].strip()) for i in range(1, len(parts), 2)]
    return [("single", raw)] if raw else []


def split_explanation(text: str) -> tuple[str, list[str]]:
    text = _normalise(text)
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if idx is None:
        return "", []
    heading = lines[idx].strip()
    paras: list[str] = []
    buf: list[str] = []
    for ln in lines[idx + 1 :]:
        if ln.strip():
            buf.append(ln.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return heading, paras


def _top6(features: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    pos = sorted(
        (f for f in features if f.get("contribution_pct", 0) > 0),
        key=lambda f: f["contribution_pct"],
        reverse=True,
    )[:6]
    neg = sorted(
        (f for f in features if f.get("contribution_pct", 0) < 0),
        key=lambda f: f["contribution_pct"],
    )[:6]
    return pos, neg


def _feat(features: Sequence[dict], key: str) -> dict | None:
    return next((f for f in features if f.get("_feat_name") == key), None)


def _direction_near(text: str, key: str, patterns: list[tuple[str, re.Pattern]], window: int = 70) -> str | None:
    lower = text.lower()
    for k, pat in patterns:
        if k != key:
            continue
        m = pat.search(lower)
        if not m:
            continue
        snippet = text[max(0, m.start() - window) : m.end() + window]
        up, down = bool(UP.search(snippet)), bool(DOWN.search(snippet))
        if up and not down:
            return "up"
        if down and not up:
            return "down"
    return None


def _paragraph1_clauses(para1: str) -> tuple[str, str]:
    """Split paragraph~1 into upward-driver and downward-driver clauses."""
    lower = para1.lower()

    if "terwijl" in lower:
        i = lower.find("terwijl")
        return para1[:i], para1[i:]

    # ``waarbij hogere X … omhoog …, en lagere Y … omlaag'' (no terwijl)
    m = re.search(r",?\s+en\s+lagere\b", lower)
    if m:
        return para1[: m.start()], para1[m.start() :]

    return para1, ""


def _clause_direction(clause: str) -> str | None:
    lower = clause.lower()
    up = bool(re.search(r"\b(hogere|verhoogde|omhoog)\b", lower))
    down = bool(re.search(r"\b(lagere|omlaag|naar beneden|beneden brengen|bijstellen)\b", lower))
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def _sentences(body: str) -> int:
    body = re.sub(r"\s+", " ", body.strip())
    if not body:
        return 0
    return len(re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü])", body))


def measure_correctness(
    text: str, features: Sequence[dict], patterns: list[tuple[str, re.Pattern]]
) -> dict[str, Any]:
    _, paras = split_explanation(text)
    p1 = paras[0] if paras else text
    up_text, down_text = _paragraph1_clauses(p1)
    up_keys = set(extract_entities(up_text, patterns))
    down_keys = set(extract_entities(down_text, patterns)) if down_text else set()
    up_clause_dir = _clause_direction(up_text)
    down_clause_dir = _clause_direction(down_text) if down_text else None

    direction_mismatches = []
    for key in set(extract_entities(p1, patterns)):
        f = _feat(features, key)
        if not f:
            continue
        pct = float(f["contribution_pct"])
        if pct == 0:
            continue

        in_up = key in up_keys
        in_down = key in down_keys
        model_dir = "up" if pct > 0 else "down"

        if in_up and not in_down:
            text_dir = up_clause_dir
        elif in_down and not in_up:
            text_dir = down_clause_dir
        elif in_up and in_down:
            continue  # ambiguous placement; skip
        else:
            text_dir = _direction_near(p1, key, patterns)

        if text_dir and text_dir != model_dir:
            direction_mismatches.append({
                "feature": f["name"],
                "contribution_pct": round(pct, 1),
                "text_direction": text_dir,
                "model_direction": model_dir,
            })

    pos6, neg6 = _top6(features)
    p1_keys = set(extract_entities(p1, patterns))
    emphasis = []
    for f in pos6:
        k = f["_feat_name"]
        emphasis.append({
            "feature": f["name"],
            "rank": pos6.index(f) + 1,
            "contribution_pct": round(f["contribution_pct"], 1),
            "named_in_paragraph_1": k in p1_keys,
            "in_upward_clause": k in up_keys,
        })
    for f in neg6:
        k = f["_feat_name"]
        emphasis.append({
            "feature": f["name"],
            "rank": neg6.index(f) + 1,
            "contribution_pct": round(f["contribution_pct"], 1),
            "named_in_paragraph_1": k in p1_keys,
            "in_downward_clause": k in down_keys,
        })

    return {
        "direction_mismatches": direction_mismatches,
        "top6_emphasis": emphasis,
    }


def measure_completeness(
    text: str, features: Sequence[dict], patterns: list[tuple[str, re.Pattern]]
) -> dict[str, Any]:
    mentioned = set(extract_entities(text, patterns))
    pos6, neg6 = _top6(features)
    return {
        "positive_contributors": [
            {"feature": f["name"], "named": f["_feat_name"] in mentioned}
            for f in pos6
        ],
        "negative_contributors": [
            {"feature": f["name"], "named": f["_feat_name"] in mentioned}
            for f in neg6
        ],
        "panel_features_named": sum(1 for f in features if f.get("_feat_name") in mentioned),
        "panel_features_total": len(features),
    }


def measure_covariate_complexity(text: str) -> dict[str, Any]:
    lower = text.lower()
    found = [
        t for t in JARGON_BLOCKLIST
        if t != "KAN" and re.search(rf"\b{re.escape(t)}\b", lower)
    ]
    if re.search(r"\bKAN\b", text):  # model acronym only; not Dutch ``kan''
        found.append("KAN")
    return {"jargon_terms_found": found}


def measure_compactness(
    text: str, patterns: list[tuple[str, re.Pattern]]
) -> dict[str, Any]:
    _, paras = split_explanation(text)
    body = " ".join(paras)
    dupes = []
    for i, para in enumerate(paras, 1):
        counts: dict[str, int] = {}
        for key in extract_entities(para, patterns):
            counts[key] = counts.get(key, 0) + 1
        for key, n in counts.items():
            if n > 1:
                dupes.append({"paragraph": i, "feature_key": key, "count": n})
    return {
        "sentence_count": _sentences(body),
        "paragraph_count": len(paras),
        "duplicate_feature_mentions": dupes,
    }


def measure_composition(text: str) -> dict[str, Any]:
    heading, paras = split_explanation(text)
    full = text.lower()
    return {
        "heading_exact_match": heading in (HEADING, HEADING_ALT),
        "heading_count": len(
            re.findall(
                r"wat betekent dit voor deze patiënt\s*[—-]\s*volgens het model", full, re.I
            )
        ),
        "body_paragraph_count": len(paras),
        "chat_invitation_markers_found": [
            m for m in CHAT_MARKERS
            if (re.search(m, full, re.I) if m.startswith(r"\b") else m in full)
        ],
        "internal_table_markers_found": [m for m in TABLE_MARKERS if m in full],
    }


def measure_confidence(text: str, expected_prob: float) -> dict[str, Any]:
    pat = re.compile(
        rf"\b{expected_prob:.1f}".replace(".", r"[,.]") + r"\s*%", re.I
    )
    matches = pat.findall(text)
    return {
        "expected_probability_pct": expected_prob,
        "probability_mentions": len(matches),
    }


def measure_baseline_run(
    label: str,
    text: str,
    features: Sequence[dict],
    patterns: list[tuple[str, re.Pattern]],
    expected_prob: float,
) -> dict[str, Any]:
    return {
        "label": label,
        "Correctness": measure_correctness(text, features, patterns),
        "Completeness": measure_completeness(text, features, patterns),
        "Covariate Complexity": measure_covariate_complexity(text),
        "Compactness": measure_compactness(text, patterns),
        "Composition": measure_composition(text),
        "Confidence": measure_confidence(text, expected_prob),
    }


def measure_consistency(
    texts: Sequence[tuple[str, str]],
    patterns: list[tuple[str, re.Pattern]],
    expected_prob: float,
) -> dict[str, Any]:
    p1_sets, p3_features, probs = [], [], []
    for label, text in texts:
        _, paras = split_explanation(text)
        p1 = paras[0] if paras else text
        p3 = paras[2] if len(paras) >= 3 else ""
        p1_sets.append(set(extract_entities(p1, patterns)))
        p3_features.append(extract_entities(p3, patterns) if p3 else [])
        prob = measure_confidence(text, expected_prob)
        probs.append({"label": label, "probability_mentions": prob["probability_mentions"]})

    shared_p1 = set.intersection(*p1_sets) if p1_sets else set()
    union_p1 = set.union(*p1_sets) if p1_sets else set()
    return {
        "n_generations": len(texts),
        "paragraph_1_entities_shared_all_runs": sorted(shared_p1),
        "paragraph_1_entities_union": sorted(union_p1),
        "paragraph_3_surprise_feature_per_run": [
            {"label": label, "entities": ents} for (label, _), ents in zip(texts, p3_features)
        ],
        "probability_mentions_per_run": probs,
        "composition_paragraph_count_per_run": [
            {"label": label, "paragraph_count": measure_composition(text)["body_paragraph_count"]}
            for label, text in texts
        ],
    }


def measure_continuity(
    baseline_text: str,
    perturbed_text: str,
    baseline_features: Sequence[dict],
    perturbed_features: Sequence[dict],
    baseline_prob: float,
    perturbed_prob: float,
    patterns: list[tuple[str, re.Pattern]],
    risk_threshold: float = 0.5383,
) -> dict[str, Any]:
    _, bp = split_explanation(baseline_text)
    _, pp = split_explanation(perturbed_text)
    b1 = bp[0] if bp else baseline_text
    p1 = pp[0] if pp else perturbed_text
    b3 = bp[2] if len(bp) >= 3 else ""
    p3 = pp[2] if len(pp) >= 3 else ""

    base_panels = {f["_feat_name"] for f in baseline_features}
    pert_panels = {f["_feat_name"] for f in perturbed_features}

    return {
        "baseline_probability_pct": round(baseline_prob * 100, 1),
        "perturbed_probability_pct": round(perturbed_prob * 100, 1),
        "probability_shift_pp": round((perturbed_prob - baseline_prob) * 100, 1),
        "same_risk_band": (baseline_prob >= risk_threshold) == (perturbed_prob >= risk_threshold),
        "detail_panels_retained": sorted(base_panels & pert_panels),
        "detail_panels_total": len(base_panels),
        "paragraph_1_entities_baseline": extract_entities(b1, patterns),
        "paragraph_1_entities_perturbed": extract_entities(p1, patterns),
        "paragraph_3_entities_baseline": extract_entities(b3, patterns) if b3 else [],
        "paragraph_3_entities_perturbed": extract_entities(p3, patterns) if p3 else [],
    }


def enrich_features(features: list[dict], name_to_key: dict[str, str]) -> list[dict]:
    return [
        {**f, "_feat_name": name_to_key.get(f.get("name", ""), f.get("name", ""))}
        for f in features
    ]


def run_audit(
    baseline_texts: Sequence[tuple[str, str]],
    perturbed_text: str,
    baseline_features: Sequence[dict],
    perturbed_features: Sequence[dict],
    baseline_prob: float,
    perturbed_prob: float,
    *,
    meta: dict | None = None,
) -> dict[str, Any]:
    gazetteer = build_gazetteer(baseline_features)
    patterns = _patterns(gazetteer)
    expected_prob = round(baseline_prob * 100, 1)

    baseline_runs = [
        measure_baseline_run(label, text, baseline_features, patterns, expected_prob)
        for label, text in baseline_texts
    ]

    return {
        "meta": meta or {},
        "procedure": "Table tab:co12_measurement",
        "ner": "gazetteer over feature-row Dutch labels + aliases",
        "feature_record": [
            {
                "feature": f["name"],
                "contribution_pct": round(float(f["contribution_pct"]), 1),
                "_feat_name": f.get("_feat_name"),
            }
            for f in baseline_features
        ],
        "baseline_runs": baseline_runs,
        "Consistency": measure_consistency(baseline_texts, patterns, expected_prob),
        "Continuity": measure_continuity(
            baseline_texts[0][1],
            perturbed_text,
            baseline_features,
            perturbed_features,
            baseline_prob,
            perturbed_prob,
            patterns,
        ),
        "not_automated": NOT_AUTOMATED,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
