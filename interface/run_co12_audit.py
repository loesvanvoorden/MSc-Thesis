#!/usr/bin/env python3
"""Run Co-12 audit → interface/results/co12_audit_report.json (raw facts only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTERFACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTERFACE_DIR.parent
DEFAULT_BASELINE = REPO_ROOT / "Thesis" / "LLManswers" / "10xgeneration.txt"
DEFAULT_PERTURBED = REPO_ROOT / "Thesis" / "LLManswers" / "pertubation.txt"
DEFAULT_OUT = INTERFACE_DIR / "results" / "co12_audit_report.json"


def _load_feature_records(perturbed: bool = False) -> tuple[list[dict], float]:
    sys.path.insert(0, str(INTERFACE_DIR))
    import app  # noqa: E402
    import perturbation  # noqa: E402

    _orig = perturbation.apply_perturbation
    app.PERTURBATION = perturbed
    app.apply_perturbation = lambda row, names: _orig(row, names, enabled=perturbed)
    app._STAP3_CHART_BUNDLE = None

    bundle = app._get_stap3_chart_bundle()
    features = app._graph4_llm_data(bundle["row_vals_raw"], bundle["chart3_indices"])
    name_to_key = {v: k for k, v in app.NL_LABEL.items()}
    from co12_audit import enrich_features  # noqa: E402

    return enrich_features(features, name_to_key), float(bundle["prob"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--perturbed", type=Path, default=DEFAULT_PERTURBED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(INTERFACE_DIR))
    from co12_audit import parse_multi_file, run_audit, write_report  # noqa: E402

    cache = INTERFACE_DIR / "results" / "co12_feature_cache.json"
    if args.skip_model and cache.is_file():
        c = json.loads(cache.read_text(encoding="utf-8"))
        baseline_features, baseline_prob = c["baseline_features"], c["baseline_prob"]
        perturbed_features, perturbed_prob = c["perturbed_features"], c["perturbed_prob"]
    else:
        baseline_features, baseline_prob = _load_feature_records(False)
        perturbed_features, perturbed_prob = _load_feature_records(True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    "baseline_features": baseline_features,
                    "perturbed_features": perturbed_features,
                    "baseline_prob": baseline_prob,
                    "perturbed_prob": perturbed_prob,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    baseline_texts = parse_multi_file(args.baseline)
    if not baseline_texts:
        print(f"No texts in {args.baseline}", file=sys.stderr)
        return 1

    report = run_audit(
        baseline_texts,
        args.perturbed.read_text(encoding="utf-8").strip(),
        baseline_features,
        perturbed_features,
        baseline_prob,
        perturbed_prob,
        meta={
            "baseline_file": str(args.baseline),
            "perturbed_file": str(args.perturbed),
            "baseline_prob": baseline_prob,
            "perturbed_prob": perturbed_prob,
        },
    )
    write_report(args.output, report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
