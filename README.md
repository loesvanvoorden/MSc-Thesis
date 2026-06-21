# FinalThesisDeliverables_LoesVanVoorden_2055036

Master's thesis deliverables (MSc Data Science in Business and Entrepreneurship). Monorepo with three main parts: thesis document, KAN training pipeline, and clinician study interface.

```
.
├── Thesis/                          # Thesis source and supporting materials
├── hospMort/                        # ICU mortality KAN — data and training
├── interface/                       # Flask study web app
├── answers/                         # Exported participant responses
├── PapersLoes/                      # Reference papers and theses
└── README.md
```

## Thesis/


| Path                                         | Contents                                            |
| -------------------------------------------- | --------------------------------------------------- |
| `LoesVanVoorden_2055036_thesis_MSc_DSBE.tex` | Main LaTeX source                                   |
| `LoesVanVoorden_2055036_thesis_MSc_DSBE.pdf` | Compiled thesis                                     |
| `ReflectionReport.pdf`                       | MSc DSBE reflection report                          |
| `references.bib`                             | Bibliography                                        |
| `compile_thesis.sh`                          | Build script (`build/` holds LaTeX aux files)       |
| `SupplementaryImages/`                       | Figures used in the thesis                          |
| `BigQueryResults/`                           | MIMIC-IV query exports (CSV)                        |
| `LLManswers/`                                | Saved LLM explanation outputs for thesis appendices |


## hospMort/


| Path                                   | Contents                                                  |
| -------------------------------------- | --------------------------------------------------------- |
| `data/`                                | MIMIC-IV cohort data (`MIMIC-IV.csv`, `.dta`) — **not in this repo** (see below) |
| `scripts/preprocess.py`                | Data loading and feature preprocessing                    |
| `scripts/train_kan.py`                 | KAN training; syncs model artefacts to `interface/model/` |
| `scripts/sweep.py`                     | Hyperparameter grid search                                |
| `scripts/refine.py`                    | Post-sweep model refinement                               |
| `scripts/build_edge_calculator.py`     | Generates the edge-patient selection HTML tool            |
| `scripts/plot_kan_annotated.py`        | KAN architecture figure for the thesis                    |
| `scripts/plot_kan_sum_to_interface.py` | KAN-to-interface pipeline figure                          |
| `scripts/model/`                       | pykan training checkpoints                                |
| `scripts/results/models/`              | Trained model weights and config                          |
| `scripts/results/metrics/`             | Training and sweep metric logs                            |
| `scripts/results/plots/`               | Generated training and spline plots                       |
| `scripts/figures/`                     | Spline panel images from training                         |


#### Data access

`hospMort/data/` is excluded from this repository (MIMIC-IV credentialed data). To reproduce the training pipeline locally, obtain your own extract and place it in `hospMort/data/` as `MIMIC-IV.csv` (and optionally `MIMIC-IV.dta`).

1. **PhysioNet credentialed access** — register at [MIMIC-IV v2.1](https://physionet.org/content/mimiciv/2.1/), complete the required [CITI *Data or Specimens Only Research*](https://physionet.org/settings/credentialing/) training, and sign the PhysioNet data use agreement.
2. **Cohort construction** — build the ICU cohort using the selection criteria from [Zhang et al. (2023)](https://www.nature.com/articles/s41598-023-38650-4): adult patients (age ≥ 18) meeting Sepsis-3 (suspected infection with SOFA ≥ 2), a single qualifying ICU stay of at least 24 hours, with input features aggregated from the first 24 hours of that stay. This project uses the same cohort definition but predicts in-hospital mortality instead of sepsis-associated delirium.
3. **Run preprocessing** — with `MIMIC-IV.csv` in place, run `python scripts/preprocess.py` from `hospMort/scripts/`.


## interface/


| Path                        | Contents                                                         |
| --------------------------- | ---------------------------------------------------------------- |
| `app.py`                    | Flask application — study flow, KAN inference, LLM explanations  |
| `prompts.py`                | LLM system and user prompts                                      |
| `perturbation.py`           | Input perturbation helpers for explanation testing               |
| `co12_audit.py`             | Co-12 explanation quality audit                                  |
| `run_co12_audit.py`         | CLI runner for the Co-12 audit                                   |
| `model/`                    | Deployed KAN model (`kan_weights.pt`, config, feature data)      |
| `data/responses.db`         | SQLite store for participant responses                           |
| `templates/`                | HTML templates for each study step                               |
| `static/`                   | CSS, logo, and standalone tools (`edge_patient_calculator.html`) |
| `results/`                  | Co-12 audit output (JSON)                                        |
| `requirements.txt`          | Python dependencies                                              |
| `.env.example`              | Environment variable template                                    |


## Quick start

**Thesis:** `cd Thesis && ./compile_thesis.sh`

**Study app (local):** `cd interface && pip install -r requirements.txt && python app.py`