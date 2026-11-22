# MoleculeMind

**An uncertainty-aware, human-in-the-loop platform for multi-objective molecular discovery.**

MoleculeMind helps a chemist decide **which molecules to measure next**, and is
honest about how much its own advice should be trusted. It predicts molecular
properties, attaches calibrated uncertainty to every prediction, warns when a
molecule falls outside what the models actually know, finds the genuine
trade-offs between competing objectives, and plans experiments against a budget.

> ### New here? Read [PROJECT_EXPLAINED_SIMPLY.md](PROJECT_EXPLAINED_SIMPLY.md).
> Plain English, no chemistry or machine-learning background assumed. Every
> technical word is explained the moment it appears, with everyday analogies,
> plus tables of exactly which tools we used and which file each piece lives in.
> **Start there.**
>
> [LEARNING_GUIDE.md](LEARNING_GUIDE.md) covers the same ground in more depth and
> with the proper terminology, for when you want the technical version.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:8000>. Interactive API documentation at `/docs`.

**The first launch takes about 3 minutes** while it trains and caches the
models. Every launch after that takes about a second. To skip the two expensive
optional models:

```bash
MOLECULEMIND_ENABLE_DMPNN=false MOLECULEMIND_ENABLE_GENERATOR=false python run.py
```

Run the tests:

```bash
python -m pytest -q        # 309 tests
```

---

## What it actually does

| Capability | What you get | Status |
|---|---|---|
| **Molecular Analyzer** | SMILES validation, 13 descriptors, Lipinski and Veber checks, 2D rendering, computed 3D conformer | Complete |
| **Property prediction** | Water solubility (logS) and clinical-trial toxicity risk, every value tagged with its provenance | Complete |
| **Conformal uncertainty** | Prediction intervals with a measured coverage guarantee; prediction sets for classification | Complete |
| **Applicability domain** | Two independent checks on whether a molecule is the kind of thing the models know, with thresholds derived from the training data | Complete |
| **Pareto Laboratory** | Non-dominated sorting over validated objectives, interactive 2D/3D trade-off plots | Complete |
| **Probabilistic Pareto** | Probability each candidate is on the front, given its calibrated uncertainty | Complete |
| **Risk slider** | Live exploit-versus-explore re-ranking | Complete |
| **Budget planner** | Cost-aware experiment plan by Expected Hypervolume Improvement, verified against BoTorch | Complete |
| **Active learning** | Gaussian-process surrogate with four acquisition strategies, simulated assays | Complete |
| **D-MPNN ensemble** | Graph neural network with calibrated per-molecule uncertainty and a learned latent space | Complete; **matches, does not beat, the baseline** |
| **Molecular generator** | SMILES language model with REINVENT-style reinforcement learning | Implemented; **MOSES pre-training pending**, so validity is 46–78% |
| **Benchmarking** | All metrics computed at startup from held-out splits; JSON/CSV report download | Complete |

---

## Measured results

Reproducible with seed 42 via `python -m pytest -q` or `GET /api/benchmark/report`.

### Solubility (Delaney ESOL, 1,117 unique molecules after deduplication)

| Model | Representation | Split | R² | RMSE |
|---|---|---|---|---|
| Extra Trees (baseline) | ECFP4 + 8 descriptors | random | **0.903** | 0.666 |
| Extra Trees | ECFP4 + 8 descriptors | Bemis–Murcko scaffold | 0.890 | 0.746 |
| D-MPNN ensemble (5 members) | molecular graph, learned | random, pinned to baseline | 0.902 | 0.669 |
| Random Forest (independent check) | ECFP4 + 8 descriptors | same held-out molecules | 0.877 | — |

### Conformal coverage (138 held-out molecules)

| Nominal | Measured | Mean interval width |
|---|---|---|
| 80% | 78.3% | 1.60 logS |
| 90% | 88.4% | 2.14 logS |
| 95% | 99.3% | 3.36 logS |

### Toxicity (ClinTox CT_TOX, 103 positives of 1,461)

| Model | ROC-AUC | PR-AUC | Baseline rate |
|---|---|---|---|
| Random Forest, single-task | 0.832 | 0.358 | 0.072 |
| D-MPNN, multi-task | 0.866 | — | — |

### Verification of the EHVI planner

Agrees with BoTorch's own `qExpectedHypervolumeImprovement` to **0.24%** on a
shared posterior, and matches closed-form Expected Improvement in the
single-objective case. Live at `GET /api/planner/validate-ehvi`.

---

## Honesty commitments

These are design rules the codebase enforces, not aspirations.

* **Every property carries a `provenance` field** — `model`, `computed`, or
  `unavailable`. Nothing is returned as a prediction unless a trained, evaluated
  model produced it.
* **Properties without a validated model return no number.** Target bioactivity
  (pIC50) and metabolic stability are marked `unavailable` with a reason,
  because no suitable dataset is bundled. Earlier versions returned hand-written
  descriptor arithmetic for both; see [PROJECT_AUDIT.md](PROJECT_AUDIT.md) §10.4.
* **Only validated objectives enter the optimisation.** Missing values raise an
  error rather than being imputed.
* **Simulated assays are labelled `mode: "simulated"`** with an explicit note
  that no laboratory experiment was performed.
* **Hidden oracle values never leave the server.** One guarded serialisation
  path, with tests that scan the whole API response.
* **No metric is hardcoded.** Everything is computed at startup from a held-out
  split, and the landing page fetches its figures from the API.
* **Assumptions are measured where possible.** The probabilistic Pareto front
  assumes a Gaussian error shape; we test that assumption on real residuals, it
  fails, and the interface says so.
* **Where an estimate cannot be made honestly, the API refuses.** Correlated
  objective sampling returns HTTP 501 rather than using an invented correlation
  matrix.

---

## Known limitations

* **Interval widths are wide.** A 90% solubility interval spans ~2.1 logS. This
  is the honest state of property prediction from ~1,100 training molecules.
* **Single seed.** All metrics come from seed 42. No confidence intervals on R².
* **The scaffold split is weaker than it looks.** ESOL contains many acyclic
  molecules, each forming its own single-member scaffold group, so that portion
  of the split behaves like a random one. Do not read 0.890 as strong
  generalisation to new chemotypes.
* **The Gaussian shape assumption behind probabilistic Pareto is violated**
  (excess kurtosis 1.416, normality rejected), making those probabilities mildly
  optimistic in the tails.
* **Generator validity is 46–78%** because MOSES is not bundled. A
  data limitation, not a code limitation — see
  [scripts/download_moses.py](scripts/download_moses.py).
* **ClinTox `CT_TOX` means "trial terminated for toxicity"**, not "molecule is
  toxic". Aspirin is labelled positive in this dataset. Predictions are
  uncalibrated probabilities and are not clinical assessments.
* **Pareto sorting is O(n²)** and will not scale past a few thousand candidates.
* **No frontend tests.** TypeScript compiles and the production build is clean,
  and the tabs were exercised manually, but there is no automated browser test.

---

## Not implemented

Deliberately stopped here rather than adding modules of diminishing value.

| Module | Why not |
|---|---|
| Docking surrogate + AutoDock Vina | Requires DOCKSTRING data and a Vina installation |
| Retrosynthesis routes | Requires AiZynthFinder |
| Target bioactivity QSAR | Requires a curated ChEMBL assay dataset |
| Counterfactual molecular editing | Unblocked, simply not built |
| Objective-conflict attribution | Unblocked, simply not built |
| MLflow experiment tracking | Unblocked, simply not built |
| Team accounts and collaboration | Needs a database and auth layer |

---

## Documentation

| File | What it is for |
|---|---|
| **[PROJECT_EXPLAINED_SIMPLY.md](PROJECT_EXPLAINED_SIMPLY.md)** | **Start here.** Plain English, no background assumed. Analogies, what-we-used and where-it-lives tables, a 20-word glossary |
| [LEARNING_GUIDE.md](LEARNING_GUIDE.md) | The same material with proper terminology and more depth, plus likely examiner questions |
| [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) | Word-for-word demo script and answers to likely questions |
| [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) | Every algorithm's mathematics, assumptions and limits |
| [PROJECT_AUDIT.md](PROJECT_AUDIT.md) | Audit of the original codebase and the defects found in it |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Per-module status, what changed at each stage |
| [TEST_REPORT.md](TEST_REPORT.md) | Test results, what each suite verifies, coverage gaps |

---

## Architecture

```
run.py                        uvicorn launcher; serves API + built frontend on :8000
backend/
  data/                       Delaney ESOL, ClinTox, Lipophilicity (CSV)
  artifacts/                  trained-model cache (gitignored, regenerated)
  app/
    main.py                   FastAPI, 24 endpoints
    core/
      config.py               every setting, overridable via MOLECULEMIND_*
      mol_analyzer.py         SMILES, descriptors, 2D/3D rendering
      dataset.py              loading, dedup, splits, featurisation
      models.py               training, persistence, prediction with provenance
      conformal.py            intervals and prediction sets
      applicability_domain.py the two trust diagnostics
      pareto.py               dominance and non-dominated sorting
      probabilistic_pareto.py Monte Carlo front, risk slider
      experiment_planner.py   hypervolume, EHVI, budget planning
      active_learning.py      GP surrogate, acquisition, hidden oracle
      molecular_graph.py      molecules as directed graphs
      dmpnn.py                the graph network, losses, ensemble
      dmpnn_service.py        multi-task training and evaluation
      generator.py            SMILES language model and RL
      generator_service.py    uncertainty-aware reward
  tests/                      309 tests across 9 files
frontend/
  src/components/             six tabs; dist/ is prebuilt and served by FastAPI
scripts/download_moses.py     how to obtain MOSES for the generator
```

### Configuration

Every setting in `config.py` is overridable by a `MOLECULEMIND_`-prefixed
environment variable:

```bash
MOLECULEMIND_SEED=7                    # change the random seed
MOLECULEMIND_ESOL_TREES=50             # smaller, faster forest
MOLECULEMIND_USE_MODEL_CACHE=false     # force a full retrain
MOLECULEMIND_ENABLE_DMPNN=false        # skip the graph network
MOLECULEMIND_ENABLE_GENERATOR=false    # skip the generator
MOLECULEMIND_LOG_LEVEL=DEBUG
```

### Frontend development

```bash
cd frontend && npm install && npm run dev
```

Vite serves on `:5173` and proxies `/api` to `:8000`, so run the backend too.
After changing components, `npm run build` refreshes `dist/`, which is what
`python run.py` serves.

---

## Stack

* **Frontend** — React 19, TypeScript, Vite, Tailwind CSS v4, Plotly.js, 3Dmol.js (vendored locally), Framer Motion
* **Backend** — Python 3.14, FastAPI, Uvicorn, Pydantic v2
* **Cheminformatics** — RDKit
* **Machine learning** — scikit-learn (Extra Trees, Random Forest, PCA), PyTorch (D-MPNN, generator)
* **Bayesian optimisation** — BoTorch, GPyTorch (Gaussian processes, exact hypervolume)
* **Testing** — pytest, httpx

---

## Data provenance

| Dataset | Source | Used for |
|---|---|---|
| Delaney ESOL (1,128 rows) | Delaney 2004, via MoleculeNet | Solubility model, applicability domain, active-learning pool |
| ClinTox (1,484 rows) | Gayvert et al. 2016, via MoleculeNet | Toxicity classifier, multi-task training |
| Lipophilicity (4,200 rows) | ChEMBL, via MoleculeNet | Present but unused |

No dataset is downloaded at runtime. All three ship with the repository.
