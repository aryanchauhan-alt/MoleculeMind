# MoleculeMind — Project Audit (Stage 1)

**Audit date:** 2026-11-22
**Commit audited:** `e209515` (branch `main`, working tree clean apart from a trailing newline in `requirements.txt`)
**Auditor method:** full read of every source file; datasets parsed with RDKit; `ModelService` booted and `ActiveLearningEngine` exercised in the project's own `venv` to reproduce reported numbers.

> **Scope note.** This audit describes what is *actually in the repository today*. The project brief (sections a–k) describes a target research architecture that is, for the most part, **not present**. Nothing in this document is inferred from file names, comments or docstrings — every claim below was traced to executed code or a reproduced run.

---

## 1. Existing project architecture

```
MoleculeMind - Copy/
├── run.py                       # uvicorn launcher, serves API + built frontend on :8000
├── requirements.txt             # 11 pinned-by-floor deps
├── README.md, PRESENTATION_GUIDE.md
├── venv/                        # Python 3.14.5
├── backend/
│   ├── data/
│   │   ├── delaney-processed.csv   (1128 rows, ESOL)
│   │   ├── clintox.csv.gz          (1484 rows)
│   │   └── lipophilicity.csv       (4200 rows — loaded by nothing)
│   └── app/
│       ├── main.py              (230 L) — FastAPI, 8 routes, static mount
│       └── core/
│           ├── mol_analyzer.py        (148 L)
│           ├── dataset.py             (105 L)
│           ├── models.py              (240 L)
│           ├── conformal.py           (127 L)
│           ├── applicability_domain.py(144 L)
│           ├── pareto.py              (103 L)
│           └── active_learning.py     (241 L)
└── frontend/                    # React 19 + Vite 8 + Tailwind 4 + TS 7
    ├── index.html               # loads 3Dmol.js 2.0.4 from cdnjs
    ├── vite.config.ts           # dev proxy /api -> 127.0.0.1:8000
    ├── dist/                    # prebuilt bundle (5.8 MB JS), served by FastAPI
    └── src/ (App.tsx + 14 components, ~6.7 kLOC)
```

**Runtime shape.** A single-process FastAPI app. `ModelService` is a module-level singleton that **trains every model synchronously at import time** (≈40–60 s on this machine). There is no model persistence, no task queue, no background workers, no database, no config file, no logging framework (`print()` only), and no tests.

**Totals:** backend ≈ 1,230 LOC; frontend ≈ 6,720 LOC. The project is roughly 15% the size the brief's architecture implies.

---

## 2. Frontend module inventory

| Component | LOC | Backend-connected? | Status |
|---|---|---|---|
| `App.tsx` | 242 | `/api/health` | Works. Hash routing, 5 tabs, health poll every 10 s |
| `LandingPage.tsx` | 753 | no | Marketing page. **Contains hardcoded metrics** (see §10.1) |
| `Header.tsx` | 134 | no | Nav only |
| `MolecularAnalyzerTab.tsx` | 535 | `POST /api/analyze` | Works |
| `PropertyPredictionTab.tsx` | 637 | `POST /api/predict` | Works; renders unvalidated properties as if validated |
| `ParetoLaboratoryTab.tsx` | 694 | `GET /api/pareto` | Works; plots include heuristic objectives |
| `ActiveLearningTab.tsx` | 535 | pool / assay / reset | Renders, but rankings are meaningless (§10.2) and it **reads oracle labels** (§10.3) |
| `BenchmarkTab.tsx` | 364 | `GET /api/benchmark` | Works. ESOL only. No download/export |
| `MoleculeViewer3D.tsx` | 219 | — | 3Dmol.js via `window.$3Dmol`; **hard fails silently if the CDN is blocked** (`console.warn`, no UI fallback) |
| `ScientificTooltip.tsx` | 314 | — | Static glossary modal. Genuinely good content |
| `DiscoveryWorkflowGuide.tsx` | 133 | — | Pipeline stepper |
| `HeroMolecularNetwork.tsx` / `QuantumMolecularOrb.tsx` | 668 | — | three.js decoration |
| `AgencyCursor.tsx` / `CyberCursor.tsx` | 158 | — | Cosmetic; `CyberCursor` is dead code (never imported) |

**Verdict:** the frontend is well-built, consistent, and — unusually for a project of this kind — **every tab is wired to a real endpoint**. There are no fake screens. The dishonesty that exists is in *labelling*, not in fabricated UI.

Not present (brief §Part 5): Uncertainty Explorer, AD Explorer, Probabilistic Pareto, Generator, Budget Planner, Docking/Synthesis, Counterfactual Editor, Conflict Explanation, Campaign Management, MLflow dashboard — **10 of the 16 required screens.**

---

## 3. Backend module inventory

| Module | What it genuinely does | Verdict |
|---|---|---|
| `mol_analyzer.py` | SMILES validate/canonicalise, 2D SVG via `rdMolDraw2D`, 3D via ETKDGv3 + MMFF94 (UFF fallback), 13 descriptors + Lipinski + Veber | **Sound.** Best module in the project |
| `dataset.py` | Loads 3 CSVs; builds 1024-bit Morgan(r=2) + 8 raw descriptors = 1032-dim features | Works. **No scaling, no dedup, no scaffold split, no caching.** `lipophilicity.csv` is loaded by no caller |
| `models.py` | Trains ExtraTrees (ESOL) + RandomForest (ClinTox) at import; serves predictions | Mixed — real models beside fabricated properties |
| `conformal.py` | Split conformal, both standard and σ-normalised, with correct `⌈(n+1)(1−α)⌉` quantile | **Mathematically correct.** Strongest scientific component |
| `applicability_domain.py` | ECFP4 Tanimoto k-NN + PCA leverage with OECD `h* = 3(p+1)/n` | Reasonable, but thresholds are hardcoded, not validated |
| `pareto.py` | O(n²) dominance, non-dominated sorting, min–max composite score | Correct algorithm, unsafe defaults (§10.5) |
| `active_learning.py` | BoTorch `SingleTaskGP` + UCB/EI over PCA features | **Broken** (§10.2) and **leaks the oracle** (§10.3) |

---

## 4. Existing API endpoints

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/api/health` | status, init flag, ESOL metrics | — |
| POST | `/api/analyze` | canonical SMILES, 2D SVG, 3D SDF, descriptors | 400 on invalid SMILES |
| POST | `/api/predict` | descriptors, solubility + conformal, AD, multi-objective profile | Mixes validated and fabricated properties in one payload with no provenance field |
| GET | `/api/benchmark` | ESOL metrics + ≤100 parity points | ESOL only; no scaffold split; no export |
| GET | `/api/pareto` | 30 candidates, ranks, frontier | **Re-predicts all 30 molecules on every request** (~2–4 s, fully blocking) |
| GET | `/api/active-learning/pool` | iteration, history, candidates | **Returns `true_pic50`, `true_solubility`, `true_toxicity` for unassayed candidates** |
| POST | `/api/active-learning/rank` | same | Duplicate of the GET route |
| POST | `/api/active-learning/assay` | revealed labels, history | Mutates global state; no auth, no campaign id |
| POST | `/api/active-learning/reset` | status | Retrains library (~5 s), blocking |

All handlers are `def`, not `async def` — FastAPI runs them in the threadpool, so the event loop survives, but there is no job model for anything long-running.

---

## 5. Existing datasets

| File | Rows | Parse failures | Used by | Provenance |
|---|---|---|---|---|
| `delaney-processed.csv` | 1128 | 0 | ESOL regressor, AD, benchmark | Delaney ESOL (MoleculeNet). Target: `measured log solubility in mols per litre` |
| `clintox.csv.gz` | 1484 | 4 | ClinTox classifier | MoleculeNet ClinTox. **112 positives (7.5%) — severe imbalance** |
| `lipophilicity.csv` | 4200 | — | **nothing** | Loader exists, no caller. `self.lipo_model` is declared and never assigned |

**Measured duplication:** 1117 unique canonical SMILES out of 1128 → **11 duplicate structures** that can straddle the train/cal/test boundary. Confirmed duplicates include `CCC1(CCC(C)C)C(=O)NC(=O)NC1=O`, `Brc1ccc2ccccc2c1`, `OCC(O)C(O)C(O)C(O)CO`.

There is **no EGFR / ChEMBL bioactivity dataset, no DOCKSTRING data, no MOSES corpus** in the repository.

---

## 6. Existing ML models and algorithms

| Model | Algorithm | Trained on | Evaluated? |
|---|---|---|---|
| ESOL solubility | `ExtraTreesRegressor(150, max_depth=20, seed=42)` | 788 ESOL compounds | **Yes** — reproduced below |
| ClinTox toxicity | `RandomForestClassifier(100, seed=42)` | `feats[:1200]` — first 1200 after filtering | **No.** No split, no ROC-AUC, no PR-AUC, no calibration, no class weighting |
| Lipophilicity | — | — | Declared, never trained |
| EGFR pIC50 | **Closed-form heuristic** | — | Not a model (§10.4) |
| Metabolic stability | **Closed-form heuristic** | — | Not a model (§10.4) |
| AL surrogate | BoTorch `SingleTaskGP` on 6 PCA components | 6 seed molecules | Fits, but is degenerate (§10.2) |

### 6.1 Reproduced ESOL metrics (this machine, `venv`, seed 42)

```
Split -> Train: 788, Cal: 170, Test: 170
Test R² = 0.894   RMSE = 0.704   MAE = 0.482
Conformal empirical coverage (σ-normalised):
  80% nominal -> 78.24%  (mean width 1.555)
  90% nominal -> 88.82%  (mean width 2.127)
  95% nominal -> 94.71%  (mean width 2.726)
```

**These numbers are real and reproducible.** The ESOL pipeline is the one part of the platform that does what it claims. Coverage sits 1–2 points under nominal, which is the expected finite-sample behaviour at n_cal = 170 and is not evidence of a bug.

Caveats that must be stated alongside them: random split (not scaffold), 11 duplicate structures unhandled, single seed, no cross-validation, no confidence interval on R².

---

## 7. Scientific libraries — declared vs. actually used

| Library | In `requirements.txt` | Installed | Actually used |
|---|---|---|---|
| rdkit 2026.09.1 | ✓ | ✓ | **Heavily** — descriptors, fingerprints, drawing, ETKDG, MMFF94, QED |
| numpy 2.5.4 / pandas 3.1.0 | ✓ | ✓ | **Yes** |
| scikit-learn 1.9.1 | ✓ | ✓ | **Yes** — ExtraTrees, RF, PCA, metrics, split |
| botorch 0.18.1 | ✓ | ✓ | **Yes** — `SingleTaskGP`, `UpperConfidenceBound`, `ExpectedImprovement` (legacy analytic classes; `LogEI` is the current recommendation) |
| gpytorch 1.15.2 | ✓ | ✓ | Transitively, via BoTorch `ExactMarginalLogLikelihood` |
| torch 2.15.0 | ✓ | ✓ | **Only** as BoTorch's tensor backend. **No neural network is defined anywhere in this project** |
| fastapi / uvicorn / pydantic | ✓ | ✓ | Yes |
| **MAPIE** | ✗ | ✗ | Not used — conformal is hand-rolled (and correctly so) |
| **MLflow** | ✗ | ✗ | Absent |
| **pytest** | ✗ | ✗ | Absent — **there is not a single test in this repository** |

The brief's claim of "PyTorch, MAPIE" in the stack is **not supported**: PyTorch appears only as a BoTorch dependency, MAPIE is not installed. The footer string "RDKit · PyTorch · BoTorch · Scikit-learn · 3Dmol.js" overstates PyTorch's role.

**Python 3.14 risk:** `torch.jit.script` emits a `FutureWarning` ("not supported in Python 3.14+ and may break") on every import. This is tolerable today but is a live hazard for the D-MPNN work in Stage 3.

---

## 8. Functionality that genuinely works

1. SMILES validation, canonicalisation, graceful 400 on garbage input.
2. 2D SVG rendering (transparent background, publication-quality).
3. 3D conformer generation — ETKDGv3 embed, MMFF94 optimisation, UFF fallback, seeded.
4. All 13 descriptors + Lipinski (≤1 violation) + Veber — spot-checked against RDKit reference behaviour.
5. ESOL solubility prediction with R² = 0.894 on a held-out 170-compound test set.
6. Split conformal prediction — the finite-sample quantile `⌈(n+1)(1−α)⌉`-th order statistic is implemented **correctly**, including the σ-normalised variant, which is already most of what brief §(b) asks for.
7. Empirical coverage evaluation on the test set.
8. Applicability-domain Tanimoto k-NN with a real nearest-analogue lookup.
9. Non-dominated sorting — the dominance and ranking logic is correct.
10. The whole frontend/backend round trip, including the single-port production serving via the static mount.

---

## 9. Features implemented but scientifically unvalidated

- **ClinTox classifier** — trains, predicts, is surfaced in the UI as "Toxicity Risk", and has **never been evaluated on anything**. With 7.5% positives and no `class_weight`, an untuned RF will be strongly majority-biased; the displayed probability is not calibrated.
- **AD thresholds** (Tanimoto 0.55 / 0.35) — plausible literature values, chosen by assertion, never validated against held-out error.
- **`reliability_score` = 0.7·Tanimoto + 0.3·exp(−leverage excess)** — an invented composite with no statistical meaning, presented in the UI as a reliability number.
- **ExtraTrees inter-tree std as σ** — a legitimate heuristic spread, not a calibrated predictive variance. It is *rescued* by the conformal layer (which is why coverage holds), but on its own it should not be read as a standard deviation.

---

## 10. Bugs and scientific weaknesses

### 10.1 — Hardcoded metrics in the UI (HIGH, honesty)
`LandingPage.tsx:113` and the hero stat block state `R² = 0.894, RMSE = 0.704, MAE = 0.482` as literal strings. They happen to match today's run, but they are not fetched from `/api/benchmark`. Any retraining silently desynchronises the marketing claim from the model. Violates brief rule 13.

### 10.2 — Active Learning acquisition is degenerate (CRITICAL, correctness)
**Reproduced.** Every unassayed candidate receives an *identical* acquisition score:

```
ucb         -> all 24 candidates score 1.414   (1 unique value)
ei          -> all 24 candidates score 0.017   (1 unique value)
uncertainty -> all 24 candidates score 0.606   (1 unique value)
```

Root cause, measured: `PCA(n_components=6)` is fitted on the **raw unscaled 1032-dim feature matrix**, where the 8 physicochemical descriptors (MolWt ≈ 100–400) dwarf the 0/1 fingerprint bits.

```
explained variance ratio = [0.909, 0.087, 0.001, 0, 0, 0]
PC1 range = [-113.5, +177.6]   (std 78.6)
```

PC1 is essentially molecular weight; 99.6% of the variance is two descriptor axes and **no structural information survives**. These unnormalised inputs are then fed to `SingleTaskGP`, whose default lengthscale prior assumes inputs in roughly `[0,1]^d`. Across a 290-unit input range every train–test pair is effectively uncorrelated, so the posterior collapses to the prior: mean 0, variance 1, for every candidate. UCB then returns `0 + √2·1 = 1.414` everywhere.

Consequence: **the entire Active Learning tab ranks candidates by nothing but their order in the hardcoded list.** The ranking does not respond to new labels. The brief's suspicion of "degenerate acquisition scores / zero acquisition values" is confirmed with a measured mechanism.

Contributing factors: no input normalisation (`Normalize`/`Standardize` transforms unused), `fit_gpytorch_mll` wrapped in a bare `except: pass` that hides fit failures, PCA refitted on every library init, 6 training points for a 6-dim GP.

### 10.3 — Hidden oracle labels are exposed over the API (CRITICAL, methodology)
`GET /api/active-learning/pool` returns `self.candidate_pool` verbatim. Each candidate dict carries `true_pic50`, `true_solubility`, `true_toxicity` **whether or not it has been assayed**. The frontend consumes them directly:

```
ActiveLearningTab.tsx:129  [...assayed].sort((a,b) => b.true_pic50 - a.true_pic50)[0]
ActiveLearningTab.tsx:244  pIC50: {bestDiscovered?.true_pic50} (Sol: {bestDiscovered?.true_solubility})
```

Any user can open DevTools and read the answer key for all 24 unassayed molecules. This invalidates the human-in-the-loop premise and directly violates brief rule 10.

### 10.4 — Fabricated properties presented as model predictions (CRITICAL, honesty)
Two of the four Pareto objectives are closed-form arithmetic on descriptors, with no training data behind them, returned inside a `multi_objective_profile` payload alongside genuine predictions and with no field distinguishing them:

```python
# models.py — "bioactivity_pic50"
simulated_pic50 = clip(5.5 + 2.5*QED + 1.0*(max_tanimoto − 0.5), 4.0, 9.8)

# models.py — "metabolic_stability_half_life_min"
metabolic_half_life = clip(60.0 − 5.0*logP − 2.0*rot_bonds + 10.0*QED, 5.0, 150.0)
```

The pIC50 expression is not EGFR-specific, not target-specific at all, and its `max_tanimoto` term is similarity to the **ESOL solubility training set** — chemically meaningless as an activity signal. The half-life formula has invented coefficients and invented units ("minutes in liver microsomes"). Both are then fed into Pareto optimisation as if they were validated objectives. Violates brief rules 4, 5, 14.

Additionally, the 30 `true_pic50` / `true_toxicity` values in the AL library (e.g. "Warfarin: pIC50 8.2, toxicity 0.38") are **invented numbers attributed to named real drugs**. They are not traceable to ChEMBL or any cited source. The `true_solubility` values do appear consistent with ESOL measurements; the other two columns do not have a source.

### 10.5 — Pareto silently substitutes 0.0 for missing objectives (HIGH)
`pareto.py:46` — `obj_matrix[i,j] = float(c.get(obj_name, 0.0))`. A missing property becomes 0.0, which for a minimised objective such as `toxicity_risk_probability` is the *best possible value* and will manufacture a false Pareto-optimal candidate. Explicitly forbidden by the brief.

### 10.6 — Data-handling defects in ESOL (MEDIUM)
- 11 duplicate canonical structures are never deduplicated before splitting → mild train/test leakage.
- No scaffold split anywhere; random split only, so reported R² is optimistic for prospective use.
- No feature scaling before PCA in the AD engine either (same flaw as §10.2 — leverage is therefore dominated by MolWt).
- Models are retrained on every process start; no persistence, no `model_version`, no dataset hash recorded.

### 10.7 — ClinTox training is arbitrary (MEDIUM)
`clintox_feats[:1200]` discards ~280 compounds by position with no shuffle and no stratification; no train/test split exists, so no metric can ever be produced; no `class_weight='balanced'` against 7.5% positives.

### 10.8 — Conformal intervals are never recalibrated (MEDIUM)
Brief §3 requires recalibration when the model changes after Active Learning. The AL engine and the conformal engine share no state; calibration happens exactly once at boot.

### 10.9 — Error handling and robustness (MEDIUM)
- `active_learning.py:~200` — `except Exception: pass` around `fit_gpytorch_mll` hides all optimiser failures.
- `models.py:_train_auxiliary_models` — catches everything and continues with `clintox_model = None`, after which the code silently falls back to a `logP`-based toxicity invention with no indication to the user.
- `/api/predict` wraps everything in a 500 with a stringified exception — leaks internals, loses the distinction between bad input and server fault.
- `MoleculeViewer3D` fails to a blank box if the 3Dmol CDN is unreachable (offline/air-gapped demo = dead viewer).

### 10.10 — Architecture (MEDIUM/LOW)
- `ModelService.get_instance()` trains at **module import**, so `uvicorn --reload`, any test collection, and any `import main` pays 40–60 s.
- `GET /api/pareto` re-runs 30 full predictions per request, uncached.
- Global mutable AL state with no campaign/session identity — two browsers share one experiment.
- `POST /api/active-learning/rank` duplicates the GET route.
- `import os` appears twice in `main.py`; `CyberCursor.tsx` is dead code.
- CORS `allow_origins=["*"]` with `allow_credentials=True`.

---

## 11. Missing proposed research features

Everything in brief Part 3 is absent. No partial scaffolding exists for any of it.

Also missing from Part 2: EGFR QSAR model, validated metabolic-stability endpoint, scaffold-split benchmarking, downloadable reports, model persistence.

---

## 12. Dependency and compatibility problems

| Issue | Severity | Detail |
|---|---|---|
| **Python 3.14 + torch 2.15** | High for Stage 3 | `torch.jit.script` warns it "may break" on 3.14+. D-MPNN training should avoid TorchScript; consider validating on 3.12 before committing |
| **pytest not installed** | High | No test infrastructure at all |
| **MLflow not installed** | Medium | Required by §(k) |
| **BoTorch 0.18 API drift** | Medium | `ExpectedImprovement` is legacy; `LogExpectedImprovement` is the current recommendation. `qNEHVI` exists in 0.18 and *is* available for §(f) — this needs an explicit compatibility check before Stage 4 |
| **No MOSES / DOCKSTRING / ChEMBL / AutoDock Vina / AiZynthFinder** | Blocking | §(c), §(f-data), §(g), §(h) are blocked on external resources |
| **`torch` is 2.5 GB of dependency for one GP** | Low | Acceptable given Stages 3–4 will use it properly |
| **5.8 MB unsplit JS bundle** | Low | three.js + plotly, no code splitting |
| **3Dmol.js from CDN** | Low | Breaks offline; should be vendored |
| `requirements.txt` has no upper bounds | Low | Reproducibility risk; no lockfile |

---

## 13. Feature-status matrix — brief methodology (a)–(k)

| § | Module | Status | Evidence |
|---|---|---|---|
| (a) | D-MPNN ensemble | **Not implemented** | No `nn.Module` anywhere in the repo; torch used only by BoTorch |
| (b) | Normalised conformal | **Partially implemented** | σ-normalised split conformal is present *and correct* in `conformal.py`. Missing: classification conformal sets, latent-space AD, 95th-percentile threshold derivation, the two-condition OOD rule |
| (b) | Advanced AD | **Partially implemented** | Tanimoto + PCA leverage present; thresholds unvalidated; no latent distance; no density-ratio reweighting |
| (c) | RL molecular generator | **Not implemented** | No generator, no MOSES, no REINVENT loop |
| (d) | Probabilistic Pareto | **Not implemented** | Deterministic Pareto only; no sampling, no P(Pareto-optimal) |
| (e) | Risk slider | **Not implemented** | No λ/κ anywhere. Note: the AL `beta` slider is a *different* thing (GP-UCB β) |
| (f) | EHVI budget planner | **Not implemented** | No hypervolume, no cost model, no budget. `strategy="pareto_hv"` is a **mislabelled heuristic** (`utility + 1.2σ`) and must be renamed |
| (g) | Docking surrogate + Vina | **Not implemented** | No docking code, no DOCKSTRING, Vina not installed |
| (h) | Synthesis feasibility | **Not implemented** | No SA score (RDKit `sascorer` not wired), no AiZynthFinder |
| (i) | Counterfactual editing | **Not implemented** | — |
| (j) | Objective-conflict explanation | **Not implemented** | No attribution; no differentiable model to attribute through |
| (k) | MLflow / audit / collaboration | **Not implemented** | No tracking, no persistence, no accounts, no export |

### Part 2 modules

| # | Module | Status | Note |
|---|---|---|---|
| 1 | Molecular Analyzer | **Completed, tests pending** | Scientifically sound; needs a test suite |
| 2a | ESOL solubility | **Implemented, validation partial** | R²=0.894 real; fix dedup + add scaffold split |
| 2b | ClinTox toxicity | **Implemented, validation pending** | Never evaluated; imbalance unaddressed |
| 2c | EGFR pIC50 | **Not implemented** | Heuristic masquerading as a model — must be removed or replaced |
| 2d | Metabolic stability | **Not implemented** | Invented formula — must be removed or replaced |
| 3 | Uncertainty / conformal | **Completed, validation partial** | Correct maths, reproduced coverage; no recalibration after AL |
| 4 | Applicability domain | **Partially implemented** | Works; thresholds unvalidated; leverage distorted by unscaled features |
| 5 | Pareto Laboratory | **Partially implemented** | Correct sorting; unsafe missing-value default; includes invalid objectives |
| 6 | Active Learning | **Broken** | Degenerate acquisition + oracle leak |
| 7 | Benchmarking | **Partially implemented** | ESOL regression only; no classification metrics, no scaffold split, no export |

---

## 14. Prioritised implementation roadmap

**P0 — correctness and honesty (Stage 2). Nothing else should ship before these.**
1. Stop serving hidden oracle labels; split the AL candidate schema into a public view and a server-side oracle store.
2. Fix the GP: standardise features, use BoTorch `Normalize`/`Standardize` input/outcome transforms, surface fit failures instead of swallowing them. Verify acquisition scores differ across candidates and respond to new labels.
3. Remove `bioactivity_pic50` and `metabolic_stability_half_life_min` from the prediction payload and from the default Pareto objectives, or mark them `status: "unavailable"` and exclude them from optimisation.
4. Deduplicate ESOL by canonical SMILES before splitting; record dataset hash + seed; re-report metrics.
5. Pareto: raise on missing objectives rather than defaulting to 0.0.
6. Drive the landing-page metrics from `/api/benchmark`.
7. Add `pytest` and a first test suite; ClinTox train/test split with ROC-AUC + PR-AUC + `class_weight`.

**P1 — foundation (Stage 2 cont.)**
8. Model persistence + `model_version` + startup load-or-train; move training out of import time.
9. Scaffold-split benchmark alongside random split.
10. Structured logging, settings module, provenance field on every predicted value.
11. Error-state handling for the 3D viewer; vendor 3Dmol.

**P2 — advanced ML (Stage 3)**
12. D-MPNN ensemble (§a) with Gaussian NLL heads, 5 members, CPU-first, small smoke config.
13. Classification conformal sets + latent-space AD + validated thresholds (§b).
14. Independent RF-fingerprint validation model (§k, first slice).

**P3 — discovery (Stage 4)**
15. Probabilistic Pareto (§d) — this depends on (a) for calibrated σ.
16. Risk slider (§e) — cheap once (d) exists.
17. EHVI planner (§f) — verify `qNEHVI` under BoTorch 0.18 first.

**P4 — resource-gated (Stage 5–6)**
18. SA score (§h, first half) — **unblocked**, RDKit ships `sascorer`.
19. Counterfactual editing (§i) — **unblocked**, needs only RDKit + the predictor.
20. Objective-conflict explanation (§j) — unblocked once (a) gives a differentiable model.
21. MLflow + audit trail (§k) — unblocked, needs `pip install mlflow`.
22. Docking (§g), retrosynthesis (§h second half), RL generator (§c), EGFR QSAR — **blocked on external data/tools**; build the integration points and mark pending.

---

## 15. Files likely to require modification

| File | Change |
|---|---|
| `backend/app/core/active_learning.py` | Rewrite: oracle separation, feature scaling, GP transforms, honest strategy names |
| `backend/app/core/models.py` | Remove fabricated properties, dedup, persistence, ClinTox evaluation, provenance |
| `backend/app/core/dataset.py` | Dedup, scaffold split, scaling, caching, dataset hashing |
| `backend/app/core/pareto.py` | Missing-value policy, objective registry |
| `backend/app/core/applicability_domain.py` | Scale features, validate thresholds, latent distance |
| `backend/app/core/conformal.py` | Classification sets, recalibration hook, configurable α |
| `backend/app/main.py` | Schemas, lifespan startup, response models, logging, new routers |
| `frontend/src/components/ActiveLearningTab.tsx` | Drop `true_*` reads; render only revealed labels |
| `frontend/src/components/LandingPage.tsx` | Fetch metrics instead of hardcoding |
| `frontend/src/components/PropertyPredictionTab.tsx` / `ParetoLaboratoryTab.tsx` | Provenance badges; hide unavailable objectives |
| `requirements.txt` | Add pytest, mlflow, pin upper bounds |
| **new** `backend/tests/` | Entire test suite |

---

## 16. Recommended tests per module

| Module | Tests |
|---|---|
| `mol_analyzer` | Valid/invalid/empty/whitespace SMILES; known descriptor values for aspirin, caffeine, benzene; Lipinski boundary at MW 500 / logP 5; stereochemistry round-trip; 3D embed success and documented failure for a strained macrocycle; SVG is well-formed XML |
| `dataset` | Row count 1128; 0 parse failures; **dedup yields 1117**; feature dim 1032; fingerprint determinism; scaffold split disjointness |
| `models` | Train/cal/test index disjointness; **no canonical SMILES appears in two partitions**; R² above a regression floor; seed reproducibility; save/load round-trip; ClinTox ROC-AUC > 0.5 with a real split |
| `conformal` | Quantile index equals `⌈(n+1)(1−α)⌉`; coverage within tolerance on synthetic Gaussian data; monotone widths across 80/90/95; normalised vs. standard agreement when σ is constant; cal/test separation enforced |
| `applicability_domain` | Identical molecule → Tanimoto 1.0, IN_DOMAIN; a long alkane vs. an ESOL training set → OUT_OF_DOMAIN; threshold boundary behaviour; leverage finite and non-negative |
| `pareto` | Dominance truth table; known 2-objective frontier by hand; minimisation direction; **missing value raises, never silently zeroes**; single candidate; all-identical candidates |
| `active_learning` | **Acquisition scores are not all equal** (direct regression test for §10.2); ranking changes after an assay; selected candidate is never already labelled; **API response contains no `true_*` key** (regression test for §10.3); GP posterior variance is positive and finite |
| Integration | `/api/health` 200; `/api/analyze` 400 on garbage; `/api/predict` schema; `/api/pareto` frontier non-empty and rank-1 consistent; assay → re-rank → history grows |

---

## 17. Honest summary

MoleculeMind is **a well-engineered demo with three genuinely sound scientific components and three serious integrity defects.**

What is real: the molecular analyser, the ESOL model (R² = 0.894, reproduced), and the conformal prediction engine (correct finite-sample maths, reproduced coverage). The frontend is good and — notably — every screen is connected to a live endpoint. There are no fake pages.

What is not real: the EGFR pIC50 and metabolic-stability "predictions" are descriptor arithmetic with invented coefficients, served inside the same payload as the real predictions and fed into Pareto optimisation. The Active Learning engine's acquisition function is mathematically degenerate — every candidate receives the identical score, so the ranking is meaningless — and the API ships the hidden oracle labels to the browser, which defeats the human-in-the-loop design the project is named for.

Against the proposed research architecture, **0 of the 11 modules (a)–(k) are implemented**, though §(b) is roughly half-covered by the existing conformal code, which is a better starting point than the brief assumes.

The honest headline: the foundation is smaller than the proposal but the parts that work, work. The P0 list in §14 is the gate — until the oracle leak, the degenerate GP and the fabricated objectives are fixed, the platform's central claims are not defensible.
