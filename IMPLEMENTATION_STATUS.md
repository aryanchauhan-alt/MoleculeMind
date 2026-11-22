# MoleculeMind — Implementation Status

**Last updated:** 2026-11-22
**Stages complete:** Stage 1 (audit), Stage 2 (P0 + P1), Stage 3, **Stage 4 complete**
**Stages outstanding:** Stages 5–8

This file records what is actually implemented and verified. Nothing here is
claimed without a test or a reproduced run behind it. See `PROJECT_AUDIT.md`
for the full pre-change audit and `TEST_REPORT.md` for the test results.

---

## Summary table

| Module | Implementation status | Algorithm | Dataset | Tests | Remaining limitations |
|---|---|---|---|---|---|
| 1. Molecular Analyzer | **Completed and tested** | RDKit parse/canonicalise, ETKDGv3 + MMFF94, 13 descriptors, Lipinski, Veber | — | 10 tests | None outstanding; 3Dmol is vendored and the viewer has error/retry states |
| 2a. ESOL solubility | **Completed and tested** | ExtraTrees (150 trees) on ECFP4 + 8 descriptors, cached artifact | Delaney ESOL, 1117 unique | 6 tests | Single seed; no cross-validation or confidence interval on R² |
| 2b. ClinTox toxicity | **Implemented and evaluated** | RandomForest, `class_weight='balanced'`, stratified 80/20 | ClinTox, CT_TOX | 2 tests | Uncalibrated probabilities; CT_TOX label semantics are narrow (see below) |
| 2c. EGFR pIC50 | **Not implemented** (was a heuristic, now removed) | — | none bundled | 2 guard tests | Blocked: needs curated ChEMBL assay data |
| 2d. Metabolic stability | **Not implemented** (was a heuristic, now removed) | — | none bundled | 2 guard tests | Blocked: needs a clearance/half-life dataset |
| 3. Uncertainty / conformal | **Completed and tested** | σ-normalised split conformal, `⌈(n+1)(1−α)⌉` quantile, recalibration after AL rounds | ESOL calibration partition | 12 tests | No classification conformal sets; recalibration does not retrain the base model |
| 4. Applicability domain | **Completed and tested** | ECFP4 Tanimoto k-NN + leverage on standardised PCA space; two-condition rule; thresholds derived from training percentiles | ESOL training partition | 11 tests | Leverage is weak on a sparse fingerprint space (10 PCs ≈ 11% variance); no learned latent distance yet |
| 5. Pareto Laboratory | **Completed and tested** | O(n²) dominance + non-dominated sorting | candidate pool | 8 tests | O(n²) will not scale past a few thousand candidates |
| 6. Active Learning | **Completed and tested** | BoTorch `SingleTaskGP` + UCB / LogEI / uncertainty / greedy, with conformal recalibration per round | ESOL held-out pool (30) | 20 tests | Single objective; base model is not retrained between rounds |
| 7. Benchmarking | **Completed and tested** | R², RMSE, MAE, conformal coverage, ROC-AUC, PR-AUC, Brier; random + scaffold split; JSON/CSV report download | ESOL, ClinTox | 8 tests | No model-vs-model comparison UI (needs a second model — Stage 3) |
| (a) D-MPNN ensemble | **Completed and tested** | Directed MPNN, 3 steps, hidden 300, mean readout, Gaussian NLL + masked BCE, 5 seeded members | ESOL + ClinTox multi-task union (2,514 molecules) | 37 tests | Matches but does not beat the fingerprint baseline at this dataset size |
| (b) Normalised conformal + latent AD | **Completed and tested** | σ-normalised split conformal; LAC / APS / randomised APS prediction sets; latent nearest-neighbour AD | ESOL, ClinTox | 31 tests | No density-ratio reweighting for distribution shift |
| (k, first slice) Independent validation | **Completed and tested** | Random Forest on fingerprints vs D-MPNN on the same held-out compounds, with disagreement flagging | ESOL | 3 tests | No independent docking validation (Stage 5); no MLflow yet (Stage 6) |
| (c) RL molecular generator | **Implemented, MOSES pre-training pending** | Char-level GRU LM + REINVENT augmented likelihood, uncertainty-aware reward, diversity filter | ESOL ∪ ClinTox SMILES (2,512) | 24 tests | Validity 46–78% vs >90% for a MOSES prior — data-limited, not code-limited |
| (d) Probabilistic Pareto | **Completed and tested** | MC over conformally calibrated Gaussians, fast non-dominated sorting, bounded pre-filter | candidate pool | 18 tests | Gaussian shape assumption rejected on real residuals (reported); correlated sampling refused for lack of joint labels |
| (e) Risk slider | **Completed and tested** | f = μ ± κσ̃, κ = κ_max·λ, live re-ranking | candidate pool | 7 tests | None outstanding |
| (f) EHVI budget planner | **Completed and tested** | MC EHVI with exact BoTorch hypervolume, cost-aware greedy | candidate pool | 24 tests | Greedy has no optimality bound under heterogeneous costs (stated) |
| (g)–(j) remaining modules | **Not implemented** | — | — | — | Stage 5; (g) and part of (h) blocked on external resources |

---

## Stage 2 — what changed

### Fixed: hidden oracle labels were served to the browser (audit §10.3)

`GET /api/active-learning/pool` previously returned `true_pic50`,
`true_solubility` and `true_toxicity` for every candidate including the
unassayed ones, so the answer key was readable in DevTools.

The engine now keeps four separate stores: a private `_oracle` dict, a
`_revealed` id set, the public candidate records, and the model's own test set
(held in `ModelService`). `public_pool()` is the only serialisation path and
raises if an oracle key ever appears in it. Two tests scan the full serialised
API payload for oracle keys; a third asserts a measurement appears only after
its assay has been executed.

Both frontend reads of `true_*` were removed.

### Fixed: the acquisition function was mathematically degenerate (audit §10.2)

Measured before: all 24 unlabelled candidates received the identical score
(UCB 1.414, EI 0.017, σ 0.606 — one unique value each). Cause: PCA was fitted on
the unscaled 1032-dim feature matrix, where molecular weight held 90.9% of the
variance and components spanned [−113, +178]; BoTorch's default lengthscale
prior assumes roughly `[0,1]^d`, so the posterior collapsed to the prior.

Now: `StandardScaler` → PCA(8) → unit-cube rescale, plus BoTorch `Normalize`
input and `Standardize` outcome transforms. `fit_gpytorch_mll` failures are
logged and exposed as `gp_fit_ok` instead of being swallowed by `except: pass`.

Measured after, on the same pool:

| Strategy | Unique scores (of 24) | Range |
|---|---|---|
| ucb | 17 | −2.292 … 1.124 |
| ei (LogEI) | 17 | −15.152 … −1.395 |
| uncertainty | 16 | 0.702 … 1.855 |
| greedy | 17 | −3.284 … −0.751 |

Ranking now responds to new labels, and `acquisition_diagnostics.degenerate` is
reported on every pool response.

### Fixed: fabricated properties presented as model predictions (audit §10.4)

`bioactivity_pic50` and `metabolic_stability_half_life_min` were closed-form
descriptor arithmetic with invented coefficients. Both are removed. They now
return `status: "unavailable"` with the reason and what would be required to
enable them, carry no numeric value, and are excluded from Pareto optimisation.
`GET /api/objectives` lists what is available and what is not.

The 30-molecule AL library with invented `true_pic50` / `true_toxicity` values
attributed to named real drugs was replaced: the pool is now 30 real Delaney
ESOL compounds held out from training and calibration, so the oracle is an
actual experimental measurement with traceable provenance.

### Fixed: Pareto silently imputed 0.0 for missing objectives (audit §10.5)

A missing value now raises `ValueError`. For a minimised objective, 0.0 was the
best possible value, so a candidate missing that property would have been
promoted onto the front on the strength of data it did not have.

### Fixed: train/test leakage and split hygiene (audit §10.6)

All datasets are deduplicated on canonical SMILES before splitting — 11
duplicate structures removed from ESOL. A 16-character content hash is recorded
with every metric set. Partition disjointness (train / calibration / test / AL
pool) is asserted by test.

### Fixed: ClinTox was never evaluated (audit §10.7)

Previously trained on `feats[:1200]` by position with no split, no metrics and
no class weighting. Now a stratified 80/20 split with `class_weight='balanced'`:

```
ROC-AUC 0.832   PR-AUC 0.358   (baseline positive rate 0.072)   Brier 0.0565
```

### Fixed: hardcoded metrics in the landing page (audit §10.1)

`R² = 0.894, RMSE = 0.704, MAE = 0.482` were literal strings in
`LandingPage.tsx`. They are now fetched from `/api/benchmark`.

### Added: scaffold-split benchmark

A parallel model is trained on a deterministic Bemis-Murcko scaffold split and
reported separately. **Honest caveat:** ESOL contains many acyclic compounds,
each of which becomes a singleton scaffold group, so this split is closer to
random than a scaffold split on a drug-like library would be. That is why its
R² (0.890) is close to the random-split R² (0.903) and the gap should not be
read as evidence of strong chemotype generalisation.

### Added: provenance on every returned value

Each property under `properties` carries `status` and `provenance`
(`model` / `computed` / `unavailable`), the model that produced it, its unit,
and any caveat. The Prediction tab renders these as badges and shows
unavailable properties as explicitly unavailable rather than hiding them.

### Added: a ClinTox memorisation flag

Spotted while verifying: aspirin returns a toxicity probability of 1.0. The
cause is not a bug — **aspirin is labelled `CT_TOX = 1` in the ClinTox dataset
itself**, and it falls in the training partition, so the value is a memorised
label. Responses now carry `in_training_set` and a `memorisation_warning`, and
the metrics explain that CT_TOX means "clinical trial terminated for toxicity",
not "this molecule is toxic".

### Added: test suite

94 tests across 5 files, all passing. See `TEST_REPORT.md`.

### Other

- API: Pydantic field constraints, specific status codes, `/api/objectives`,
  cached Pareto response, structured logging, CORS narrowed from `*`.
- Frontend: Pareto plots are now axis-driven (X/Y/Z selectable) instead of
  hardcoded to removed properties; typecheck and production build both clean.

---

## Stage 2 P1 — what changed

### Added: configuration module

`backend/app/core/config.py` declares every tunable in one frozen dataclass.
Each is overridable by a `MOLECULEMIND_`-prefixed environment variable, and
`/api/config` reports the configuration a run actually used. `training_fingerprint()`
isolates the settings that affect model identity from presentation-only ones.

### Added: model persistence and load-or-train startup

Trained artifacts are pickled to `backend/artifacts/` under a key that hashes
the dataset **file bytes**, the model version, the cache format and every
training-relevant setting. Hashing file bytes rather than the parsed dataset
keeps the key cheap, so a cache hit skips RDKit featurisation of ~2,600
molecules rather than paying for it just to learn the cache is valid.

| Start | Time | Source |
|---|---|---|
| Cold | 3.8 s | trained from scratch |
| Warm | ~0.0 s | loaded from cache |

Writes are atomic (`os.replace` from a `.tmp`), a corrupt or mismatched cache
falls back to training, and a write failure logs a warning and leaves no partial
file rather than breaking startup. `backend/artifacts/` is gitignored.

RDKit's `FingerprintGenerator64` is a Boost.Python object and cannot be pickled;
`DatasetManager` and `ApplicabilityDomainEngine` now drop it in `__getstate__`
and rebuild it from `(radius, fpSize)` in `__setstate__`.

### Fixed: training no longer happens at module import

`ModelService.get_instance()` previously ran at import time, so every
`uvicorn --reload`, test collection and plain `import main` paid the full
training cost. Services are now built in the FastAPI `lifespan` hook, with a
`services()` accessor that lazily starts them if a route is reached without the
hook having run. `import backend.app.main` now returns immediately.

### Fixed: applicability-domain features were unscaled (audit §10.6)

The AD engine's PCA was fitted on the raw 1032-dim matrix, so — exactly as in
the Active Learning bug — molecular weight carried 90.9% of the variance and
"leverage" was effectively a molecular-weight test. Features are now
standardised before PCA.

### Fixed: applicability-domain thresholds were asserted, not validated

The hardcoded Tanimoto cut-offs of 0.55 / 0.35 are gone. Both thresholds are
now derived at fit time from the training set's own leave-one-out diagnostics:

| Threshold | Derivation | Value on ESOL |
|---|---|---|
| Similarity | 5th percentile of leave-one-out nearest-neighbour Tanimoto | 0.250 |
| Leverage | 95th percentile of training leverage | 0.0583 |
| *(reference)* OECD h\* = 3(p+1)/n | rule of thumb, reported not used | 0.0423 |

Classification follows the methodology's two-condition rule: **out of domain
only when both** diagnostics fail, **borderline** when exactly one does. On the
training set this flags 0.5% of molecules, so the engine does not reject the
data it was fitted on. `/api/applicability-domain/diagnostics` exposes the full
derivation.

`reliability_score` was renamed `domain_index` (the old key is retained for
compatibility) and now carries an explicit caveat that it is a ranking composite,
not a calibrated probability of correctness.

**Honest limitation:** with standardised features, 10 principal components
capture only ~11% of the variance of a mostly-sparse 1032-dim fingerprint space.
Leverage is therefore a much weaker signal than similarity here. The right fix is
a learned latent representation (Stage 3); until then this is documented rather
than hidden.

### Fixed: conformal intervals were never recalibrated after Active Learning (audit §10.8)

`ModelService.recalibrate_conformal()` folds newly revealed labels into the
calibration residuals and recomputes the quantiles. The AL engine calls it after
every assay (interval configurable), and `/api/benchmark` exposes the full
calibration history with a generation counter.

Two caveats are returned in the payload rather than left implicit: the base model
is **not** retrained, so this corrects the calibration set and not the predictor;
and labels chosen by an acquisition function are not an i.i.d. sample, so strict
exchangeability with the original calibration set does not hold and post-recalibration
coverage is approximate.

### Fixed: 3D viewer failed silently to a blank box

3Dmol.js 2.0.4 is vendored to `frontend/public/3Dmol-min.js` (470 KB) and served
from the app's own origin, so the viewer works offline and air-gapped. The
component now has explicit loading, error and empty states: it polls for the
bundle with a timeout, shows a specific message and a Retry button if the library
or the render fails, and surfaces the backend's `sdf_error` when RDKit could not
embed a conformer at all — noting that the 2D structure and all predictions are
unaffected.

### Added: downloadable benchmark reports

`GET /api/benchmark/report?format=json` returns a full record — configuration,
both splits, ClinTox metrics, AD derivation, calibration history, parity table
and five explicit honesty notes — with a `Content-Disposition` attachment header.
`format=csv` returns the parity table for spreadsheet use.

### Added: structured logging

All `print()` calls in the model service are now `logging` calls on the
`moleculemind.*` hierarchy, with the level set by `MOLECULEMIND_LOG_LEVEL`.

### Tests

94 → **126 tests**, all passing. The new file
`backend/tests/test_persistence_and_config.py` (24 tests) covers settings
overrides, cache round-trip and every cache failure mode, the derived AD
thresholds, and recalibration.

---

## Stage 3 — what changed

### Added: D-MPNN ensemble (§a)

`backend/app/core/molecular_graph.py` and `backend/app/core/dmpnn.py` implement
a Directed Message Passing Neural Network from scratch, to the specified
architecture: 3 message-passing steps (configurable 2–5), hidden size 300, mean
readout, two-layer feed-forward head with ReLU and dropout 0.1, Adam from
`lr = 1e-3` with warm-up and decay, batch size 64, early stopping, 5
independently seeded members.

**The message passing is verified against its mathematical definition, not
eyeballed.** A subtly wrong graph network still trains and still produces
plausible numbers, so `test_directed_message_excludes_reverse_edge` recomputes

    m_e = Σ_{k ∈ N(v) \ {w}} h_{(k→v)}    for e = (v→w)

by explicit enumeration over every edge and asserts agreement with the
vectorised `incoming[src] − h[rev]` form to 1e-5. A companion test asserts the
reverse-edge subtraction actually changes the result, so a regression that drops
it fails loudly. Permutation invariance, batching invariance and bondless
molecules (methane, lone ions) are each covered.

Full mathematics is in [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) §2.

### Added: genuine multi-task learning with a masked loss

ESOL and ClinTox are joined on canonical SMILES into one table with **NaN where
a label is absent**:

| | count |
|---|---|
| Molecules in the union | 2,514 |
| With a logS label | 1,117 |
| With a CT_TOX label | 1,461 |
| With **both** labels | 64 |

Only 64 molecules carry both labels, so the benefit comes through the shared
encoder rather than through correlated heads. **Measured effect: ClinTox ROC-AUC
0.832 → 0.858.** The masked loss is verified by setting a masked entry to 1e6
and asserting the loss does not move.

### Added: ensemble uncertainty decomposition

`Var_total = mean(σ²ₘ) + var(μₘ)` — aleatoric plus epistemic.
`test_variance_decomposition_is_exact` recomputes both terms from the individual
members and asserts equality to 1e-6. Measured on the ESOL test partition:
aleatoric ≈ 0.397, epistemic ≈ 0.034, with predicted σ correlating with absolute
error at r ≈ 0.33–0.43.

**A real bug this caught:** ensemble members were not reproducible for a fixed
seed, because `torch.manual_seed` ran inside `fit()` while weight initialisation
happens in the constructor — so initialisation drew from ambient RNG state. Now
seeded before construction.

### Added: conformal prediction sets for classification (§b)

Three methods in `ClassificationConformalEngine`, all split-conformal with the
`⌈(n+1)(1−α)⌉` quantile. Measured on a 3-class synthetic problem (4,000
calibration / 6,000 test):

| Method | 80% | 90% | 95% | mean set @90% |
|---|---|---|---|---|
| LAC | 79.9% | 89.8% | 94.3% | 1.42 |
| APS deterministic | 99.2% | 99.8% | 100.0% | 2.69 |
| APS randomised | 80.0% | 89.9% | 94.8% | 1.57 |

**Worth flagging:** the deterministic APS variant **over-covers badly** — a
nominal-80% set achieving 99.2% coverage with mean size 2.36 of 3 is valid but
useless. The randomised variant from the paper fixes it. Both are kept and the
difference is documented rather than quietly shipping the conservative one.

### Added: latent-space applicability domain (§b)

The D-MPNN's molecule embeddings become the AD's second diagnostic, replacing
descriptor-space leverage when available — which is what the methodology
specifies. Threshold = 95th percentile of training leave-one-out latent
distances (measured 0.356 in 300 dimensions).

**A case that justifies having two independent diagnostics:** a C36
straight-chain alkane returns Tanimoto similarity **1.000** against the ESOL
training set, because Morgan fingerprints of long alkanes are bit-identical to
short ones — every atom environment is the same CH2. The similarity test passes
completely. Its latent distance is 0.481 against a 0.356 threshold, so the
learned representation catches what the fingerprint cannot and the molecule is
correctly flagged BORDERLINE.

### Added: independent model comparison (§k, first slice)

A Random Forest on fingerprints is trained as an independent check — a different
model family on a different representation — and scored on the **same held-out
compounds**. The D-MPNN's ESOL test partition is pinned to the baseline's, so the
comparison is not across two different random splits.

Measured: Pearson r = 0.94 between the two models, with 25.4% of test compounds
disagreeing by more than the D-MPNN's own σ. The payload states plainly that
agreement between models trained on the same data is weak evidence of
reliability, not proof of correctness.

### Fixed: the classification head had no held-out evaluation

Pinning the test partition to the baseline's ESOL compounds gave a fair
regression comparison but left the classification head with almost no CT_TOX
labels to score, so no classification metric could be computed. A stratified
slice of ClinTox-only molecules is now added to the test partition; it cannot
affect the regression comparison because those molecules have no logS label. A
task that still cannot be scored reports `status: "not_evaluated"` with a reason
rather than silently omitting itself.

### Fixed: numpy types reaching the JSON encoder

Freshly-trained metrics embedded the raw numpy label matrix and `numpy.float32`
scalars, which FastAPI cannot serialise — surfacing as an opaque 500 at request
time, and *only* on a cold start, because the warm path had round-tripped through
JSON. The metrics payload is now sanitised at the point it is built, and the raw
label matrix never enters it.

### Fixed: warm starts reported a trained model with no metrics

The D-MPNN checkpoint carried weights but not the evaluation record, so after a
restart the API reported `status: "trained"` with no metrics. Metrics and the
full conformal calibration state are now persisted in a sidecar beside the
checkpoint, and a **missing sidecar invalidates the load** — recomputing metrics
against a different split, or serving a model with none, would both be
misleading.

### Measured results

| Model | Representation | Split | R² | RMSE |
|---|---|---|---|---|
| ExtraTrees (baseline) | ECFP4 + descriptors | random | 0.903 | 0.666 |
| ExtraTrees | ECFP4 + descriptors | scaffold | 0.890 | 0.746 |
| D-MPNN ensemble (5) | molecular graph | random, pinned | 0.899 | 0.680 |
| Random Forest (independent) | ECFP4 + descriptors | same compounds | 0.877 | — |

The graph network **matches but does not beat** the fingerprint baseline on
solubility (0.899 vs 0.903, well inside the noise of a 138-compound test set).
That is the honest result and the expected one at this dataset size. Its value
here is the calibrated per-molecule σ, the learned latent space for the
applicability domain, and the multi-task gain on classification — not raw
regression accuracy.

### Cost

| Start | Time | Notes |
|---|---|---|
| Cold | ~95 s | 5-member ensemble trained on CPU, then checkpointed |
| Warm | ~1 s | weights + metrics sidecar loaded from cache |

Set `MOLECULEMIND_ENABLE_DMPNN=false` to skip it entirely; the Stage 2 platform
then runs unchanged rather than breaking.

### Frontend

The Property Prediction tab gained an independent cross-check card showing both
models side by side, the aleatoric/epistemic split, and the conformal prediction
sets. The Benchmarking tab gained a model-comparison table (all four models,
both splits), the uncertainty decomposition, the σ-tracks-error correlation, the
ClinTox classification section with conformal set coverage, and report download
buttons. Both degrade to an explicit "D-MPNN unavailable" notice with the reason
when the ensemble is absent.

### Tests

126 → **204 tests**, all passing. New: `test_dmpnn.py` (37) and
`test_stage3_integration.py` (31), plus 7 new API tests.

---

## Stage 4 — what changed

### Added: probabilistic Pareto front (§d)

`backend/app/core/probabilistic_pareto.py`. A conformal interval is converted to
a Gaussian scale by width matching, `σ̃ = q̂σ/z`; candidates are sampled **jointly**
so one candidate's good luck is evaluated against another's bad luck; the
non-dominated set is recomputed per draw; the probability is the fraction of
draws in which a candidate appears in it. A Monte Carlo standard error is
returned with every result.

**The Gaussian assumption is measured, not assumed.** On this platform's own
ESOL calibration residuals, excess kurtosis is **1.416** and the
Kolmogorov–Smirnov test **rejects** normality. The interval widths stay valid;
the shape inside them does not, which makes the probabilities mildly optimistic
in the tails. The frontend shows this as an amber warning whenever it fires —
which, on the bundled data, is always.

**Correlated sampling is refused rather than faked.** The code supports a
correlation matrix via a Cholesky factor, but the API returns **HTTP 501**:
only 64 molecules carry both a solubility and a toxicity label, too few to
estimate the matrix honestly. Inventing one would look more rigorous while being
less trustworthy.

### Added: uncertainty-aware risk slider (§e)

`f = μ + κσ̃` for maximised objectives and `μ − κσ̃` for minimised ones, with
`κ = κ_max·λ`. The sign flip is essential and tested: optimism about toxicity
means assuming a *lower* value, and getting it backwards would penalise
uncertainty on exactly the objectives that should reward it. Live re-ranking via
a debounced slider, labelled throughout as an optimistic ranking device that is
**not** a calibrated prediction.

### Added: cost-aware EHVI experiment planner (§f)

`backend/app/core/experiment_planner.py`. `EHVI(x) = E[HV(P ∪ {y}) − HV(P)]`,
with each hypervolume term computed **exactly** by BoTorch's box decomposition
and only the expectation sampled.

**This is verified to be EHVI, not an EHVI-shaped heuristic** — the brief's most
insistent requirement, so it is checked against two independent ground truths:

1. **Single-objective reduction to Expected Improvement.** EHVI reduces
   analytically to EI in one dimension; the estimator matches the closed form to
   within ~2 Monte Carlo standard errors across four parameter settings. A
   weighted `μ + kσ` score cannot reproduce the EI curve, so this test alone
   separates a real estimator from a heuristic.
2. **Cross-check against BoTorch's own `qExpectedHypervolumeImprovement`** on a
   shared GP posterior: **max relative difference 0.24%**. Exposed live at
   `GET /api/planner/validate-ehvi`.

BoTorch 0.18.1 does ship `qEHVI`, `qNEHVI` and `qLogEHVI`, so the Monte Carlo
route is a deliberate choice, not a fallback: `qEHVI` integrates over a *GP
posterior*, and this platform's distributions come from a D-MPNN ensemble
calibrated by conformal prediction. Routing them through `qEHVI` would mean
fitting a GP to our own predictions and integrating over that instead —
discarding the calibration which is the point.

An assay informs only some objectives; the rest are held at their mean with
**zero** variance, because measuring solubility tells you nothing new about
toxicity.

Verified: budget never exceeded, no (molecule, assay) pair repeated, expected
gain non-decreasing in budget, early stop when no improvement remains, and
changing assay costs genuinely reshapes the plan (cheap toxicity moved it from 5
solubility assays to 7 toxicity + 3 solubility). Limitations are stated in the
payload: greedy has a `(1 − 1/e)` guarantee only in the unit-cost case, mean
imputation is optimistic, and hypervolume depends on the reported reference point.

### Added: molecular generator with RL (§c) — honestly data-limited

`backend/app/core/generator.py` and `generator_service.py`. Character-level GRU
language model with a round-tripping tokeniser (multi-character atoms handled),
maximum-likelihood pre-training, then REINVENT augmented likelihood
`Loss = (log P_prior + σR − log P_agent)²` with a Murcko-scaffold diversity
filter.

**MOSES pre-training is pending and the cost is measured, not hidden.** MOSES
(~1.9M molecules) is not bundled and is not downloaded automatically. The prior
trains on the 2,512 SMILES available locally, three orders of magnitude fewer:

| Temperature | Validity | Uniqueness | Novelty | Internal diversity |
|---|---|---|---|---|
| 0.7 | 0.780 | 0.936 | — | — |
| 1.0 | 0.463–0.592 | 0.971 | 0.72 | 0.91 |
| 1.2 | 0.435 | 0.966 | — | — |

A MOSES-pretrained prior of this architecture typically exceeds 90% validity.
Uniqueness, novelty and diversity are respectable; **validity is the casualty**,
and `scripts/download_moses.py` documents how to close the gap without changing
any other code.

**The reward uses the LOWER conformal bound, not the mean.** Rewarding the mean
would teach the generator to exploit regions where the model is merely
uncertain — the exact failure mode uncertainty-aware generation exists to
prevent. Measured over 20 RL steps: validity 0.48 → 0.67, reward on fresh
samples 0.747 → 0.791, diversity held at 0.91.

**An honest tension:** lower-bound ranking biases selection towards the training
distribution, because that is where the model is confident. The top-screened
generated candidates are consequently small, simple, often already-known
molecules. That is the reward working as designed and is a real trade-off
against novelty, not a bug.

### Fixed: generator could not be deep-copied

`MolecularGenerator` held an RDKit `FingerprintGenerator64`, which is a
Boost.Python object and cannot be pickled or deep-copied — the same class of
problem found in Stage 2. Caught by a test that deep-copies the prior to avoid
mutating a shared fixture. `__getstate__`/`__setstate__` now drop and rebuild it.

### Frontend

A new **Discovery Engine** tab with four panels: Probabilistic Pareto (a
probability-coloured scatter with σ̃ error bars, plus the residual-normality
warning), Risk Slider (live debounced λ with re-ranking), Budget Planner (assay
catalogue, budget slider, per-selection EHVI with standard errors), and
Generator (temperature control, the four generative metrics, and screened
candidates with conformal intervals). Every panel renders its own caveats from
the backend payload, and each has explicit loading, error-with-retry and
unavailable states.

### Cost

| Stage | Cold | Warm |
|---|---|---|
| Baseline + D-MPNN | ~95 s | ~1 s |
| + generator prior | ~170 s | ~1 s |

`MOLECULEMIND_ENABLE_GENERATOR=false` and `MOLECULEMIND_ENABLE_DMPNN=false` skip
either independently.

### Tests

204 → **309 tests**, all passing. New: `test_stage4.py` (81) plus 24 new API
tests.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, model version, D-MPNN and latent-AD availability |
| GET | `/api/models/dmpnn` | ensemble architecture, metrics, training report |
| GET | `/api/models/comparison` | baseline vs D-MPNN vs independent RF |
| POST | `/api/predict/dmpnn` | graph-network prediction with intervals and sets |
| POST | `/api/pareto/probabilistic` | P(Pareto optimal) per candidate, with MC standard error |
| POST | `/api/pareto/risk` | risk-slider ranking at a given λ |
| GET | `/api/planner/assays` | assay catalogue with costs (all simulated) |
| POST | `/api/planner/plan` | cost-aware EHVI experiment plan |
| GET | `/api/planner/validate-ehvi` | live cross-check against BoTorch `qEHVI` |
| GET | `/api/generator/status` | generator architecture and MOSES limitation |
| POST | `/api/generator/generate` | sample, validate and screen molecules |
| POST | `/api/generator/reinforce` | run an RL campaign |
| GET | `/api/config` | configuration and cache key for reproducibility |
| POST | `/api/analyze` | validation, 2D SVG, 3D conformer, descriptors |
| POST | `/api/predict` | properties with provenance, conformal intervals, AD |
| GET | `/api/objectives` | which objectives are available vs. unavailable |
| GET | `/api/benchmark` | metrics, both splits, AD derivation, calibration history |
| GET | `/api/benchmark/report` | downloadable JSON or CSV report |
| GET | `/api/applicability-domain/diagnostics` | how AD thresholds were derived |
| GET | `/api/pareto` | multi-objective front over validated objectives only |
| GET/POST | `/api/active-learning/pool` · `/rank` | ranked candidates, no hidden labels |
| POST | `/api/active-learning/assay` | simulated assay + conformal recalibration |
| POST | `/api/active-learning/reset` | restart the campaign |

---

## Not done yet

**Stages 3–6:** every module (a)–(k). Of these, the following are *unblocked*
and can be built with what is already installed: normalised conformal
classification sets (b), D-MPNN ensemble (a), probabilistic Pareto (d), risk
slider (e), EHVI planner (f, after a BoTorch 0.18 `qNEHVI` compatibility check),
SA score (h, first half), counterfactual editing (i), MLflow and audit trail (k).

**Blocked on external resources:** EGFR QSAR (ChEMBL data), RL generator
(MOSES corpus + compute), docking surrogate and Vina (DOCKSTRING + AutoDock
Vina), retrosynthesis (AiZynthFinder).

---

## How to run

```bash
pip install -r requirements.txt
python run.py                 # http://127.0.0.1:8000
```

```bash
python -m pytest -q           # 309 tests
```

Useful environment overrides (all prefixed `MOLECULEMIND_`):

```bash
# Fast start: skip the graph network entirely
MOLECULEMIND_ENABLE_DMPNN=false python run.py

# Smaller ensemble / fewer epochs
MOLECULEMIND_DMPNN_MEMBERS=2 MOLECULEMIND_DMPNN_EPOCHS=20 python run.py

# Force a full retrain
MOLECULEMIND_USE_MODEL_CACHE=false python run.py

# Skip the generator (saves ~75 s on a cold start)
MOLECULEMIND_ENABLE_GENERATOR=false python run.py
```

Frontend development with hot reload (backend must be running on :8000):

```bash
cd frontend && npm install && npm run dev
```
