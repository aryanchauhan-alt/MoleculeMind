# MoleculeMind — Test Report

**Run date:** 2026-11-22
**Command:** `python -m pytest -q`
**Environment:** macOS (darwin 27.0.0), Python 3.14.5, pytest 9.2.0, RDKit 2026.09.1,
scikit-learn 1.9.1, torch 2.15.0, BoTorch 0.18.1

## Result

```
309 passed in 124.18s
```

**309 executed · 309 passed · 0 failed · 0 skipped · 0 xfailed**

Verified from a **cold start** (`rm -rf backend/artifacts`) as well as warm, because
two Stage 3 defects appeared only on the cold path — see "Bugs the tests caught" below.

| File | Tests | Covers |
|---|---:|---|
| `backend/tests/test_mol_analyzer.py` | 10 | SMILES validation, descriptors, Lipinski/Veber, 2D SVG, 3D conformers |
| `backend/tests/test_dataset_and_models.py` | 19 | Dataset integrity, deduplication, scaffold split, ESOL and ClinTox models, provenance |
| `backend/tests/test_conformal_ad_pareto.py` | 18 | Conformal quantiles and coverage, applicability domain, Pareto dominance and sorting |
| `backend/tests/test_active_learning.py` | 18 | GP posterior, acquisition functions, oracle separation, assay safety |
| `backend/tests/test_persistence_and_config.py` | 24 | Settings overrides, model cache round-trip and failure modes, derived AD thresholds, conformal recalibration |
| `backend/tests/test_dmpnn.py` | 37 | Graph construction, directed message passing, heads, masked multi-task loss, training, ensemble variance |
| `backend/tests/test_stage3_integration.py` | 31 | Conformal prediction sets, latent-space AD, D-MPNN service, independent comparison |
| `backend/tests/test_stage4.py` | 81 | Probabilistic Pareto, risk slider, hypervolume, EHVI, planner, generator |
| `backend/tests/test_api_integration.py` | 45 | All twenty-five HTTP endpoints end to end against the real app |

The suite trains real models rather than mocking them — a reduced "quick"
configuration (50 trees instead of 150; a 2-member D-MPNN with hidden size 64 and
8 epochs instead of 5 members at 300/80) — which is why it takes ~73 s warm and
~170 s from cold.

---

## Regression guards for the critical audit findings

Three tests exist specifically to prevent the defects in `PROJECT_AUDIT.md` from
returning. Each was confirmed to **fail against the pre-fix code**.

| Audit item | Test | Asserts |
|---|---|---|
| §10.2 degenerate acquisition | `test_acquisition_is_discriminative` (×4 strategies) | `degenerate is False` and at least half the pool receives distinct scores. Pre-fix: 1 unique score across 24 candidates |
| §10.3 oracle leak | `test_al_pool_never_ships_hidden_labels` | Serialises the entire `/api/active-learning/pool` payload and scans it for every oracle key |
| §10.3 oracle leak | `test_measurement_appears_only_after_an_assay` | A candidate's `measured_value` is `None` before its assay and matches the private oracle after |
| §10.4 fabricated properties | `test_unvalidated_properties_return_no_number` | `bioactivity_pic50` and `metabolic_stability` carry no `value` key |
| §10.5 silent zero imputation | `test_missing_objective_raises_instead_of_imputing_zero` | `ValueError` rather than a 0.0 default |
| §10.6 split leakage | `test_partitions_do_not_share_structures` | Train / test / AL-pool canonical SMILES sets are pairwise disjoint |
| §10.6 duplicates | `test_no_duplicate_canonical_smiles_survive` | 11 duplicates removed; all remaining structures unique |
| §10.6 unscaled AD features | `test_ad_features_are_standardised` | PC1 carries < 50% of variance. Pre-fix it carried 90.9%, making leverage a molecular-weight test |
| §10.8 stale calibration | `test_assay_triggers_recalibration` | An AL round advances the calibration generation instead of reusing the old quantile |
| §10.1 hardcoded metrics | `test_benchmark_report_json_download` | The report's honesty notes state nothing is hardcoded, and every metric is present and computed |

---

## Verified scientific properties

**Conformal coverage on synthetic exchangeable data.** With 2000 calibration and
4000 test points drawn from the same distribution, empirical coverage lands
within 3 percentage points of nominal at the 80%, 90% and 95% levels — the
property split conformal is supposed to deliver.

**Finite-sample quantile.** `test_quantile_uses_finite_sample_correction` checks
the returned quantile equals the `⌈(n+1)(1−α)⌉`-th order statistic exactly, for
three α values, rather than a plain empirical percentile.

**Normalised vs. standard conformal.** With a constant σ the two formulations
produce identical interval widths, as they must.

**Scaffold split integrity.** No Bemis-Murcko scaffold group is split across
partitions; the split is deterministic across runs.

**Pareto correctness.** A five-case dominance truth table (including the
equal-vectors and incomparable cases), a hand-computed three-point frontier, and
direction handling for minimised objectives.

---

## Measured model metrics (from this run)

| Metric | Random split | Scaffold split |
|---|---|---|
| ESOL R² | 0.903 | 0.890 |
| ESOL RMSE | 0.666 | 0.746 |
| ESOL MAE | 0.484 | — |
| Test compounds | 138 | 167 |

| ClinTox | Value |
|---|---|
| ROC-AUC | 0.832 |
| PR-AUC | 0.358 |
| Baseline positive rate | 0.072 |
| Brier score | 0.0565 |

Conformal empirical coverage on the ESOL test partition: 80% → 78.2%,
90% → 88.8%, 95% → 94.7% (1–2 points under nominal, the expected finite-sample
behaviour at n_cal ≈ 168).

---

## Stage 2 P1 verification

**Model persistence.** `test_save_and_load_round_trip` trains a model, reloads
it from cache in a second service instance, and asserts the predictions,
applicability-domain verdict and metrics are identical. Failure modes are
tested too: a corrupt cache file falls back to training rather than raising, a
cache write failure leaves no `.tmp` artifact behind, and caching can be turned
off.

Measured cold vs. warm start (full-size model, this machine):

| Start | Time | Source |
|---|---|---|
| Cold | 3.8 s | trained from scratch |
| Warm | ~0.0 s | loaded from cache |

The cache key hashes the dataset *file bytes* rather than the parsed dataset, so
a cache hit skips RDKit featurisation of ~2,600 molecules entirely. The key also
covers the model version, cache format and every training-relevant setting, so a
stale model can never be served against changed data or changed configuration.

**Training no longer happens at import.** `test_importing_the_app_does_not_train`
confirms services are assigned by the FastAPI lifespan hook. `import
backend.app.main` now returns immediately with `model_service is None`.

**Applicability-domain thresholds are derived, not asserted.** Measured on the
ESOL training partition: similarity threshold 0.250 (5th percentile of
leave-one-out nearest-neighbour similarity), leverage threshold 0.0583 (95th
percentile of training leverage; the OECD rule-of-thumb h\* = 0.0423 is reported
alongside for reference). Under the two-condition rule, 0.5% of training
molecules are flagged out of domain — the engine does not reject the data it was
fitted on.

**Conformal recalibration.** An assay now extends the calibration set and
advances a calibration generation counter, with the full history exposed on
`/api/benchmark`. Both the attached and unattached cases are tested, and the
response states plainly that the base model was not retrained and that actively
selected labels break strict exchangeability.


---

## Stage 3 verification

### The message-passing mathematics

A subtly wrong graph network still trains and still produces plausible numbers,
so the directed message is checked against its definition rather than by
inspecting a loss curve.

`test_directed_message_excludes_reverse_edge` recomputes

    m_e = Σ_{k ∈ N(v) \ {w}} h_{(k→v)}    for e = (v→w)

by explicit enumeration over every edge of a ring-plus-branch molecule and
asserts agreement with the vectorised `incoming[src] − h[rev]` form to 1e-5.
`test_message_is_not_merely_the_incoming_sum` asserts the reverse-edge
subtraction measurably changes the result, so a regression that drops it fails
instead of silently degrading the model.

Also verified: the reverse-edge map is an involution that swaps source and
destination and never maps an edge to itself; it survives batching with index
offsets; embeddings are invariant to atom renumbering and to batch composition;
molecules with no bonds produce finite output; an element outside the feature
vocabulary (platinum) is encoded rather than crashing.

### The losses

| Test | Asserts |
|---|---|
| `test_gaussian_nll_matches_closed_form` | NLL equals `0.5[log σ² + (y−μ)²/σ²]` to 1e-6 |
| `test_gaussian_nll_is_minimised_at_the_true_variance` | NLL is minimal at `σ² = (y−μ)²`, so the head learns a real error scale |
| `test_masked_loss_ignores_missing_labels` | A masked entry set to 1e6 does not change the loss |
| `test_masked_bce_ignores_missing_labels` | Masked BCE equals `log 2` for a single logit-0 / target-1 pair |
| `test_fully_masked_loss_is_zero` | An all-masked batch contributes nothing |

### Ensemble uncertainty

`test_variance_decomposition_is_exact` recomputes `mean(σ²ₘ)` and `var(μₘ)` from
the individual members and asserts `Var_total` equals their sum to 1e-8, with
`std = √Var`. Epistemic variance is asserted non-negative and aleatoric strictly
positive. Members are confirmed independently seeded and to produce genuinely
different predictions.

### Conformal prediction sets

Measured on a 3-class synthetic problem, 4,000 calibration / 6,000 test:

| Method | 80% | 90% | 95% | mean set @90% |
|---|---|---|---|---|
| LAC | 79.9% | 89.8% | 94.3% | 1.42 |
| APS deterministic | 99.2% | 99.8% | 100.0% | 2.69 |
| APS randomised | 80.0% | 89.9% | 94.8% | 1.57 |

`test_deterministic_aps_is_conservative_not_wrong` pins the over-coverage
explicitly, so the conservative variant can never be mistaken for the exact one.
Also tested: the finite-sample quantile, the conservative fallback when `n` is
too small for the level (`ceil((n+1)(1−α)) > n` → threshold 1.0), monotone set
size in confidence, 1-D binary probability input, and reproducibility of
randomised sets for a fixed `random_state`.

### Latent-space applicability domain

Tested: row-count mismatch raises rather than silently misaligning; a training
point has distance ~0 from itself; a far point exceeds the threshold; a dimension
mismatch raises; and the AD rule genuinely **switches its second condition** to
latent distance when an embedding is supplied, falling back to leverage when it
is not.

### D-MPNN service

Tested: the multi-task join produces NaN (never imputed) for absent labels; both
tasks are evaluated; variance is decomposed in the served metrics; predictions
carry provenance and a conformal interval bracketing the point estimate; invalid
SMILES raise; the independent comparison runs and uses a genuinely different
representation; `summary()` reports `not_trained` with a reason and **no metrics
key** when untrained; a partly unparseable embedding request returns `None`
rather than a silently shorter matrix; checkpoint round-trip reproduces
predictions to 1e-6; and a missing checkpoint returns `False` rather than raising.


---

## Stage 4 verification

### EHVI is verified to be EHVI

A "weighted utility plus uncertainty" heuristic can be dressed up to look like
EHVI, so the estimator is checked against two independent ground truths.

**1. Single-objective reduction to Expected Improvement.** In one dimension EHVI
reduces analytically to EI. `test_single_objective_ehvi_equals_closed_form_ei`
checks four parameter settings against the closed form
`(μ−best)·Φ(z) + σ·φ(z)`:

| μ | σ | best | MC EHVI | closed-form EI | difference |
|---|---|---|---|---|---|
| 1.0 | 0.5 | 0.8 | 0.31406 | 0.31522 | 0.0012 |
| 0.5 | 1.0 | 1.2 | 0.14167 | 0.14288 | 0.0012 |
| 2.0 | 0.3 | 1.0 | 0.99930 | 1.00003 | 0.0007 |
| 0.0 | 1.0 | 0.0 | 0.39680 | 0.39894 | 0.0021 |

All within ~2 Monte Carlo standard errors. **A `μ + kσ` score cannot reproduce
this curve**, so the test separates a real estimator from a heuristic.

**2. Cross-check against BoTorch.** `test_ehvi_matches_botorch_qehvi` fits a GP
so both implementations integrate over the *same* posterior:

```
botorch : [0.0, 0.0, 0.177936, 0.149688, 0.000156]
ours    : [0.0, 0.0, 0.178362, 0.149869, 0.000124]
max relative difference: 0.0024
```

Also asserted: a deterministic outcome gives exactly the analytic HVI; a
dominated certain point gives exactly 0; EHVI is non-negative and rises with a
better mean; hypervolume is correct for maximised and minimised directions,
monotone under adding points, and unchanged by adding a dominated point.

### Planner behaviour

| Test | Asserts |
|---|---|
| `test_plan_never_exceeds_the_budget` | Four budgets; selected costs sum to the reported total and never exceed B |
| `test_no_molecule_assay_pair_is_repeated` | No pair selected twice, even with repeats allowed |
| `test_expected_gain_is_non_decreasing_in_budget` | More budget never yields less expected gain |
| `test_planner_stops_when_no_improvement_remains` | A 1000-unit budget stops early rather than spending pointlessly |
| `test_cheaper_assays_change_the_selected_mix` | Cost genuinely drives selection, not just ranking |
| `test_plan_reports_reference_point_and_limitations` | Reference point and the greedy caveat are in the payload |

### Probabilistic Pareto

Asserted: a certain dominant candidate has probability exactly 1.0; wide
intervals give a dominated candidate a real chance (>0.2) where tight ones do
not (<0.05); zero σ reproduces the deterministic front exactly; probabilities
stay in [0,1]; front members score above non-members on average; correlated
sampling changes the answer; the pre-filter removes only genuinely hopeless
candidates and spares any whose optimistic corner is competitive; and the
approximations list names the Gaussian assumption and the pool-relativity.

`test_normality_report_detects_heavy_tails` confirms the diagnostic distinguishes
Gaussian from t(3) residuals and marks the latter inadequate.

### Risk slider

Asserted: λ=0 returns the means unchanged; a maximised objective gains
`+κσ̃`; a **minimised objective gains `−κσ̃`** (the sign flip — getting this
backwards would invert the slider's purpose); κ scales linearly with λ; a
high-variance candidate climbs to rank 1 as λ rises; λ outside [0,1] raises; and
the payload states it is not a calibrated prediction.

### Generator

Asserted: the tokeniser handles `Cl`/`Br`/bracket atoms and **returns an empty
list rather than a corrupted one** for characters it cannot match; vocabulary
round-trips through serialisation; every molecule reported valid actually parses
*and is canonical*, so deduplication is meaningful; `n_valid + n_invalid` equals
the number requested, so nothing is silently dropped; every rejection carries a
reason; metric denominators are documented in the payload; lower temperature
does not reduce validity; novel molecules are genuinely absent from the training
corpus; the diversity filter pushes observed mean reward below a constant reward
of 1.0; RL raises mean reward on a trivial substructure objective; a malformed
reward function raises; and the checkpoint round-trips to identical samples.

`test_prior_trains_and_reports_its_data_limitation` asserts the MOSES shortfall
is declared in the payload with `moses_pretraining: "pending"`.

---

## Bugs the tests caught

Four defects were found by the tests themselves during Stage 3, not by
inspection:

1. **Ensemble members were not reproducible for a fixed seed.**
   `torch.manual_seed` ran inside `fit()`, but weight initialisation happens in
   the constructor, so initialisation drew from ambient global RNG state. Caught
   by `test_training_is_reproducible_for_a_fixed_seed`.
2. **`numpy.float32` reached the JSON encoder.** Freshly-trained metrics embedded
   the raw numpy label matrix and numpy scalars, which FastAPI cannot serialise.
   It surfaced as an opaque 500 **only on a cold start**, because the warm path
   had round-tripped through JSON and cleaned the types — which is why the suite
   is now run from cold as well as warm.
3. **Warm starts served a "trained" model with no metrics.** The checkpoint held
   weights but not the evaluation record. Caught by
   `test_dmpnn_status_endpoint_is_honest`.
4. **The classification head had no held-out evaluation.** Pinning the test
   partition to the baseline's ESOL compounds left almost no CT_TOX labels to
   score. Caught by the same test, via a missing `toxicity` metrics key.

A fifth, in Stage 4: **`MolecularGenerator` could not be deep-copied**, because
it held an RDKit `FingerprintGenerator64` (a Boost.Python object). Caught by a
test that deep-copies the prior to avoid mutating a shared module-scoped
fixture — the same class of defect as the Stage 2 pickling problem, in a new
place.

A sixth was caught by a strict oracle-leak scan in Stage 2 and is worth noting
here because it recurred in spirit: the campaign payload contained the *string*
`measured_logs` as an objective key name, which tripped the substring scan. The
key was renamed rather than the test weakened.

---

## Known gaps in test coverage

These are stated rather than papered over:

1. **No frontend tests.** The React components have no unit or component tests.
   TypeScript compilation (`tsc -b`) and the production build both pass clean,
   and the five tabs were exercised manually against a live server, but there is
   no automated browser-level test.
2. **No cross-validation or seed-sweep.** Every metric comes from a single seed
   (42). The reported R² has no confidence interval.
3. **Leverage is a weak diagnostic on this representation.** Now that features
   are standardised, 10 principal components capture only ~11% of the variance
   of a 1032-dimensional mostly-sparse fingerprint space. The leverage test is
   therefore far less informative than the similarity test, which is part of why
   the two-condition rule flags so little. A learned latent representation
   (Stage 3) is the right fix; this is documented rather than papered over.
4. **Modules (g)–(j) have no tests** because they have no implementations
   (Stage 5).
5. **The D-MPNN is tested at a reduced configuration.** The suite trains 2
   members at hidden size 64 for 8 epochs, not 5 at 300 for 80. The full
   configuration's numbers are reproduced in
   [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) §9 but are not asserted
   by a test, because a ~95 s training run per invocation does not belong in a
   unit suite.
6. **ClinTox conformal set coverage is measured on 218 test molecules with 15
   positives.** Those coverage figures are noisy; the tight coverage assertions
   are made against synthetic data where the sample size supports them.
7. **The generator is tested at a tiny configuration.** The suite trains a
   64-unit single-layer GRU on a 112-SMILES corpus for 25 epochs. Validity from
   that model is not meaningful, so the tests assert *accounting* properties
   (every rejection explained, every reported-valid molecule actually parses)
   rather than performance thresholds. The real prior's measured validity is in
   [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) §7b.
8. **The RL campaign is tested for mechanism, not for chemistry.** Reward rises
   on a trivial "contains nitrogen" objective; no test asserts that an RL
   campaign produces better *molecules*, because that would require experimental
   validation the project does not have.
5. **AD thresholds are tested for behaviour, not for correctness.** The tests
   confirm an identical molecule is in-domain and an exotic one is flagged; they
   do not validate that 0.55/0.35 are the right cut-offs, because that validation
   has not been done.
6. **Conformal coverage on real data is tested with a ±10 point tolerance**,
   which is loose. It is a smoke test against gross miscalibration, not a tight
   guarantee.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest -q              # full suite
python -m pytest -q backend/tests/test_active_learning.py -v   # the AL guards
```
