# MoleculeMind — Scientific Methodology

Every algorithm implemented in this platform, with its mathematics, its
assumptions, and what it does **not** claim. Numbers quoted are measured on this
machine with `MOLECULEMIND_SEED=42`; they are reproducible via
`python -m pytest -q` and `GET /api/benchmark/report`.

---

## 1. Molecular representation

### 1.1 Fingerprint + descriptor features (baseline models)

A 1024-bit Morgan fingerprint (ECFP4, radius 2) concatenated with 8
physicochemical descriptors — molecular weight, MolLogP, TPSA, H-bond donors,
H-bond acceptors, rotatable bonds, ring count, heavy-atom count. Dimension 1032.

**Critical caveat.** These features are returned *unscaled*. Any scale-sensitive
consumer must standardise them first. On the raw matrix, molecular weight alone
accounts for **90.9%** of the variance, which silently reduced both the PCA
leverage test and the active-learning Gaussian process to functions of molecular
weight. Both defects are documented in `PROJECT_AUDIT.md` §10.2 and §10.6 and are
now fixed with `StandardScaler`.

### 1.2 Molecular graph (D-MPNN)

Atoms carry one-hot encodings of atomic number, total degree, formal charge,
total H count, hybridisation and chiral tag, plus aromaticity, ring membership
and scaled mass — 48 features. Bonds carry bond type, stereo configuration,
conjugation and ring membership — 13 features. Every one-hot has a trailing
"other" slot, so an element outside the vocabulary (platinum, say) is encoded
rather than crashing the featuriser.

---

## 2. Directed Message Passing Neural Network

### 2.1 Why *directed*

Each chemical bond becomes two directed edges. For a directed edge
`e = (v → w)`:

**Edge initialisation**

$$h_e^0 = \mathrm{ReLU}\!\left(W_i \left[x_v \; ; \; f_{vw}\right]\right)$$

**Message** — the sum of messages arriving at `v`, *excluding* the reverse edge:

$$m_e^t = \sum_{k \in N(v) \setminus \{w\}} h_{(k \to v)}^t = \left(\sum_{k \in N(v)} h_{(k \to v)}^t\right) - h_{(w \to v)}^t$$

The subtraction is the entire point of the directed formulation. Without it a
message flows straight back along the bond it arrived on, and the network
amplifies 2-step walks instead of exploring the graph.

It is implemented as `incoming[edge_src] - h[edge_rev]`, which is the closed form
on the right. Because a subtly wrong graph network still trains and still
produces plausible numbers, this is **verified numerically rather than by
inspection**: `test_directed_message_excludes_reverse_edge` computes the message
by explicit enumeration over all edges and asserts agreement to 1e-5, and
`test_message_is_not_merely_the_incoming_sum` asserts the subtraction actually
changes the result, so a regression that drops it fails loudly.

**Edge update**, with a residual onto the initial embedding:

$$h_e^{t+1} = \mathrm{ReLU}\!\left(h_e^0 + W_h m_e^t\right)$$

**Atom readout**

$$m_v = \sum_{k \in N(v)} h_{(k \to v)}^T, \qquad h_v = \mathrm{ReLU}\!\left(W_o \left[x_v \; ; \; m_v\right]\right)$$

**Molecule embedding** — mean over the molecule's atoms.

Additional verified properties: permutation invariance (atom renumbering leaves
the embedding unchanged), batching invariance (a batched embedding equals the
individual ones), and finite output for molecules with no bonds (methane, lone
ions).

### 2.2 Configuration

3 message-passing steps, hidden size 300, mean readout, two-layer feed-forward
head with ReLU and dropout 0.1, Adam from `lr = 1e-3` with 2 epochs of linear
warm-up then exponential decay to `0.1 × lr`, batch size 64, gradient clipping at
norm 5, early stopping with patience 12 on a held-out inner validation split.
Depth is configurable from 2 to 5 and is covered by a parametrised test.

### 2.3 Multi-task learning with a masked loss

ESOL (regression, logS) and ClinTox (classification, CT_TOX) are different
datasets. They are joined on canonical SMILES into one table with **NaN where a
label is absent**:

| | count |
|---|---|
| Molecules in the union | 2,514 |
| With a logS label | 1,117 |
| With a CT_TOX label | 1,461 |
| With **both** labels | 64 |

Only 64 molecules carry both labels, so the benefit here comes almost entirely
through the **shared encoder**: a molecule with only a solubility label still
contributes gradient to the representation the classification head reads from.
Measured effect on ClinTox: ROC-AUC **0.832 → 0.858** versus the single-task
Random Forest.

Regression loss (Gaussian NLL, up to an additive constant):

$$\mathcal{L} = \frac{1}{|M|}\sum_{i \in M} \tfrac{1}{2}\left[\log \sigma_i^2 + \frac{(y_i - \mu_i)^2}{\sigma_i^2}\right]$$

Classification loss: binary cross-entropy with logits, masked identically.
Missing labels contribute exactly nothing — verified by
`test_masked_loss_ignores_missing_labels`, which sets a masked entry to 1e6 and
asserts the loss is unchanged. `test_gaussian_nll_is_minimised_at_the_true_variance`
confirms the NLL is minimised at `σ² = (y − μ)²`, i.e. the head is genuinely
learning an error scale rather than an arbitrary positive number.

Regression targets are standardised per task internally and mapped back to
original units at prediction time, with variance rescaled by `std²` since
`Var(aX) = a² Var(X)`.

### 2.4 Ensemble uncertainty decomposition

Five independently seeded members. Total predictive variance:

$$\mathrm{Var}_{\text{total}} = \underbrace{\frac{1}{M}\sum_m \sigma_m^2}_{\text{aleatoric}} + \underbrace{\frac{1}{M}\sum_m (\mu_m - \bar{\mu})^2}_{\text{epistemic}}$$

(Lakshminarayanan et al. 2017). The first term is the mean noise each member
predicts; the second is disagreement between members.
`test_variance_decomposition_is_exact` recomputes both terms from the individual
members and asserts equality to 1e-6.

Measured on the ESOL test partition: aleatoric ≈ 0.397, epistemic ≈ 0.034. The
predicted σ correlates with absolute test error at **r ≈ 0.33–0.43**, so the
model has real, if modest, knowledge of where it is weak.

**A seeding bug worth recording.** Ensemble members were initially *not*
reproducible for a fixed seed: `torch.manual_seed` was called inside `fit()`, but
weight initialisation happens in the constructor, so initialisation drew from
ambient global RNG state. Caught by
`test_training_is_reproducible_for_a_fixed_seed`; the seed is now set before the
model is constructed.

---

## 3. Conformal prediction

### 3.1 Normalised split conformal for regression

Nonconformity score, prediction interval, and the finite-sample quantile:

$$s_i = \frac{|y_i - \mu_i|}{\sigma_i}, \qquad C(x) = \mu(x) \pm \hat{q}\,\sigma(x), \qquad \hat{q} = s_{(\lceil (n+1)(1-\alpha)\rceil)}$$

The `⌈(n+1)(1−α)⌉`-th order statistic — not a plain empirical percentile — is
what makes the guarantee hold at finite `n`. Asserted exactly by
`test_quantile_uses_finite_sample_correction` for three α values.

Two sources of σ are used, and the difference matters:

* **Baseline models** use the inter-tree spread of the ExtraTrees ensemble. This
  is a heuristic dispersion measure, *not* a calibrated predictive standard
  deviation. It works because the conformal layer supplies the actual coverage
  guarantee; on its own it should not be read as a standard deviation.
* **D-MPNN** uses σ from a head trained by Gaussian NLL to predict its own error
  scale, plus the epistemic component. This is the methodology's intended form.

Validation on synthetic exchangeable data (2,000 calibration / 4,000 test):
empirical coverage lands within 3 percentage points of nominal at 80%, 90% and
95%. With a constant σ the normalised and standard formulations give identical
widths, as they must.

### 3.2 Prediction sets for classification

Three methods, all split-conformal with the same finite-sample quantile.

**LAC** (Sadinle et al. 2019) — score `s = 1 − p̂(y|x)`; set is
`{y : 1 − p̂(y|x) ≤ q̂}`. Smallest average sets, but coverage is uneven across
inputs: easy inputs are over-covered and hard ones under-covered.

**APS** (Romano et al. 2020) — accumulate sorted probability mass down to the
true label. Two variants, and the distinction is *not* cosmetic:

$$\text{score}(x, y, u) = \sum_{y' : p(y') > p(y)} p(y') + u \cdot p(y), \qquad u \sim \mathcal{U}(0,1)$$

* **Deterministic** (`u` omitted, crossing label included in full) is
  **conservative — it over-covers**. Measured on a 3-class synthetic problem, the
  nominal-80% set achieved **99.2% coverage with mean size 2.36 of 3** — valid
  but nearly useless.
* **Randomised** is the published form and hits the nominal level.

Measured coverage (3-class synthetic, 4,000 calibration / 6,000 test):

| Method | 80% | 90% | 95% | mean set size @90% |
|---|---|---|---|---|
| LAC | 79.9% | 89.8% | 94.3% | 1.42 |
| APS deterministic | 99.2% | 99.8% | 100.0% | 2.69 |
| APS randomised | 80.0% | 89.9% | 94.8% | 1.57 |

Randomised APS gives slightly larger sets than LAC in exchange for more even
coverage across inputs — the usual trade-off, and the reason both are offered.
Because its sets are themselves random, `random_state` is exposed for
reproducibility.

### 3.3 What conformal coverage does **not** mean

A 90% interval or set provides **marginal** coverage: averaged over the test
distribution, 90% of intervals contain the truth. It does **not** mean this
particular molecule's value lies in its interval with probability 0.90. No method
here calibrates per-input coverage, and none repairs distribution shift — which
is precisely why the applicability domain is an independent diagnostic rather
than folded into the interval.

### 3.4 Recalibration after active learning

Once an assay reveals labels, the calibration set is no longer the one the stale
quantile came from. `recalibrate_conformal()` folds new labels into the residuals
and recomputes the quantiles, tracked by a generation counter exposed on
`/api/benchmark`.

Two caveats are returned in the payload rather than left implicit: the base model
is **not** retrained, so this corrects the calibration set and not the predictor;
and labels chosen by an acquisition function are not an i.i.d. sample of the
candidate distribution, so strict exchangeability does not hold and
post-recalibration coverage is approximate.

---

## 4. Applicability domain

Two **independent** diagnostics, with thresholds derived from the training set
rather than asserted.

### 4.1 Structural similarity

Maximum ECFP4 Tanimoto similarity to the training set. Threshold = the 5th
percentile of each training molecule's *leave-one-out* nearest-neighbour
similarity, so ~95% of training molecules sit at or above it. Measured on ESOL:
**0.250**.

### 4.2 Second condition

**With a learned latent space** (the methodology's choice): Euclidean distance
from the query's D-MPNN embedding to its nearest training embedding. Threshold =
95th percentile of training leave-one-out latent distances. Measured: **0.356**
in a 300-dimensional space.

**Without one** (fallback): leverage in a PCA space fitted on *standardised*
features,

$$h = z^\top (Z^\top Z)^{-1} z$$

with threshold = 95th percentile of the training leverage distribution (measured
0.0583). The OECD rule of thumb `h* = 3(p+1)/n = 0.0423` is computed and reported
for reference but is **not** what the classification uses.

**Honest limitation.** With standardised features, 10 principal components
capture only ~11% of the variance of a mostly-sparse 1032-dimensional
fingerprint space, so leverage is a much weaker signal than similarity. This is
exactly why the learned latent space is preferred when available.

### 4.3 Two-condition rule

A candidate is **OUT_OF_DOMAIN** only when **both** diagnostics fail,
**BORDERLINE** when exactly one does, **IN_DOMAIN** when neither. Failing one of
two independent tests is weaker evidence than failing both, and collapsing them
would discard that distinction. On the ESOL training set, 0.5% of molecules are
flagged — the engine does not reject the data it was fitted on.

**A case that shows why two diagnostics are needed.** A C36 straight-chain
alkane returns Tanimoto similarity **1.000** against the ESOL training set:
Morgan fingerprints of long alkanes are bit-identical to those of short ones,
because every atom environment is the same CH2. The similarity test therefore
passes completely. Its latent distance is 0.481 against a 0.356 threshold, so the
learned representation catches what the fingerprint cannot, and the molecule is
correctly flagged BORDERLINE.

### 4.4 `domain_index` is not a probability

A bounded composite, `0.6 × similarity + 0.4 × exp(−excess/threshold)`, intended
for ranking only. It is **not** a calibrated probability that the prediction is
correct, and the API says so in an `index_caveat` field. The earlier name
`reliability_score` invited exactly that misreading.

---

## 5. Multi-objective optimisation

Pareto dominance over objectives normalised to a common direction: `a` dominates
`b` iff `a` is at least as good on every objective and strictly better on at
least one. Non-dominated sorting assigns ranks; rank 1 is the Pareto front.
Complexity is O(n²) in candidates, which is fine at the scale here and will not
scale past a few thousand.

**Missing values raise an error; they are never imputed.** Substituting 0.0 (the
original behaviour) is actively dangerous: for a *minimised* objective such as
toxicity probability, 0.0 is the best possible value, so a candidate missing that
property would be promoted onto the Pareto front on the strength of data it does
not have.

**Only validated objectives are optimised.** Solubility (model), QED
(deterministic RDKit calculation) and ClinTox toxicity (model). Target
bioactivity (pIC50) and metabolic stability are reported as `unavailable` with a
reason, carry no numeric value, and are excluded — see §7.

---

## 6. Active learning (simulated laboratory mode)

A Gaussian process surrogate (BoTorch `SingleTaskGP`) over standardised PCA
features mapped to the unit cube, with `Normalize` input and `Standardize`
outcome transforms. Acquisition functions: UCB (`μ + √β·σ`),
LogExpectedImprovement, pure uncertainty sampling, and greedy exploitation.

**The bug this replaced.** The GP previously received PCA components from the
*unscaled* feature matrix, spanning [−113, +178] with molecular weight carrying
90.9% of the variance. BoTorch's default lengthscale prior assumes inputs in
roughly `[0,1]^d`, so every train/test pair was effectively uncorrelated, the
posterior collapsed to the prior, and **all 24 candidates received an identical
acquisition score** (UCB = 0 + √2·1 = 1.414). The tab was ranking by list order.
After the fix, 17 of 24 candidates receive distinct scores and the ranking
responds to new labels. `test_acquisition_is_discriminative` guards this.

**Oracle separation.** Four separate stores: a private oracle dict, a revealed-id
set, the public candidate records, and the model's own test set.
`public_pool()` is the only serialisation path and raises if an oracle key would
escape. Two tests scan the entire serialised API payload for oracle keys.

**Honest labelling.** The "assay" reveals a withheld Delaney ESOL measurement. It
is a retrospective simulation, declared as `mode: "simulated"` with an explicit
note that no laboratory experiment was performed. The candidate pool is 30 real
held-out ESOL compounds, so the oracle is a genuine experimental measurement with
traceable provenance — replacing an earlier library of invented pIC50 and
toxicity values attributed to named real drugs.


---

## 5b. Probabilistic Pareto optimisation

Deterministic Pareto sorting treats each predicted value as a fact. A candidate
just inside the front with tight intervals and one just outside with wide
intervals are being separated by noise, and a deterministic front cannot say
which is which.

### Converting a conformal interval into a Gaussian scale

Conformal prediction gives a calibrated *interval*, not a distribution. To
sample, a scale is needed:

$$\tilde{\sigma} = \frac{\hat{q}\,\sigma}{z}, \qquad z = \Phi^{-1}(1 - \alpha/2)$$

This picks the Gaussian whose central $(1-\alpha)$ interval has the **same
width** as the conformal interval. At $\alpha = 0.1$, $z = 1.645$.

**This approximation has a direction, and it matters.** Conformal intervals are
distribution-free: the width is honest, the *shape* inside it is not specified.
Imposing a Gaussian shape by width matching asserts something the conformal
procedure never claimed. Where residuals are heavier-tailed than Gaussian —
common in molecular property prediction, where a few compounds are badly
mispredicted — this **understates** the chance of an extreme outcome, so Pareto
probabilities are **optimistic** for candidates whose membership hinges on a
tail event.

Rather than leave that unexamined, `residual_normality_report()` measures it.
**Measured on this platform's ESOL calibration residuals: excess kurtosis
1.416, and the Kolmogorov–Smirnov test rejects normality.** The frontend
displays this as an amber warning whenever it fires, which on the bundled data
is always. The honest summary is: the interval widths are calibrated, the
Gaussian shape inside them is not, and the resulting probabilities are mildly
optimistic in the tails.

### Independence across objectives

Marginal conformal intervals, one per objective, do **not** define a jointly
calibrated multivariate distribution. Sampling each objective independently
assumes zero residual correlation, which is generally false: predictions for the
same molecule share a representation and share training data.

Correlated sampling is implemented (via a Cholesky factor) but the API
**refuses** it with HTTP 501 rather than inventing a correlation matrix: only 64
molecules in this platform carry both a solubility and a toxicity label, which
is too few to estimate the matrix honestly. Supplying a fabricated correlation
would make the output look more rigorous while being less trustworthy.

### What the probability means

$P(\text{Pareto optimal})$ is the fraction of sampled joint outcomes in which a
candidate is non-dominated **by the other candidates in this pool, under this
model**. It is not a probability that the molecule is a good drug, nor that it
would survive against a different candidate set. Adding or removing candidates
changes every value. A Monte Carlo standard error is returned so a caller can
tell a real difference from sampling noise.

### Pre-filter

A candidate whose *optimistic* corner $\mu + k\tilde{\sigma}$ is still dominated
by another's *pessimistic* corner $\mu - k\tilde{\sigma}$ cannot plausibly reach
the front, and is skipped. At $k = 4$ the discarded probability mass is $\approx
3\times10^{-5}$ per objective per candidate — a bounded, stated error, not a
silent approximation.

---

## 5c. Uncertainty-aware risk slider

$$\kappa = \kappa_{\max}\lambda, \qquad f = \mu + \kappa\tilde{\sigma} \;\;(\text{maximised}), \qquad f = \mu - \kappa\tilde{\sigma} \;\;(\text{minimised})$$

The **sign flip for minimised objectives is essential**: being optimistic about
toxicity means assuming a *lower* value. Getting it backwards would make the
slider penalise uncertainty on exactly the objectives it should reward it on.
`test_minimised_objective_gets_an_optimistic_decrease` pins this.

$\lambda = 0$ is pure exploitation (rank by the mean); $\lambda = 1$ gives
uncertainty maximum weight. This is an **optimistic ranking device, not a
prediction**: $\mu + 2\sigma$ carries no coverage guarantee and promises no
improvement. It says "rank this candidate as if it came out well", which is how
exploration is bought.

---

## 6b. Expected Hypervolume Improvement planner

### The quantity

The hypervolume $HV(P)$ of a front $P$ relative to a reference point $r$ is the
volume of objective space dominated by $P$ and bounded by $r$. Improvement from
an outcome $y$ and its expectation are

$$HVI(y) = HV(P \cup \{y\}) - HV(P) \geq 0, \qquad EHVI(x) = \mathbb{E}_{y \sim p(y|x)}\left[HVI(y)\right]$$

Each $HV$ term is computed **exactly** by BoTorch's box decomposition; only the
outer expectation is Monte Carlo, with a reported standard error.

### Why Monte Carlo rather than BoTorch's `qEHVI`

BoTorch's `qEHVI`/`qNEHVI` are available here (0.18.1) and are excellent, but
they integrate over a **BoTorch model's GP posterior**. This platform's
predictive distributions come from a D-MPNN ensemble whose scale has been
calibrated by conformal prediction. Routing them through `qEHVI` would require
fitting a GP surrogate to our own predictions and integrating over *that*
posterior — discarding the calibration which is the whole point.

### Verified to be EHVI, not an EHVI-shaped heuristic

This is the claim the brief is most insistent about, so it is checked against
two independent ground truths:

**1. Single-objective reduction to Expected Improvement.** With one objective,
EHVI reduces analytically to EI, $(\mu - best)\Phi(z) + \sigma\phi(z)$. Measured:

| μ | σ | best | MC EHVI | closed-form EI | difference |
|---|---|---|---|---|---|
| 1.0 | 0.5 | 0.8 | 0.31406 | 0.31522 | 0.0012 |
| 0.5 | 1.0 | 1.2 | 0.14167 | 0.14288 | 0.0012 |
| 2.0 | 0.3 | 1.0 | 0.99930 | 1.00003 | 0.0007 |
| 0.0 | 1.0 | 0.0 | 0.39680 | 0.39894 | 0.0021 |

All within ~2 Monte Carlo standard errors. A weighted $\mu + k\sigma$ score
cannot reproduce the EI curve, so this test alone separates a real estimator
from a heuristic.

**2. Cross-check against BoTorch's `qExpectedHypervolumeImprovement`.** Both
integrate over the *same* fitted GP posterior:

```
botorch : [0.0, 0.0, 0.177936, 0.149688, 0.000156]
ours    : [0.0, 0.0, 0.178362, 0.149869, 0.000124]
max relative difference: 0.0024
```

**0.24%** — residual disagreement is Monte Carlo noise plus BoTorch's QMC
sampling, not a difference in the quantity computed. Exposed live at
`GET /api/planner/validate-ehvi`.

Additional checks: a deterministic outcome gives exactly the analytic HVI; a
dominated certain point gives exactly 0; EHVI is non-negative; hypervolume is
correct for maximised and minimised directions and is monotone under adding
points.

### Cost-aware greedy selection

EHVI is evaluated for every (molecule, assay) pair, the best EHVI-per-unit-cost
is selected, the front is updated with an imputed outcome, and the loop repeats
until the budget is exhausted or nothing offers positive improvement.

An assay informs only some objectives. Objectives it does not inform are held at
their predicted mean with **zero** variance — measuring solubility tells you
nothing new about toxicity, so only the informed objectives contribute
uncertainty to the improvement.

**Stated limitations, not implied ones:**

* **Greedy is not optimal.** Hypervolume improvement is submodular, giving
  greedy a $(1 - 1/e)$ guarantee in the **unit-cost** case; with heterogeneous
  costs the cost-weighted rule here has **no such bound**.
* **Imputation is optimistic.** The `"mean"` strategy assumes an assay returns
  exactly the prediction, which under-rewards a second measurement in the same
  region. A `"sample"` strategy is offered.
* **Hypervolume depends on the reference point**, which is reported with every
  plan. Comparing hypervolumes computed against different reference points is
  meaningless.

Verified behaviour: budget never exceeded; no (molecule, assay) pair repeated;
expected gain non-decreasing in budget; the planner stops early when no
improvement remains; and changing assay costs genuinely reshapes the selected
mix (making toxicity cheap moved the plan from 5 solubility assays to 7 toxicity
+ 3 solubility).

---

## 7b. Molecular generator

### Architecture

Character-level tokenisation with multi-character atom tokens handled (`Cl`,
`Br`, bracket atoms), a GRU language model (2 × 256), teacher-forced
maximum-likelihood pre-training, then REINVENT-style RL. The tokeniser
round-trips: if tokens do not reassemble the input exactly, the string is
rejected rather than silently corrupted.

### MOSES pre-training is pending — and the consequence is measured

The methodology specifies pre-training on MOSES (~1.9M molecules). **MOSES is
not bundled and is not downloaded automatically.** The prior is trained on the
~2,500 SMILES available locally (ESOL ∪ ClinTox), three orders of magnitude
fewer.

The consequence is concrete and reported rather than hidden:

| Temperature | Validity | Uniqueness |
|---|---|---|
| 0.7 | 0.780 | 0.936 |
| 1.0 | 0.463–0.592 | 0.971 |
| 1.2 | 0.435 | 0.966 |

A MOSES-pretrained prior of the same architecture typically exceeds 90%
validity. Novelty 0.72, internal diversity 0.91 and 44 unique scaffolds per 300
samples are respectable; **validity is the casualty of the data shortfall**, and
`scripts/download_moses.py` documents how to close it without changing any other
code.

### REINVENT augmented likelihood

$$\log P_{\text{aug}}(S) = \log P_{\text{prior}}(S) + \sigma R(S), \qquad \mathcal{L} = \left[\log P_{\text{aug}}(S) - \log P_{\text{agent}}(S)\right]^2$$

The prior term is what prevents reward hacking. A bare policy-gradient agent
maximising $R$ drifts into degenerate strings that score well and are not
molecules; anchoring to the prior's likelihood keeps output chemically
plausible, and $\sigma$ sets how far the agent may stray. A diversity filter
discounts repeated Murcko scaffolds so reward cannot be farmed from one good
molecule.

### The reward uses the LOWER conformal bound

$$R = w_{\text{valid}}\cdot\text{scoreable} + w_{\text{obj}}\cdot\overline{\text{normalised lower bound}} + w_{\text{ad}}\cdot\text{domain} + w_{\text{nov}}\cdot\text{first seen}$$

**This is the most important choice in the reward.** Rewarding the predicted
*mean* teaches the generator to find regions where the model is uncertain and
happens to predict well — it learns to exploit the model's blind spots, which is
exactly the failure mode uncertainty-aware generation exists to prevent.
Rewarding the lower bound makes uncertainty a cost rather than a free option.
The applicability-domain term does the same job from the other direction.

Measured over a 20-step campaign: mean reward 0.297 → 0.371 within RL batches,
validity 0.48 → 0.67 on fresh samples, reward on sampled molecules 0.747 →
0.791, internal diversity held at 0.91. Before/after metrics are both reported
so a campaign that raises reward while destroying validity or diversity is
visible.

**An honest tension worth naming.** Ranking by the lower bound biases selection
towards the training distribution, because that is where the model is confident.
In practice the top-screened generated candidates are small, simple, often
already-known molecules with tight intervals. That is the lower-bound reward
working as designed, and it is a real trade-off against novelty, not a bug. A
campaign that genuinely wants novel chemistry must either accept wider intervals
or extend the applicability domain with new data.

Rising reward means the agent found molecules **this platform's own models score
highly with confidence**. It is not experimental validation, and none of these
molecules has been synthesised or assayed.

---

## 7. Properties deliberately **not** modelled

| Property | Status | Why |
|---|---|---|
| Target bioactivity (pIC50) | `unavailable` | No curated target-specific assay dataset is bundled. The previous value was `clip(5.5 + 2.5·QED + 1.0·(maxTanimoto − 0.5), 4, 9.8)` — not fitted to any data, not target-specific, and its similarity term measured closeness to an *aqueous solubility* training set, which carries no information about binding affinity. |
| Metabolic stability | `unavailable` | No dataset bundled. The previous value was `clip(60 − 5·logP − 2·rotBonds + 10·QED, 5, 150)` with invented coefficients and invented units ("minutes in liver microsomes"). |

Both now return `status: "unavailable"` with a reason and what would be required
to enable them. Reinstating either requires a curated dataset and a trained,
evaluated model.

---

## 8. Dataset semantics and known traps

**ClinTox CT_TOX does not mean "this molecule is toxic."** It marks compounds
whose clinical trial was terminated for toxicity reasons in the source records.
Aspirin is labelled `CT_TOX = 1` in this dataset. A prediction of 1.0 for aspirin
from the Random Forest is therefore not a bug — aspirin is in the training
partition and the value is a memorised label. Responses carry `in_training_set`
and a `memorisation_warning` for exactly this case.

**PR-AUC must be read against the base rate.** ClinTox is 7.5% positive. A
PR-AUC of 0.358 against a baseline of 0.072 is a real signal; read against
ROC-AUC it would look poor. The test partition holds only 15 positives, so these
figures are noisy.

**Deduplication before splitting.** The raw Delaney file contains 11 duplicate
canonical structures. Left in, they straddle the train/test boundary and inflate
reported performance. All datasets are deduplicated on canonical SMILES before
any split, and partition disjointness is asserted by test.

**Scaffold splitting on ESOL is weaker than it looks.** The deterministic
Bemis-Murcko split reports R² 0.890 against the random split's 0.903, but ESOL
contains many acyclic compounds and each becomes a *singleton* scaffold group, so
the split degenerates towards random for that portion. The small gap should not
be read as evidence of strong chemotype generalisation.

---

## 9. Measured results summary

| Model | Representation | Split | R² | RMSE |
|---|---|---|---|---|
| ExtraTrees (baseline) | ECFP4 + descriptors | random | 0.903 | 0.666 |
| ExtraTrees | ECFP4 + descriptors | Bemis-Murcko scaffold | 0.890 | 0.746 |
| D-MPNN ensemble (5) | molecular graph, learned | random, pinned to baseline | 0.899 | 0.680 |
| Random Forest (independent) | ECFP4 + descriptors | same held-out compounds | 0.877 | — |

| Classification (ClinTox CT_TOX) | ROC-AUC | PR-AUC | baseline |
|---|---|---|---|
| Random Forest, single-task | 0.832 | 0.358 | 0.072 |
| D-MPNN, multi-task | 0.858 | 0.277 | 0.069 |

The graph network **matches but does not beat** the fingerprint baseline on
solubility (0.899 vs 0.903 — a difference well inside the noise of a
138-compound test set). This is the honest result and is what the literature
would predict at this dataset size: D-MPNNs need more data than ~1,100 molecules
to pull ahead of well-tuned fingerprint baselines. Its value here is the
calibrated per-molecule σ, the learned latent space for the applicability domain,
and the multi-task gain on classification — not raw regression accuracy.

**Independent-model agreement:** Pearson r = 0.94 between the D-MPNN and the
Random Forest; 25.4% of test compounds disagree by more than the D-MPNN's own σ.
Agreement is weak evidence of reliability, **not** proof of correctness: two
models trained on the same data share its biases and can agree while both being
wrong. Disagreement is the more informative signal.

---

## References

* Yang et al. (2019), *Analyzing Learned Molecular Representations for Property Prediction*, J. Chem. Inf. Model. — D-MPNN architecture
* Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS — ensemble variance decomposition
* Sadinle et al. (2019), *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels*, JASA — LAC
* Romano et al. (2020), *Classification with Valid and Adaptive Coverage*, NeurIPS — APS
* Delaney (2004), *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*, J. Chem. Inf. Comput. Sci.
* Gayvert et al. (2016) / MoleculeNet (Wu et al. 2018) — ClinTox
* Bickerton et al. (2012), *Quantifying the chemical beauty of drugs*, Nat. Chem. — QED
* OECD (2007), *Guidance Document on the Validation of (Q)SAR Models* — leverage applicability domain
* Deb et al. (2002), *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*, IEEE Trans. Evol. Comput. — fast non-dominated sorting
* Emmerich et al. (2006) / Daulton et al. (2020), *Differentiable Expected Hypervolume Improvement*, NeurIPS — EHVI and qEHVI
* Olivecrona et al. (2017), *Molecular de-novo design through deep reinforcement learning*, J. Cheminform. — REINVENT augmented likelihood
* Polykovskiy et al. (2020), *Molecular Sets (MOSES)*, Front. Pharmacol. — generative benchmark (not bundled; see scripts/download_moses.py)
