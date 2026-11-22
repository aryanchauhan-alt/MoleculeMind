# MoleculeMind — Learning Guide

**Read this first. It explains the whole project from the ground up, assuming
no prior knowledge of chemistry or machine learning.**

Every section follows the same shape: *the problem a scientist actually has* →
*what we do about it* → *where the code lives* → *what it does NOT claim*. By
the end you should be able to explain any part of this project to an examiner
without looking anything up.

Numbers quoted here are real, measured on this machine with seed 42, and
reproducible with `python -m pytest -q`.

---

## Contents

0. [The problem we are solving](#0-the-problem-we-are-solving)
1. [How a molecule lives inside a computer](#1-how-a-molecule-lives-inside-a-computer)
2. [Predicting a property](#2-predicting-a-property)
3. [Why one number is not enough](#3-why-one-number-is-not-enough)
4. [Knowing when not to trust the model at all](#4-knowing-when-not-to-trust-the-model-at-all)
5. [Wanting several things at once](#5-wanting-several-things-at-once)
6. [Choosing the next experiment](#6-choosing-the-next-experiment)
7. [Spending a budget](#7-spending-a-budget)
8. [Inventing new molecules](#8-inventing-new-molecules)
9. [The graph neural network — and why it did not win](#9-the-graph-neural-network--and-why-it-did-not-win)
10. [The three defects we found and fixed](#10-the-three-defects-we-found-and-fixed)
11. [Map of the codebase](#11-map-of-the-codebase)
12. [Running and demonstrating it](#12-running-and-demonstrating-it)
13. [Glossary](#13-glossary)
14. [Hard questions you might be asked](#14-hard-questions-you-might-be-asked)

---

## 0. The problem we are solving

A medicinal chemist wants a molecule that will become a drug. It must satisfy
many requirements at once:

* dissolve in water well enough to be absorbed (**solubility**),
* not be toxic,
* look generally drug-like,
* and bind to the biological target.

There are around 10⁶⁰ plausible drug-like molecules. Nobody can test them. A
laboratory experiment ("assay") costs money and takes days, so a chemist might
afford a few dozen in a project.

So the real question is never "what is the best molecule?" It is:

> **Given a limited budget, which handful of molecules should I put in the lab
> next — and how much should I trust the computer's advice?**

That second half is the whole point of this project. A model that says
"solubility = −2.1" is nearly useless on its own. The chemist needs to know
*how wrong it might be*, and *whether this molecule is even the kind of thing
the model has seen before*. A confident wrong answer wastes a month.

MoleculeMind is a decision-support tool for exactly that. It is **not** a drug
discovery engine; it is a tool for deciding what to measure next, which is
honest about what it does not know.

---

## 1. How a molecule lives inside a computer

### 1.1 SMILES — molecules as text

A molecule is an arrangement of atoms and bonds. **SMILES** is a way of writing
that arrangement as a line of text.

| SMILES | Molecule | How to read it |
|---|---|---|
| `CCO` | ethanol | carbon–carbon–oxygen chain |
| `c1ccccc1` | benzene | lowercase = aromatic ring, the `1`s close the ring |
| `CC(=O)Oc1ccccc1C(=O)O` | aspirin | `(=O)` is a double-bonded oxygen branch |

Rules worth knowing: letters are atoms, brackets `()` are branches, `=` and `#`
are double and triple bonds, digits open and close rings, lowercase means
aromatic, and square brackets like `[nH]` carry extra detail such as charge or
attached hydrogens.

### 1.2 Canonicalisation — one molecule, one spelling

The same molecule can be written many ways. `OCC` and `CCO` are both ethanol.
This matters enormously: if we are not careful, the *same molecule* appears in
both our training data and our test data under two different spellings, and we
fool ourselves into thinking the model is better than it is.

So every SMILES entering this system is converted to a single **canonical** form
by RDKit. This is the first thing `/api/analyze` does.

> **We found this exact bug.** The raw Delaney dataset contains **11 duplicate
> structures** written differently. They were being split across train and test.
> See §10.

### 1.3 RDKit — the chemistry library

[RDKit](https://www.rdkit.org/) does all the actual chemistry: parsing SMILES,
rejecting nonsense, drawing 2D pictures, generating 3D coordinates, and
calculating properties. We never reimplement chemistry; we call RDKit.

### 1.4 Descriptors — simple numbers about a molecule

A **descriptor** is one measurable number. We use 8:

| Descriptor | Meaning in plain words |
|---|---|
| Molecular weight | How heavy it is |
| LogP | How much it prefers oil over water (fat-solubility) |
| TPSA | How much of its surface is polar (affects gut absorption) |
| H-bond donors / acceptors | How many hydrogen bonds it can make |
| Rotatable bonds | How floppy it is |
| Ring count | How many rings |
| Heavy atom count | Atoms other than hydrogen |

Plus two famous rule-of-thumb checks:

* **Lipinski's Rule of Five** — oral drugs usually have MW ≤ 500, LogP ≤ 5,
  donors ≤ 5, acceptors ≤ 10. We allow one violation, as is conventional.
* **Veber's rule** — rotatable bonds ≤ 10 and TPSA ≤ 140.

These are *heuristics from observation*, not laws. Plenty of real drugs break
them.

### 1.5 Fingerprints — molecules as bit vectors

Descriptors lose the structure. A **Morgan fingerprint** (also called ECFP4)
keeps some of it:

1. Look at each atom and its neighbours out to 2 bonds away.
2. Hash each of those little neighbourhoods into a number.
3. Set that bit to 1 in a 1024-slot array.

The result is 1024 ones and zeros saying "these substructures are present". Two
similar molecules share many bits. **Tanimoto similarity** measures that
overlap: shared bits ÷ total bits used, giving 0 (nothing in common) to 1
(identical fingerprints).

So our feature vector is **1024 fingerprint bits + 8 descriptors = 1032
numbers** per molecule.

### 1.6 A trap that cost us twice

Those 1032 numbers are on wildly different scales. Fingerprint bits are 0 or 1.
Molecular weight is around 100–400.

Any algorithm that cares about distance or variance will be **completely
dominated by molecular weight**. We measured it: on the raw matrix, molecular
weight accounts for **90.9%** of all variance. We hit this bug twice, in two
different modules (§10.2 and §4.2). The fix is always the same: standardise
features (subtract the mean, divide by the standard deviation) before anything
scale-sensitive.

**Code:** [`mol_analyzer.py`](backend/app/core/mol_analyzer.py),
[`dataset.py`](backend/app/core/dataset.py)

### 1.7 3D structures — an honesty note

We also generate 3D coordinates (ETKDG embedding, then MMFF94 force-field
optimisation). This is a **computed low-energy guess** at the shape, not an
experimentally determined structure. The API says so in a `sdf_note` field. A
real structure comes from X-ray crystallography.

---

## 2. Predicting a property

### 2.1 The dataset

**Delaney ESOL**: 1,128 molecules with experimentally measured water solubility,
reported as **logS** — the base-10 logarithm of how many moles dissolve per
litre. More negative = less soluble. Roughly −8 (practically insoluble) to +2
(very soluble).

After removing the 11 duplicates: **1,117 unique molecules**.

### 2.2 Splitting the data, and why it is the most important step

We divide the molecules into four groups that **never overlap**:

| Group | Size | Purpose |
|---|---|---|
| Train | 781 | The model learns from these |
| Calibration | 168 | Measures how wrong the model tends to be (§3) |
| Test | 138 | Final honest score; touched once |
| Active-learning pool | 30 | Pretend-unknown molecules for §6 |

If a molecule appears in two groups, every number afterwards is a lie. The model
would be graded on an answer it had already memorised. This is called **data
leakage** and it is the single most common way machine learning papers go wrong.
`test_partitions_do_not_share_structures` asserts the four groups are disjoint
by canonical SMILES.

### 2.3 The model

**Extra Trees** (extremely randomised trees): 150 decision trees, each trained
on the data with random split points, their predictions averaged. Why this
rather than a neural network? It is fast (under 4 seconds), needs no tuning, and
is very strong on small tabular datasets. It is a genuinely hard baseline — see
§9.

### 2.4 How well does it do?

```
R² = 0.903     RMSE = 0.666     MAE = 0.484     (138 test molecules)
```

What these mean:

* **R²** = fraction of the variation in solubility the model explains. 1.0 is
  perfect, 0 is no better than always guessing the average. 0.903 is good.
* **RMSE** = typical error size, in logS units. 0.666 means a typical prediction
  is off by about two-thirds of a log unit — a factor of ~4.6 in actual
  concentration. That sounds bad, and it is worth being clear-eyed about: this
  model is useful for *ranking and triage*, not for precise numbers.
* **MAE** = average absolute error, less sensitive to outliers than RMSE.

### 2.5 Random split vs scaffold split

A random split can put two close cousins from the same chemical family on
opposite sides. The model then only has to *interpolate*, which is easy.

A **scaffold split** groups molecules by their core ring structure (the
"Bemis–Murcko scaffold") and keeps whole families on one side. The model must
generalise to chemistry it has never seen — much harder, and much closer to real
prospective use.

```
random split:    R² = 0.903
scaffold split:  R² = 0.890
```

**Be honest about this result.** The gap is small, but that is partly an
artefact: ESOL contains many chain-like molecules with no rings at all, and each
of those becomes its own one-member "family", so for that portion the scaffold
split behaves like a random one. Do **not** present 0.890 as proof of strong
generalisation to new chemistry.

### 2.6 The second model: toxicity

**ClinTox** — 1,461 molecules labelled `CT_TOX`, 103 of them positive (7.1%).

**The label means something narrower than it sounds.** `CT_TOX = 1` means *this
compound's clinical trial was terminated for toxicity reasons*. It does not mean
"this molecule is poisonous". Aspirin is labelled `CT_TOX = 1` in this dataset.

```
ROC-AUC = 0.832     PR-AUC = 0.358     (baseline positive rate 0.072)
```

* **ROC-AUC** = probability the model ranks a random positive above a random
  negative. 0.5 is chance.
* **PR-AUC** must be read against the **base rate**. 0.358 looks poor next to
  0.832, but random guessing would score 0.072, so the model is ~5× better than
  chance at this.

Because only 7% are positive, we set `class_weight="balanced"` so the model is
not rewarded for simply predicting "not toxic" every time.

If you ask the platform about aspirin it returns toxicity 1.0 — because aspirin
is in the training set with that label. The API flags this honestly with
`in_training_set: true` and a `memorisation_warning`.

**Code:** [`models.py`](backend/app/core/models.py)

---

## 3. Why one number is not enough

### 3.1 The problem

"Solubility = −2.1" tells a chemist nothing about risk. Is it −2.1 ± 0.1 or
−2.1 ± 2.0? Those lead to completely different decisions.

### 3.2 The naive answer, and why it is not enough

Our 150 trees each give a prediction. Their spread looks like an uncertainty —
if they disagree, the model is unsure. We do use that spread, but it is **a
heuristic, not a calibrated standard deviation**. There is no guarantee that
"±1 spread" contains the truth 68% of the time.

### 3.3 Conformal prediction — the real answer

**Conformal prediction** turns any model's rough uncertainty into an interval
with a mathematical guarantee. The idea is beautifully simple:

1. Take the **calibration set** — 168 molecules the model never trained on.
2. For each, record how wrong the model was, scaled by how unsure it claimed to
   be: `s = |actual − predicted| / spread`.
3. Sort those 168 scores. Find the one at the 90th percentile — call it `q̂`.
4. For a brand-new molecule: the interval is `prediction ± q̂ × spread`.

That is it. The logic: if a new molecule is drawn from the same population as
the calibration molecules, its error score should land in the same distribution,
so 90% of the time it lands below `q̂`.

One subtlety that matters: we do not take a plain 90th percentile. We take the
**⌈(n+1)(1−α)⌉-th** smallest value — the 153rd of 168 — which is what makes the
guarantee exact for a finite sample rather than approximate.

### 3.4 Does it work? Yes, measured

| Nominal | Measured coverage | Mean interval width |
|---|---|---|
| 80% | 78.3% | 1.60 logS |
| 90% | 88.4% | 2.14 logS |
| 95% | 99.3% | 3.36 logS |

The 80% and 90% levels land within 2 points of target. The 95% figure
over-covers, which is the expected behaviour when a single extreme calibration
residual sets the quantile at small `n` — with 168 calibration points the 95%
level depends on just a handful of the worst cases.

Notice the **widths**. A 90% interval spans 2.14 logS units. That is wide. This
is the honest truth about molecular property prediction on 1,100 training
molecules, and it is exactly the information a chemist needs that a bare
prediction hides.

### 3.5 What the guarantee does NOT say

This is the single most important misunderstanding to avoid.

> A 90% conformal interval does **not** mean "there is a 90% chance this
> molecule's true solubility is in this interval."

It means: **across many molecules**, about 90% of the intervals will contain the
truth. This is called **marginal coverage**. Some individual molecules are
systematically over-covered and others under-covered; the guarantee is about the
average, not about your molecule.

It also assumes the new molecule is **exchangeable** with the calibration set —
drawn from the same kind of chemistry. When that assumption fails, the guarantee
simply does not apply. Which brings us to §4.

### 3.6 Prediction sets for yes/no questions

For classification you cannot have an interval, so conformal prediction gives a
**set of labels** instead. Three methods are implemented. Measured on synthetic
3-class data:

| Method | 80% | 90% | 95% | mean set size @90% |
|---|---|---|---|---|
| LAC | 79.9% | 89.8% | 94.3% | 1.42 |
| APS (plain) | 99.2% | 99.8% | 100.0% | 2.69 |
| APS (randomised) | 80.0% | 89.9% | 94.8% | 1.57 |

A set can come back **empty** (no label was plausible) or **full** (the model
cannot tell). Both are informative.

**Note the middle row.** Plain APS massively over-covers — 99.2% when you asked
for 80%, with sets containing 2.4 of 3 labels. That is *valid* (coverage is a
lower bound) but useless. The randomised version from the original paper fixes
it. We kept both and documented the difference rather than quietly shipping the
broken-feeling one.

### 3.7 Recalibration

When active learning (§6) reveals new measurements, the calibration set is no
longer the one `q̂` was computed from. Reusing it would silently misstate
coverage, so we refit. Two caveats we state in the response: the underlying
model is not retrained, and actively-chosen molecules are not a random sample,
so coverage afterwards is approximate.

**Code:** [`conformal.py`](backend/app/core/conformal.py)

---

## 4. Knowing when not to trust the model at all

### 4.1 The problem

A model trained on small organic molecules will cheerfully give you a confident
prediction for a protein, an organometallic complex, or a polymer. It has no
idea it is out of its depth. Conformal intervals will not save you, because
their exchangeability assumption has been violated.

The **applicability domain** is a separate, independent check: *is this molecule
the kind of thing the model has seen?*

### 4.2 Two independent tests

**Test 1 — structural similarity.** Compute the maximum Tanimoto similarity
between the query and every training molecule. If the closest thing in training
is barely similar, we are extrapolating.

**Test 2 — descriptor-space position.** Compute **leverage**, a measure of how
far from the centre of the training cloud this molecule sits.

> This is where we hit the molecular-weight trap for the second time. Leverage
> was computed on unstandardised features, so it was effectively just asking "is
> this molecule unusually heavy?". Standardising fixed it.

### 4.3 Where the thresholds come from

Earlier versions hardcoded "similarity below 0.55 = suspicious". Where does 0.55
come from? Nowhere — it was asserted. The right threshold depends entirely on
how diverse your training set is.

So we **derive** them from the training data itself. For every training
molecule, ask "how similar is my nearest neighbour?" (excluding itself). That
gives 781 numbers. Take the 5th percentile: **0.250**. By construction, 95% of
training molecules have a neighbour at least that close, so a query below it is
less typical of this training set than almost every training molecule is.

Same logic for leverage, using the 95th percentile: **0.0583**.

### 4.4 The verdict rule

* **Both** tests fail → `OUT_OF_DOMAIN`
* **One** test fails → `BORDERLINE`
* **Neither** fails → `IN_DOMAIN`

Failing one of two independent checks is weaker evidence than failing both, so
we keep the distinction rather than collapsing it. Sanity check: on the training
set itself only **0.5%** of molecules get flagged, so the engine does not reject
the very data it was built from.

### 4.5 A worked example showing why two tests are needed

Ask the platform about a 36-carbon straight-chain alkane (`CCCCCC...`).

Its Tanimoto similarity to the training set is **1.000** — perfect. Why?
Fingerprints record *local* atom environments, and every carbon in a long chain
has exactly the same environment as every carbon in a short chain. The
fingerprint of C36 is bit-identical to that of hexane. **The similarity test is
completely blind here.**

But its learned latent distance (§4.6) is 0.481 against a threshold of 0.358, so
the second test catches it and the molecule is correctly flagged `BORDERLINE`.

This is the single clearest justification for the two-test design, and a good
example to have ready.

### 4.6 The latent-space version

When the neural network of §9 is available, we use something better than
leverage: the distance in the network's own **learned representation** of
molecules. The network has learned which structural features matter for the
property, so distance in that space is more meaningful than distance in raw
descriptor space. Threshold: 95th percentile of training distances, **0.358**.

### 4.7 What this is NOT

The API returns a `domain_index` between 0 and 1. It is a convenient score for
sorting. It is **not** a probability that the prediction is right, and the
response says so in an `index_caveat` field. An earlier version called it
`reliability_score`, which invited exactly that misreading.

**Code:** [`applicability_domain.py`](backend/app/core/applicability_domain.py)

---

## 5. Wanting several things at once

### 5.1 The problem

A chemist wants high solubility **and** low toxicity **and** good drug-likeness.
These fight each other — the changes that make a molecule dissolve better often
make it less drug-like. There is no single best molecule.

You could average the objectives into one score, but then you have invented
weights ("solubility is 1.5× as important as toxicity") that you cannot justify.

### 5.2 Pareto dominance

Instead, ask a question with no weights in it:

> Molecule **A dominates** molecule **B** if A is at least as good as B on
> *every* objective, and strictly better on *at least one*.

If A dominates B, nobody should ever pick B — A beats it outright. The molecules
that **nothing** dominates form the **Pareto front**. These are the genuine
trade-offs: to improve any objective from here, you must give something up.

Small example, both objectives maximised:

| Molecule | Solubility | Drug-likeness | Verdict |
|---|---|---|---|
| A | 1 | 5 | on the front |
| B | 5 | 1 | on the front |
| C | 3 | 3 | on the front |
| D | 1 | 1 | **dominated** by all three |

A, B and C are incomparable without stating preferences. D is simply worse.

### 5.3 Two rules we enforce

**Direction matters.** Toxicity should be *minimised*, solubility *maximised*.
We flip the sign of minimised objectives so everything becomes "bigger is
better". Getting this backwards silently inverts your conclusions.

**Missing values are an error, never a default.** An earlier version substituted
`0.0` for a missing objective. For toxicity, which we minimise, `0.0` is the
*best possible value* — so a molecule with no toxicity prediction got promoted
onto the Pareto front on the strength of data it did not have. It now raises an
exception.

### 5.4 The probabilistic version

Here is the subtle problem with the above: every objective is a *prediction with
an interval*. A molecule just inside the front with tight intervals and one just
outside with wide intervals are being separated by noise.

So we ask a better question: **with what probability is this molecule on the
front?**

1. Convert each conformal interval into a sampling scale: `σ̃ = q̂σ / z`. This
   picks the bell curve whose middle 90% is exactly as wide as our calibrated
   interval.
2. Draw a random possible outcome for **every** molecule at once (500 times).
3. Each time, work out which molecules are non-dominated.
4. A molecule's probability = the fraction of draws in which it was on the front.

Sampling all molecules together matters: within one draw, molecule A's good luck
is judged against molecule B's bad luck, which is exactly the comparison you
want.

### 5.5 Two honest caveats, both visible in the UI

**The bell-curve shape is an assumption, and we measured that it is wrong.**
Conformal prediction is *distribution-free*: it guarantees the interval's
**width**, never its shape. We impose a bell curve to be able to sample. On our
actual calibration residuals, excess kurtosis is **1.416** (0 for a true bell
curve) and a statistical test rejects normality. Real residuals have **heavier
tails** — more extreme errors than a bell curve predicts. So our probabilities
are mildly **optimistic** for molecules whose front membership depends on a rare
large error. The frontend shows an amber warning saying this.

**Objectives are sampled independently, which is also not quite right.**
Separate intervals for solubility and toxicity do not define a joint
distribution; sampling them independently assumes their errors are uncorrelated,
which is unlikely as they share a model and training data. The code *supports*
correlated sampling, but the API **refuses** it with HTTP 501, because only 64
molecules in our data have both labels — far too few to estimate the correlation
honestly. A made-up correlation matrix would look more rigorous while being less
trustworthy.

### 5.6 The risk slider

`κ = κ_max × λ`, then score each objective as:

* maximised: `μ + κσ̃` (assume it turned out well)
* minimised: `μ − κσ̃` (assume it turned out well — i.e. *lower*)

Drag λ from 0 to 1 and the ranking changes live. λ = 0 ranks by the prediction
alone (**exploit** — play safe). λ = 1 gives uncertainty full weight
(**explore** — favour molecules that *might* be great).

The sign flip for minimised objectives is essential. Being optimistic about
toxicity means assuming a *lower* number. Get it backwards and the slider
punishes uncertainty on exactly the objectives where it should reward it.

`μ + 2σ` is **not a prediction** and carries no guarantee. It is a deliberately
optimistic sorting key.

**Code:** [`pareto.py`](backend/app/core/pareto.py),
[`probabilistic_pareto.py`](backend/app/core/probabilistic_pareto.py)

---

## 6. Choosing the next experiment

### 6.1 The setup

We have 30 molecules whose real measured solubility exists in the Delaney
dataset but which the model has never seen. We reveal 6 at the start and hide
the other 24. The question: **which should we "measure" next?**

This is **active learning**: the model chooses its own training data.

### 6.2 The surrogate

A **Gaussian process** gives, for any molecule, both a prediction and a genuine
uncertainty that grows the further you get from measured examples. That property
is what makes it the standard choice here.

### 6.3 Acquisition functions — four ways to choose

| Strategy | Rule | Behaviour |
|---|---|---|
| **UCB** | `μ + √β·σ` | Balanced; β tunes the balance |
| **Expected Improvement** | expected gain over the best so far | Balanced, naturally cautious |
| **Uncertainty sampling** | `σ` alone | Pure exploration; learns the landscape |
| **Greedy** | `μ` alone | Pure exploitation; no learning |

### 6.4 Simulated laboratory mode — said plainly

Clicking "run assay" **reveals a measurement that already exists** in the
Delaney dataset and was being withheld. **No experiment is performed.** This is
a retrospective simulation, and the API labels every response
`mode: "simulated"` with an explicit note.

This matters for honesty: a simulated assay that reveals a known answer is a
legitimate way to evaluate an active-learning strategy, and a dishonest thing to
call a laboratory result.

### 6.5 Keeping the answers hidden

Four separate stores:

1. `_oracle` — the real measurements, **server-side only, never serialised**
2. `_revealed` — which have been released
3. `candidates` — the public records, carrying a value only once revealed
4. the model's own test set — held elsewhere entirely

`public_pool()` is the only way data leaves the engine, and it raises an
exception if an oracle key would escape. Two tests scan the entire API response
for forbidden keys. This is in direct response to a defect we found — see §10.1.

**Code:** [`active_learning.py`](backend/app/core/active_learning.py)

---

## 7. Spending a budget

### 7.1 The problem

You have 8 budget units. A solubility assay costs 1, a toxicity panel costs 3.
Which molecule–assay pairs should you buy?

### 7.2 Hypervolume — scoring a whole Pareto front

Pick a deliberately bad **reference point**. The **hypervolume** is the area (in
2D) or volume (in higher dimensions) between your Pareto front and that
reference point. A better front covers more area. One number for the quality of
a whole set of trade-offs.

### 7.3 Expected Hypervolume Improvement

```
HVI(y)  = HV(front + new point y) − HV(front)        how much a result helps
EHVI(x) = average of HVI(y) over y drawn from our predictive distribution
```

So EHVI is "how much do we *expect* this experiment to improve our trade-off
front?" We compute each hypervolume **exactly** (BoTorch's box decomposition)
and average over 128 sampled outcomes.

### 7.4 How we proved this is really EHVI

The brief we worked from insisted on this, and rightly: it is easy to write a
`μ + k·σ` score and call it EHVI. So we checked against two things we could not
influence.

**Check 1 — it must reduce to a known formula.** With a single objective, EHVI
collapses mathematically to **Expected Improvement**, which has a closed form.
Measured:

| μ | σ | best | our EHVI | exact EI |
|---|---|---|---|---|
| 1.0 | 0.5 | 0.8 | 0.31406 | 0.31522 |
| 2.0 | 0.3 | 1.0 | 0.99930 | 1.00003 |

Within sampling error. A `μ + kσ` score **cannot** reproduce that curve, so this
test alone distinguishes a real implementation from a heuristic.

**Check 2 — agree with BoTorch's own implementation.** We fit a Gaussian process
so both can integrate the same distribution, and compare:

```
BoTorch : [0.0, 0.0, 0.177936, 0.149688, 0.000156]
ours    : [0.0, 0.0, 0.178362, 0.149869, 0.000124]
largest relative difference: 0.24%
```

Live at `GET /api/planner/validate-ehvi`.

### 7.5 Why not just use BoTorch's qEHVI directly?

Fair question, and the answer is the interesting part. BoTorch's `qEHVI`
integrates over a **Gaussian process posterior**. Our uncertainties do not come
from a GP — they come from a neural ensemble calibrated by conformal prediction.
Using `qEHVI` would mean fitting a GP to our own predictions and integrating
over *that*, throwing away the calibration that is the entire point of the
project. So we kept our calibrated distribution and verified our maths against
theirs.

### 7.6 Choosing greedily, and admitting the limitation

Evaluate EHVI for every molecule–assay pair, pick the best value-per-cost,
pretend we got the predicted result, recompute, repeat until the money runs out.

Hypervolume improvement is **submodular** (each extra experiment helps a bit
less), which gives greedy selection a classic `(1 − 1/e)` ≈ 63%-of-optimal
guarantee — **but only when all costs are equal**. With different assay prices,
no such guarantee exists. We say so in the response. Good plans, not provably
optimal ones.

One nice detail: an assay only informs some objectives. Measuring solubility
tells you nothing new about toxicity, so uninformed objectives contribute zero
uncertainty to the improvement.

**Code:** [`experiment_planner.py`](backend/app/core/experiment_planner.py)

---

## 8. Inventing new molecules

### 8.1 The idea

Instead of screening a fixed list, generate new structures. Since SMILES are
text, a model that learns to write SMILES one character at a time can invent
molecules — the same mechanism as predictive text.

### 8.2 Two phases

**Phase 1 — learn the language.** Train a recurrent network (a GRU) to predict
the next character of real SMILES. It learns that `(` needs `)`, that ring
digits come in pairs, that `Cl` is one atom not two.

**Phase 2 — learn to aim.** Reinforcement learning, REINVENT-style:

```
Loss = ( log P_prior(S) + σ·Reward(S) − log P_agent(S) )²
```

The agent is pulled towards high reward, but tethered to the original model's
sense of what a molecule looks like. **That tether is what stops reward
hacking** — without it, an agent maximising reward alone drifts into gibberish
strings that happen to score well.

### 8.3 The reward uses the LOWER bound, deliberately

```
R = validity + mean(normalised LOWER conformal bound) + domain term − duplicate penalty
```

Rewarding the *predicted mean* would teach the generator to hunt for places
where the model is uncertain and happens to guess high — it learns to exploit
our blind spots, which is precisely the failure uncertainty-aware generation
exists to prevent. Rewarding the **lower** bound means uncertainty is a cost.

A diversity filter discounts repeated ring scaffolds, so the agent cannot farm
reward by submitting one good molecule repeatedly.

### 8.4 Results, honestly

| Temperature | Validity | Uniqueness | Novelty | Diversity |
|---|---|---|---|---|
| 0.7 | 78.0% | 93.6% | — | — |
| 1.0 | 46–59% | 97.1% | 72% | 0.91 |
| 1.2 | 43.5% | 96.6% | — | — |

The metrics: **validity** = fraction RDKit can parse; **uniqueness** = fraction
not repeats; **novelty** = fraction absent from training; **diversity** = 1 −
average pairwise similarity.

**Validity is poor, and we know exactly why.** The method calls for pre-training
on MOSES — 1.9 million molecules. MOSES is not bundled (130 MB, and the project
must work offline), so the model learned from the **2,512** SMILES we have
locally. That is 750× less data. A MOSES-trained model of the same design
exceeds 90% validity.

This is a **data limitation, not a code limitation**.
[`scripts/download_moses.py`](scripts/download_moses.py) explains how to close
it, and nothing else would need to change. Uniqueness, novelty and diversity are
all healthy; validity is the casualty.

RL does work: over 20 steps validity rose 48% → 67% and reward 0.747 → 0.791
with diversity holding at 0.91.

### 8.5 An honest tension worth understanding

Because we rank by the lower bound, we favour molecules the model is *confident*
about — and it is confident about things resembling its training data. So the
top-ranked generated molecules come out small, simple, and often already known.

That is the reward **working as designed**, and a genuine trade-off against
novelty rather than a bug. A campaign that truly wants novel chemistry must
either accept wider intervals or extend the applicability domain with new
experimental data. Being able to articulate this trade-off is worth more than
hiding it.

**Code:** [`generator.py`](backend/app/core/generator.py),
[`generator_service.py`](backend/app/core/generator_service.py)

---

## 9. The graph neural network — and why it did not win

### 9.1 The motivation

Fingerprints are fixed: somebody decided what substructures to count. A **graph
neural network** instead *learns* what structural features matter for the
property in question.

### 9.2 How a D-MPNN works

Treat the molecule as a graph: atoms are nodes, bonds are edges. Then pass
messages along the bonds.

For a directed edge from atom `v` to atom `w`:

1. **Start:** build a message from `v`'s features and the bond's features.
2. **Pass (×3):** each edge's new message = the sum of messages arriving at `v`,
   **minus the one that came back from `w`**.
3. **Gather:** each atom sums the messages arriving at it.
4. **Readout:** average over all atoms → one vector describing the molecule.
5. **Predict:** a small two-layer network maps that vector to the property.

**Step 2's subtraction is the whole trick.** Without it, a message flows
straight back along the bond it arrived on, and the network just echoes itself
instead of exploring the molecule. That is what "directed" means in D-MPNN.

### 9.3 How we verified it rather than trusting it

A subtly broken graph network still trains and still produces plausible numbers
— which makes it dangerous. So the test suite recomputes that message by brute
force, looping over every edge and summing exactly what the mathematical
definition says, then asserts it matches the fast vectorised version to within
10⁻⁵. A second test asserts the subtraction actually changes the answer, so if
anyone deletes it the suite fails loudly.

Also verified: renumbering the atoms does not change the output; batching
molecules together gives the same answers as one at a time; methane (no bonds at
all) does not crash it.

### 9.4 Predicting its own error

Each prediction outputs a mean **and** a variance, trained with **Gaussian
negative log-likelihood** — a loss that rewards the model for being
realistically unsure. We train five networks from different random starts and
split the total uncertainty in two:

```
total variance = average of each network's own predicted variance   (aleatoric)
               + how much the five networks disagree                (epistemic)
```

* **Aleatoric** = noise in the data; more data will not remove it. Measured 0.389.
* **Epistemic** = our ignorance; more data *would* reduce it. Measured 0.039.

Useful distinction: high epistemic uncertainty means "measure this and I will
improve". High aleatoric means "measuring this will not help much".

### 9.5 Multi-task learning

ESOL and ClinTox are different datasets. We joined them into one table with
**blanks** where a label is missing, and used a **masked loss** so missing
labels contribute nothing at all. Of 2,514 molecules, only 64 have both labels.

So the benefit flows through the **shared representation**: a molecule with only
a solubility label still helps the network learn chemistry that the toxicity
head then reads from. Measured: ClinTox ROC-AUC **0.832 → 0.866**.

### 9.6 The honest result

```
Extra Trees baseline : R² = 0.903
D-MPNN ensemble      : R² = 0.902
Random Forest check  : R² = 0.877
```

**The neural network did not beat the baseline.** 0.902 versus 0.903 on a
138-molecule test set is a difference far inside the noise.

This is not a failure of the implementation — it is the expected result at this
data scale, and the literature agrees: D-MPNNs need tens of thousands of
molecules to overtake well-tuned fingerprint baselines. With ~1,100, a strong
tree ensemble is very hard to beat.

**So why keep it?** Three things it provides that Extra Trees does not:

1. A **calibrated per-molecule σ** from a model trained to predict its own error.
2. A **learned latent space** for the applicability domain — the thing that
   caught the C36 alkane in §4.5.
3. The **multi-task gain** on toxicity.

Being able to say "we built it properly, measured it honestly, and it matched
rather than beat the baseline — here is why, and here is what it still gives us"
is a stronger position than a suspiciously large improvement.

### 9.7 Independent cross-checking

We also train a Random Forest on fingerprints — a different model family on a
different representation — and compare predictions on identical molecules.
Agreement: r = 0.944, with 25% of molecules disagreeing by more than the
network's own σ.

Disagreement is the useful signal: it flags molecules where the *choice of
representation* is doing the work. Agreement is only weak evidence, because both
models share the same training data and therefore the same biases. They can
agree and both be wrong.

**Code:** [`molecular_graph.py`](backend/app/core/molecular_graph.py),
[`dmpnn.py`](backend/app/core/dmpnn.py),
[`dmpnn_service.py`](backend/app/core/dmpnn_service.py)

---

## 10. The three defects we found and fixed

This is the most valuable part of the project to be able to discuss. The
starting codebase looked complete and worked when you clicked it. Three things
were wrong in ways that invalidated its central claims. Full detail in
[PROJECT_AUDIT.md](PROJECT_AUDIT.md).

### 10.1 The answers were being sent to the browser

`GET /api/active-learning/pool` returned `true_pic50`, `true_solubility` and
`true_toxicity` for **every** candidate, including the ones supposedly hidden.
The frontend read them directly. Anyone could open developer tools and see the
complete answer key.

That destroys the entire premise: a human-in-the-loop system where the loop can
see the answers is not evaluating anything.

**Fix:** four separate stores, one guarded serialisation path, and tests that
scan the whole API response for forbidden keys.

### 10.2 Every candidate got an identical score

The active-learning tab ranked candidates. Measured:

```
UCB         : all 24 candidates scored 1.414
EI          : all 24 scored 0.017
uncertainty : all 24 scored 0.606
```

One unique value each. **The ranking was meaningless** — the displayed order was
just the order of the hardcoded list.

The diagnosis took some digging. PCA was run on the raw 1032-dimensional
features, where molecular weight holds 90.9% of the variance, so the components
spanned roughly −113 to +178. Gaussian processes in BoTorch assume inputs around
0 to 1. Across a 290-unit range the GP concluded every molecule was unrelated to
every other, so its posterior fell back to the prior: mean 0, variance 1, for
everything. UCB then returned `0 + √2 × 1 = 1.414` for all of them.

**Fix:** standardise, then PCA, then rescale to the unit cube, plus BoTorch's
own input/output transforms. Now 17 of 24 score distinctly and the ranking
responds to new data. A test asserts the scores are not all equal, so this
cannot silently return.

### 10.3 Two "predictions" were invented arithmetic

```python
pIC50 = clip(5.5 + 2.5*QED + 1.0*(max_tanimoto − 0.5), 4.0, 9.8)
half_life = clip(60 − 5*logP − 2*rot_bonds + 10*QED, 5.0, 150.0)
```

Neither was fitted to any data. The coefficients were chosen by hand. The first
was presented as biological activity against a target, but is not
target-specific at all — and its similarity term measures resemblance to a
*solubility* dataset, which carries no information about binding. Both were
returned alongside genuine predictions, with nothing to distinguish them, and
both fed into the Pareto optimisation.

**Fix:** both removed. They now return `status: "unavailable"` with a reason and
what would be needed to enable them, carry no number, and are excluded from
optimisation. Every property now carries a `provenance` field: `model`,
`computed`, or `unavailable`.

The 30-molecule library also had invented `true_pic50` and `true_toxicity`
values attributed to real named drugs. Replaced with real held-out Delaney
measurements.

### 10.4 Plus

11 duplicate structures crossing the train/test boundary; ClinTox trained on an
arbitrary slice with no evaluation at all; Pareto substituting `0.0` for missing
values (the *best* value for a minimised objective); R² hardcoded as a string in
the landing page; the applicability domain's leverage hitting the same
unscaled-feature bug; conformal calibration never refreshed after active
learning.

---

## 11. Map of the codebase

Read in this order — each builds on the one before.

| # | File | Lines | What it does |
|---|---|---|---|
| 1 | `core/mol_analyzer.py` | 148 | SMILES in, descriptors and pictures out. **Start here.** |
| 2 | `core/dataset.py` | 305 | Loading, deduplicating, splitting, featurising |
| 3 | `core/conformal.py` | 415 | Intervals and prediction sets |
| 4 | `core/applicability_domain.py` | 465 | The two trust diagnostics |
| 5 | `core/pareto.py` | 123 | Dominance and non-dominated sorting |
| 6 | `core/models.py` | 928 | Trains everything, serves predictions with provenance |
| 7 | `core/active_learning.py` | 458 | GP, acquisition functions, the hidden oracle |
| 8 | `core/probabilistic_pareto.py` | 454 | Monte Carlo front, risk slider |
| 9 | `core/experiment_planner.py` | 635 | Hypervolume, EHVI, budget planning |
| 10 | `core/molecular_graph.py` | 267 | Molecules as graphs |
| 11 | `core/dmpnn.py` | 689 | The network, losses, ensemble |
| 12 | `core/dmpnn_service.py` | 755 | Multi-task training, evaluation, comparison |
| 13 | `core/generator.py` | 733 | SMILES language model and RL |
| 14 | `core/generator_service.py` | 381 | Reward built from our own models |
| 15 | `core/config.py` | 163 | Every setting, overridable by environment |
| 16 | `main.py` | 936 | 24 HTTP endpoints |

**Frontend** (`frontend/src/components/`): six tabs — Analyzer, Prediction,
Pareto Lab, Discovery Engine (the §5–8 tools), Active Learning, Benchmarking.

**Tests** (`backend/tests/`): 309 tests across 9 files.

**Suggested first session:** read `mol_analyzer.py` top to bottom (it is short
and entirely concrete), then `conformal.py`, then run
`pytest backend/tests/test_conformal_ad_pareto.py -v` and read the test names.
The tests are written to be readable as documentation.

---

## 12. Running and demonstrating it

### Install and run

```bash
pip install -r requirements.txt
python run.py
```

Then open <http://127.0.0.1:8000>. Interactive API docs at `/docs`.

**First launch takes about 3 minutes** — it trains everything and caches it.
Every later launch takes about 1 second. To skip the heavy parts:

```bash
MOLECULEMIND_ENABLE_DMPNN=false MOLECULEMIND_ENABLE_GENERATOR=false python run.py
```

### Tests

```bash
python -m pytest -q
```

### A 10-minute demonstration route

1. **Analyzer** — paste `CC(=O)Oc1ccccc1C(=O)O` (aspirin). Show descriptors,
   2D, 3D. Note that the 3D shape is computed, not measured.
2. **Prediction** — same molecule. Point at the **90% interval**, not the number.
   Show the provenance badges, and the two properties marked *unavailable* —
   explain that they used to be fake.
3. **Applicability Domain** — try a 36-carbon chain
   (`CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC`). Similarity says 1.000, latent
   distance catches it. This is the best single story in the project.
4. **Pareto Lab** — show the front, explain dominance with two molecules.
5. **Discovery Engine → Probabilistic Pareto** — show probabilities instead of
   yes/no, and read out the amber warning about heavy tails.
6. **Discovery Engine → Risk Slider** — drag λ from 0 to 1, watch the ranking
   change.
7. **Discovery Engine → Budget Planner** — set a budget, show the plan, then hit
   `/api/planner/validate-ehvi` to prove the maths against BoTorch.
8. **Active Learning** — run an assay, show the ranking update. Say "simulated".
9. **Benchmarking** — the four-model comparison. Say out loud that the neural
   network *matched* rather than beat the baseline.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Acquisition function** | Rule scoring how useful it would be to measure a molecule |
| **Aleatoric uncertainty** | Noise in the data; more data will not help |
| **Applicability domain** | Whether a molecule resembles the training data enough to trust a prediction |
| **Canonical SMILES** | The one agreed spelling of a molecule |
| **Conformal prediction** | Turning rough uncertainty into intervals with a coverage guarantee |
| **Coverage** | Fraction of intervals that actually contain the truth |
| **Descriptor** | A single calculated number about a molecule |
| **D-MPNN** | Directed Message Passing Neural Network — learns features from the molecular graph |
| **Dominance** | A beats B if A is ≥ on every objective and > on at least one |
| **ECFP4 / Morgan fingerprint** | 1024-bit vector of which substructures are present |
| **EHVI** | Expected Hypervolume Improvement — expected gain in front quality |
| **Epistemic uncertainty** | Our ignorance; more data would reduce it |
| **Exchangeability** | New data coming from the same distribution as calibration data |
| **Gaussian process** | Model giving predictions plus uncertainty that grows away from data |
| **Hypervolume** | Area/volume a Pareto front covers; one score for a whole front |
| **Leverage** | How far a molecule sits from the centre of the training cloud |
| **logS** | log₁₀ of solubility in mol/L; more negative = less soluble |
| **Marginal coverage** | The guarantee holds on average over molecules, not for each one |
| **Masked loss** | Loss that ignores missing labels instead of imputing them |
| **Pareto front** | The set of molecules nothing dominates |
| **PR-AUC** | Precision–recall area; must be read against the base rate |
| **Provenance** | Where a number came from: model, calculation, or unavailable |
| **QED** | Quantitative Estimate of Drug-likeness, 0–1 |
| **R²** | Fraction of variance explained; 1 perfect, 0 useless |
| **RDKit** | The open-source cheminformatics library doing all the chemistry |
| **REINVENT** | RL method keeping a generator tethered to a pre-trained prior |
| **RMSE** | Typical error size, in the property's own units |
| **Scaffold split** | Splitting by core ring structure; harder and more realistic |
| **SMILES** | Text representation of a molecule |
| **Tanimoto similarity** | Fingerprint overlap, 0 to 1 |
| **UCB** | Upper Confidence Bound, `μ + √β·σ` |

---

## 14. Hard questions you might be asked

**"Does your model actually work?"**
It predicts water solubility with R² = 0.903 and a typical error of 0.67 log
units on 138 held-out molecules. That is good enough to rank and triage
candidates, not to replace a measurement. The 90% interval is 2.1 log units
wide, and I would quote that interval rather than the point prediction.

**"Your neural network is no better than your decision trees. Why build it?"**
Correct, and expected at 1,100 training molecules — the literature puts the
crossover much higher. I kept it for three things trees cannot give: a
calibrated per-molecule uncertainty from a model trained to predict its own
error, a learned latent space that catches molecules fingerprints cannot
distinguish (the C36 alkane example), and a multi-task gain that lifted toxicity
ROC-AUC from 0.832 to 0.866.

**"What does a 90% interval mean?"**
That across many molecules, about 90% of such intervals contain the true value.
It does *not* mean this molecule has a 90% chance of falling in its interval.
And it assumes the molecule is exchangeable with my calibration set — which is
why the applicability domain is a separate check.

**"Is the 90% probability of Pareto optimality trustworthy?"**
It is relative to this candidate pool and this model, and it rests on imposing a
bell-curve shape on distribution-free intervals. I measured that assumption and
it is violated — excess kurtosis 1.416, normality rejected — so the
probabilities are mildly optimistic in the tails. The interface says so. I also
refused to offer correlated sampling, because only 64 molecules have both labels
and I would have had to invent the correlation matrix.

**"Is your EHVI real EHVI?"**
Yes, and I verified it two ways rather than asserting it. In one dimension EHVI
must reduce to Expected Improvement, and mine matches the closed form to within
sampling error — a `μ + kσ` heuristic cannot do that. And it agrees with
BoTorch's own qEHVI to 0.24% on a shared posterior. There is a live endpoint
that runs the comparison.

**"Why is your generator's validity only 46%?"**
Because the method calls for pre-training on 1.9 million MOSES molecules and I
trained on the 2,512 I had locally — 750× less data. The architecture, the RL
objective and the evaluation are all implemented; the data is the gap, and
`scripts/download_moses.py` documents how to close it. I report the measured
number rather than the one the method would reach.

**"What was wrong with the original code?"**
Three things that invalidated its claims. It sent the hidden answers to the
browser. Its acquisition function gave all 24 candidates an identical score, so
the ranking was meaningless — I traced that to unstandardised PCA inputs
breaking the Gaussian process's assumptions. And two of its four optimisation
objectives were hand-written arithmetic presented as trained models. All three
are fixed with regression tests.

**"What would you do next?"**
Not add modules. Get MOSES for the generator, run multiple seeds so I can put
confidence intervals on R² rather than quoting one number, and find a real
bioactivity dataset so the pIC50 objective can come back as a genuine model
instead of staying marked unavailable.

**"What are you least confident about?"**
The scaffold-split result. It reads as though the model generalises well to new
chemotypes, but ESOL has many acyclic molecules that each form their own
single-member scaffold group, so for that portion the split degenerates towards
random. I would not claim strong chemotype generalisation from it.
