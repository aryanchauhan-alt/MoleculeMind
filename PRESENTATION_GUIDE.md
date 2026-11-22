# MoleculeMind — Presentation and Demo Guide

**A word-for-word script for presenting this project, with the numbers that are
actually true.**

Every figure here is measured on the current code with seed 42 and reproducible
via `python -m pytest -q` or `GET /api/benchmark/report`. If you quote a number
from this file, the live application will agree with you.

> **Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md) first.** This file tells you what
> to *say*; that one tells you what you actually *know*. You cannot defend this
> project from this file alone.

---

## 1. The 30-second pitch

> "A medicinal chemist can afford maybe a few dozen lab experiments in a
> project, out of around 10⁶⁰ possible drug-like molecules. So the real question
> is never 'what is the best molecule' — it's 'which handful should I measure
> next, and how much should I trust the computer's advice?'
>
> MoleculeMind is a decision-support tool for exactly that. It predicts
> molecular properties, but more importantly it attaches a *calibrated*
> uncertainty to every prediction, warns you when a molecule is outside what the
> models actually know, finds the genuine trade-offs between competing
> objectives, and plans experiments against a fixed budget.
>
> The part I'm most pleased with isn't a model — it's that I audited the original
> codebase and found three defects that invalidated its central claims. I'll show
> you those."

**Why lead with the audit:** it is the most defensible work in the project and
the hardest thing to fake. Anyone can import scikit-learn; finding that an
acquisition function silently returned an identical score for every candidate,
and tracing it to a feature-scaling bug breaking a Gaussian process's
assumptions, is engineering judgement.

---

## 2. The core problem, in one slide

Traditional molecular ML hands you a single number: "solubility = −2.1". That is
nearly useless to a chemist, for three reasons:

| Problem | What goes wrong | What we do |
|---|---|---|
| **No uncertainty** | Is it −2.1 ± 0.1 or ± 2.0? Completely different decisions. | Conformal prediction intervals with measured coverage |
| **Confident extrapolation** | The model will happily predict for a molecule unlike anything it trained on | Applicability domain — two independent trust checks |
| **One objective at a time** | Real projects need solubility *and* low toxicity *and* drug-likeness | Pareto optimisation, probabilistic version included |

---

## 3. The six modules in plain English

### Module 1 — Molecular Analyzer

Type a molecule as text (SMILES), get back its structure in 2D and 3D plus 13
calculated properties and the Lipinski and Veber drug-likeness checks.

**Say this:** "The 3D shape is an ETKDG embedding optimised with the MMFF94 force
field. It's a *computed* low-energy guess, not an experimentally determined
structure — a real one would come from X-ray crystallography. The API says so in
the response."

### Module 2 — Property Prediction

Two trained models: water solubility (logS) and clinical-trial toxicity risk.

```
Solubility:  R² = 0.903   RMSE = 0.666   MAE = 0.484   (138 held-out molecules)
Toxicity:    ROC-AUC = 0.832   PR-AUC = 0.358   (baseline positive rate 0.072)
```

**Say this:** "RMSE of 0.666 log units means a typical prediction is off by a
factor of about 4.6 in actual concentration. That's honest — this model is for
ranking and triage, not for precise numbers. And PR-AUC has to be read against
the base rate: 0.358 against a 0.072 baseline is about five times better than
chance, even though it looks small next to the ROC-AUC."

**Every property carries a `provenance` field** — `model`, `computed`, or
`unavailable`. Two properties are deliberately marked unavailable; see §4.

### Module 3 — Conformal Prediction

Turns the model's rough uncertainty into an interval with a real guarantee.

1. Take 168 molecules the model never trained on (the calibration set).
2. Record how wrong it was on each, scaled by how unsure it claimed to be.
3. Take the ⌈(n+1)(1−α)⌉-th smallest of those scores — the 153rd of 168.
4. New molecule's interval = prediction ± that quantile × its spread.

| Nominal | Measured coverage | Mean interval width |
|---|---|---|
| 80% | 78.3% | 1.60 logS |
| 90% | 88.4% | 2.14 logS |
| 95% | 99.3% | 3.36 logS |

**Say this, and say it carefully:** "A 90% interval does *not* mean this molecule
has a 90% chance of falling inside it. It means that across many molecules, about
90% of such intervals contain the truth. That's called marginal coverage. And it
assumes the new molecule is exchangeable with my calibration set — which is
exactly why the applicability domain is a *separate* check."

**If asked about the 95% row:** "It over-covers. With only 168 calibration
points, the 95% quantile is set by a handful of the worst residuals, so it ends
up conservative. The 80% and 90% levels land within two points of target."

### Module 4 — Applicability Domain

Two independent checks, with thresholds **derived from the training data**, not
asserted:

| Check | Threshold | Where it comes from |
|---|---|---|
| Max ECFP4 Tanimoto similarity | 0.250 | 5th percentile of training leave-one-out nearest-neighbour similarity |
| Latent-space distance | 0.358 | 95th percentile of training leave-one-out distances |

Both fail → `OUT_OF_DOMAIN`. One fails → `BORDERLINE`. Neither → `IN_DOMAIN`.
On the training set itself only **0.5%** get flagged, so the engine doesn't
reject the data it was built from.

**Say this:** "Earlier versions hardcoded 'similarity below 0.55 is suspicious'.
Where does 0.55 come from? Nowhere — it was asserted. The right threshold depends
entirely on how diverse your training set is, so I derive it from the training
set's own statistics."

### Module 5 — Pareto Laboratory

> "A dominates B if A is at least as good on *every* objective and strictly
> better on at least one. If A dominates B, nobody should ever pick B. The
> molecules nothing dominates form the Pareto front — those are the real
> trade-offs. The point is it requires no invented weights."

Two rules we enforce: minimised objectives have their sign flipped, and **a
missing value raises an error rather than defaulting to zero** — because for
toxicity, which we minimise, zero is the *best possible* value.

### Module 6 — Active Learning

A Gaussian process picks which molecule to measure next, using one of four
acquisition strategies: UCB (`μ + √β·σ`), Expected Improvement, pure uncertainty
sampling, or greedy.

**Say this clearly:** "This is **simulated laboratory mode**. Clicking 'run
assay' reveals a measurement that already exists in the Delaney dataset and was
being withheld from the model. No experiment is performed. The API labels every
response `mode: simulated`. That's a legitimate way to evaluate an
active-learning strategy, and a dishonest thing to call a lab result."

### Plus — the Discovery Engine tab

Four uncertainty-aware decision tools: probabilistic Pareto, the risk slider,
the EHVI budget planner, and the molecular generator. Covered in §5 of the demo.

---

## 4. Academic honesty — what is real and what is not

Put this slide up. It is your strongest material, not your weakest.

| Capability | Status | Detail |
|---|---|---|
| Solubility (logS) | **Trained model** | Extra Trees on 781 Delaney molecules, R² = 0.903 |
| Toxicity (CT_TOX) | **Trained model** | Random Forest, class-weighted, ROC-AUC = 0.832 |
| QED drug-likeness | **Deterministic calculation** | RDKit implementation of Bickerton et al. 2012 |
| Conformal intervals | **Real, verified** | Coverage measured on held-out data |
| Applicability domain | **Real, data-derived thresholds** | Not literature constants |
| Active learning | **Real GP** | BoTorch `SingleTaskGP`, four acquisition functions |
| EHVI planner | **Real, cross-verified** | Agrees with BoTorch's own `qEHVI` to 0.24% |
| D-MPNN ensemble | **Real, matched the baseline** | R² = 0.902 vs 0.903 — honest null result |
| Molecular generator | **Real, data-limited** | Validity 46–78%; MOSES not bundled |
| **Target bioactivity (pIC50)** | **Unavailable — no model** | Removed; was invented arithmetic |
| **Metabolic stability** | **Unavailable — no model** | Removed; was invented arithmetic |
| Simulated assays | **Labelled as simulated** | Reveal withheld measurements |
| Docking, retrosynthesis | **Not implemented** | Blocked on external tools |

**The line to say about the last two unavailable rows:**

> "The original code returned a pIC50 from this formula:
> `clip(5.5 + 2.5·QED + 1.0·(similarity − 0.5), 4, 9.8)`. It was never fitted to
> any data — the coefficients were chosen by hand. It wasn't target-specific at
> all, and its similarity term measured resemblance to a *solubility* dataset,
> which carries no information about binding affinity. It was being served
> alongside genuine predictions with nothing to distinguish them, and it was
> feeding into the Pareto optimisation.
>
> I removed it. Those properties now return `status: unavailable` with a reason
> and a note on what data would be needed to enable them. I'd rather show you a
> gap than a number I can't defend."

---

## 5. Demo script, step by step (10 minutes)

Start the server beforehand — the first launch trains and caches models (~3
minutes), every launch after is about a second.

```bash
python run.py        # then open http://127.0.0.1:8000
```

### Step 1 — Molecular Analyzer (1 min)

Paste aspirin: `CC(=O)Oc1ccccc1C(=O)O`

> "SMILES in, validated and canonicalised by RDKit. Canonicalisation matters
> more than it sounds — the same molecule can be written several ways, and if
> you're not careful the same molecule lands in both your training and test sets.
> The raw Delaney file had 11 such duplicates and they were being split across
> partitions. I deduplicate before splitting."

Show the 2D, rotate the 3D, point at the descriptors and the Lipinski verdict.

### Step 2 — Property Prediction (2 min)

Same molecule.

> "Here's the prediction — but look at the **90% interval**, not the number.
> It spans about two log units. That width is the honest state of property
> prediction from 1,100 training molecules, and it's exactly what a bare
> prediction hides from you."

Point at the provenance badges. Then scroll to the two unavailable cards.

> "These two are marked unavailable. They used to return invented arithmetic.
> That's the slide I showed you earlier."

Scroll to the D-MPNN cross-check card.

> "And here's a second, completely different model — a graph neural network —
> predicting the same molecule. When two different model families on two
> different representations agree, that's weak evidence the prediction isn't an
> artefact of how I encoded the molecule. When they disagree by more than the
> network's own σ, that's the interesting signal."

### Step 3 — Applicability Domain (1.5 min) ← **your best story**

Paste a 36-carbon chain: `CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC`

> "Watch this. Tanimoto similarity to the training set comes back as **1.000** —
> perfect. Why? Fingerprints record *local* atom environments, and every carbon
> in a long chain has exactly the same environment as every carbon in a short
> one. The fingerprint of this is bit-identical to hexane. **The similarity test
> is completely blind here.**
>
> But the learned latent distance is 0.481 against a threshold of 0.358, so the
> second check catches it and the molecule is correctly flagged BORDERLINE. That
> is the single clearest argument for having two independent diagnostics instead
> of one."

### Step 4 — Pareto Laboratory (1 min)

> "Three objectives: maximise solubility, maximise drug-likeness, minimise
> toxicity. The front is the set nothing dominates. Pick any two molecules on it
> and neither beats the other — you need a human preference to choose, which is
> the point of a human-in-the-loop system."

### Step 5 — Discovery Engine (3 min)

**Probabilistic Pareto:**

> "The deterministic front treats each prediction as a fact. But every objective
> has an interval, so a molecule just inside the front with tight intervals and
> one just outside with wide intervals are being separated by noise. So instead:
> sample all candidates jointly 500 times, and report the *fraction* of draws in
> which each is non-dominated."

Point at the amber warning.

> "And here's the honest bit. To sample, I have to impose a bell-curve shape on
> intervals that conformal prediction deliberately leaves distribution-free. I
> *tested* that assumption on my actual residuals — excess kurtosis 1.416,
> normality rejected. Real errors have heavier tails than a bell curve, so these
> probabilities are mildly optimistic for molecules whose membership depends on a
> rare large error. The interface tells you that rather than hiding it."

**Risk slider:** drag λ from 0 to 1.

> "λ = 0 ranks by the prediction alone — play safe. λ = 1 gives uncertainty full
> weight — favour molecules that *might* be excellent. Note the sign flips for
> minimised objectives: being optimistic about toxicity means assuming a *lower*
> number."

**Budget planner:** set a budget, build the plan.

> "Assays have costs. This picks molecule–assay pairs to maximise Expected
> Hypervolume Improvement — how much we expect the trade-off front to improve —
> subject to the budget."

Then open `/api/planner/validate-ehvi` in a new tab.

> "And because it's easy to write a `μ + kσ` score and *call* it EHVI, I verified
> it two ways. In one dimension EHVI must reduce to Expected Improvement, which
> has a closed form, and mine matches it. And it agrees with BoTorch's own qEHVI
> to 0.24% on a shared posterior. This endpoint runs that comparison live."

**Generator:** generate at temperature 0.8.

> "A character-level language model that learned to write SMILES, then tuned by
> reinforcement learning. Validity is around 60% here, and I know exactly why:
> the method calls for pre-training on 1.9 million MOSES molecules and I trained
> on the 2,512 I have locally. That's 750× less data. It's a data limitation, not
> a code limitation — `scripts/download_moses.py` documents how to close it."

### Step 6 — Active Learning (1 min)

Select the top-ranked candidate, run the assay.

> "The ranking updates because the Gaussian process has been refit on the
> enlarged labelled set, and the conformal calibration is refreshed too —
> reusing the old quantile after the data changed would silently misstate
> coverage. Again: simulated, revealing a withheld measurement."

### Step 7 — Benchmarking (30 sec)

> "Four models on the same held-out molecules. And I want to point at this row
> deliberately: the graph neural network got R² 0.902 against the Extra Trees
> baseline's 0.903. **It did not win.** That's the expected result at 1,100
> training molecules — the literature puts the crossover much higher. I kept it
> for the calibrated per-molecule uncertainty, the learned latent space that
> caught that long alkane, and a multi-task gain that lifted toxicity ROC-AUC
> from 0.832 to 0.866. Not for regression accuracy."

Every number on this page is computed at startup from a held-out split. Nothing
is hardcoded — and the report downloads as JSON or CSV.

---

## 6. Likely questions, with answers you can defend

**Q: "Why conformal prediction rather than a Bayesian neural network or MC dropout?"**
> "Conformal prediction is distribution-free and model-agnostic — it wraps any
> model and gives a finite-sample coverage guarantee that I can *measure*. A
> Bayesian neural network's credible intervals are only as good as its prior and
> its approximate inference, and MC dropout's uncertainty has no coverage
> guarantee at all. I can show you empirically that my 90% intervals cover 88.4%
> of held-out molecules. Neither alternative lets me make that check so directly."

**Q: "What's the assumption behind conformal prediction, and when does it break?"**
> "Exchangeability — the new molecule has to come from the same distribution as
> the calibration set. It breaks exactly when you move to new chemistry, which
> is the common case in drug discovery. That's not a flaw I can patch inside
> conformal prediction, so I built the applicability domain as an independent
> check on whether the assumption is plausible for a given query."

**Q: "Why Extra Trees instead of a deep learning model?"**
> "I built both. Extra Trees gets R² 0.903; the D-MPNN ensemble gets 0.902 on
> the same held-out molecules. Tree ensembles are extremely strong on small
> tabular datasets, and 1,100 molecules is small. I kept the neural network for
> three things trees can't give me — a calibrated per-molecule σ from a model
> trained to predict its own error, a learned latent space for the applicability
> domain, and a multi-task gain on toxicity — but I'm not going to claim it beat
> the baseline when it didn't."

**Q: "What features did you use?"**
> "1024-bit Morgan fingerprints at radius 2, plus eight physicochemical
> descriptors — 1032 numbers. And a warning from experience: those are on wildly
> different scales. Fingerprint bits are 0 or 1, molecular weight is in the
> hundreds. On the raw matrix, molecular weight accounts for 90.9% of all
> variance. I hit that bug twice in two different modules before I learned to
> standardise everything that's scale-sensitive."

**Q: "How does active learning balance exploration and exploitation?"**
> "Four strategies. UCB scores `μ + √β·σ`, so β tunes the balance directly.
> Expected Improvement computes the expected gain over the best measurement so
> far. Uncertainty sampling uses σ alone — pure exploration. Greedy uses μ alone
> — pure exploitation. The Discovery Engine's risk slider exposes the same
> trade-off continuously through λ."

**Q: "Is your EHVI really EHVI?"**
> "Yes, and I verified it rather than asserting it. Two independent checks: in
> one dimension EHVI reduces analytically to Expected Improvement and mine
> matches the closed form to within sampling error — a `μ + kσ` heuristic cannot
> reproduce that curve. And it agrees with BoTorch's own implementation to 0.24%
> on a shared posterior. There's a live endpoint that runs the comparison."

**Q: "Why not just use BoTorch's qEHVI directly?"**
> "Because qEHVI integrates over a Gaussian process posterior, and my
> uncertainties don't come from a GP — they come from a neural ensemble
> calibrated by conformal prediction. Using qEHVI would mean fitting a GP to my
> own predictions and integrating over *that*, which throws away the calibration
> that's the whole point of the project. So I kept my calibrated distribution and
> verified my maths against theirs."

**Q: "What was actually wrong with the original code?"**
> "Three things that invalidated its claims. First, it sent the hidden answers to
> the browser — the active-learning endpoint returned the true values for every
> candidate including the supposedly hidden ones, so you could read the answer
> key in developer tools. Second, its acquisition function gave all 24 candidates
> an *identical* score — UCB returned 1.414 for every one — so the ranking was
> meaningless. I traced that to PCA on unstandardised features: components
> spanned −113 to +178, BoTorch's Gaussian process assumes inputs near 0 to 1, so
> the posterior collapsed to the prior. Third, two of its four objectives were
> hand-written arithmetic presented as trained models. All three are fixed with
> regression tests so they can't silently come back."

**Q: "How do you know your tests are meaningful?"**
> "309 tests, and the important ones check mathematics against independent
> ground truth rather than against themselves. The graph network's message
> passing is recomputed by brute force over every edge and compared to the fast
> version. EHVI is checked against a closed-form formula and against BoTorch.
> Conformal coverage is verified on synthetic data where I know the answer. And
> four bugs were found by the tests during development, including that my
> ensemble wasn't reproducible for a fixed seed because I was seeding after
> constructing the model instead of before."

**Q: "What are you least confident about?"**
> "The scaffold-split result. R² 0.890 versus 0.903 on a random split reads as
> though the model generalises well to new chemical series. But ESOL contains
> many chain-like molecules with no rings, and each becomes its own
> single-member scaffold group, so for that portion the scaffold split degenerates
> towards a random one. I would not claim strong generalisation to new chemotypes
> from that number."

**Q: "What would you do next?"**
> "Not add modules — I deliberately stopped. Three things: get MOSES so the
> generator's validity reflects the method rather than my data budget; run
> multiple seeds so I can put a confidence interval on R² instead of quoting one
> number; and find a curated bioactivity dataset so pIC50 can come back as a
> genuine model rather than staying marked unavailable."

---

## 7. Numbers to memorise

| Quantity | Value |
|---|---|
| ESOL molecules (after dedup) | 1,117 (from 1,128; **11 duplicates removed**) |
| Split | 781 train / 168 calibration / 138 test / 30 AL pool |
| Solubility R², RMSE | **0.903**, 0.666 |
| Scaffold-split R² | 0.890 |
| 90% conformal coverage, width | **88.4%**, 2.14 logS |
| ClinTox ROC-AUC / PR-AUC / baseline | 0.832 / 0.358 / 0.072 |
| D-MPNN R² (5-member ensemble) | **0.902** — matched, did not beat |
| D-MPNN multi-task toxicity ROC-AUC | 0.866 (up from 0.832) |
| Independent Random Forest R² | 0.877, agreement r = 0.944 |
| AD thresholds (similarity / latent) | 0.250 / 0.358 |
| Training molecules flagged out-of-domain | 0.5% |
| EHVI agreement with BoTorch | **0.24%** |
| Residual excess kurtosis | 1.416 (normality rejected) |
| Generator corpus / validity | 2,512 SMILES / 46–78% |
| Tests | **309**, all passing |

---

## 8. Five things never to say

1. ~~"The model is 90% accurate."~~ → "90% of intervals contain the truth, on average across molecules."
2. ~~"We ran assays."~~ → "Simulated assays that reveal withheld measurements."
3. ~~"Our neural network outperforms the baseline."~~ → "It matched it — 0.902 versus 0.903 — which is the expected result at this data scale."
4. ~~"We predict bioactivity."~~ → "Bioactivity is marked unavailable; we have no dataset for it."
5. ~~"The molecule is 85% likely to be on the Pareto front."~~ → "In 85% of sampled outcomes it was non-dominated *by this candidate pool, under this model*."
