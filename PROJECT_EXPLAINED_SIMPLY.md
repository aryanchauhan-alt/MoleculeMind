# MoleculeMind — Explained Simply

**No chemistry background needed. No machine learning background needed.**

Every hard word is explained the moment it appears. Read it in order. If a
sentence doesn't make sense, it's a problem with this document, not with you.

At the end of each part there are two boxes: **What we used** (the tools) and
**Where it lives** (the files). Your teacher asked for those specifically.

---

## First, the one-paragraph version

Scientists who invent medicines have a problem. There are more possible
medicine-like molecules than there are atoms in the galaxy, and testing just one
of them in a laboratory costs money and takes days. So they can only test a
handful. Our software looks at molecules on a computer, predicts how they'll
behave, and helps the scientist decide **which few to test next**. The important
part — and the whole point of this project — is that our software also says
**how much it trusts its own answers**, and admits when it doesn't know.

That's it. That's the project.

---

## The one idea that matters most

Imagine two weather apps.

- App A says: **"Tomorrow: 20°C."**
- App B says: **"Tomorrow: 20°C, give or take 2 degrees. And by the way, you're
  in Antarctica — I've never seen weather like this, so don't rely on me."**

App B is more useful, even though App A sounds more confident.

**Our project is App B.** Most molecule-prediction software is App A. If you
remember one thing, remember this.

---

# Part 1 — Teaching a computer to read molecules

## 1.1 The problem

A molecule is a group of atoms joined by bonds. Water is two hydrogens and one
oxygen. Aspirin has 21 atoms. A computer can't look at a drawing, so we need to
write molecules as text.

## 1.2 SMILES — molecules written as text

**SMILES** is a way of typing a molecule as a line of text. It's just a code.

| You type | You get | How to read it |
|---|---|---|
| `CCO` | alcohol (ethanol) | C = carbon, O = oxygen. So: carbon–carbon–oxygen |
| `c1ccccc1` | a benzene ring | the `1`s mean "this is where the ring closes" |
| `CC(=O)Oc1ccccc1C(=O)O` | aspirin | `(...)` is a branch, `=O` is a double-bonded oxygen |

You don't need to be able to write SMILES. You need to know that **it's text, and
text is something a computer can handle.**

## 1.3 The same molecule, two spellings — a trap

Here's something important. `CCO` and `OCC` are the **same molecule** written
backwards. Both are alcohol.

Why does that matter? Imagine you're a teacher making an exam. You accidentally
put a question on the exam that was already in the homework. A student who
memorised the homework gets it right without understanding anything. Their score
is a lie.

The same thing happens here. If the same molecule appears in our "study" data
and our "exam" data under two different spellings, our software looks cleverer
than it is.

**So the first thing we do with every molecule is convert it to one standard
spelling.** This is called *canonicalising*.

> **We actually found this bug.** Our dataset had 1,128 molecules. When we
> converted them all to standard spelling, 11 turned out to be duplicates of
> others. They were being split between study and exam data. We removed them,
> leaving 1,117.

## 1.4 Turning a molecule into numbers

A computer model can't learn from text. It needs numbers. We produce two kinds.

### Kind 1: Descriptors — simple facts about the molecule

A **descriptor** is one number describing the molecule. We calculate 8:

| Descriptor | In plain words |
|---|---|
| Molecular weight | How heavy it is |
| LogP | Does it prefer oil or water? |
| TPSA | How much of its surface is "sticky" to water |
| H-bond donors / acceptors | How many weak bonds it can form with other molecules |
| Rotatable bonds | How bendy and floppy it is |
| Ring count | How many rings it has |
| Heavy atom count | How many atoms (not counting hydrogen) |

### Kind 2: Fingerprints — a checklist of parts

Think about describing a face. You could tick boxes on a list: *glasses? beard?
curly hair? brown eyes?* Two people who tick similar boxes look similar.

A **molecular fingerprint** does exactly that for molecules. The computer looks
at every atom and its immediate neighbours, and ticks a box for each little
pattern it finds. We use a list of **1024 boxes**. Each box is 1 (pattern
present) or 0 (absent).

Two molecules that tick many of the same boxes are chemically similar. We measure
that overlap with a number between 0 and 1 called **Tanimoto similarity**:

- 1.0 = identical checklists
- 0.0 = nothing in common

**So each molecule becomes 1024 checkboxes + 8 facts = 1032 numbers.**

## 1.5 A mistake we made twice

Those 1032 numbers are on wildly different scales. The checkboxes are 0 or 1.
Molecular weight is a number like 180 or 350.

Imagine averaging someone's height in millimetres (1,750) with their age (20).
The height completely swamps the age — the average is basically just the height.

The same thing happened to us. We measured it: molecular weight alone accounted
for **90.9%** of all the variation in our numbers. Anything that cared about
distance or spread was effectively ignoring the chemistry and just asking "is
this molecule heavy?"

**The fix** is called *standardising*: for each of the 1032 numbers, subtract its
average and divide by its spread. Now everything is on a comparable scale.

We hit this bug in **two different parts of the project** before we learned to
always standardise first. It's the single most instructive mistake in here.

> **What we used**
> **RDKit** — a free, widely-used chemistry library. It reads SMILES, rejects
> nonsense, draws molecule pictures, works out 3D shapes, and calculates all the
> descriptors and fingerprints. We never wrote chemistry code ourselves; we call
> RDKit for all of it.

> **Where it lives**
> `backend/app/core/mol_analyzer.py` — reading SMILES, calculating descriptors,
> making pictures (148 lines, the simplest file in the project — **read this one
> first**)
> `backend/app/core/dataset.py` — loading data, removing duplicates, turning
> molecules into the 1032 numbers

---

# Part 2 — Making a prediction

## 2.1 What we predict

**Water solubility** — how well a molecule dissolves in water. This matters
because a medicine that won't dissolve can't get into your blood.

It's measured on a scale called **logS**. Roughly:

- `+1` = dissolves very easily (like sugar)
- `−2` = dissolves a bit
- `−6` = basically won't dissolve (like candle wax)

## 2.2 The data

We use a public dataset called **Delaney ESOL**: 1,128 molecules whose real
solubility was measured in actual laboratories and published in 2004. After
removing the 11 duplicates: **1,117 molecules**.

This is real experimental data. We did not make it up or simulate it.

## 2.3 Splitting the data — the most important step

We divide our 1,117 molecules into four groups that **never overlap**:

| Group | How many | What it's for |
|---|---|---|
| **Training** | 781 | The computer learns from these |
| **Calibration** | 168 | Used to work out how wrong the computer usually is (Part 3) |
| **Test** | 138 | The final exam. We look at these **once**, at the end |
| **Hidden pool** | 30 | Pretend-unknown molecules for Part 6 |

**Why this matters so much:** going back to the exam analogy — if a molecule
appears in both the training group and the test group, we're testing the computer
on something it already memorised. Every number we report afterwards would be
meaningless. In machine learning this mistake is called **data leakage**, and
it's the most common way these projects go wrong.

We have an automatic check that fails loudly if the four groups ever overlap.

## 2.4 The model — what actually does the predicting

We use something called **Extra Trees**. Here's the idea, built up in three
steps:

**Step 1: a decision tree.** A flowchart of yes/no questions.

```
Is molecular weight above 300?
├── Yes → Does it have a benzene ring?
│         ├── Yes → predict logS = −4.2
│         └── No  → predict logS = −3.1
└── No  → Is LogP above 2?
          ├── Yes → predict logS = −2.0
          └── No  → predict logS = −0.5
```

The computer builds this flowchart automatically by finding which questions best
separate soluble molecules from insoluble ones.

**Step 2: a forest.** One flowchart is unreliable — it might latch onto a quirk
of the training data. So we build **150 different flowcharts**, each from a
slightly different angle, and average their answers. Like asking 150 people and
taking the average rather than trusting one person.

**Step 3: "Extra".** Each tree picks its question thresholds partly at random.
Counter-intuitively this makes the group *more* reliable, because the trees make
different mistakes that cancel out.

**Why not a neural network?** We built one too (Part 8) — it didn't do better.
Forests of trees are very hard to beat on small datasets, and 1,117 molecules is
small.

## 2.5 How well does it work?

On the 138 test molecules it had never seen:

```
R² = 0.903        RMSE = 0.666        MAE = 0.484
```

What those mean:

- **R² (say "R squared")** — how much of the variation the model explains.
  **1.0 = perfect, 0.0 = useless** (no better than always guessing the average).
  **0.903 is good.**
- **RMSE** — the typical size of the mistake, in logS units. Ours is 0.666.
  **Be honest about this:** 0.666 on a log scale means a typical prediction is
  off by a factor of about 4.6 in the real amount that dissolves. This model is
  good for **sorting molecules into better and worse**, not for exact numbers.
- **MAE** — the plain average mistake size. Similar idea, less affected by a few
  very bad predictions.

## 2.6 An easy exam and a hard exam

Our test above was a **random split** — we shuffled the molecules and dealt them
out. But molecules come in families, like brothers and sisters. A random shuffle
can put one brother in training and the other in the test, so the computer only
has to recognise a close relative. That's easy.

A **scaffold split** keeps whole families together on one side. Now the computer
must handle chemistry it has genuinely never seen. Much harder, and much closer
to real use.

```
Random split (easy exam):    R² = 0.903
Scaffold split (hard exam):  R² = 0.890
```

**Don't oversell this.** The scores look close, but that's partly an accident: our
dataset has lots of simple chain-like molecules with no rings at all, and each of
those counts as a family of one — so for those, the "hard" exam isn't actually
hard. We should not claim our model generalises brilliantly to new chemistry on
this evidence.

## 2.7 The second model — toxicity

We also predict toxicity risk, using a public dataset called **ClinTox**: 1,461
molecules, of which 103 are flagged.

**Read the label carefully.** The flag means *"this compound's clinical trial was
stopped for safety reasons"*. It does **not** mean "this molecule is poisonous".
Aspirin is flagged in this dataset.

Scores:

```
ROC-AUC = 0.832        PR-AUC = 0.358   (random guessing would get 0.072)
```

- **ROC-AUC** — the chance the model correctly ranks a flagged molecule above an
  unflagged one. 0.5 = coin flip, 1.0 = perfect. 0.832 is decent.
- **PR-AUC** — must be compared against **what random guessing would score**.
  Only 7% of our molecules are flagged, so random guessing scores 0.072. We score
  0.358, which is about 5× better than chance. It looks like a low number but
  isn't.

Because only 7% are flagged, we tell the model to treat the rare flagged cases as
more important — otherwise it could score 93% just by always saying "not toxic",
which would be useless.

> **What we used**
> **scikit-learn** — the standard free machine learning library for Python. It
> provides Extra Trees, Random Forest, and all the scoring maths (R², RMSE,
> ROC-AUC).
> **pandas** — reads the data files.
> **NumPy** — does the number crunching.

> **Where it lives**
> `backend/app/core/models.py` — trains both models, saves them, makes
> predictions

---

# Part 3 — Saying how sure we are

## 3.1 Why a single number isn't enough

"Solubility = −2.1" tells a scientist nothing about risk. Is it −2.1 give or take
0.1, or give or take 2.0? Those two situations lead to completely different
decisions about whether to spend money on a lab test.

## 3.2 The simple idea

Here's the trick, and it's genuinely simple:

> **Check how wrong you usually are, then quote that as your margin of error.**

Like a bus that's typically 5 minutes late. You don't say "the bus arrives at
3:00". You say "the bus arrives at 3:00, give or take 5 minutes" — because you
watched it for a month and that's what you measured.

## 3.3 How we do it — conformal prediction

**Conformal prediction** is the proper name. Four steps:

1. Take the **168 calibration molecules**. The model has never trained on these,
   but we know their real answers.
2. For each one, record **how wrong the model was**.
3. Sort those 168 errors from smallest to largest. Look at the one 90% of the way
   up the list. Call it the *margin*.
4. For any new molecule: **answer = prediction ± margin.**

Why does this work? If the new molecule is similar in nature to our 168
calibration molecules, its error should be about the same size as theirs. So 90%
of the time, the truth lands inside.

(One technical detail, for completeness: we don't take exactly the 90th item. We
take the 153rd of 168, which comes from a formula that makes the guarantee exact
rather than approximate when you only have 168 examples.)

## 3.4 Does it actually work? We measured it

| We promised | We actually got | How wide the range was |
|---|---|---|
| 80% | 78.3% | 1.60 logS |
| 90% | 88.4% | 2.14 logS |
| 95% | 99.3% | 3.36 logS |

The 80% and 90% rows land within 2 points of the promise. **This is a real
result, measured on molecules the model never saw.**

Look at the **widths** though. Our 90% range is 2.14 logS units wide. That's
wide! But that *is the honest truth* about predicting molecule properties from
1,117 examples. A single number hides that from you; we show it.

(The 95% row over-covers — we promised 95% and got 99.3%. That's because with
only 168 calibration molecules, the 95% margin is decided by just a few of the
very worst cases, so it ends up overly cautious. Being cautious isn't wrong, but
we report it rather than hide it.)

## 3.5 What the guarantee does NOT mean — important

This is the thing most likely to be asked, and most likely to be got wrong.

> A 90% range does **not** mean "there's a 90% chance this molecule's real answer
> is in this range."

It means: **across lots of molecules, about 90% of the ranges we give will
contain the truth.** Some individual molecules get ranges that are too generous,
others too tight. The promise is about the average, not about your particular
molecule.

It also only holds if the new molecule is **the same kind of thing** as our
calibration molecules. If you hand it something wildly different, the promise
simply doesn't apply. Which is exactly what Part 4 is for.

> **What we used**
> Nothing! We wrote this ourselves from the mathematics. It's about 400 lines of
> Python using only NumPy. There are libraries that do it (one is called MAPIE),
> but the maths is simple enough to implement directly and we wanted to be sure
> it was right.

> **Where it lives**
> `backend/app/core/conformal.py`

---

# Part 4 — Knowing when to say "I don't know"

## 4.1 The problem

A doctor who has only ever treated adults shouldn't confidently prescribe for a
newborn. But a computer model has no such instinct — hand it anything and it
produces a confident-looking number.

So we add a separate check: **is this molecule the kind of thing our model
actually knows about?** The technical name is the **applicability domain**.

## 4.2 Two separate tests

**Test 1 — "Have I seen anything like this?"**
Compare the new molecule's fingerprint checklist against all 781 training
molecules. Find the most similar one. If even the closest match is barely
similar, we're guessing.

**Test 2 — "Is this an unusual molecule overall?"**
Check whether the molecule sits far from the middle of the cloud of training
molecules.

> This is where we hit the scaling bug the second time. Before standardising,
> Test 2 was effectively just asking "is this molecule unusually heavy?"

## 4.3 Where our cut-off numbers come from

An earlier version of this project said "similarity below 0.55 is suspicious".
Where did 0.55 come from? **Nowhere.** Someone just picked it.

We do it properly. We ask: *for each of our 781 training molecules, how similar
is its own closest neighbour?* That gives us 781 numbers. We take the value that
95% of them beat: **0.250**.

So the cut-off is **measured from our own data**, not invented. If a new molecule
scores below 0.250, it's less typical of our training set than nearly every
training molecule is.

## 4.4 The verdict

- **Both tests fail** → `OUT OF DOMAIN` (don't trust this)
- **One test fails** → `BORDERLINE` (be careful)
- **Neither fails** → `IN DOMAIN` (reasonable to trust)

Sanity check: when we run this on our own training molecules, only **0.5%** get
flagged. Good — the system doesn't reject the very data it learned from.

## 4.5 The best story in the project

Type a long chain of 36 carbons into the app:
`CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC`

**Test 1 reports similarity 1.000 — a perfect match.** How?

Remember fingerprints tick boxes for *small local patterns*. Every carbon in a
long chain has exactly the same neighbours as every carbon in a short chain. So
the checklist for a 36-carbon chain is **identical** to the checklist for a
6-carbon chain. **Test 1 is completely blind here.**

But Test 2 catches it, and the molecule is correctly flagged `BORDERLINE`.

**This is the clearest reason why we use two independent tests instead of one.**
If you demonstrate one thing to your teacher, demonstrate this.

> **What we used**
> RDKit (for the fingerprints and similarity) and scikit-learn (for the
> statistical part). The threshold-deriving logic we wrote ourselves.

> **Where it lives**
> `backend/app/core/applicability_domain.py`

---

# Part 5 — Wanting several things at once

## 5.1 The problem

A medicine needs to dissolve well **and** not be toxic **and** be drug-like.
These fight each other — changes that help one often hurt another.

You *could* mash them into one score, like "solubility counts double". But then
you've invented those weights out of thin air, and someone will rightly ask why.

## 5.2 The phone-shopping idea

You're buying a phone. You care about **low price** and **good camera**.

| Phone | Price | Camera |
|---|---|---|
| A | £200 | ★★ |
| B | £800 | ★★★★★ |
| C | £400 | ★★★ |
| D | £800 | ★★ |

**Phone D is clearly silly** — it costs the same as B but has a worse camera.
Nobody should buy D. B beats it outright.

But A, B and C? You genuinely can't say which is best without knowing how much
*you* care about price versus camera. They're all reasonable choices.

That's the whole idea:

> **A beats B** ("dominates" it) if A is at least as good on *everything* and
> better on *at least one thing*. The options nobody beats are the **real
> choices**. Everything else can be thrown away.

The set of real choices is called the **Pareto front**. We find it for molecules
exactly as you'd find it for phones, and we need no invented weights.

## 5.3 Two rules we had to get right

**Direction.** More solubility is good. More toxicity is bad. We flip the toxicity
numbers so that "bigger is better" everywhere. Get this backwards and every
conclusion inverts.

**Missing values are an error, never a guess.** The old version of this project
filled in `0` when a value was missing. For toxicity — where lower is better —
`0` is the **best possible score**! So a molecule with no toxicity data was being
promoted as a top choice because of data it didn't have. We now raise an error
instead.

## 5.4 The improved version — probabilities instead of yes/no

Here's a subtlety. Every number in that table is a *prediction with a margin of
error*. So a molecule just inside the "real choices" list and one just outside
might only be separated by noise.

So we ask a better question: **how likely is this molecule to be a real choice?**

How we do it:

1. Take each prediction and its margin of error.
2. Roll dice to pick a plausible value for **every** molecule at once.
3. Work out which molecules are "real choices" this time.
4. Repeat 500 times.
5. A molecule's score = **the fraction of the 500 rounds where it was a real
   choice.**

A molecule that's a real choice in 487 of 500 rounds scores 0.974. One that
scraped in 150 times scores 0.300. Far more useful than a yes/no.

## 5.5 Something we're honest about

To "roll dice" we have to assume the errors follow a standard bell curve. But we
checked, and **on our actual data they don't** — real errors have more extreme
cases than a bell curve predicts.

So we added an **orange warning in the app** that says so. Our probabilities are
slightly too optimistic for molecules whose status depends on a rare big error.

We also *refuse* to offer one feature a user might want (accounting for errors in
two properties moving together), because we'd need data from molecules with both
measurements and we only have 64 of those — far too few. **Rather than invent a
number, the app returns an error explaining why.**

## 5.6 The risk slider

A slider from 0 to 1 that changes how the list is sorted:

- **0 = play safe.** Sort by the prediction alone.
- **1 = take a chance.** Favour molecules that *might* be brilliant — ones with
  big margins of error, which could turn out much better than predicted.

Drag it and the ranking updates live. Useful because early in a project you want
to take chances, and late on you want safe bets.

> **What we used**
> NumPy and SciPy (standard maths libraries). The Pareto logic we wrote ourselves
> — it's about 120 lines.

> **Where it lives**
> `backend/app/core/pareto.py` — the basic version
> `backend/app/core/probabilistic_pareto.py` — the dice-rolling version and the
> slider

---

# Part 6 — Choosing what to test next

## 6.1 The setup

We take 30 molecules whose real measured answers exist in our dataset but which
the model has **never seen**. We reveal 6 at the start. The other 24 stay hidden.

The question: **which should we "test" next?**

This is called **active learning** — the model chooses its own next lesson.

## 6.2 The four strategies

We use a model (a **Gaussian process**) that gives both a prediction and a
genuine "how unsure am I" number that grows the further you get from things it's
seen.

| Strategy | What it picks | Like a student who... |
|---|---|---|
| **Greedy** | highest prediction | only revises what they already know |
| **Uncertainty** | most unsure | only studies what confuses them |
| **UCB** | a mix of both | sensibly balances the two |
| **Expected Improvement** | best chance of beating the current best | focuses on what could improve their grade |

## 6.3 Say this clearly: it's a simulation

Clicking "run assay" **reveals an answer that was already in our dataset and
being hidden from the model.** No laboratory experiment happens. The software
labels every response `mode: "simulated"`.

This is a perfectly respectable way to test whether a strategy works — you
pretend not to know answers you do know, then check if the strategy finds the
good molecules quickly. It would be dishonest to call it a lab result, so we
don't.

## 6.4 Keeping the answers hidden properly

The hidden answers are kept in a locked box on the server. There is exactly
**one** door they can leave through, and that door checks every time that no
hidden answer is escaping. Two automatic tests search the entire web response for
leaked answers.

We built it this way because of a bug we found — see Part 9.

> **What we used**
> **BoTorch** and **GPyTorch** — libraries from Meta for this exact kind of
> "choose the next experiment" problem. They provide the Gaussian process and the
> strategies.

> **Where it lives**
> `backend/app/core/active_learning.py`

---

# Part 7 — Spending a budget

## 7.1 The problem

You have 8 units of budget. A solubility test costs 1 unit. A toxicity panel
costs 3. **Which tests should you buy?**

## 7.2 Measuring how good a set of choices is

Go back to the phones. Your "real choices" cover a certain amount of ground —
cheap-but-basic through to expensive-but-excellent. If you discovered a new phone
that was both cheaper *and* better than one of your current options, your set of
choices would cover **more ground**.

We measure that ground as an area, called **hypervolume**. Bigger area = better
set of options. One number for the quality of all your choices together.

## 7.3 Choosing the test that helps most

For each possible test we ask: **how much do we expect our area to grow if we run
this?** The proper name is **Expected Hypervolume Improvement** (EHVI).

Then: pick the test with the best improvement-per-cost, pretend we got the
predicted result, recalculate, repeat until the money runs out.

## 7.4 How we proved our maths is right

This is worth telling your teacher, because it's easy to write something that
*looks* like this calculation and is actually just "prediction + a bit of
uncertainty" in disguise.

We checked it two ways against things we couldn't influence:

**Check 1.** With only one property to optimise, this calculation must simplify
to a different, older formula that has an exact answer. We compared:

| Our answer | The exact formula |
|---|---|
| 0.31406 | 0.31522 |
| 0.99930 | 1.00003 |

A fake "prediction + uncertainty" score **cannot** produce those numbers.

**Check 2.** We compared against BoTorch's own professional implementation of the
same calculation. **Largest difference: 0.24%.**

There's a button in the app (`/api/planner/validate-ehvi`) that runs Check 2 live.
Click it in front of your teacher.

## 7.5 What we admit about it

Our method picks the best test, then the next best, and so on. That's called
**greedy** and it's **not guaranteed to find the perfect plan** — occasionally
two slightly-worse tests together would beat the best one. We say so in the
software's own output rather than pretending otherwise.

> **What we used**
> **BoTorch** — for the exact area calculation, which is fiddly geometry we
> didn't want to get wrong. The expected-improvement part we wrote ourselves, so
> it works with our own margins of error.

> **Where it lives**
> `backend/app/core/experiment_planner.py`

---

# Part 8 — The two extra models (and an honest result)

## 8.1 The neural network

Fingerprints are a **fixed** checklist — somebody decided in advance which
patterns to count. A **neural network** can instead *work out for itself* which
shapes matter.

Ours treats the molecule as a map: atoms are towns, bonds are roads. Then it
passes messages along the roads. Each town gathers what its neighbours tell it,
adds its own information, and passes it on. After three rounds, every town knows
about its wider neighbourhood. Average all the towns together and you have a
description of the whole molecule — one the computer invented itself.

The technical name is **D-MPNN** (Directed Message Passing Neural Network).

**One clever detail:** when a town passes a message along a road, it leaves out
whatever came *in* on that same road. Otherwise messages bounce back and forth
and the network just hears its own echo. That's what the "Directed" means.

We verified this is correct by recalculating it the slow, obvious way — looping
over every road by hand — and checking it matches.

## 8.2 The honest result

```
Extra Trees (the simple forest):  R² = 0.903
Neural network:                   R² = 0.902
```

**The neural network did not win.**

This is not a failure of our code — it's the expected result. Neural networks
need tens of thousands of examples to beat a good forest. We have 1,117.

**So why keep it?** Three things the forest can't give us:

1. A proper "how unsure am I" number for each individual molecule.
2. Its self-invented description of molecules, which is what **caught the
   36-carbon chain** in Part 4.5.
3. It learns solubility and toxicity *together*, and sharing knowledge between
   the two lifted the toxicity score from **0.832 to 0.866**.

Being able to say *"we built it properly, measured it honestly, it tied — and
here's why we kept it"* is a stronger position than claiming a win.

## 8.3 The molecule inventor

Since molecules are text, a model that learns to **write** text can invent
molecules — the same technology as your phone's predictive text.

Two stages:

1. **Learn the language.** Show it real molecules until it learns the grammar —
   that brackets must close, that ring numbers come in pairs.
2. **Learn to aim.** Reward it for inventing molecules our other models score
   highly, while keeping it tethered to proper-looking chemistry so it can't
   cheat by producing gibberish that happens to score well.

**One deliberate choice worth explaining:** we reward it based on the
**pessimistic end** of our margin of error, not the prediction itself. If we
rewarded the prediction, it would learn to hunt for molecules where our models
are *confused* and happen to guess high. Rewarding the pessimistic end means
uncertainty costs it something.

## 8.4 Honest results again

| Setting | How many invented molecules are valid |
|---|---|
| Cautious | 78% |
| Normal | 46–59% |

**That's poor, and we know exactly why.** The proper method says to first train
on **1.9 million** molecules from a dataset called MOSES. We don't have it (it's
130 MB and we wanted the project to work without internet), so we trained on the
**2,512** molecules we do have. **That's 750 times less data.**

The code is complete; the *data* is the gap. `scripts/download_moses.py` explains
how to fix it, and nothing else would need changing. We report what we measured,
not what the method could achieve.

> **What we used**
> **PyTorch** — the standard free library for building neural networks. Both the
> network and the molecule-inventor are written in it.

> **Where it lives**
> `backend/app/core/molecular_graph.py` — turning molecules into maps
> `backend/app/core/dmpnn.py` — the neural network
> `backend/app/core/generator.py` — the molecule inventor

---

# Part 9 — Three serious bugs we found and fixed

**This is the most valuable part of the project.** The code we started with
*looked* finished and worked when you clicked it. Three things were badly wrong.

## Bug 1 — The answers were being sent to the browser

The "hidden" molecules in Part 6 weren't hidden at all. The server was sending
their real answers to the web page for **every** molecule, including the secret
ones. Anyone could press F12 in their browser and read the complete answer key.

**Why it's serious:** the entire point is that a human and a computer work
together *without* knowing the answers. If the answers are visible, nothing is
being tested.

**Fix:** locked box, one guarded exit, automatic tests that search every response
for leaks.

## Bug 2 — Every molecule got exactly the same score

The software ranked 24 molecules by how useful they'd be to test. We measured
what it actually produced:

```
all 24 molecules scored 1.414
```

**Every single one identical.** The "ranking" was just the order they happened to
be written in the file. It looked like it was working. It was doing nothing.

**Finding the cause took real detective work.** It traces back to the scaling
problem from Part 1.5. The numbers fed to the model ranged from −113 to +178, but
that type of model expects numbers between roughly 0 and 1. Given such huge
numbers, it concluded every molecule was totally unrelated to every other, gave
up, and returned the same default answer for all of them.

**Fix:** standardise the numbers first. Now 17 of the 24 get different scores and
the ranking changes sensibly when new information arrives. A test now checks the
scores aren't all identical, so this can't come back silently.

## Bug 3 — Two "predictions" were invented arithmetic

The old code claimed to predict how strongly a molecule binds to its biological
target. Here's what it actually did:

```
answer = 5.5 + 2.5 × (drug-likeness) + 1.0 × (similarity − 0.5)
```

**That's not a model.** Nobody trained it on anything. Someone picked those
numbers by hand. Worse, the "similarity" part measured similarity to a
**solubility** dataset, which tells you nothing whatsoever about binding.

And it was being displayed right next to our genuine predictions with nothing to
tell them apart, and fed into the choose-the-best-molecule calculation.

**Fix:** deleted. Those properties now say **"unavailable — no model exists"**
with an explanation of what data would be needed. Every property we *do* report
is now tagged with where it came from: a trained model, a direct calculation, or
unavailable.

## And a few more

- 11 duplicate molecules split across study and exam data
- The toxicity model had never been scored **at all** — no test, no number
- Missing values being replaced with `0`, which for toxicity means "perfect"
- The homepage had `R² = 0.894` typed into it as text, so it couldn't ever update
- The applicability domain had the same scaling bug as Bug 2
- Margins of error never recalculated after new information arrived

---

# Part 10 — Everything we used

| Tool | What it is | What we used it for |
|---|---|---|
| **Python 3.14** | Programming language | Everything on the server |
| **RDKit** | Free chemistry library | Reading molecules, drawing them, 3D shapes, descriptors, fingerprints |
| **scikit-learn** | Free machine learning library | Extra Trees, Random Forest, the scoring maths |
| **NumPy** | Number crunching | All the maths underneath |
| **pandas** | Spreadsheet handling | Reading the data files |
| **SciPy** | Scientific maths | Statistics for the probability calculations |
| **PyTorch** | Neural network library | The neural network and the molecule-inventor |
| **BoTorch / GPyTorch** | Choose-the-next-experiment libraries | Gaussian process, strategies, exact area calculation |
| **FastAPI** | Web server framework | The 24 web addresses the app provides |
| **Uvicorn** | Web server | Actually runs it |
| **Pydantic** | Input checking | Rejects bad input before it reaches our code |
| **pytest** | Testing framework | Runs our 309 automatic checks |
| **React 19 + TypeScript** | Website framework | The six screens you click on |
| **Vite** | Website build tool | Packages the website |
| **Tailwind CSS** | Styling | How it looks |
| **Plotly.js** | Charting | The interactive graphs |
| **3Dmol.js** | 3D molecule viewer | The rotatable molecules |

**What we wrote ourselves rather than using a library:** conformal prediction
(the margins of error), the applicability-domain logic and its data-derived
cut-offs, the Pareto "real choices" logic, the probability-by-dice-rolling
version, the expected-improvement calculation, the neural network's
message-passing, and the molecule-inventor's training loop.

**Data** — all three datasets ship with the project; nothing is downloaded while
running:

| Dataset | Rows | Where it came from | Used for |
|---|---|---|---|
| Delaney ESOL | 1,128 | Published 2004, real lab measurements | Solubility, applicability domain, hidden pool |
| ClinTox | 1,484 | Published 2016, real trial records | Toxicity |
| Lipophilicity | 4,200 | From ChEMBL | Included but not used |

---

# Part 11 — Where everything lives

Read in this order. Each builds on the one before.

| Order | File | Lines | What it does |
|---|---|---|---|
| **1** | `core/mol_analyzer.py` | 148 | Molecule in, facts and pictures out. **Start here — it's short and concrete.** |
| 2 | `core/dataset.py` | 305 | Loading data, removing duplicates, splitting, making numbers |
| 3 | `core/conformal.py` | 415 | Margins of error |
| 4 | `core/applicability_domain.py` | 465 | The two "do I know this?" tests |
| 5 | `core/pareto.py` | 123 | The "real choices" logic |
| 6 | `core/models.py` | 928 | Trains the models, makes predictions |
| 7 | `core/active_learning.py` | 458 | Choosing what to test next, the locked box |
| 8 | `core/probabilistic_pareto.py` | 454 | Dice-rolling probabilities, the slider |
| 9 | `core/experiment_planner.py` | 635 | Budget planning |
| 10 | `core/molecular_graph.py` | 267 | Molecules as maps |
| 11 | `core/dmpnn.py` | 689 | The neural network |
| 12 | `core/generator.py` | 733 | The molecule inventor |
| 13 | `core/config.py` | 163 | All the settings in one place |
| 14 | `main.py` | 936 | The 24 web addresses |

**The website:** `frontend/src/components/` — six screens (Analyzer, Prediction,
Pareto, Discovery Engine, Active Learning, Benchmarking).

**The tests:** `backend/tests/` — 309 automatic checks in 9 files.

## How to run it

```bash
pip install -r requirements.txt
python run.py
```

Then open <http://127.0.0.1:8000> in a browser.

The **first** start takes about 3 minutes because it trains everything and saves
it. Every start after that takes about 1 second.

To run the tests:

```bash
python -m pytest -q
```

---

# Part 12 — The 20 words you need

| Word | Simple meaning |
|---|---|
| **SMILES** | A molecule written as a line of text |
| **RDKit** | The free chemistry library we use for everything chemical |
| **Descriptor** | One number describing a molecule (weight, bendiness…) |
| **Fingerprint** | A 1024-box checklist of which small patterns a molecule has |
| **Tanimoto similarity** | How much two checklists overlap, 0 to 1 |
| **logS** | The solubility scale. Higher = dissolves better |
| **Training data** | The molecules the computer learns from |
| **Test data** | Molecules kept back for the final exam |
| **Data leakage** | Accidentally testing on something it already studied. Ruins everything |
| **Standardising** | Rescaling numbers so big ones don't drown out small ones |
| **Decision tree** | A flowchart of yes/no questions |
| **Extra Trees** | 150 such flowcharts, averaged |
| **R²** | How much of the variation the model explains. 1 = perfect, 0 = useless |
| **RMSE** | The typical size of the mistake |
| **Conformal prediction** | Measuring how wrong you usually are, and quoting it as a margin |
| **Coverage** | How often the margin actually contains the truth |
| **Applicability domain** | The check for "is this molecule the kind of thing I know?" |
| **Pareto front** | The options nobody beats outright — the real choices |
| **Active learning** | Letting the model choose its own next lesson |
| **Hypervolume** | The "ground covered" by your set of real choices |
| **Simulated assay** | Revealing an answer we hid. **Not** a real lab experiment |

---

# Part 13 — If you can only remember five things

1. **The project helps a scientist choose which molecules to test next — and
   says how much to trust its own advice.** That second half is the point.

2. **It predicts solubility with R² = 0.903**, and gives a margin of error that
   we *measured* contains the truth 88.4% of the time when we promised 90%.

3. **It admits when it doesn't know.** Best example: the 36-carbon chain, where
   the similarity test says "perfect match" and is completely wrong, and only the
   second test catches it.

4. **We found three serious bugs in the starting code** — the answers were being
   sent to the browser, the ranking gave all 24 molecules an identical score, and
   two "predictions" were invented arithmetic with no model behind them.

5. **Where it's honest about limits:** the neural network tied with the simple
   forest rather than beating it, the molecule-inventor's 46% validity is because
   we lack the big dataset, and the app shows an orange warning where one of our
   statistical assumptions doesn't hold.
