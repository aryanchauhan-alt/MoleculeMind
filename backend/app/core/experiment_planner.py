"""
Cost-aware experiment planner built on Expected Hypervolume Improvement.

The problem
-----------
Given a candidate pool with calibrated predictive distributions, a set of assays
with per-assay costs, and a total budget `B`, choose which (molecule, assay)
pairs to run so as to advance the Pareto front as much as possible, subject to

    sum of selected assay costs <= B.

What EHVI actually is
---------------------
The hypervolume `HV(P)` of a Pareto front `P` relative to a reference point `r`
is the volume of objective space dominated by `P` and bounded by `r`. It is the
standard scalar summary of "how good is this front". Hypervolume improvement
from observing an outcome `y` is

    HVI(y) = HV(P union {y}) - HV(P)        >= 0

and Expected Hypervolume Improvement is its expectation under the predictive
distribution:

    EHVI(x) = E_{y ~ p(y|x)} [ HVI(y) ]

This module computes that expectation by Monte Carlo over the **calibrated**
predictive distribution, while computing each `HV` term **exactly** using
BoTorch's box-decomposition `Hypervolume`. So the hypervolume arithmetic is not
approximated at all; only the outer expectation is, with a reported standard
error.

Why Monte Carlo rather than BoTorch's `qEHVI`
---------------------------------------------
BoTorch's `qEHVI` / `qNEHVI` are available in this environment (0.18.1) and are
excellent, but they take a **BoTorch model** and integrate over *its* GP
posterior. Our predictive distributions do not come from a GP: they come from a
D-MPNN ensemble whose scale has been calibrated by conformal prediction. Routing
them through `qEHVI` would require fitting a GP surrogate to our own predictions
and would integrate over that surrogate's posterior instead of the calibrated
one - discarding the calibration that is the point of the platform.

The estimator here is therefore genuine EHVI by definition, evaluated against the
calibrated distribution. `validate_against_botorch_qehvi()` cross-checks it
against BoTorch's implementation on a shared GP posterior, where both apply, so
the claim is testable rather than asserted. In the single-objective case EHVI
reduces analytically to Expected Improvement, and
`test_single_objective_ehvi_equals_closed_form_ei` checks the estimator against
the closed-form EI formula.

This is NOT a weighted utility-plus-uncertainty heuristic. `mu + k*sigma` scores
are in `probabilistic_pareto.risk_adjusted_score` and are clearly labelled as a
ranking device there.

Greedy sequential selection
---------------------------
EHVI is evaluated for every (molecule, assay) pair; the best is selected; the
front is updated with an imputed outcome; and the process repeats until the
budget is exhausted or no selection offers positive improvement.

Greedy is not optimal. Hypervolume improvement is submodular in the set of
observations, so greedy selection carries the standard (1 - 1/e) guarantee
relative to the optimum **for the unit-cost case**; with heterogeneous costs the
cost-weighted greedy rule used here has no such clean bound. This is stated
rather than implied: the planner returns good plans, not provably optimal ones.

Imputation after selection
--------------------------
Once a pair is selected we do not know its outcome, but the next EHVI evaluation
needs an updated front. Two strategies:

* `"mean"` - insert the predictive mean. Optimistic: it assumes the assay returns
  exactly the prediction, so it under-rewards a second measurement in the same
  region of objective space.
* `"sample"` - insert a draw from the predictive distribution, averaged over
  several restarts. More faithful to the uncertainty but slower and stochastic.

The default is `"mean"`, and the limitation is reported in the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from scipy import stats

DEFAULT_EHVI_SAMPLES = 128


# ------------------------------------------------------------------ assays

@dataclass
class Assay:
    """One measurable endpoint with a cost and the objectives it informs."""

    name: str
    cost: float
    objectives: List[str]
    description: str = ""

    def __post_init__(self):
        if self.cost <= 0:
            raise ValueError(f"Assay '{self.name}' must have a positive cost.")
        if not self.objectives:
            raise ValueError(f"Assay '{self.name}' must inform at least one objective.")


@dataclass
class PlannedExperiment:
    candidate_id: str
    candidate_index: int
    assay: str
    cost: float
    ehvi: float
    ehvi_per_unit_cost: float
    ehvi_standard_error: float
    cumulative_cost: float
    front_hypervolume_before: float
    front_hypervolume_after_imputation: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "assay": self.assay,
            "cost": round(self.cost, 4),
            "ehvi": round(self.ehvi, 6),
            "ehvi_per_unit_cost": round(self.ehvi_per_unit_cost, 6),
            "ehvi_standard_error": round(self.ehvi_standard_error, 6),
            "cumulative_cost": round(self.cumulative_cost, 4),
            "front_hypervolume_before": round(self.front_hypervolume_before, 6),
            "front_hypervolume_after_imputation": round(
                self.front_hypervolume_after_imputation, 6
            ),
        }


# -------------------------------------------------------------- hypervolume

def _as_maximisation(Y: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    signs = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)
    return np.asarray(Y, dtype=np.float64) * signs


def compute_hypervolume(
    Y: np.ndarray, ref_point: np.ndarray, maximize: Sequence[bool]
) -> float:
    """
    Exact hypervolume of the non-dominated subset of `Y` relative to `ref_point`.

    Both are converted to a maximisation frame first. Points that do not dominate
    the reference point contribute nothing, which BoTorch handles internally.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    if Y.size == 0:
        return 0.0

    Y_max = _as_maximisation(Y, maximize)
    ref_max = _as_maximisation(np.asarray(ref_point, dtype=np.float64).reshape(1, -1), maximize)[0]

    t = torch.as_tensor(Y_max, dtype=torch.double)
    ref = torch.as_tensor(ref_max, dtype=torch.double)

    # Keep only points that strictly dominate the reference point.
    valid = (t > ref).all(dim=1)
    if not bool(valid.any()):
        return 0.0
    t = t[valid]
    front = t[is_non_dominated(t)]
    return float(Hypervolume(ref_point=ref).compute(front))


def infer_reference_point(
    Y: np.ndarray, maximize: Sequence[bool], padding: float = 0.1
) -> np.ndarray:
    """
    A reference point just outside the worst observed value per objective.

    The hypervolume value depends on this choice, so it is reported with every
    plan. Padding is relative to each objective's observed range (with an
    absolute fallback for a degenerate range), which keeps the reference
    scale-appropriate per objective.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    maximize = np.asarray(maximize, dtype=bool)
    worst = np.where(maximize, Y.min(axis=0), Y.max(axis=0))
    spread = Y.max(axis=0) - Y.min(axis=0)
    pad = np.where(spread > 1e-12, spread * padding, np.maximum(np.abs(worst), 1.0) * padding)
    return np.where(maximize, worst - pad, worst + pad)


# --------------------------------------------------------------------- EHVI

def monte_carlo_ehvi(
    mu: np.ndarray,
    sigma: np.ndarray,
    front: np.ndarray,
    ref_point: np.ndarray,
    maximize: Sequence[bool],
    n_samples: int = DEFAULT_EHVI_SAMPLES,
    random_state: int = 0,
    correlation: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    EHVI for one candidate: `E[HV(front u {y}) - HV(front)]`.

    Returns `(ehvi, standard_error)`. The hypervolume terms are exact; only the
    expectation is sampled, and its standard error is returned so a caller can
    tell a real difference from sampling noise.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma must have the same shape.")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative.")

    m = mu.size
    base_hv = compute_hypervolume(front, ref_point, maximize) if len(front) else 0.0

    rng = np.random.default_rng(random_state)
    chol = None
    if correlation is not None:
        R = np.asarray(correlation, dtype=np.float64)
        if R.shape != (m, m):
            raise ValueError(f"correlation must be ({m}, {m}).")
        chol = np.linalg.cholesky(R + 1e-10 * np.eye(m))

    improvements = np.empty(n_samples, dtype=np.float64)
    front_arr = np.atleast_2d(np.asarray(front, dtype=np.float64)) if len(front) else np.zeros((0, m))

    for s in range(n_samples):
        z = rng.standard_normal(m)
        if chol is not None:
            z = chol @ z
        y = mu + sigma * z
        augmented = np.vstack([front_arr, y.reshape(1, -1)]) if front_arr.size else y.reshape(1, -1)
        improvements[s] = max(
            compute_hypervolume(augmented, ref_point, maximize) - base_hv, 0.0
        )

    ehvi = float(improvements.mean())
    se = float(improvements.std(ddof=1) / math.sqrt(n_samples)) if n_samples > 1 else 0.0
    return ehvi, se


def closed_form_ei(mu: float, sigma: float, best: float) -> float:
    """
    Closed-form Expected Improvement, used to validate the MC estimator.

    EHVI with a single objective reduces exactly to EI with `best` the incumbent
    (or the reference point when the front is empty):

        EI = (mu - best) * Phi(z) + sigma * phi(z),   z = (mu - best) / sigma
    """
    if sigma <= 0:
        return max(mu - best, 0.0)
    z = (mu - best) / sigma
    return float((mu - best) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z))


def validate_against_botorch_qehvi(
    n_points: int = 6, seed: int = 0
) -> Dict[str, Any]:
    """
    Cross-check the MC estimator against BoTorch's `qExpectedHypervolumeImprovement`.

    A GP is fitted to synthetic 2-objective data so that both implementations can
    integrate over the *same* posterior; the MC estimator then uses that
    posterior's mean and variance. Agreement confirms the estimator computes EHVI
    and not something merely EHVI-shaped.
    """
    try:
        import warnings as _warnings

        from botorch.acquisition.multi_objective import (
            qExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            FastNondominatedPartitioning,
        )
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(seed)
        X = torch.rand(n_points, 2, dtype=torch.double)
        Y = torch.stack([X[:, 0] + 0.3 * X[:, 1], 1.0 - X[:, 1]], dim=-1).double()

        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=2))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

        ref = torch.tensor([-0.5, -0.5], dtype=torch.double)
        partitioning = FastNondominatedPartitioning(ref_point=ref, Y=Y)
        # BoTorch warns that qEHVI has numerical issues versus qLogEHVI for
        # *optimisation*. We are comparing VALUES, not optimising, and qLogEHVI
        # returns log-EHVI, so the non-log form is the correct comparator here.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            acq = qExpectedHypervolumeImprovement(
                model=gp,
                ref_point=ref.tolist(),
                partitioning=partitioning,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4096])),
            )

        X_test = torch.rand(5, 2, dtype=torch.double)
        botorch_vals = acq(X_test.unsqueeze(1)).detach().numpy()

        with torch.no_grad():
            post = gp.posterior(X_test)
            mu = post.mean.numpy()
            sd = post.variance.sqrt().numpy()

        front = Y.numpy()
        ours, ses = [], []
        for i in range(len(X_test)):
            v, se = monte_carlo_ehvi(
                mu[i], sd[i], front, ref.numpy(), [True, True],
                n_samples=4096, random_state=seed + i,
            )
            ours.append(v)
            ses.append(se)

        ours = np.array(ours)
        diffs = np.abs(ours - botorch_vals)
        scale = max(float(np.abs(botorch_vals).max()), 1e-9)
        return {
            "status": "ok",
            "botorch_ehvi": [round(float(v), 6) for v in botorch_vals],
            "monte_carlo_ehvi": [round(float(v), 6) for v in ours],
            "monte_carlo_standard_errors": [round(float(v), 6) for v in ses],
            "max_absolute_difference": round(float(diffs.max()), 6),
            "max_relative_difference": round(float(diffs.max() / scale), 4),
            "agrees": bool(diffs.max() / scale < 0.1),
            "note": (
                "Both integrate over the same fitted GP posterior. Residual "
                "disagreement is Monte Carlo noise plus BoTorch's QMC sampling, "
                "not a difference in the quantity being computed."
            ),
        }
    except Exception as e:  # pragma: no cover - diagnostic path
        return {"status": "failed", "error": str(e)}


# ------------------------------------------------------------------ planner

@dataclass
class ExperimentPlan:
    selected: List[PlannedExperiment] = field(default_factory=list)
    total_cost: float = 0.0
    budget: float = 0.0
    initial_hypervolume: float = 0.0
    final_hypervolume: float = 0.0
    ref_point: List[float] = field(default_factory=list)
    stopped_because: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected": [e.as_dict() for e in self.selected],
            "n_selected": len(self.selected),
            "total_cost": round(self.total_cost, 4),
            "budget": round(self.budget, 4),
            "budget_remaining": round(self.budget - self.total_cost, 4),
            "budget_utilisation": (
                round(self.total_cost / self.budget, 4) if self.budget > 0 else 0.0
            ),
            "initial_hypervolume": round(self.initial_hypervolume, 6),
            "final_hypervolume_expected": round(self.final_hypervolume, 6),
            "expected_hypervolume_gain": round(
                self.final_hypervolume - self.initial_hypervolume, 6
            ),
            "reference_point": [round(v, 4) for v in self.ref_point],
            "stopped_because": self.stopped_because,
            "diagnostics": self.diagnostics,
        }


class ExperimentPlanner:
    """Greedy, cost-aware EHVI planner over (molecule, assay) pairs."""

    def __init__(
        self,
        objectives: Sequence[str],
        maximize: Sequence[bool],
        assays: Sequence[Assay],
        n_ehvi_samples: int = DEFAULT_EHVI_SAMPLES,
        random_state: int = 0,
    ):
        if len(objectives) != len(maximize):
            raise ValueError("objectives and maximize must have the same length.")
        if not assays:
            raise ValueError("At least one assay must be provided.")
        unknown = [
            o for a in assays for o in a.objectives if o not in set(objectives)
        ]
        if unknown:
            raise ValueError(f"Assays reference unknown objectives: {sorted(set(unknown))}")

        self.objectives = list(objectives)
        self.maximize = list(maximize)
        self.assays = list(assays)
        self.n_ehvi_samples = n_ehvi_samples
        self.random_state = random_state

    def plan(
        self,
        candidate_ids: Sequence[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        budget: float,
        observed_front: Optional[np.ndarray] = None,
        ref_point: Optional[np.ndarray] = None,
        imputation: str = "mean",
        allow_repeat_molecule: bool = False,
        correlation: Optional[np.ndarray] = None,
    ) -> ExperimentPlan:
        """
        Build a plan whose selected assay costs sum to at most `budget`.

        A (molecule, assay) pair is never selected twice. By default a molecule
        appears at most once in the whole plan; `allow_repeat_molecule=True`
        permits several different assays on the same molecule.
        """
        mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        n, m = mu.shape

        if m != len(self.objectives):
            raise ValueError(
                f"mu has {m} objectives but the planner was built with {len(self.objectives)}."
            )
        if sigma.shape != mu.shape:
            raise ValueError("sigma must have the same shape as mu.")
        if len(candidate_ids) != n:
            raise ValueError("candidate_ids must have one entry per row of mu.")
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}.")
        if imputation not in ("mean", "sample"):
            raise ValueError("imputation must be 'mean' or 'sample'.")
        if not np.isfinite(mu).all():
            raise ValueError("mu contains non-finite values.")

        front = (
            np.atleast_2d(np.asarray(observed_front, dtype=np.float64))
            if observed_front is not None and len(observed_front)
            else np.zeros((0, m))
        )

        if ref_point is None:
            pool = np.vstack([mu, front]) if front.size else mu
            ref_point = infer_reference_point(pool, self.maximize)
        ref_point = np.asarray(ref_point, dtype=np.float64)

        initial_hv = compute_hypervolume(front, ref_point, self.maximize) if front.size else 0.0
        plan = ExperimentPlan(
            budget=float(budget),
            initial_hypervolume=initial_hv,
            final_hypervolume=initial_hv,
            ref_point=ref_point.tolist(),
        )

        rng = np.random.default_rng(self.random_state)
        used_pairs: set = set()
        used_molecules: set = set()
        spent = 0.0
        current_front = front.copy()
        iteration = 0
        cheapest = min(a.cost for a in self.assays)

        while True:
            iteration += 1
            remaining = budget - spent
            if remaining < cheapest - 1e-12:
                plan.stopped_because = (
                    f"Remaining budget {remaining:.4f} is below the cheapest assay "
                    f"cost {cheapest:.4f}."
                )
                break

            best: Optional[Tuple[float, float, float, int, Assay]] = None
            current_hv = compute_hypervolume(current_front, ref_point, self.maximize)

            for assay in self.assays:
                if assay.cost > remaining + 1e-12:
                    continue
                # Objectives this assay does not inform stay at their predicted
                # mean with zero variance: measuring solubility tells you nothing
                # new about toxicity, so only the informed objectives contribute
                # uncertainty to the improvement.
                informs = np.array(
                    [o in set(assay.objectives) for o in self.objectives], dtype=bool
                )
                for i in range(n):
                    if (candidate_ids[i], assay.name) in used_pairs:
                        continue
                    if not allow_repeat_molecule and candidate_ids[i] in used_molecules:
                        continue

                    sigma_eff = np.where(informs, sigma[i], 0.0)
                    value, se = monte_carlo_ehvi(
                        mu[i], sigma_eff, current_front, ref_point, self.maximize,
                        n_samples=self.n_ehvi_samples,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                        correlation=correlation,
                    )
                    ratio = value / assay.cost
                    if best is None or ratio > best[0]:
                        best = (ratio, value, se, i, assay)

            if best is None:
                plan.stopped_because = (
                    "No affordable (molecule, assay) pair remains that has not "
                    "already been selected."
                )
                break

            ratio, value, se, idx, assay = best
            if value <= 1e-12:
                plan.stopped_because = (
                    "The best remaining option has no expected hypervolume "
                    "improvement, so further spending is not justified."
                )
                break

            # Impute the outcome so the next round sees an updated front.
            if imputation == "mean":
                imputed = mu[idx].copy()
            else:
                informs = np.array(
                    [o in set(assay.objectives) for o in self.objectives], dtype=bool
                )
                sigma_eff = np.where(informs, sigma[idx], 0.0)
                imputed = mu[idx] + sigma_eff * rng.standard_normal(m)

            new_front = (
                np.vstack([current_front, imputed.reshape(1, -1)])
                if current_front.size
                else imputed.reshape(1, -1)
            )
            new_hv = compute_hypervolume(new_front, ref_point, self.maximize)

            spent += assay.cost
            used_pairs.add((candidate_ids[idx], assay.name))
            used_molecules.add(candidate_ids[idx])
            current_front = new_front

            plan.selected.append(
                PlannedExperiment(
                    candidate_id=candidate_ids[idx],
                    candidate_index=idx,
                    assay=assay.name,
                    cost=assay.cost,
                    ehvi=value,
                    ehvi_per_unit_cost=ratio,
                    ehvi_standard_error=se,
                    cumulative_cost=spent,
                    front_hypervolume_before=current_hv,
                    front_hypervolume_after_imputation=new_hv,
                )
            )
            plan.final_hypervolume = new_hv

            if iteration > 10_000:  # pragma: no cover - safety valve
                plan.stopped_because = "Iteration limit reached."
                break

        plan.total_cost = spent
        plan.diagnostics = {
            "n_candidates": int(n),
            "n_objectives": int(m),
            "objectives": self.objectives,
            "maximize": self.maximize,
            "assays": [
                {"name": a.name, "cost": a.cost, "objectives": a.objectives}
                for a in self.assays
            ],
            "n_ehvi_samples": self.n_ehvi_samples,
            "imputation": imputation,
            "allow_repeat_molecule": allow_repeat_molecule,
            "correlated_sampling": correlation is not None,
            "method": (
                "Monte Carlo EHVI: E[HV(P u {y}) - HV(P)] over the calibrated "
                "predictive distribution, with each HV term computed exactly by "
                "BoTorch box decomposition."
            ),
            "limitations": [
                "Greedy selection. Hypervolume improvement is submodular, which "
                "gives greedy a (1 - 1/e) guarantee in the unit-cost case; with "
                "heterogeneous assay costs the cost-weighted rule used here has "
                "no such bound. Plans are good, not provably optimal.",
                f"Outcomes are imputed by the '{imputation}' strategy between "
                "selections. 'mean' assumes an assay returns exactly the "
                "prediction, which under-rewards a second measurement in the same "
                "region of objective space.",
                "Hypervolume depends on the reference point, which is reported "
                "with the plan. Comparing hypervolumes computed against different "
                "reference points is meaningless.",
                "The predictive distribution is Gaussian with a conformally "
                "calibrated scale; see probabilistic_pareto for what that assumes.",
            ],
        }
        if not plan.stopped_because:
            plan.stopped_because = "Budget exhausted."
        return plan


DEFAULT_ASSAYS: List[Assay] = [
    Assay(
        name="solubility_shake_flask",
        cost=1.0,
        objectives=["solubility_logs"],
        description="Thermodynamic aqueous solubility by shake-flask (simulated).",
    ),
    Assay(
        name="toxicity_panel",
        cost=3.0,
        objectives=["toxicity_risk_probability"],
        description="Cell-viability toxicity panel (simulated).",
    ),
    Assay(
        name="combined_profile",
        cost=3.5,
        objectives=["solubility_logs", "toxicity_risk_probability"],
        description="Combined solubility and toxicity profiling (simulated).",
    ),
]
