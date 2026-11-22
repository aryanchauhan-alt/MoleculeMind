"""
Probabilistic Pareto optimisation and uncertainty-aware ranking.

Standard Pareto sorting treats each predicted objective value as a fact. It is
not: every value carries a calibrated uncertainty. A candidate that sits just
inside the front with tight intervals and one that sits just outside with wide
intervals are being told apart by noise, and a deterministic front cannot say
which is which.

This module answers a different question: **with what probability is this
candidate on the Pareto front?**

--------------------------------------------------------------------------
1. Converting a conformal interval into a Gaussian scale
--------------------------------------------------------------------------
Conformal prediction gives a calibrated interval `mu +- q_hat * sigma`, not a
distribution. To sample we need a scale. The methodology's conversion is

    sigma_tilde = q_hat * sigma / z,      z = Phi^-1(1 - alpha/2)

This chooses the Gaussian whose central (1-alpha) interval has the *same width*
as the conformal interval. At alpha = 0.1, z = 1.645.

**This is an approximation, and its direction matters.** Conformal intervals are
distribution-free: the calibrated width is honest, but the *shape* inside it is
not specified. Matching widths to a Gaussian imposes a shape the conformal
procedure never claimed. Where true residuals are heavier-tailed than Gaussian -
common for molecular property prediction, where a handful of compounds are badly
mispredicted - this *understates* the chance of an extreme outcome, so
`probability_pareto_optimal` is optimistic for candidates whose membership hinges
on a tail event. `residual_normality_report()` measures the deviation on real
calibration residuals rather than leaving the assumption unexamined.

--------------------------------------------------------------------------
2. Independence across objectives
--------------------------------------------------------------------------
Marginal conformal intervals, one per objective, do **not** define a jointly
calibrated multivariate distribution. Sampling each objective independently
assumes zero residual correlation between them, which is generally false:
solubility and toxicity predictions for the same molecule share a representation
and share training data, so their errors correlate.

Two sampling modes are therefore provided:

* `correlation=None` - independent marginals (the default, and the
  approximation the methodology describes).
* `correlation=R` - a correlation matrix estimated from joint calibration
  residuals, used via a Cholesky factor so draws respect the observed coupling.

Neither is jointly calibrated in the conformal sense. The correlated mode is
strictly better justified than the independent one when residuals are available
to estimate `R` from; it still inherits the Gaussian shape assumption above.

--------------------------------------------------------------------------
3. What `probability_pareto_optimal` means
--------------------------------------------------------------------------
It is the fraction of sampled joint outcomes in which the candidate is
non-dominated **by the other candidates in this pool, under this model**. It is
not a probability that the molecule is a good drug, nor that it would be
non-dominated against a different candidate set, nor a calibrated statement
about physical reality. Adding or removing candidates changes every value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

DEFAULT_N_SAMPLES = 200


# --------------------------------------------------------------- scale

def conformal_to_gaussian_scale(
    sigma: np.ndarray,
    q_hat: float,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    sigma_tilde = q_hat * sigma / z, with z = Phi^-1(1 - alpha/2).

    Matches the Gaussian's central (1-alpha) interval width to the calibrated
    conformal width. See the module docstring for what this assumes.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if q_hat <= 0:
        raise ValueError(f"q_hat must be positive, got {q_hat}.")
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return np.asarray(sigma, dtype=np.float64) * q_hat / z


def residual_normality_report(
    residuals: np.ndarray, sigma: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Measure how far calibration residuals depart from the Gaussian shape that
    `conformal_to_gaussian_scale` imposes.

    Reports excess kurtosis (0 for a Gaussian; positive means heavier tails and
    therefore an *optimistic* Pareto probability), skewness, and the
    Kolmogorov-Smirnov statistic against a fitted normal.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    if sigma is not None:
        s = np.asarray(sigma, dtype=np.float64).ravel()
        r = r / np.maximum(s, 1e-12)
    if r.size < 8:
        return {"status": "insufficient_data", "n": int(r.size)}

    standardised = (r - r.mean()) / (r.std() or 1.0)
    ks_stat, ks_p = stats.kstest(standardised, "norm")
    excess_kurtosis = float(stats.kurtosis(r, fisher=True))

    return {
        "status": "ok",
        "n": int(r.size),
        "excess_kurtosis": round(excess_kurtosis, 3),
        "skewness": round(float(stats.skew(r)), 3),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(ks_p), 4),
        "gaussian_approximation_adequate": bool(ks_p > 0.05),
        "interpretation": (
            "Excess kurtosis above 0 means residuals are heavier-tailed than the "
            "Gaussian the sampler assumes, so Pareto probabilities are optimistic "
            "for candidates whose membership depends on a tail event. A KS p-value "
            "below 0.05 rejects the Gaussian shape outright; the calibrated interval "
            "WIDTH remains valid either way, only the assumed shape inside it does not."
        ),
    }


# ------------------------------------------------------------ dominance

def _to_maximisation(values: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    """Flip minimisation objectives so every column is 'larger is better'."""
    signs = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)
    return np.asarray(values, dtype=np.float64) * signs


def non_dominated_mask(values: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    """
    Boolean mask of non-dominated rows, vectorised.

    `a` dominates `b` iff a >= b on every objective and a > b on at least one.
    A row is non-dominated iff no other row dominates it. Duplicated optima are
    all reported as non-dominated, which is correct: neither strictly beats the
    other.
    """
    Y = _to_maximisation(values, maximize)
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # ge[i, j] = row j is >= row i on every objective
    #  gt[i, j] = row j is >  row i on at least one objective
    ge = np.all(Y[None, :, :] >= Y[:, None, :], axis=2)
    gt = np.any(Y[None, :, :] > Y[:, None, :], axis=2)
    dominated_by_someone = np.any(ge & gt, axis=1)
    return ~dominated_by_someone


def fast_non_dominated_sort(
    values: np.ndarray, maximize: Sequence[bool]
) -> np.ndarray:
    """
    Deb's fast non-dominated sort: 1-based front rank per row.

    Rank 1 is the Pareto front, rank 2 the front of what remains, and so on.
    """
    Y = _to_maximisation(values, maximize)
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    ge = np.all(Y[:, None, :] >= Y[None, :, :], axis=2)
    gt = np.any(Y[:, None, :] > Y[None, :, :], axis=2)
    dominates = ge & gt  # dominates[i, j] = i dominates j

    domination_count = dominates.sum(axis=0)  # how many dominate j
    ranks = np.zeros(n, dtype=int)
    current = np.where(domination_count == 0)[0]
    rank = 1
    remaining = domination_count.copy()

    while current.size:
        ranks[current] = rank
        remaining[current] = -1  # retire this front
        newly_freed: List[int] = []
        for i in current:
            for j in np.where(dominates[i])[0]:
                if remaining[j] > 0:
                    remaining[j] -= 1
                    if remaining[j] == 0:
                        newly_freed.append(j)
        current = np.unique(np.array(newly_freed, dtype=int)) if newly_freed else np.array([], dtype=int)
        rank += 1

    # Any row never freed (possible only with numerical ties) gets the last rank.
    ranks[ranks == 0] = rank
    return ranks


# -------------------------------------------------------------- pre-filter

def hopeless_mask(
    mu: np.ndarray,
    sigma_tilde: np.ndarray,
    maximize: Sequence[bool],
    k: float = 4.0,
) -> np.ndarray:
    """
    Flag candidates that cannot plausibly reach the front, to skip in sampling.

    A candidate's *optimistic* corner is `mu + k*sigma_tilde` on every maximised
    objective. If that optimistic corner is still dominated by some other
    candidate's *pessimistic* corner (`mu - k*sigma_tilde`), then even a
    k-sigma-favourable draw for the first and a k-sigma-unfavourable draw for the
    second leaves it dominated.

    This is a sound pre-filter, not an exact one: at k = 4 the discarded
    probability mass is ~3e-5 per objective per candidate. It is a speed
    optimisation whose error is bounded and stated, and `k` is configurable.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma_tilde = np.asarray(sigma_tilde, dtype=np.float64)
    signs = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)

    optimistic = _to_maximisation(mu, maximize) + k * sigma_tilde * np.abs(signs)
    pessimistic = _to_maximisation(mu, maximize) - k * sigma_tilde * np.abs(signs)

    ge = np.all(pessimistic[None, :, :] >= optimistic[:, None, :], axis=2)
    gt = np.any(pessimistic[None, :, :] > optimistic[:, None, :], axis=2)
    return np.any(ge & gt, axis=1)


# ------------------------------------------------------------- sampling

@dataclass
class ProbabilisticParetoResult:
    probability: np.ndarray          # (n,) P(non-dominated)
    deterministic_rank: np.ndarray   # (n,) rank from the mean predictions
    deterministic_front: np.ndarray  # (n,) bool, rank == 1
    n_samples: int
    prefiltered: np.ndarray          # (n,) bool, skipped by the pre-filter
    diagnostics: Dict[str, Any]

    def as_records(self) -> List[Dict[str, Any]]:
        return [
            {
                "probability_pareto_optimal": round(float(self.probability[i]), 4),
                "deterministic_pareto_rank": int(self.deterministic_rank[i]),
                "is_deterministic_pareto_optimal": bool(self.deterministic_front[i]),
                "prefiltered_as_hopeless": bool(self.prefiltered[i]),
            }
            for i in range(len(self.probability))
        ]


def probability_pareto_optimal(
    mu: np.ndarray,
    sigma_tilde: np.ndarray,
    maximize: Sequence[bool],
    n_samples: int = DEFAULT_N_SAMPLES,
    correlation: Optional[np.ndarray] = None,
    random_state: int = 0,
    prefilter_k: Optional[float] = 4.0,
) -> ProbabilisticParetoResult:
    """
    Monte Carlo probability that each candidate is Pareto non-dominated.

    For each of `n_samples` draws, sample a joint objective vector for every
    candidate and recompute the non-dominated set; the probability is the
    fraction of draws in which a candidate appears in it.

    All candidates are resampled together in each draw, so the comparison within
    a draw is self-consistent: candidate A's good luck and candidate B's bad luck
    are evaluated against each other, which is the whole point.
    """
    mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
    sigma_tilde = np.atleast_2d(np.asarray(sigma_tilde, dtype=np.float64))
    n, m = mu.shape

    if sigma_tilde.shape != mu.shape:
        raise ValueError(
            f"sigma_tilde shape {sigma_tilde.shape} must match mu shape {mu.shape}."
        )
    if len(maximize) != m:
        raise ValueError(
            f"maximize has {len(maximize)} entries but there are {m} objectives."
        )
    if np.any(sigma_tilde < 0):
        raise ValueError("sigma_tilde must be non-negative.")
    if not np.isfinite(mu).all():
        raise ValueError("mu contains non-finite values; candidates must be complete.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    deterministic_rank = fast_non_dominated_sort(mu, maximize)
    deterministic_front = deterministic_rank == 1

    if n == 0:
        return ProbabilisticParetoResult(
            np.zeros(0), deterministic_rank, deterministic_front, n_samples,
            np.zeros(0, dtype=bool), {"n_candidates": 0},
        )

    prefiltered = (
        hopeless_mask(mu, sigma_tilde, maximize, k=prefilter_k)
        if prefilter_k is not None and n > 1
        else np.zeros(n, dtype=bool)
    )

    rng = np.random.default_rng(random_state)
    chol: Optional[np.ndarray] = None
    if correlation is not None:
        R = np.asarray(correlation, dtype=np.float64)
        if R.shape != (m, m):
            raise ValueError(f"correlation must be ({m}, {m}), got {R.shape}.")
        # Jitter keeps an estimated matrix positive definite.
        chol = np.linalg.cholesky(R + 1e-10 * np.eye(m))

    counts = np.zeros(n, dtype=np.int64)
    for _ in range(n_samples):
        z = rng.standard_normal(size=(n, m))
        if chol is not None:
            z = z @ chol.T
        draw = mu + sigma_tilde * z
        mask = non_dominated_mask(draw, maximize)
        # A pre-filtered candidate keeps probability 0 by construction.
        mask &= ~prefiltered
        counts += mask

    probability = counts / float(n_samples)

    # Monte Carlo standard error on a proportion, so the user can judge whether
    # a difference between two candidates is real or sampling noise.
    se = np.sqrt(np.maximum(probability * (1.0 - probability), 0.0) / n_samples)

    return ProbabilisticParetoResult(
        probability=probability,
        deterministic_rank=deterministic_rank,
        deterministic_front=deterministic_front,
        n_samples=n_samples,
        prefiltered=prefiltered,
        diagnostics={
            "n_candidates": int(n),
            "n_objectives": int(m),
            "n_samples": int(n_samples),
            "n_prefiltered": int(prefiltered.sum()),
            "prefilter_k": prefilter_k,
            "correlated_sampling": correlation is not None,
            "max_monte_carlo_standard_error": round(float(se.max()), 4),
            "mean_monte_carlo_standard_error": round(float(se.mean()), 4),
            "approximations": [
                "Gaussian shape imposed on distribution-free conformal intervals "
                "by width matching (sigma_tilde = q_hat * sigma / z).",
                (
                    "Objectives sampled with an estimated correlation matrix."
                    if correlation is not None
                    else "Objectives sampled independently; marginal conformal "
                         "intervals do not define a jointly calibrated distribution."
                ),
                f"Monte Carlo with S = {n_samples}; max standard error "
                f"{float(se.max()):.4f}.",
                "Probability is relative to THIS candidate pool and this model. "
                "Adding or removing candidates changes every value.",
            ],
        },
    )


# ----------------------------------------------------------- risk slider

def risk_adjusted_score(
    mu: np.ndarray,
    sigma_tilde: np.ndarray,
    maximize: Sequence[bool],
    lam: float,
    kappa_max: float = 2.0,
) -> np.ndarray:
    """
    Optimistic score for the exploration/exploitation slider.

        kappa = kappa_max * lambda
        f     = mu + kappa * sigma_tilde          (maximised objectives)
        f     = mu - kappa * sigma_tilde          (minimised objectives)

    For a *minimised* objective, being optimistic means assuming a **lower**
    value, so the sign flips. Getting this backwards would make the slider
    penalise uncertainty on exactly those objectives it should reward it on.

    `lambda = 0` is pure exploitation (rank by the mean alone); `lambda = 1`
    gives uncertainty its maximum weight.

    This is an *optimistic ranking device*, not a prediction. `mu + 2*sigma` is
    not a forecast and carries no coverage guarantee; it says "rank this
    candidate as if it came out well", which is how exploration is bought.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}.")
    if kappa_max < 0:
        raise ValueError(f"kappa_max must be non-negative, got {kappa_max}.")

    mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
    sigma_tilde = np.atleast_2d(np.asarray(sigma_tilde, dtype=np.float64))
    if sigma_tilde.shape != mu.shape:
        raise ValueError("sigma_tilde must have the same shape as mu.")
    if len(maximize) != mu.shape[1]:
        raise ValueError("maximize length must match the number of objectives.")

    kappa = kappa_max * lam
    signs = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)
    return mu + signs * kappa * sigma_tilde


def rank_with_risk(
    mu: np.ndarray,
    sigma_tilde: np.ndarray,
    maximize: Sequence[bool],
    lam: float,
    kappa_max: float = 2.0,
) -> Dict[str, Any]:
    """Risk-adjusted scores plus the Pareto front they imply."""
    scores = risk_adjusted_score(mu, sigma_tilde, maximize, lam, kappa_max)
    ranks = fast_non_dominated_sort(scores, maximize)

    # A single scalar for list ordering: min-max normalise each objective on the
    # risk-adjusted scale, then average. Reported as a relative score, not a
    # physical quantity.
    adjusted = _to_maximisation(scores, maximize)
    lo = adjusted.min(axis=0)
    hi = adjusted.max(axis=0)
    normalised = (adjusted - lo) / np.maximum(hi - lo, 1e-12)
    composite = normalised.mean(axis=1)

    return {
        "lambda": lam,
        "kappa": kappa_max * lam,
        "kappa_max": kappa_max,
        "scores": scores,
        "pareto_rank": ranks,
        "composite_score": composite,
        "note": (
            "f = mu + kappa*sigma_tilde for maximised objectives and "
            "mu - kappa*sigma_tilde for minimised ones. This is an optimistic "
            "ranking device for trading exploitation against exploration. It is "
            "not a calibrated prediction and guarantees no improvement."
        ),
    }
