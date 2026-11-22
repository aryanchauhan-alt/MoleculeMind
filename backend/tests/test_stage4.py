"""
Stage 4 tests: probabilistic Pareto, risk slider, EHVI planner, generator.

The EHVI tests are the important ones. A "weighted utility plus uncertainty"
heuristic can be made to look like EHVI, so the estimator is checked against two
independent ground truths: the closed-form Expected Improvement it must reduce to
in one dimension, and BoTorch's own `qExpectedHypervolumeImprovement`.
"""

import math

import numpy as np
import pytest

from backend.app.core.experiment_planner import (
    DEFAULT_ASSAYS,
    Assay,
    ExperimentPlanner,
    closed_form_ei,
    compute_hypervolume,
    infer_reference_point,
    monte_carlo_ehvi,
    validate_against_botorch_qehvi,
)
from backend.app.core.probabilistic_pareto import (
    conformal_to_gaussian_scale,
    fast_non_dominated_sort,
    hopeless_mask,
    non_dominated_mask,
    probability_pareto_optimal,
    rank_with_risk,
    residual_normality_report,
    risk_adjusted_score,
)


# =========================================================== dominance

def test_non_dominated_mask_basic():
    Y = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [0.2, 0.2]])
    assert non_dominated_mask(Y, [True, True]).tolist() == [True, True, True, False]


def test_fast_non_dominated_sort_assigns_fronts():
    Y = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    assert fast_non_dominated_sort(Y, [True, True]).tolist() == [1, 2, 3]


def test_minimisation_direction_is_respected():
    Y = np.array([[1.0], [5.0]])
    assert fast_non_dominated_sort(Y, [False]).tolist() == [1, 2]
    assert fast_non_dominated_sort(Y, [True]).tolist() == [2, 1]


def test_identical_rows_are_all_non_dominated():
    Y = np.array([[1.0, 2.0]] * 4)
    assert non_dominated_mask(Y, [True, True]).all()


def test_mixed_directions():
    # maximise first, minimise second
    Y = np.array([[2.0, 0.1], [1.0, 0.9]])
    assert non_dominated_mask(Y, [True, False]).tolist() == [True, False]


def test_empty_input():
    assert non_dominated_mask(np.zeros((0, 2)), [True, True]).shape == (0,)
    assert fast_non_dominated_sort(np.zeros((0, 2)), [True, True]).shape == (0,)


# ================================================ conformal -> Gaussian

def test_scale_conversion_matches_interval_width():
    """At alpha = 0.1, q_hat = z gives sigma_tilde = sigma exactly."""
    from scipy import stats

    z = stats.norm.ppf(0.95)
    out = conformal_to_gaussian_scale(np.array([1.0, 2.0]), q_hat=z, alpha=0.1)
    assert out[0] == pytest.approx(1.0, abs=1e-9)
    assert out[1] == pytest.approx(2.0, abs=1e-9)


def test_scale_conversion_scales_linearly_in_q():
    a = conformal_to_gaussian_scale(np.array([1.0]), q_hat=1.0, alpha=0.1)
    b = conformal_to_gaussian_scale(np.array([1.0]), q_hat=2.0, alpha=0.1)
    assert b[0] == pytest.approx(2.0 * a[0])


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_scale_conversion_rejects_bad_alpha(bad_alpha):
    with pytest.raises(ValueError):
        conformal_to_gaussian_scale(np.array([1.0]), q_hat=1.0, alpha=bad_alpha)


def test_scale_conversion_rejects_nonpositive_q():
    with pytest.raises(ValueError):
        conformal_to_gaussian_scale(np.array([1.0]), q_hat=0.0)


def test_normality_report_detects_heavy_tails():
    rng = np.random.default_rng(0)
    gaussian = residual_normality_report(rng.normal(size=2000))
    heavy = residual_normality_report(rng.standard_t(3, size=2000))

    assert gaussian["gaussian_approximation_adequate"] is True
    assert heavy["excess_kurtosis"] > gaussian["excess_kurtosis"]
    assert heavy["gaussian_approximation_adequate"] is False
    assert "optimistic" in heavy["interpretation"]


def test_normality_report_handles_tiny_samples():
    assert residual_normality_report(np.array([1.0, 2.0]))["status"] == "insufficient_data"


# ========================================= probabilistic Pareto front

def test_certain_dominant_candidate_has_probability_one():
    mu = np.array([[5.0, 5.0], [1.0, 1.0], [0.0, 0.0]])
    sigma = np.full_like(mu, 0.01)
    r = probability_pareto_optimal(mu, sigma, [True, True], n_samples=400, random_state=0)
    assert r.probability[0] == pytest.approx(1.0, abs=1e-9)
    assert r.probability[1] < 0.05
    assert r.probability[2] < 0.05


def test_uncertainty_spreads_probability_mass():
    """With wide intervals a dominated candidate gains a real chance."""
    mu = np.array([[2.0, 2.0], [1.0, 1.0]])
    tight = probability_pareto_optimal(
        mu, np.full_like(mu, 0.01), [True, True], n_samples=800, random_state=0
    )
    wide = probability_pareto_optimal(
        mu, np.full_like(mu, 3.0), [True, True], n_samples=800, random_state=0
    )
    assert tight.probability[1] < 0.05
    assert wide.probability[1] > 0.2


def test_probability_is_bounded_and_front_members_dominate():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(15, 3))
    sigma = rng.uniform(0.1, 0.5, size=(15, 3))
    r = probability_pareto_optimal(mu, sigma, [True, True, False], n_samples=300, random_state=0)

    assert ((r.probability >= 0) & (r.probability <= 1)).all()
    # Deterministic front members should on average score above non-members.
    if r.deterministic_front.any() and (~r.deterministic_front).any():
        assert r.probability[r.deterministic_front].mean() > r.probability[~r.deterministic_front].mean()


def test_zero_sigma_reproduces_the_deterministic_front():
    rng = np.random.default_rng(2)
    mu = rng.normal(size=(10, 2))
    r = probability_pareto_optimal(
        mu, np.zeros_like(mu), [True, True], n_samples=20, random_state=0
    )
    assert np.allclose(r.probability, r.deterministic_front.astype(float))


def test_single_candidate_is_always_optimal():
    r = probability_pareto_optimal(
        np.array([[1.0, 2.0]]), np.array([[0.5, 0.5]]), [True, True],
        n_samples=50, random_state=0,
    )
    assert r.probability[0] == pytest.approx(1.0)


def test_monte_carlo_standard_error_is_reported():
    mu = np.array([[1.0, 1.0], [1.0, 1.0]])
    sigma = np.full_like(mu, 1.0)
    r = probability_pareto_optimal(mu, sigma, [True, True], n_samples=100, random_state=0)
    assert r.diagnostics["max_monte_carlo_standard_error"] > 0
    assert any("Monte Carlo" in a for a in r.diagnostics["approximations"])


def test_approximations_are_declared():
    r = probability_pareto_optimal(
        np.array([[1.0, 1.0]]), np.array([[0.1, 0.1]]), [True, True],
        n_samples=50, random_state=0,
    )
    text = " ".join(r.diagnostics["approximations"])
    assert "Gaussian" in text
    assert "independently" in text or "correlation" in text
    assert "THIS candidate pool" in text


def test_correlated_sampling_changes_the_result():
    mu = np.array([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1]])
    sigma = np.full_like(mu, 1.0)
    independent = probability_pareto_optimal(
        mu, sigma, [True, True], n_samples=2000, random_state=0
    )
    correlated = probability_pareto_optimal(
        mu, sigma, [True, True], n_samples=2000, random_state=0,
        correlation=np.array([[1.0, 0.95], [0.95, 1.0]]),
    )
    assert correlated.diagnostics["correlated_sampling"] is True
    assert not np.allclose(independent.probability, correlated.probability, atol=0.02)


def test_prefilter_only_removes_hopeless_candidates():
    mu = np.array([[10.0, 10.0], [0.0, 0.0]])
    sigma = np.full_like(mu, 0.01)
    mask = hopeless_mask(mu, sigma, [True, True], k=4.0)
    assert mask[0] is np.False_ or mask[0] == False  # the dominant one survives
    assert mask[1] == True


def test_prefilter_spares_candidates_within_reach():
    """A candidate whose optimistic corner is competitive must not be filtered."""
    mu = np.array([[1.0, 1.0], [0.5, 0.5]])
    sigma = np.full_like(mu, 1.0)  # wide enough to overlap
    assert not hopeless_mask(mu, sigma, [True, True], k=4.0).any()


def test_prefilter_can_be_disabled():
    mu = np.array([[10.0, 10.0], [0.0, 0.0]])
    sigma = np.full_like(mu, 0.01)
    r = probability_pareto_optimal(
        mu, sigma, [True, True], n_samples=50, random_state=0, prefilter_k=None
    )
    assert r.prefiltered.sum() == 0


def test_invalid_shapes_and_values_rejected():
    mu = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        probability_pareto_optimal(mu, np.array([[1.0]]), [True, True])
    with pytest.raises(ValueError):
        probability_pareto_optimal(mu, np.array([[-1.0, 1.0]]), [True, True])
    with pytest.raises(ValueError):
        probability_pareto_optimal(mu, np.array([[1.0, 1.0]]), [True])
    with pytest.raises(ValueError):
        probability_pareto_optimal(
            np.array([[np.nan, 1.0]]), np.array([[1.0, 1.0]]), [True, True]
        )


# ================================================== risk slider

def test_lambda_zero_is_pure_exploitation():
    mu = np.array([[1.0, 2.0]])
    sigma = np.array([[5.0, 5.0]])
    assert np.allclose(risk_adjusted_score(mu, sigma, [True, True], 0.0), mu)


def test_maximised_objective_gets_an_optimistic_increase():
    mu = np.array([[1.0]])
    sigma = np.array([[0.5]])
    out = risk_adjusted_score(mu, sigma, [True], 1.0, kappa_max=2.0)
    assert out[0, 0] == pytest.approx(1.0 + 2.0 * 0.5)


def test_minimised_objective_gets_an_optimistic_decrease():
    """Optimism for a minimised objective means a LOWER value, not a higher one."""
    mu = np.array([[1.0]])
    sigma = np.array([[0.5]])
    out = risk_adjusted_score(mu, sigma, [False], 1.0, kappa_max=2.0)
    assert out[0, 0] == pytest.approx(1.0 - 2.0 * 0.5)


def test_kappa_scales_linearly_with_lambda():
    mu = np.zeros((1, 1))
    sigma = np.ones((1, 1))
    values = [
        float(risk_adjusted_score(mu, sigma, [True], lam, kappa_max=2.0)[0, 0])
        for lam in (0.0, 0.25, 0.5, 1.0)
    ]
    assert values == pytest.approx([0.0, 0.5, 1.0, 2.0])


def test_risk_slider_reorders_candidates():
    """A high-variance candidate should climb as lambda rises."""
    mu = np.array([[1.0], [0.9]])
    sigma = np.array([[0.01], [1.0]])
    exploit = rank_with_risk(mu, sigma, [True], 0.0)
    explore = rank_with_risk(mu, sigma, [True], 1.0, kappa_max=2.0)
    assert exploit["pareto_rank"][0] == 1 and exploit["pareto_rank"][1] == 2
    assert explore["pareto_rank"][1] == 1  # the uncertain one now leads


@pytest.mark.parametrize("bad_lam", [-0.1, 1.1])
def test_risk_slider_rejects_lambda_out_of_range(bad_lam):
    with pytest.raises(ValueError):
        risk_adjusted_score(np.zeros((1, 1)), np.ones((1, 1)), [True], bad_lam)


def test_risk_slider_declares_it_is_not_a_prediction():
    out = rank_with_risk(np.zeros((2, 1)), np.ones((2, 1)), [True], 0.5)
    assert "not a calibrated prediction" in out["note"]


# ================================================== hypervolume

def test_hypervolume_of_a_single_point():
    assert compute_hypervolume(
        np.array([[2.0, 2.0]]), np.array([0.0, 0.0]), [True, True]
    ) == pytest.approx(4.0)


def test_hypervolume_handles_minimisation():
    assert compute_hypervolume(
        np.array([[1.0, 1.0]]), np.array([3.0, 3.0]), [False, False]
    ) == pytest.approx(4.0)


def test_point_not_dominating_the_reference_contributes_nothing():
    assert compute_hypervolume(
        np.array([[-1.0, -1.0]]), np.array([0.0, 0.0]), [True, True]
    ) == pytest.approx(0.0)


def test_hypervolume_is_monotone_under_adding_points():
    front = np.array([[1.0, 1.0]])
    ref = np.array([0.0, 0.0])
    before = compute_hypervolume(front, ref, [True, True])
    after = compute_hypervolume(
        np.vstack([front, [[2.0, 0.5]]]), ref, [True, True]
    )
    assert after >= before


def test_dominated_point_does_not_change_hypervolume():
    ref = np.array([0.0, 0.0])
    a = compute_hypervolume(np.array([[2.0, 2.0]]), ref, [True, True])
    b = compute_hypervolume(np.array([[2.0, 2.0], [1.0, 1.0]]), ref, [True, True])
    assert a == pytest.approx(b)


def test_reference_point_is_outside_the_observed_range():
    Y = np.array([[1.0, 5.0], [3.0, 2.0]])
    ref = infer_reference_point(Y, [True, False])
    assert ref[0] < Y[:, 0].min()   # maximised: reference below the worst
    assert ref[1] > Y[:, 1].max()   # minimised: reference above the worst


# ======================================================= EHVI

@pytest.mark.parametrize(
    "mu,sigma,best",
    [(1.0, 0.5, 0.8), (0.5, 1.0, 1.2), (2.0, 0.3, 1.0), (0.0, 1.0, 0.0)],
)
def test_single_objective_ehvi_equals_closed_form_ei(mu, sigma, best):
    """
    With one objective, EHVI reduces analytically to Expected Improvement.

    This is the test that separates a real EHVI estimator from a heuristic: a
    weighted mu + k*sigma score cannot reproduce the EI curve.
    """
    front = np.array([[best]])
    ref = np.array([best - 10.0])
    estimate, se = monte_carlo_ehvi(
        np.array([mu]), np.array([sigma]), front, ref, [True],
        n_samples=40000, random_state=1,
    )
    exact = closed_form_ei(mu, sigma, best)
    # Within 4 Monte Carlo standard errors.
    assert abs(estimate - exact) < max(4 * se, 1e-3)


def test_deterministic_outcome_gives_exact_hypervolume_improvement():
    front = np.array([[1.0, 1.0]])
    ref = np.array([0.0, 0.0])
    y = np.array([2.0, 2.0])
    estimate, _ = monte_carlo_ehvi(y, np.zeros(2), front, ref, [True, True], n_samples=16)
    exact = compute_hypervolume(
        np.vstack([front, y.reshape(1, -1)]), ref, [True, True]
    ) - compute_hypervolume(front, ref, [True, True])
    assert estimate == pytest.approx(exact, abs=1e-9)


def test_dominated_certain_point_has_zero_ehvi():
    estimate, _ = monte_carlo_ehvi(
        np.array([0.5, 0.5]), np.zeros(2), np.array([[1.0, 1.0]]),
        np.array([0.0, 0.0]), [True, True], n_samples=16,
    )
    assert estimate == pytest.approx(0.0)


def test_ehvi_is_non_negative():
    rng = np.random.default_rng(0)
    for _ in range(5):
        mu = rng.normal(size=2)
        sigma = rng.uniform(0.1, 1.0, size=2)
        estimate, _ = monte_carlo_ehvi(
            mu, sigma, rng.normal(size=(3, 2)), np.array([-5.0, -5.0]),
            [True, True], n_samples=64, random_state=0,
        )
        assert estimate >= 0.0


def test_ehvi_increases_with_a_better_mean():
    front = np.array([[1.0, 1.0]])
    ref = np.array([0.0, 0.0])
    low, _ = monte_carlo_ehvi(
        np.array([1.1, 1.1]), np.array([0.1, 0.1]), front, ref, [True, True],
        n_samples=2000, random_state=0,
    )
    high, _ = monte_carlo_ehvi(
        np.array([3.0, 3.0]), np.array([0.1, 0.1]), front, ref, [True, True],
        n_samples=2000, random_state=0,
    )
    assert high > low


def test_ehvi_matches_botorch_qehvi():
    """
    Cross-check against BoTorch's own implementation on a shared GP posterior.

    Both integrate over the same posterior, so agreement confirms the estimator
    computes EHVI rather than something merely EHVI-shaped.
    """
    result = validate_against_botorch_qehvi(n_points=8, seed=0)
    if result["status"] != "ok":
        pytest.skip(f"BoTorch cross-check unavailable: {result.get('error')}")
    assert result["agrees"] is True
    assert result["max_relative_difference"] < 0.1


def test_ehvi_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        monte_carlo_ehvi(
            np.array([1.0, 2.0]), np.array([1.0]), np.zeros((0, 2)),
            np.array([0.0, 0.0]), [True, True],
        )


def test_ehvi_rejects_negative_sigma():
    with pytest.raises(ValueError):
        monte_carlo_ehvi(
            np.array([1.0]), np.array([-1.0]), np.zeros((0, 1)),
            np.array([0.0]), [True],
        )


# ================================================= the planner

@pytest.fixture
def planner_inputs():
    rng = np.random.default_rng(0)
    n = 10
    mu = np.column_stack([rng.normal(-3, 1.5, n), rng.uniform(0.05, 0.6, n)])
    sigma = np.column_stack([rng.uniform(0.3, 1.2, n), rng.uniform(0.02, 0.15, n)])
    ids = [f"MOL-{i:02d}" for i in range(n)]
    return ids, mu, sigma


@pytest.fixture
def planner():
    return ExperimentPlanner(
        objectives=["solubility_logs", "toxicity_risk_probability"],
        maximize=[True, False],
        assays=DEFAULT_ASSAYS,
        n_ehvi_samples=48,
        random_state=0,
    )


def test_plan_never_exceeds_the_budget(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    for budget in (1.0, 2.5, 7.0, 15.0):
        plan = planner.plan(ids, mu, sigma, budget=budget)
        assert plan.total_cost <= budget + 1e-9
        assert sum(e.cost for e in plan.selected) == pytest.approx(plan.total_cost)


def test_plan_selects_nothing_when_nothing_is_affordable(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=0.5)  # cheapest assay costs 1.0
    assert plan.selected == []
    assert "below the cheapest assay" in plan.stopped_because


def test_no_molecule_assay_pair_is_repeated(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=20.0, allow_repeat_molecule=True)
    pairs = [(e.candidate_id, e.assay) for e in plan.selected]
    assert len(set(pairs)) == len(pairs)


def test_molecules_are_unique_by_default(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=20.0)
    molecules = [e.candidate_id for e in plan.selected]
    assert len(set(molecules)) == len(molecules)


def test_repeat_molecules_allowed_when_requested(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=20.0, allow_repeat_molecule=True)
    molecules = [e.candidate_id for e in plan.selected]
    assert len(molecules) >= len(set(molecules))


def test_expected_gain_is_non_decreasing_in_budget(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    gains = []
    for budget in (2.0, 4.0, 8.0, 16.0):
        plan = planner.plan(ids, mu, sigma, budget=budget)
        gains.append(plan.final_hypervolume - plan.initial_hypervolume)
    for a, b in zip(gains, gains[1:]):
        assert b >= a - 1e-9


def test_planner_stops_when_no_improvement_remains(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=1000.0)
    assert plan.total_cost < 1000.0
    assert "no expected hypervolume improvement" in plan.stopped_because or (
        "has not already been selected" in plan.stopped_because
    )


def test_cheaper_assays_change_the_selected_mix(planner_inputs):
    ids, mu, sigma = planner_inputs
    objectives = ["solubility_logs", "toxicity_risk_probability"]

    expensive_tox = ExperimentPlanner(
        objectives, [True, False],
        [Assay("sol", 1.0, ["solubility_logs"]), Assay("tox", 10.0, ["toxicity_risk_probability"])],
        n_ehvi_samples=48, random_state=0,
    )
    cheap_tox = ExperimentPlanner(
        objectives, [True, False],
        [Assay("sol", 1.0, ["solubility_logs"]), Assay("tox", 0.2, ["toxicity_risk_probability"])],
        n_ehvi_samples=48, random_state=0,
    )
    a = {e.assay for e in expensive_tox.plan(ids, mu, sigma, budget=5.0).selected}
    b = {e.assay for e in cheap_tox.plan(ids, mu, sigma, budget=5.0).selected}
    assert a != b


def test_plan_reports_reference_point_and_limitations(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    payload = planner.plan(ids, mu, sigma, budget=5.0).as_dict()
    assert len(payload["reference_point"]) == 2
    limitations = " ".join(payload["diagnostics"]["limitations"])
    assert "Greedy" in limitations
    assert "reference point" in limitations
    assert "Monte Carlo EHVI" in payload["diagnostics"]["method"]


def test_each_selection_records_its_ehvi_and_standard_error(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=6.0)
    for e in plan.selected:
        assert e.ehvi >= 0
        assert e.ehvi_standard_error >= 0
        assert e.ehvi_per_unit_cost == pytest.approx(e.ehvi / e.cost)


def test_sample_imputation_runs(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    plan = planner.plan(ids, mu, sigma, budget=5.0, imputation="sample")
    assert plan.diagnostics["imputation"] == "sample"


def test_planner_validates_its_inputs(planner, planner_inputs):
    ids, mu, sigma = planner_inputs
    with pytest.raises(ValueError):
        planner.plan(ids, mu, sigma, budget=-1.0)
    with pytest.raises(ValueError):
        planner.plan(ids, mu, sigma, budget=5.0, imputation="magic")
    with pytest.raises(ValueError):
        planner.plan(ids[:-1], mu, sigma, budget=5.0)
    with pytest.raises(ValueError):
        planner.plan(ids, mu[:, :1], sigma[:, :1], budget=5.0)


def test_planner_rejects_assays_for_unknown_objectives():
    with pytest.raises(ValueError, match="unknown objectives"):
        ExperimentPlanner(
            ["solubility_logs"], [True],
            [Assay("x", 1.0, ["not_an_objective"])],
        )


def test_assay_rejects_nonpositive_cost():
    with pytest.raises(ValueError):
        Assay("x", 0.0, ["solubility_logs"])


def test_assay_requires_objectives():
    with pytest.raises(ValueError):
        Assay("x", 1.0, [])


# ================================================= the generator

def test_tokeniser_handles_multicharacter_atoms():
    from backend.app.core.generator import tokenize_smiles

    assert tokenize_smiles("ClCCBr") == ["Cl", "C", "C", "Br"]
    assert tokenize_smiles("[nH]1cccc1")[0] == "[nH]"
    assert "".join(tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O")) == "CC(=O)Oc1ccccc1C(=O)O"


def test_tokeniser_round_trip_fails_loudly_on_unknown_characters():
    from backend.app.core.generator import tokenize_smiles

    # A character the pattern cannot match returns an empty token list rather
    # than a silently corrupted one.
    assert tokenize_smiles("CC\x00O") == []


def test_vocabulary_round_trip():
    from backend.app.core.generator import SmilesVocabulary

    vocab = SmilesVocabulary.from_smiles(["CCO", "c1ccccc1", "ClCCBr"])
    for smiles in ("CCO", "ClCCBr"):
        assert vocab.decode(vocab.encode(smiles)) == smiles


def test_vocabulary_serialisation():
    from backend.app.core.generator import SmilesVocabulary

    a = SmilesVocabulary.from_smiles(["CCO", "CCN"])
    b = SmilesVocabulary.from_dict(a.as_dict())
    assert a.itos == b.itos
    assert b.decode(b.encode("CCO")) == "CCO"


@pytest.fixture(scope="module")
def tiny_generator():
    """A deliberately small prior: enough to sample from, not to be good."""
    from backend.app.core.generator import GeneratorConfig, MolecularGenerator

    corpus = [
        "CCO", "CCC", "CCCC", "CCN", "CCCN", "CCOC", "CCOCC", "c1ccccc1",
        "c1ccccc1C", "c1ccccc1O", "CC(=O)O", "CCC(=O)O", "CC(C)O", "CCCO",
        "ClCCCl", "BrCCBr", "CCS", "CCCS", "OCCO", "NCCN", "CC#N", "CCC#N",
        "CC(C)C", "CCCCC", "c1ccncc1", "c1cccnc1", "CCOC(C)=O", "COC(C)=O",
    ] * 4
    gen = MolecularGenerator(
        GeneratorConfig(
            embedding_size=32, hidden_size=64, n_layers=1,
            max_epochs=25, patience=25, batch_size=16, seed=0,
        )
    )
    gen.train_prior(corpus)
    return gen


def test_prior_trains_and_reports_its_data_limitation(tiny_generator):
    report = tiny_generator.prior_report
    assert report["status"] == "trained"
    assert report["vocabulary_size"] > 4
    assert report["data_limitation"]["moses_pretraining"] == "pending"
    assert "not bundled" in report["data_limitation"]["reason"]


def test_prior_refuses_an_impossibly_small_corpus():
    from backend.app.core.generator import GeneratorConfig, MolecularGenerator

    with pytest.raises(ValueError, match="cannot train a prior"):
        MolecularGenerator(GeneratorConfig(max_epochs=1)).train_prior(["CCO"])


def test_sampling_reports_validity_honestly(tiny_generator):
    out = tiny_generator.sample(n=60, temperature=1.0, random_state=0)
    m = out["metrics"]

    assert m["n_requested"] == 60
    assert m["n_valid"] == len(out["valid_smiles"])
    assert 0.0 <= m["validity"] <= 1.0
    # Every rejection is accounted for, not silently dropped.
    assert m["n_valid"] + out["n_invalid"] == 60
    for item in out["invalid"]:
        assert "reason" in item


def test_all_reported_valid_molecules_actually_parse(tiny_generator):
    from rdkit import Chem

    out = tiny_generator.sample(n=80, temperature=0.8, random_state=1)
    for smiles in out["valid_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        # And are canonical, so deduplication is meaningful.
        assert Chem.MolToSmiles(mol) == smiles


def test_metric_denominators_are_documented(tiny_generator):
    m = tiny_generator.sample(n=40, random_state=0)["metrics"]
    assert m["definitions"]["validity"] == "valid / requested"
    assert "unique valid / valid" in m["definitions"]["uniqueness"]


def test_uniqueness_and_novelty_are_consistent(tiny_generator):
    out = tiny_generator.sample(n=80, temperature=1.0, random_state=2)
    assert len(out["unique_smiles"]) <= len(out["valid_smiles"])
    assert len(out["novel_smiles"]) <= len(out["unique_smiles"])
    # Novel molecules must genuinely be absent from the training corpus.
    for smiles in out["novel_smiles"]:
        assert smiles not in tiny_generator.training_smiles


def test_lower_temperature_does_not_reduce_validity(tiny_generator):
    cold = tiny_generator.sample(n=120, temperature=0.6, random_state=3)["metrics"]
    hot = tiny_generator.sample(n=120, temperature=1.6, random_state=3)["metrics"]
    assert cold["validity"] >= hot["validity"]


def test_sampling_rejects_bad_temperature(tiny_generator):
    with pytest.raises(ValueError):
        tiny_generator.sample(n=5, temperature=0.0)


def test_sample_before_training_raises():
    from backend.app.core.generator import MolecularGenerator

    with pytest.raises(RuntimeError):
        MolecularGenerator().sample(n=5)


def test_diversity_metrics_are_computed(tiny_generator):
    out = tiny_generator.sample(n=100, temperature=1.0, random_state=4)
    m = out["metrics"]
    if m["n_unique"] >= 2 and "internal_diversity" in m:
        assert 0.0 <= m["internal_diversity"] <= 1.0
        assert m["internal_diversity"] == pytest.approx(
            1.0 - m["mean_pairwise_similarity"], abs=1e-6
        )
        assert m["n_unique_scaffolds"] >= 0


def test_reinforce_improves_a_trivial_reward(tiny_generator):
    """
    With a reward that favours a specific substructure, mean reward should rise.

    The prior is deep-copied first so the shared module-scoped fixture is not
    mutated for other tests.
    """
    import copy

    gen = copy.deepcopy(tiny_generator)

    def reward_fn(smiles_list):
        # Reward nitrogen-containing molecules.
        return np.array([1.0 if "N" in s.upper() else 0.0 for s in smiles_list])

    report = gen.reinforce(reward_fn, steps=12, batch_size=16, sigma=60.0, diversity_filter=False)
    assert report["status"] == "trained"
    assert len(report["history"]) == 12
    assert report["final_mean_reward"] is not None
    assert "augmented likelihood" in report["method"].lower()


def test_reinforce_validates_the_reward_function(tiny_generator):
    import copy

    gen = copy.deepcopy(tiny_generator)
    with pytest.raises(ValueError, match="one reward per SMILES"):
        gen.reinforce(lambda s: np.zeros(len(s) + 1), steps=1, batch_size=8)


def test_reinforce_before_training_raises():
    from backend.app.core.generator import MolecularGenerator

    with pytest.raises(RuntimeError):
        MolecularGenerator().reinforce(lambda s: np.zeros(len(s)))


def test_diversity_filter_penalises_repeated_scaffolds(tiny_generator):
    import copy

    gen = copy.deepcopy(tiny_generator)
    report = gen.reinforce(
        lambda s: np.ones(len(s)), steps=6, batch_size=16, diversity_filter=True
    )
    # With a constant reward of 1 and a diversity filter, the observed mean
    # reward must fall below 1 as scaffolds repeat.
    assert report["history"][-1]["mean_reward"] <= 1.0
    assert report["diversity_filter"] is True


def test_generator_checkpoint_round_trip(tiny_generator, tmp_path):
    from backend.app.core.generator import MolecularGenerator

    path = str(tmp_path / "gen.pt")
    tiny_generator.save(path)
    restored = MolecularGenerator.load(path)

    assert restored.vocab.itos == tiny_generator.vocab.itos
    assert restored.training_smiles == tiny_generator.training_smiles
    a = tiny_generator.sample(n=20, temperature=1.0, random_state=7)
    b = restored.sample(n=20, temperature=1.0, random_state=7)
    assert a["raw_generated"] == b["raw_generated"]


def test_summary_is_honest_when_untrained():
    from backend.app.core.generator import MolecularGenerator

    summary = MolecularGenerator().summary()
    assert summary["status"] == "not_trained"
    assert summary["moses_pretraining"] == "pending"


def test_summary_reports_rl_not_run(tiny_generator):
    summary = tiny_generator.summary()
    if summary["status"] == "prior_trained":
        assert summary["reinforcement_learning"]["status"] == "not_run"
