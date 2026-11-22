"""Conformal prediction, applicability domain and Pareto optimisation tests."""

import math
import numpy as np
import pytest
from rdkit import Chem

from backend.app.core.conformal import ConformalPredictionEngine
from backend.app.core.pareto import ParetoLaboratory


# -------------------------------------------------------------- conformal

def test_quantile_uses_finite_sample_correction():
    """q must be the ceil((n+1)(1-alpha))-th order statistic (1-indexed)."""
    engine = ConformalPredictionEngine()
    n = 100
    scores = np.arange(1.0, n + 1.0)  # sorted 1..100
    for alpha in (0.2, 0.1, 0.05):
        expected_rank = math.ceil((n + 1) * (1 - alpha))
        expected = float(scores[min(expected_rank, n) - 1])
        assert engine._compute_quantile(scores, alpha) == expected


def test_quantile_is_monotone_in_confidence():
    engine = ConformalPredictionEngine()
    scores = np.sort(np.abs(np.random.default_rng(0).normal(size=500)))
    q80 = engine._compute_quantile(scores, 0.2)
    q90 = engine._compute_quantile(scores, 0.1)
    q95 = engine._compute_quantile(scores, 0.05)
    assert q80 <= q90 <= q95


def test_empirical_coverage_matches_nominal_on_synthetic_data():
    """Split conformal must deliver marginal coverage on exchangeable data."""
    rng = np.random.default_rng(7)
    n_cal, n_test = 2000, 4000

    y_cal = rng.normal(size=n_cal)
    cal_pred = np.zeros(n_cal)  # unbiased but uninformative predictor
    y_test = rng.normal(size=n_test)
    test_pred = np.zeros(n_test)

    engine = ConformalPredictionEngine()
    engine.calibrate(y_cal, cal_pred, None)
    cov = engine.evaluate_test_coverage(y_test, test_pred, None)

    for level, key in ((0.80, "80%"), (0.90, "90%"), (0.95, "95%")):
        empirical = cov[key]["empirical_coverage"] / 100.0
        assert abs(empirical - level) < 0.03, f"{key}: got {empirical}"


def test_normalised_conformal_matches_standard_when_sigma_constant():
    """With a constant sigma, s = |y-mu|/sigma rescales but does not change widths."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=1000)
    pred = np.zeros(1000)
    sigma = np.full(1000, 2.0)

    std_engine = ConformalPredictionEngine()
    std_engine.calibrate(y, pred, None)
    norm_engine = ConformalPredictionEngine()
    norm_engine.calibrate(y, pred, sigma)

    std_w = std_engine.predict_intervals(0.0, None)["intervals"]["90%"]["interval_width"]
    norm_w = norm_engine.predict_intervals(0.0, 2.0)["intervals"]["90%"]["interval_width"]
    assert std_w == pytest.approx(norm_w, abs=1e-6)


def test_interval_widths_increase_with_confidence():
    rng = np.random.default_rng(11)
    engine = ConformalPredictionEngine()
    engine.calibrate(rng.normal(size=400), np.zeros(400), None)
    iv = engine.predict_intervals(1.0, None)["intervals"]
    assert iv["80%"]["interval_width"] <= iv["90%"]["interval_width"] <= iv["95%"]["interval_width"]


def test_interval_is_centred_on_the_prediction():
    rng = np.random.default_rng(5)
    engine = ConformalPredictionEngine()
    engine.calibrate(rng.normal(size=300), np.zeros(300), None)
    iv = engine.predict_intervals(2.5, None)["intervals"]["90%"]
    assert (iv["lower_bound"] + iv["upper_bound"]) / 2 == pytest.approx(2.5, abs=1e-3)


def test_real_model_coverage_is_close_to_nominal(model_service):
    cov = model_service.esol_metrics["conformal_test_coverage"]
    for key, nominal in (("80%", 80.0), ("90%", 90.0), ("95%", 95.0)):
        empirical = cov[key]["empirical_coverage"]
        # Finite-sample slack at n_cal ~ 170 on a ~140-compound test set.
        assert abs(empirical - nominal) < 10.0, f"{key}: {empirical}"


# ------------------------------------------------------ applicability domain

def test_identical_molecule_is_in_domain(model_service):
    train_smiles = model_service.ad_engine.train_smiles[0]
    mol = Chem.MolFromSmiles(train_smiles)
    feat = model_service.dataset_manager.extract_features_from_mol(mol)
    a = model_service.ad_engine.assess(mol, feat)
    assert a["metrics"]["max_tanimoto_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert a["status"] == "IN_DOMAIN"
    assert a["nearest_analog"]["smiles"] == train_smiles


def test_exotic_molecule_is_flagged(model_service):
    """A large metal-containing cage is far from the Delaney small-molecule domain."""
    smiles = "[Fe+2].[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N"
    mol = Chem.MolFromSmiles(smiles)
    feat = model_service.dataset_manager.extract_features_from_mol(mol)
    a = model_service.ad_engine.assess(mol, feat)
    assert a["status"] in {"OUT_OF_DOMAIN", "BORDERLINE"}
    assert a["metrics"]["max_tanimoto_similarity"] < 0.55


def test_ad_metrics_are_finite_and_bounded(model_service):
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    feat = model_service.dataset_manager.extract_features_from_mol(mol)
    a = model_service.ad_engine.assess(mol, feat)
    m = a["metrics"]
    assert 0.0 <= m["max_tanimoto_similarity"] <= 1.0
    assert 0.0 <= m["mean_top5_tanimoto"] <= 1.0
    assert m["leverage"] >= 0 and math.isfinite(m["leverage"])
    assert 0.0 <= a["reliability_score"] <= 1.0


# ----------------------------------------------------------------- pareto

@pytest.fixture
def lab():
    return ParetoLaboratory()


def test_dominance_truth_table(lab):
    mx = np.array([True, True])
    assert lab.dominates(np.array([2.0, 2.0]), np.array([1.0, 1.0]), mx)
    assert not lab.dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]), mx)
    # Equal on all objectives -> no strict improvement -> no domination.
    assert not lab.dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]), mx)
    # Better on one, worse on the other -> incomparable.
    assert not lab.dominates(np.array([2.0, 0.0]), np.array([1.0, 1.0]), mx)
    # Equal on one, better on the other -> dominates.
    assert lab.dominates(np.array([2.0, 1.0]), np.array([1.0, 1.0]), mx)


def test_minimisation_direction_is_respected(lab):
    mn = np.array([False])
    assert lab.dominates(np.array([1.0]), np.array([5.0]), mn)
    assert not lab.dominates(np.array([5.0]), np.array([1.0]), mn)


def test_known_two_objective_frontier(lab):
    candidates = [
        {"id": "A", "x": 1.0, "y": 5.0},  # non-dominated
        {"id": "B", "x": 5.0, "y": 1.0},  # non-dominated
        {"id": "C", "x": 3.0, "y": 3.0},  # non-dominated
        {"id": "D", "x": 1.0, "y": 1.0},  # dominated by all three
    ]
    out = lab.compute_pareto_front(candidates, ["x", "y"], [True, True])
    ranks = {c["id"]: c["pareto_rank"] for c in out}
    assert ranks["A"] == ranks["B"] == ranks["C"] == 1
    assert ranks["D"] == 2
    assert sum(c["is_pareto_optimal"] for c in out) == 3


def test_missing_objective_raises_instead_of_imputing_zero(lab):
    """Regression test for PROJECT_AUDIT 10.5."""
    candidates = [
        {"id": "A", "tox": 0.5, "sol": -1.0},
        {"id": "B", "sol": -2.0},  # 'tox' absent
    ]
    with pytest.raises(ValueError, match="no usable value"):
        lab.compute_pareto_front(candidates, ["tox", "sol"], [False, True])


def test_nan_objective_raises(lab):
    candidates = [{"id": "A", "x": 1.0}, {"id": "B", "x": float("nan")}]
    with pytest.raises(ValueError):
        lab.compute_pareto_front(candidates, ["x"], [True])


def test_mismatched_objective_and_direction_lengths_raise(lab):
    with pytest.raises(ValueError):
        lab.compute_pareto_front([{"id": "A", "x": 1.0}], ["x"], [True, False])


def test_single_and_empty_candidate_sets(lab):
    assert lab.compute_pareto_front([], ["x"], [True]) == []
    out = lab.compute_pareto_front([{"id": "A", "x": 1.0}], ["x"], [True])
    assert len(out) == 1 and out[0]["is_pareto_optimal"] is True


def test_all_identical_candidates_are_all_optimal(lab):
    candidates = [{"id": str(i), "x": 1.0, "y": 2.0} for i in range(4)]
    out = lab.compute_pareto_front(candidates, ["x", "y"], [True, True])
    assert all(c["is_pareto_optimal"] for c in out)
