"""
Active Learning tests.

The first two tests are direct regression guards for the two critical defects
recorded in PROJECT_AUDIT.md:
  * 10.2 - every candidate received an identical acquisition score
  * 10.3 - hidden oracle labels were serialised to the client
"""

import copy
import pytest

from backend.app.core.active_learning import ActiveLearningEngine, _ORACLE_KEYS


@pytest.fixture
def engine(model_service):
    """A fresh engine per test; the campaign is stateful."""
    return ActiveLearningEngine(
        dataset_manager=model_service.dataset_manager,
        pool_payload=copy.deepcopy(model_service.al_pool_payload),
    )


# ------------------------------------------------- 10.2: acquisition quality

@pytest.mark.parametrize("strategy", ["ucb", "ei", "uncertainty", "greedy"])
def test_acquisition_is_discriminative(engine, strategy):
    """
    Regression guard. Previously the GP received unscaled PCA inputs spanning
    ~290 units, the posterior collapsed to the prior, and all 24 unlabelled
    candidates scored identically (UCB = 1.414 for every one of them).
    """
    engine.rank_candidates(strategy=strategy)
    diag = engine.acquisition_spread()
    assert diag["n"] > 5
    assert diag["degenerate"] is False
    # A working GP should separate most of the pool, not just a couple of points.
    assert diag["n_unique"] >= max(3, diag["n"] // 2)
    assert diag["max"] > diag["min"]


def test_gp_posterior_variance_is_positive_and_finite(engine):
    import math

    engine.rank_candidates(strategy="ucb")
    sigmas = [
        c["uncertainty_sigma"] for c in engine.public_pool() if not c["assayed"]
    ]
    assert sigmas
    assert all(s is not None and s > 0 and math.isfinite(s) for s in sigmas)


def test_gp_fit_status_is_reported_not_swallowed(engine):
    engine.rank_candidates(strategy="ucb")
    assert engine.last_fit_ok is not None  # True or False, never silently unknown


def test_ranking_responds_to_new_labels(engine):
    engine.rank_candidates(strategy="ucb")
    before = {
        c["id"]: c["acquisition_score"]
        for c in engine.public_pool()
        if not c["assayed"]
    }
    top = max(before, key=lambda k: before[k])

    engine.run_assay([top])
    after = {
        c["id"]: c["acquisition_score"]
        for c in engine.public_pool()
        if not c["assayed"]
    }

    shared = set(before) & set(after)
    assert shared, "pool should not be exhausted"
    # Conditioning on a new observation must move at least some scores.
    assert any(before[k] != after[k] for k in shared)


@pytest.mark.parametrize("strategy", ["ucb", "ei", "uncertainty", "greedy"])
def test_all_strategies_produce_scores(engine, strategy):
    pool = engine.rank_candidates(strategy=strategy)
    unlabelled = [c for c in pool if not c["assayed"]]
    assert all(c["acquisition_score"] is not None for c in unlabelled)


def test_unknown_strategy_falls_back_to_ucb(engine):
    pool = engine.rank_candidates(strategy="nonsense")
    assert any(c["acquisition_score"] is not None for c in pool if not c["assayed"])


# -------------------------------------------------- 10.3: oracle separation

def test_public_pool_contains_no_oracle_keys(engine):
    """Regression guard: `true_pic50`/`measured_logs` must never be serialised."""
    for record in engine.public_pool():
        assert not (_ORACLE_KEYS & set(record.keys()))


def test_unassayed_candidates_expose_no_measurement(engine):
    for record in engine.public_pool():
        if not record["assayed"]:
            assert record["measured_value"] is None
            assert record["assay_round"] is None


def test_measurement_appears_only_after_an_assay(engine):
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])

    before = next(c for c in engine.public_pool() if c["id"] == target)
    assert before["measured_value"] is None

    engine.run_assay([target])

    after = next(c for c in engine.public_pool() if c["id"] == target)
    assert after["assayed"] is True
    assert after["measured_value"] is not None
    assert after["measured_value"] == pytest.approx(engine._oracle[target], abs=1e-3)


def test_oracle_is_held_privately(engine):
    """Every candidate has a stored measurement even though none is published."""
    assert len(engine._oracle) == len(engine.candidates)
    published = [
        c["measured_value"] for c in engine.public_pool() if c["measured_value"] is not None
    ]
    assert len(published) == len(engine._revealed) < len(engine._oracle)


def test_campaign_state_carries_no_oracle(engine):
    import json

    state = json.dumps(engine.campaign_state("ucb", 2.0))
    for key in _ORACLE_KEYS:
        assert key not in state


# ------------------------------------------------------------ assay safety

def test_already_assayed_candidate_is_not_reselected(engine):
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])
    engine.run_assay([target])

    result = engine.run_assay([target])
    assert result["status"] == "no_op"
    assert target in result["rejected_already_assayed"]


def test_labelled_candidates_carry_no_acquisition_score(engine):
    engine.rank_candidates(strategy="ucb")
    for c in engine.public_pool():
        if c["assayed"]:
            assert c["acquisition_score"] is None


def test_unknown_candidate_id_is_rejected(engine):
    result = engine.run_assay(["NOT-A-REAL-ID"])
    assert result["status"] == "no_op"
    assert "NOT-A-REAL-ID" in result["rejected_unknown"]


def test_assay_advances_iteration_and_history(engine):
    start_iter = engine.iteration
    start_len = len(engine.history)
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])

    engine.run_assay([target])

    assert engine.iteration == start_iter + 1
    assert len(engine.history) == start_len + 1
    assert engine.history[-1]["round"] == engine.iteration


def test_simulated_mode_is_declared(engine):
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])
    result = engine.run_assay([target])
    assert result["mode"] == "simulated"
    assert "no laboratory experiment" in result["mode_note"].lower()
    assert engine.campaign_state("ucb", 2.0)["mode"] == "simulated"


def test_reset_restores_the_initial_campaign(engine):
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])
    engine.run_assay([target])
    assert engine.iteration == 1

    engine.reset()
    assert engine.iteration == 0
    assert sum(c["assayed"] for c in engine.public_pool()) == engine.N_SEED


def test_pool_is_disjoint_from_model_training_data(engine, model_service):
    pool = {c["smiles"] for c in engine.public_pool()}
    train = set(model_service.ad_engine.train_smiles)
    assert pool and not (pool & train)
