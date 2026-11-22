"""
Stage 3 integration: conformal prediction sets, latent-space applicability
domain, and the D-MPNN service with independent model comparison.
"""

import numpy as np
import pytest
from rdkit import Chem

from backend.app.core.conformal import ClassificationConformalEngine


# ------------------------------------------- classification conformal sets

@pytest.fixture(scope="module")
def synthetic_classification():
    """A 3-class problem with an informative but imperfect classifier."""
    rng = np.random.default_rng(0)

    def make(n):
        y = rng.integers(0, 3, n)
        logits = rng.normal(size=(n, 3))
        logits[np.arange(n), y] += 1.5
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return y, e / e.sum(axis=1, keepdims=True)

    y_cal, p_cal = make(4000)
    y_test, p_test = make(6000)
    engine = ClassificationConformalEngine(class_names=["A", "B", "C"])
    engine.calibrate(y_cal, p_cal)
    return engine, y_test, p_test


def test_lac_achieves_nominal_coverage(synthetic_classification):
    engine, y_test, p_test = synthetic_classification
    cov = engine.evaluate_coverage(y_test, p_test)["lac"]
    for key, nominal in (("80%", 80.0), ("90%", 90.0), ("95%", 95.0)):
        assert abs(cov[key]["empirical_coverage"] - nominal) < 3.0, f"{key}: {cov[key]}"


def test_randomized_aps_achieves_nominal_coverage(synthetic_classification):
    engine, y_test, p_test = synthetic_classification
    cov = engine.evaluate_coverage(y_test, p_test)["aps_randomized"]
    for key, nominal in (("80%", 80.0), ("90%", 90.0), ("95%", 95.0)):
        assert abs(cov[key]["empirical_coverage"] - nominal) < 3.0, f"{key}: {cov[key]}"


def test_deterministic_aps_is_conservative_not_wrong(synthetic_classification):
    """
    Unrandomised APS over-covers by construction. That is valid (coverage is a
    lower bound) but it must be visible, not mistaken for exactness.
    """
    engine, y_test, p_test = synthetic_classification
    cov = engine.evaluate_coverage(y_test, p_test)
    det = cov["aps"]["90%"]
    rand = cov["aps_randomized"]["90%"]
    assert det["empirical_coverage"] >= 90.0
    assert det["empirical_coverage"] > rand["empirical_coverage"]
    assert det["avg_set_size"] > rand["avg_set_size"]


def test_lac_gives_the_smallest_sets(synthetic_classification):
    engine, y_test, p_test = synthetic_classification
    cov = engine.evaluate_coverage(y_test, p_test)
    assert cov["lac"]["90%"]["avg_set_size"] <= cov["aps_randomized"]["90%"]["avg_set_size"]


def test_conformal_quantile_uses_finite_sample_correction():
    import math

    engine = ClassificationConformalEngine()
    scores = np.arange(1.0, 101.0)
    for alpha in (0.2, 0.1, 0.05):
        expected = float(scores[math.ceil(101 * (1 - alpha)) - 1])
        assert engine._conformal_quantile(scores, alpha) == expected


def test_too_little_calibration_data_falls_back_conservatively():
    """With n too small for the level, the threshold must be maximally safe."""
    engine = ClassificationConformalEngine()
    # n = 5 cannot support 95% coverage: ceil(6 * 0.95) = 6 > 5.
    assert engine._conformal_quantile(np.linspace(0, 1, 5), 0.05) == 1.0


def test_set_sizes_grow_with_confidence(synthetic_classification):
    engine, _, p_test = synthetic_classification
    sizes = {
        c: np.mean([len(s) for s in engine.predict_set(p_test, c, "lac")])
        for c in (0.80, 0.90, 0.95)
    }
    assert sizes[0.80] <= sizes[0.90] <= sizes[0.95]


def test_binary_probabilities_are_accepted():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.2, 3000), 0.01, 0.99)
    engine = ClassificationConformalEngine(class_names=["safe", "toxic"])
    engine.calibrate(y[:1500], p[:1500])  # 1-D input
    cov = engine.evaluate_coverage(y[1500:], p[1500:])
    assert abs(cov["lac"]["90%"]["empirical_coverage"] - 90.0) < 4.0


def test_describe_set_is_human_readable(synthetic_classification):
    engine, _, _ = synthetic_classification
    d = engine.describe_set(np.array([0.1, 0.7, 0.2]), 0.90, "lac")
    assert set(d["labels"]) <= {"A", "B", "C"}
    assert d["set_size"] == len(d["label_indices"])
    assert "not a per-molecule probability" in d["interpretation"]


def test_randomized_sets_are_reproducible_for_a_fixed_state(synthetic_classification):
    engine, _, p_test = synthetic_classification
    a = engine.predict_set(p_test[:50], 0.90, "aps_randomized", random_state=3)
    b = engine.predict_set(p_test[:50], 0.90, "aps_randomized", random_state=3)
    assert a == b


def test_unknown_method_and_uncalibrated_level_rejected(synthetic_classification):
    engine, _, p_test = synthetic_classification
    with pytest.raises(ValueError):
        engine.predict_set(p_test[:5], 0.90, "nonsense")
    with pytest.raises(ValueError):
        engine.predict_set(p_test[:5], 0.77, "lac")


def test_predict_set_before_calibrate_raises():
    with pytest.raises(RuntimeError):
        ClassificationConformalEngine().predict_set(np.array([[0.5, 0.5]]))


def test_mismatched_calibration_lengths_rejected():
    with pytest.raises(ValueError):
        ClassificationConformalEngine().calibrate(
            np.array([0, 1]), np.array([[0.5, 0.5]])
        )


# ------------------------------------------- latent applicability domain

def test_latent_space_requires_matching_row_count(model_service):
    engine = model_service.ad_engine
    with pytest.raises(ValueError):
        engine.fit_latent(np.zeros((3, 8)))


def test_latent_fit_and_distance(model_service):
    """Fit a synthetic latent space and check the threshold and distances."""
    from backend.app.core.applicability_domain import ApplicabilityDomainEngine

    engine = model_service.ad_engine
    n = engine.n_samples
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 16))

    diag = engine.fit_latent(emb, percentile=95.0)
    try:
        assert diag["enabled"] is True
        assert diag["embedding_dim"] == 16
        assert diag["latent_threshold"] > 0
        assert engine.has_latent_space

        # A point copied from the training set has distance ~0.
        assert engine.latent_distance(emb[0]) == pytest.approx(0.0, abs=1e-9)
        # A far-away point must exceed the threshold.
        assert engine.latent_distance(np.full(16, 1e3)) > engine.latent_threshold
        # Dimension mismatch is an error, not a silent wrong answer.
        with pytest.raises(ValueError):
            engine.latent_distance(np.zeros(5))
    finally:
        # Restore state so later tests see the real (or absent) latent space.
        engine.train_latent = None
        engine.latent_threshold = 0.0
        engine.latent_diagnostics = {}
        engine.threshold_diagnostics.pop("latent_space", None)


def test_latent_distance_becomes_the_second_condition(model_service):
    """With a latent space fitted, the AD rule must switch its second test."""
    engine = model_service.ad_engine
    n = engine.n_samples
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(n, 12))
    engine.fit_latent(emb, percentile=95.0)
    try:
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        feat = model_service.dataset_manager.extract_features_from_mol(mol)

        with_latent = engine.assess(mol, feat, latent_vector=emb[0])
        assert with_latent["conditions"]["second_condition"] == (
            "latent-space nearest-neighbour distance"
        )
        assert with_latent["metrics"]["latent_distance"] is not None
        assert with_latent["conditions"]["latent_distance_fails"] is False

        # An extreme embedding fails the second condition.
        far = engine.assess(mol, feat, latent_vector=np.full(12, 1e3))
        assert far["conditions"]["latent_distance_fails"] is True

        # Without a latent vector it falls back to leverage.
        without = engine.assess(mol, feat)
        assert without["conditions"]["second_condition"] == "descriptor-space leverage"
        assert without["metrics"]["latent_distance"] is None
    finally:
        engine.train_latent = None
        engine.latent_threshold = 0.0
        engine.latent_diagnostics = {}
        engine.threshold_diagnostics.pop("latent_space", None)


def test_diagnostics_report_latent_state(model_service):
    d = model_service.ad_engine.diagnostics()
    assert "latent_space" in d
    assert "enabled" in d["latent_space"]


# --------------------------------------------------- D-MPNN service

@pytest.fixture(scope="module")
def dmpnn_service(data_dir):
    """A deliberately tiny service: 2 members, 8 epochs, hidden 64."""
    from backend.app.core.dataset import DatasetManager
    from backend.app.core.dmpnn_service import DMPNNService

    service = DMPNNService(DatasetManager(data_dir=data_dir), quick=True, n_members=2)
    service.train()
    return service


def test_multitask_table_joins_both_datasets(dmpnn_service):
    table = dmpnn_service.build_multitask_table()
    assert table["n_molecules"] > table["n_solubility_labels"]
    assert table["n_molecules"] > table["n_toxicity_labels"]
    assert table["n_solubility_labels"] == 1117
    assert table["n_toxicity_labels"] > 1000
    # The union is what makes this multi-task: most molecules have one label.
    assert table["n_with_both_labels"] < table["n_molecules"]
    assert table["targets"].shape == (table["n_molecules"], 2)
    # Missing labels must be NaN, never imputed.
    assert np.isnan(table["targets"]).any()


def test_service_trains_and_reports_both_tasks(dmpnn_service):
    m = dmpnn_service.metrics
    assert dmpnn_service.is_trained
    assert m["solubility"]["r2_score"] > 0.3  # a tiny smoke config, not a target
    assert m["solubility"]["n_test"] > 0
    assert "conformal_coverage" in m["solubility"]
    assert m["toxicity"]["roc_auc"] > 0.4


def test_variance_is_decomposed_in_the_metrics(dmpnn_service):
    s = dmpnn_service.metrics["solubility"]
    assert s["mean_aleatoric_variance"] > 0
    assert s["mean_epistemic_variance"] >= 0
    assert s["mean_sigma"] > 0


def test_service_prediction_shape_and_provenance(dmpnn_service):
    r = dmpnn_service.predict("CC(=O)Oc1ccccc1C(=O)O")
    sol = r["solubility_logs"]
    assert sol["provenance"] == "model"
    assert sol["status"] == "available"
    assert len(sol["member_predictions"]) == 2
    iv = sol["conformal"]["intervals"]["90%"]
    assert iv["lower_bound"] < sol["value"] < iv["upper_bound"]
    assert 0.0 <= r["toxicity"]["value"] <= 1.0


def test_service_prediction_includes_conformal_sets(dmpnn_service):
    r = dmpnn_service.predict("CCO")
    sets = r["toxicity"].get("conformal_sets")
    if sets is None:
        pytest.skip("Too few calibration labels for classification conformal sets.")
    for method in ("lac", "aps_randomized"):
        assert sets[method]["set_size"] == len(sets[method]["label_indices"])
        assert sets[method]["confidence"] == 0.90


def test_service_rejects_invalid_smiles(dmpnn_service):
    with pytest.raises(ValueError):
        dmpnn_service.predict("not_a_molecule")


def test_independent_comparison_runs(dmpnn_service):
    c = dmpnn_service.comparison
    assert c["status"] == "ok"
    assert "RandomForest" in c["independent_model"]
    agree = c["prediction_agreement"]
    assert -1.0 <= agree["pearson_r"] <= 1.0
    assert agree["mean_absolute_difference"] >= 0
    assert 0.0 <= agree["fraction_exceeding_dmpnn_sigma"] <= 1.0
    assert len(c["largest_disagreements"]) <= 5
    assert "not a correctness guarantee" in c["caveat"]


def test_comparison_uses_a_different_representation(dmpnn_service):
    """The independent model must not be another graph network."""
    c = dmpnn_service.comparison
    assert "Morgan fingerprint" in c["independent_model"]
    assert c["dmpnn"]["r2_score"] is not None
    assert c["independent"]["r2_score"] is not None


def test_summary_is_honest_when_untrained(data_dir):
    from backend.app.core.dataset import DatasetManager
    from backend.app.core.dmpnn_service import DMPNNService

    service = DMPNNService(DatasetManager(data_dir=data_dir), quick=True, n_members=1)
    summary = service.summary()
    assert summary["status"] == "not_trained"
    assert "reason" in summary
    assert "metrics" not in summary


def test_summary_describes_the_architecture(dmpnn_service):
    a = dmpnn_service.summary()["architecture"]
    assert a["readout"] == "mean over atoms"
    assert a["regression_loss"] == "Gaussian negative log-likelihood"
    assert a["message_passing_steps"] == 3
    assert a["n_members"] == 2


def test_embeddings_have_the_configured_width(dmpnn_service):
    emb = dmpnn_service.embed_smiles(["CCO", "CC(=O)Oc1ccccc1C(=O)O"])
    assert emb is not None
    assert emb.shape == (2, dmpnn_service.metrics["config"]["hidden_size"])


def test_embedding_reports_failure_rather_than_silently_dropping(dmpnn_service):
    """A partly unparseable list must not return a silently shorter matrix."""
    assert dmpnn_service.embed_smiles(["CCO", "not_a_molecule"]) is None


def test_checkpoint_round_trip(dmpnn_service, tmp_path):
    key = "testkey"
    dmpnn_service.save(str(tmp_path), key)

    from backend.app.core.dataset import DatasetManager
    from backend.app.core.dmpnn_service import DMPNNService

    restored = DMPNNService(DatasetManager(data_dir=dmpnn_service.dm.data_dir), quick=True)
    assert restored.load(str(tmp_path), key) is True
    assert restored.is_trained

    a = dmpnn_service.ensemble.predict([__import__(
        "backend.app.core.molecular_graph", fromlist=["smiles_to_graph"]
    ).smiles_to_graph("CCO")])
    b = restored.ensemble.predict([__import__(
        "backend.app.core.molecular_graph", fromlist=["smiles_to_graph"]
    ).smiles_to_graph("CCO")])
    assert np.allclose(a["mu"], b["mu"], atol=1e-6)


def test_load_missing_checkpoint_returns_false(data_dir, tmp_path):
    from backend.app.core.dataset import DatasetManager
    from backend.app.core.dmpnn_service import DMPNNService

    service = DMPNNService(DatasetManager(data_dir=data_dir), quick=True)
    assert service.load(str(tmp_path), "nonexistent") is False
    assert service.is_trained is False
