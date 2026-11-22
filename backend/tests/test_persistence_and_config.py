"""Stage 2 P1: configuration, model persistence, AD thresholds, recalibration."""

import os
import numpy as np
import pytest
from rdkit import Chem

from backend.app.core.config import Settings, settings
from backend.app.core.models import ModelService, MODEL_VERSION


# ------------------------------------------------------------------ config

def test_settings_are_serialisable_and_complete():
    d = settings.as_dict()
    for key in ("random_seed", "data_dir", "artifact_dir", "esol_n_estimators"):
        assert key in d
    assert isinstance(d["conformal_levels"], list)


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("MOLECULEMIND_SEED", "1234")
    monkeypatch.setenv("MOLECULEMIND_ESOL_TREES", "7")
    monkeypatch.setenv("MOLECULEMIND_USE_MODEL_CACHE", "false")
    s = Settings()
    assert s.random_seed == 1234
    assert s.esol_n_estimators == 7
    assert s.use_model_cache is False


def test_malformed_environment_value_falls_back(monkeypatch):
    monkeypatch.setenv("MOLECULEMIND_SEED", "not-a-number")
    assert Settings().random_seed == 42


def test_training_fingerprint_tracks_training_relevant_settings():
    fp = settings.training_fingerprint()
    assert "random_seed" in fp and "esol_n_estimators" in fp
    # Presentation-only settings must not affect model identity.
    assert "log_level" not in fp and "cors_origins" not in fp


# ------------------------------------------------------------- persistence

def test_cache_key_is_stable_and_config_sensitive(model_service):
    assert model_service.cache_key() == model_service.cache_key()

    other = ModelService(data_dir=model_service.dataset_manager.data_dir, quick=False)
    # `quick` changes the trained artifact, so it must change the key.
    assert other.cache_key() != model_service.cache_key()


def test_cache_key_is_cheap(model_service):
    """A cache hit must not require featurising the datasets first."""
    fresh = ModelService(data_dir=model_service.dataset_manager.data_dir, quick=True)
    key = fresh.cache_key()
    assert len(key) == 20
    # Computing the key must not have populated the parsed-dataset caches.
    assert fresh.dataset_manager._esol_cache is None


def test_save_and_load_round_trip(tmp_path, data_dir):
    artifacts = str(tmp_path / "artifacts")

    first = ModelService(data_dir=data_dir, quick=True, artifact_dir=artifacts)
    first.initialize()
    assert first.loaded_from_cache is False
    assert os.path.exists(first.cache_path())

    second = ModelService(data_dir=data_dir, quick=True, artifact_dir=artifacts)
    second.initialize()
    assert second.loaded_from_cache is True

    # A restored model must behave identically.
    a = first.predict_molecule("CC(=O)Oc1ccccc1C(=O)O")
    b = second.predict_molecule("CC(=O)Oc1ccccc1C(=O)O")
    assert a["properties"]["solubility_logs"]["value"] == b["properties"]["solubility_logs"]["value"]
    assert a["applicability_domain"]["status"] == b["applicability_domain"]["status"]
    assert a["applicability_domain"]["metrics"] == b["applicability_domain"]["metrics"]
    assert first.esol_metrics["r2_score"] == second.esol_metrics["r2_score"]


def test_cache_can_be_disabled(tmp_path, data_dir):
    artifacts = str(tmp_path / "artifacts")
    ModelService(data_dir=data_dir, quick=True, artifact_dir=artifacts).initialize()

    svc = ModelService(
        data_dir=data_dir, quick=True, artifact_dir=artifacts, use_cache=False
    )
    svc.initialize()
    assert svc.loaded_from_cache is False


def test_corrupt_cache_falls_back_to_training(tmp_path, data_dir):
    artifacts = str(tmp_path / "artifacts")
    first = ModelService(data_dir=data_dir, quick=True, artifact_dir=artifacts)
    first.initialize()

    with open(first.cache_path(), "wb") as fh:
        fh.write(b"this is not a pickle")

    second = ModelService(data_dir=data_dir, quick=True, artifact_dir=artifacts)
    second.initialize()  # must not raise
    assert second.loaded_from_cache is False
    assert second.esol_metrics["r2_score"] > 0.8


def test_cache_write_failure_does_not_break_startup(tmp_path, data_dir, monkeypatch):
    svc = ModelService(
        data_dir=data_dir, quick=True, artifact_dir=str(tmp_path / "a"), use_cache=False
    )
    monkeypatch.setattr(
        "backend.app.core.models.pickle.dump",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    svc.initialize()  # must not raise
    assert svc.is_initialized is True
    # No partial artifact left behind.
    assert not os.path.exists(svc.cache_path() + ".tmp")


def test_model_version_is_recorded(model_service):
    assert model_service.model_version == MODEL_VERSION
    assert model_service.esol_metrics["model_version"] == MODEL_VERSION


# -------------------------------------------- applicability-domain thresholds

def test_thresholds_are_derived_not_hardcoded(model_service):
    d = model_service.ad_engine.diagnostics()
    assert d["n_training_molecules"] > 100
    assert 0.0 < d["similarity_threshold"] < 1.0
    assert d["leverage_threshold"] > 0
    # The old hardcoded Tanimoto cut-offs must not reappear.
    assert d["similarity_threshold"] not in (0.55, 0.35)
    assert d["similarity_percentile"] == settings.ad_similarity_percentile


def test_similarity_threshold_matches_its_percentile(model_service):
    """~95% of training compounds should sit at or above the 5th-percentile cut-off."""
    d = model_service.ad_engine.diagnostics()
    loo = d["training_loo_similarity"]
    assert loo["min"] <= d["similarity_threshold"] <= loo["max"]
    assert loo["median"] >= d["similarity_threshold"]


def test_few_training_molecules_are_flagged(model_service):
    """The two-condition rule should not reject the training set it was fitted on."""
    assert model_service.ad_engine.diagnostics()["training_flag_rate"] < 0.10


def test_ad_features_are_standardised(model_service):
    """Leverage must not be dominated by molecular weight (audit 10.6)."""
    engine = model_service.ad_engine
    assert engine.scaler is not None
    # With standardised inputs no single component should carry ~91% of variance.
    top = float(engine.pca.explained_variance_ratio_[0])
    assert top < 0.5, f"PC1 carries {top:.1%} of the variance; features look unscaled"


def test_two_condition_rule_is_reported(model_service):
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    feat = model_service.dataset_manager.extract_features_from_mol(mol)
    a = model_service.ad_engine.assess(mol, feat)
    c = a["conditions"]
    assert {"similarity_fails", "leverage_fails", "second_condition", "rule"} <= set(c)
    # Without a fitted latent space, leverage is the second condition.
    assert c["second_condition"] == "descriptor-space leverage"
    assert c["latent_distance_fails"] is None
    if c["similarity_fails"] and c["leverage_fails"]:
        assert a["status"] == "OUT_OF_DOMAIN"
    elif c["similarity_fails"] or c["leverage_fails"]:
        assert a["status"] == "BORDERLINE"
    else:
        assert a["status"] == "IN_DOMAIN"


def test_domain_index_is_not_called_a_probability(model_service):
    mol = Chem.MolFromSmiles("CCO")
    feat = model_service.dataset_manager.extract_features_from_mol(mol)
    a = model_service.ad_engine.assess(mol, feat)
    assert 0.0 <= a["domain_index"] <= 1.0
    assert "not a calibrated probability" in a["index_caveat"]


def test_assess_before_fit_raises():
    from backend.app.core.applicability_domain import ApplicabilityDomainEngine

    with pytest.raises(RuntimeError):
        ApplicabilityDomainEngine().assess(Chem.MolFromSmiles("CCO"), np.zeros(1032))


def test_fit_rejects_a_degenerate_training_set():
    from backend.app.core.applicability_domain import ApplicabilityDomainEngine

    with pytest.raises(ValueError):
        ApplicabilityDomainEngine().fit(["CCO"], ["x"], [1.0], np.zeros((1, 1032)))


# ------------------------------------------------------------ recalibration

def test_recalibration_extends_the_calibration_set(model_service):
    before = model_service.conformal_engine.cal_size
    gen_before = model_service.calibration_generation

    result = model_service.recalibrate_conformal(
        ["CCO", "c1ccccc1O"], [-0.31, -0.40], trigger="unit test"
    )

    assert result["extra_labels"] == 2
    assert result["calibration_set_size"] == before + 2
    assert model_service.conformal_engine.cal_size == before + 2
    assert model_service.calibration_generation == gen_before + 1
    assert model_service.calibration_history[-1]["trigger"] == "unit test"
    assert "approximate" in result["caveat"]


def test_recalibration_skips_unparseable_molecules(model_service):
    result = model_service.recalibrate_conformal(
        ["not_a_molecule"], [1.0], trigger="unit test"
    )
    assert result["extra_labels"] == 0
    assert result["skipped_unparseable"] == ["not_a_molecule"]


def test_recalibration_rejects_mismatched_lengths(model_service):
    with pytest.raises(ValueError):
        model_service.recalibrate_conformal(["CCO", "CCC"], [1.0])


def test_assay_triggers_recalibration(model_service):
    """An AL round must refit calibration rather than reuse a stale quantile."""
    from backend.app.core.active_learning import ActiveLearningEngine
    import copy

    engine = ActiveLearningEngine(
        dataset_manager=model_service.dataset_manager,
        pool_payload=copy.deepcopy(model_service.al_pool_payload),
        model_service=model_service,
    )
    gen_before = model_service.calibration_generation
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])

    result = engine.run_assay([target])

    assert result["recalibration"]["status"] == "ok"
    assert model_service.calibration_generation == gen_before + 1


def test_engine_without_model_service_reports_skipped(al_engine):
    """The campaign still runs unattached, but says calibration was not refreshed."""
    import copy
    from backend.app.core.active_learning import ActiveLearningEngine

    engine = ActiveLearningEngine(
        dataset_manager=al_engine.dm,
        pool_payload=copy.deepcopy(al_engine.pool_payload),
        model_service=None,
    )
    target = next(c["id"] for c in engine.public_pool() if not c["assayed"])
    result = engine.run_assay([target])
    assert result["recalibration"]["status"] == "skipped"
