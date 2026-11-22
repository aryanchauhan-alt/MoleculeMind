"""Dataset integrity, split hygiene and property-model tests."""

import numpy as np
import pytest
from rdkit import Chem

from backend.app.core.dataset import (
    FEATURE_DIM,
    compute_data_hash,
    murcko_scaffold,
    scaffold_split,
)


# --------------------------------------------------------------- dataset

def test_esol_loads_and_deduplicates(dataset_manager):
    ds = dataset_manager.get_esol_dataset()
    assert ds.n_raw_rows == 1128
    assert ds.n_unparseable == 0
    # The raw Delaney file contains 11 repeated canonical structures.
    assert ds.n_duplicates_removed == 11
    assert len(ds) == 1117


def test_no_duplicate_canonical_smiles_survive(dataset_manager):
    ds = dataset_manager.get_esol_dataset()
    assert len(set(ds.smiles)) == len(ds.smiles)


def test_feature_dimension_and_determinism(dataset_manager):
    ds = dataset_manager.get_esol_dataset()
    assert ds.X.shape == (len(ds), FEATURE_DIM)
    a = dataset_manager.extract_features_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    b = dataset_manager.extract_features_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert np.array_equal(a, b)


def test_features_rejected_for_invalid_smiles(dataset_manager):
    assert dataset_manager.extract_features_from_smiles("not_a_molecule") is None


def test_data_hash_is_order_independent_but_content_sensitive():
    s = ["CCO", "CCC"]
    y = np.array([1.0, 2.0])
    assert compute_data_hash(s, y) == compute_data_hash(s[::-1], y[::-1])
    assert compute_data_hash(s, y) != compute_data_hash(s, np.array([1.0, 2.5]))


def test_clintox_loads_and_deduplicates(dataset_manager):
    ds = dataset_manager.get_clintox_dataset()
    assert ds is not None
    assert len(ds) > 1000
    assert set(np.unique(ds.y)) <= {0.0, 1.0}
    assert len(set(ds.smiles)) == len(ds.smiles)


# -------------------------------------------------------- scaffold split

def test_murcko_scaffold_basic():
    assert murcko_scaffold("c1ccccc1CCN") == "c1ccccc1"
    assert murcko_scaffold("CCCC") == ""  # acyclic


def test_scaffold_split_partitions_are_disjoint_and_complete():
    smiles = [
        "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O", "C1CCCCC1", "C1CCCCC1C",
        "CCCC", "CCCCC", "c1ccncc1", "c1ccncc1C", "CCO",
    ]
    tr, cal, te = scaffold_split(smiles)
    assert set(tr) | set(cal) | set(te) == set(range(len(smiles)))
    assert not (set(tr) & set(cal)) and not (set(tr) & set(te)) and not (set(cal) & set(te))


def test_scaffold_split_never_splits_a_scaffold_group():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "C1CCCCC1", "C1CCCCC1C", "CCO"]
    tr, cal, te = scaffold_split(smiles)
    assignment = {}
    for bucket, idxs in enumerate((tr, cal, te)):
        for i in idxs:
            assignment[i] = bucket
    groups = {}
    for i, s in enumerate(smiles):
        scaf = murcko_scaffold(s)
        if scaf:
            groups.setdefault(scaf, []).append(i)
    for scaf, members in groups.items():
        assert len({assignment[i] for i in members}) == 1, f"scaffold {scaf} was split"


def test_scaffold_split_is_deterministic():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "C1CCCCC1", "CCO", "CCCC", "c1ccncc1"]
    assert scaffold_split(smiles) == scaffold_split(smiles)


# ----------------------------------------------------------- ESOL model

def test_esol_model_trained_and_scored(model_service):
    m = model_service.esol_metrics
    assert m["n_samples"] == 1117
    # A reduced-tree quick model; this is a regression floor, not a target.
    assert m["r2_score"] > 0.80
    assert m["rmse"] < 1.0
    assert m["mae"] < 0.8
    assert m["provenance"]["n_duplicates_removed"] == 11
    assert m["random_seed"] == 42
    assert len(m["provenance"]["data_hash"]) == 16


def test_partitions_do_not_share_structures(model_service):
    """Train, calibration, test and the AL pool must be structurally disjoint."""
    train_smiles = set(model_service.ad_engine.train_smiles)
    test_smiles = {p["smiles"] for p in model_service.test_parity_data}
    al_smiles = {c["smiles"] for c in model_service.al_pool_payload["compounds"]}

    assert train_smiles and test_smiles and al_smiles
    assert not (train_smiles & test_smiles), "train/test structure overlap"
    assert not (train_smiles & al_smiles), "the AL oracle leaks into training"
    assert not (test_smiles & al_smiles), "AL pool overlaps the evaluation set"


def test_scaffold_split_metrics_reported(model_service):
    s = model_service.scaffold_metrics
    assert s.get("status") != "failed"
    assert "r2_score" in s and s["test_samples"] > 0


def test_clintox_is_actually_evaluated(model_service):
    c = model_service.clintox_metrics
    assert c["status"] == "available"
    # Must beat chance; this is a sanity floor, not a performance claim.
    assert c["roc_auc"] > 0.6
    # PR-AUC is only meaningful against the base rate.
    assert c["pr_auc"] > c["baseline_pr_auc"]
    assert c["test_samples"] > 0


def test_prediction_payload_carries_provenance(model_service):
    r = model_service.predict_molecule("CC(=O)Oc1ccccc1C(=O)O")
    assert r["properties"]["solubility_logs"]["provenance"] == "model"
    assert r["properties"]["qed_drug_likeness"]["provenance"] == "computed"
    for key, prop in r["properties"].items():
        assert prop["status"] in {"available", "unavailable"}
        if prop["status"] == "available":
            assert "value" in prop and prop["value"] is not None
        else:
            assert "reason" in prop and "value" not in prop


def test_unvalidated_properties_return_no_number(model_service):
    """The removed descriptor heuristics must not reappear as values."""
    r = model_service.predict_molecule("CC(=O)Oc1ccccc1C(=O)O")
    for key in ("bioactivity_pic50", "metabolic_stability"):
        prop = r["properties"][key]
        assert prop["status"] == "unavailable"
        assert "value" not in prop


def test_prediction_is_reproducible(model_service):
    a = model_service.predict_molecule("CCO")["properties"]["solubility_logs"]["value"]
    b = model_service.predict_molecule("CCO")["properties"]["solubility_logs"]["value"]
    assert a == b


def test_prediction_rejects_invalid_smiles(model_service):
    with pytest.raises(ValueError):
        model_service.predict_molecule("not_a_molecule")


def test_available_objectives_exclude_unvalidated(model_service):
    keys = {o["key"] for o in model_service.available_objectives()}
    assert "solubility_logs" in keys
    assert "bioactivity_pic50" not in keys
    assert "metabolic_stability" not in keys
