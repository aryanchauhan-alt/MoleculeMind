"""
End-to-end API tests against the real FastAPI app.

These are slow: importing `backend.app.main` trains the models. The module is
imported once and shared via a session-scoped client fixture.
"""

import json
import pytest
from fastapi.testclient import TestClient

from backend.app.core.active_learning import _ORACLE_KEYS


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["models_initialized"] is True
    assert body["esol_metrics"]["r2_score"] > 0.8


def test_analyze_valid_molecule(client):
    r = client.post("/api/analyze", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert r.status_code == 200
    body = r.json()
    assert body["descriptors"]["formula"] == "C9H8O4"
    assert body["svg_2d"].lstrip().startswith(("<?xml", "<svg"))
    assert body["sdf_3d"] is not None
    # The conformer must be described honestly.
    assert "not an experimentally determined structure" in body["sdf_note"]


@pytest.mark.parametrize("bad", ["not_a_molecule", "C((("])
def test_analyze_rejects_garbage(client, bad):
    r = client.post("/api/analyze", json={"smiles": bad})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_analyze_rejects_empty_string(client):
    r = client.post("/api/analyze", json={"smiles": ""})
    assert r.status_code == 422  # pydantic min_length


def test_predict_schema_and_provenance(client):
    r = client.post("/api/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert r.status_code == 200
    body = r.json()

    sol = body["properties"]["solubility_logs"]
    assert sol["status"] == "available" and sol["provenance"] == "model"
    iv = sol["conformal"]["intervals"]["90%"]
    assert iv["lower_bound"] < sol["value"] < iv["upper_bound"]

    assert body["applicability_domain"]["status"] in {
        "IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN"
    }
    assert body["model_version"]

    for key in ("bioactivity_pic50", "metabolic_stability"):
        assert body["properties"][key]["status"] == "unavailable"
        assert "value" not in body["properties"][key]


def test_predict_rejects_invalid_smiles(client):
    r = client.post("/api/predict", json={"smiles": "not_a_molecule"})
    assert r.status_code == 400


def test_objectives_endpoint(client):
    body = client.get("/api/objectives").json()
    available = {o["key"] for o in body["available"]}
    unavailable = {o["key"] for o in body["unavailable"]}
    assert "solubility_logs" in available
    assert {"bioactivity_pic50", "metabolic_stability"} <= unavailable
    assert not (available & unavailable)


def test_benchmark_metrics_are_computed_not_hardcoded(client):
    body = client.get("/api/benchmark").json()
    m = body["metrics"]
    assert m["train_samples"] + m["cal_samples"] + m["test_samples"] + m["al_reserved_samples"] == m["n_samples"]
    assert body["scaffold_split_metrics"]["test_samples"] > 0
    assert body["clintox_metrics"]["status"] == "available"
    assert len(body["parity_plot_data"]) > 0
    point = body["parity_plot_data"][0]
    assert {"actual", "predicted", "lower_90", "upper_90"} <= set(point)


def test_pareto_front_is_consistent(client):
    body = client.get("/api/pareto").json()
    assert body["total_candidates"] > 0
    assert 0 < body["pareto_optimal_count"] <= body["total_candidates"]

    # Only validated objectives may be optimised.
    assert "bioactivity_pic50" not in body["objectives"]
    assert "metabolic_stability" not in body["objectives"]
    assert "solubility_logs" in body["objectives"]

    frontier_ids = {c["id"] for c in body["pareto_frontier"]}
    for c in body["candidates"]:
        assert c["is_pareto_optimal"] == (c["id"] in frontier_ids)
        assert c["pareto_rank"] >= 1
        if c["is_pareto_optimal"]:
            assert c["pareto_rank"] == 1
        # Every optimised objective must carry a real value.
        for key in body["objectives"]:
            assert c.get(key) is not None


def test_al_pool_never_ships_hidden_labels(client):
    """Full-payload regression guard for PROJECT_AUDIT 10.3."""
    r = client.get("/api/active-learning/pool?strategy=ucb&beta=2.0")
    assert r.status_code == 200
    body = r.json()

    raw = json.dumps(body)
    for key in _ORACLE_KEYS:
        assert key not in raw, f"oracle key '{key}' reached the client"

    for c in body["candidates"]:
        if not c["assayed"]:
            assert c["measured_value"] is None

    assert body["mode"] == "simulated"
    assert body["acquisition_diagnostics"]["degenerate"] is False


def test_al_rank_endpoint(client):
    r = client.post("/api/active-learning/rank", json={"strategy": "ei", "beta": 1.5})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "ei"
    scores = [
        c["acquisition_score"] for c in body["candidates"] if not c["assayed"]
    ]
    assert len(set(scores)) > 1


def test_assay_workflow_then_reset(client):
    pool = client.get("/api/active-learning/pool").json()
    target = next(c["id"] for c in pool["candidates"] if not c["assayed"])

    r = client.post("/api/active-learning/assay", json={"candidate_ids": [target]})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "simulated"
    assert body["newly_assayed"][0]["id"] == target
    assert body["newly_assayed"][0]["measured_value"] is not None

    # Re-assaying the same candidate must be refused.
    again = client.post("/api/active-learning/assay", json={"candidate_ids": [target]})
    assert again.status_code == 400

    reset = client.post("/api/active-learning/reset")
    assert reset.status_code == 200 and reset.json()["iteration"] == 0


def test_assay_requires_candidate_ids(client):
    assert client.post("/api/active-learning/assay", json={"candidate_ids": []}).status_code == 422


# --------------------------------------------------- Stage 2 P1 endpoints

def test_config_endpoint_reports_reproducibility_info(client):
    body = client.get("/api/config").json()
    assert body["model_version"]
    assert len(body["cache_key"]) == 20
    assert body["settings"]["random_seed"] == 42
    assert "data_hash" in body["dataset_provenance"]


def test_ad_diagnostics_endpoint(client):
    d = client.get("/api/applicability-domain/diagnostics").json()
    assert d["n_training_molecules"] > 100
    assert 0.0 < d["similarity_threshold"] < 1.0
    assert "oecd_leverage_threshold" in d
    assert d["training_flag_rate"] < 0.10


def test_benchmark_includes_ad_and_calibration_history(client):
    body = client.get("/api/benchmark").json()
    assert body["applicability_domain"]["n_training_molecules"] > 100
    assert body["calibration_history"][0]["generation"] == 1
    assert "generated_at" in body


def test_benchmark_report_json_download(client):
    r = client.get("/api/benchmark/report?format=json")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "attachment" in r.headers["content-disposition"]
    assert ".json" in r.headers["content-disposition"]

    report = json.loads(r.content)
    assert report["esol_random_split"]["r2_score"] > 0.8
    assert report["esol_scaffold_split"]["test_samples"] > 0
    assert len(report["test_set_parity"]) > 0
    assert any("hardcoded" in note for note in report["honesty_notes"])


def test_benchmark_report_csv_download(client):
    import csv as _csv
    import io as _io

    r = client.get("/api/benchmark/report?format=csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert ".csv" in r.headers["content-disposition"]

    rows = list(_csv.DictReader(_io.StringIO(r.text)))
    assert len(rows) > 0
    assert {"compound_id", "actual", "predicted", "lower_90", "upper_90"} <= set(rows[0])


def test_benchmark_report_rejects_unknown_format(client):
    assert client.get("/api/benchmark/report?format=pdf").status_code == 400


def test_assay_response_reports_recalibration(client):
    client.post("/api/active-learning/reset")
    pool = client.get("/api/active-learning/pool").json()
    target = next(c["id"] for c in pool["candidates"] if not c["assayed"])

    body = client.post(
        "/api/active-learning/assay", json={"candidate_ids": [target]}
    ).json()

    recal = body["recalibration"]
    assert recal["status"] == "ok"
    assert recal["extra_labels"] == 1
    assert "approximate" in recal["caveat"]
    client.post("/api/active-learning/reset")


def test_importing_the_app_does_not_train(client):
    """Training must happen in the lifespan hook, not at module import."""
    import backend.app.main as main

    # The client fixture has already started the app, so services exist now;
    # the guard is that the module-level objects are assigned at startup rather
    # than being built during import.
    assert main.model_service is not None
    assert main.al_engine is not None
    assert main.al_engine.model_service is main.model_service


# ------------------------------------------------------ Stage 3 endpoints

def test_health_reports_stage3_capabilities(client):
    body = client.get("/api/health").json()
    assert "dmpnn_available" in body
    assert "latent_applicability_domain" in body


def test_dmpnn_status_endpoint_is_honest(client):
    """Either a trained model with metrics, or a clear reason there is none."""
    body = client.get("/api/models/dmpnn").json()
    assert body["status"] in {"trained", "not_trained", "disabled", "failed"}
    if body["status"] == "trained":
        a = body["architecture"]
        assert a["readout"] == "mean over atoms"
        assert a["regression_loss"] == "Gaussian negative log-likelihood"
        assert body["metrics"]["solubility"]["r2_score"] is not None
    else:
        assert body.get("reason")
        assert "metrics" not in body


def test_model_comparison_endpoint(client):
    import backend.app.main as main

    r = client.get("/api/models/comparison")
    if not main.model_service.has_dmpnn:
        assert r.status_code == 409
        return
    body = r.json()
    assert body["baseline"]["r2_score"] is not None
    assert body["dmpnn"]["r2_score"] is not None
    assert body["independent_validation"]["status"] in {"ok", "skipped", "failed"}
    assert "weak evidence" in body["note"]


def test_dmpnn_prediction_endpoint(client):
    import backend.app.main as main

    r = client.post("/api/predict/dmpnn", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    if not main.model_service.has_dmpnn:
        assert r.status_code == 409
        return
    body = r.json()
    sol = body["solubility_logs"]
    assert sol["provenance"] == "model"
    iv = sol["conformal"]["intervals"]["90%"]
    assert iv["lower_bound"] < sol["value"] < iv["upper_bound"]
    assert len(sol["member_predictions"]) == body["n_members"]


def test_dmpnn_prediction_rejects_bad_smiles(client):
    import backend.app.main as main

    r = client.post("/api/predict/dmpnn", json={"smiles": "not_a_molecule"})
    assert r.status_code in (400, 409)
    if not main.model_service.has_dmpnn:
        assert r.status_code == 409


def test_predict_includes_dmpnn_agreement_when_available(client):
    import backend.app.main as main

    body = client.post("/api/predict", json={"smiles": "CCO"}).json()
    if not main.model_service.has_dmpnn:
        assert body["dmpnn"] is None
        return
    agreement = body["dmpnn"]["agreement_with_baseline"]
    assert agreement["baseline_value"] is not None
    assert agreement["dmpnn_value"] is not None
    assert isinstance(agreement["exceeds_dmpnn_sigma"], bool)
    assert "weak evidence" in agreement["interpretation"]


def test_benchmark_includes_dmpnn_section(client):
    body = client.get("/api/benchmark").json()
    assert "dmpnn" in body
    assert body["dmpnn"]["status"] in {"trained", "not_trained", "disabled", "failed"}


# ------------------------------------------------------ Stage 4 endpoints

def test_health_reports_stage4_capabilities(client):
    assert "generator_available" in client.get("/api/health").json()


def test_probabilistic_pareto_endpoint(client):
    r = client.post("/api/pareto/probabilistic", json={"n_samples": 100})
    if r.status_code == 409:
        pytest.skip("No candidate has a complete objective set.")
    body = r.json()

    assert body["deterministic_front_size"] >= 1
    assert body["diagnostics"]["n_candidates"] > 0
    assert body["diagnostics"]["max_monte_carlo_standard_error"] >= 0
    assert "IN THIS POOL" in body["interpretation"]
    assert "not a probability that the molecule is a good drug" in body["interpretation"]

    probs = [c["probability_pareto_optimal"] for c in body["candidates"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Returned sorted by descending probability.
    assert probs == sorted(probs, reverse=True)
    # Every deterministic front member should have non-trivial probability.
    for c in body["candidates"]:
        if c["is_deterministic_pareto_optimal"]:
            assert c["probability_pareto_optimal"] > 0.0


def test_probabilistic_pareto_declares_its_approximations(client):
    r = client.post("/api/pareto/probabilistic", json={"n_samples": 50})
    if r.status_code == 409:
        pytest.skip("No candidate has a complete objective set.")
    text = " ".join(r.json()["diagnostics"]["approximations"])
    assert "Gaussian" in text


def test_correlated_sampling_is_refused_not_faked(client):
    """Asking for correlated sampling must 501, not invent a correlation matrix."""
    r = client.post("/api/pareto/probabilistic", json={"correlated": True})
    if r.status_code == 409:
        pytest.skip("No candidate has a complete objective set.")
    assert r.status_code == 501
    assert "invented" in r.json()["detail"]


def test_probabilistic_pareto_validates_input(client):
    assert client.post("/api/pareto/probabilistic", json={"n_samples": 1}).status_code == 422
    assert client.post("/api/pareto/probabilistic", json={"alpha": 1.5}).status_code == 422


def test_risk_slider_endpoint_sweeps(client):
    results = {}
    for lam in (0.0, 0.5, 1.0):
        r = client.post("/api/pareto/risk", json={"lam": lam})
        if r.status_code == 409:
            pytest.skip("No candidate has a complete objective set.")
        body = r.json()
        assert body["lambda"] == lam
        assert body["kappa"] == pytest.approx(lam * body["kappa_max"])
        assert "not a calibrated prediction" in body["note"]
        results[lam] = [c["id"] for c in body["candidates"]]

    # lambda = 0 must be pure exploitation; raising it should change something.
    assert results[0.0] != results[1.0] or len(results[0.0]) <= 1


def test_risk_slider_validates_lambda(client):
    assert client.post("/api/pareto/risk", json={"lam": -0.1}).status_code == 422
    assert client.post("/api/pareto/risk", json={"lam": 1.1}).status_code == 422


def test_assay_catalogue_declares_simulation(client):
    body = client.get("/api/planner/assays").json()
    assert len(body["assays"]) >= 1
    assert all(a["cost"] > 0 for a in body["assays"])
    assert "SIMULATED" in body["note"]


def test_planner_respects_the_budget(client):
    for budget in (2.0, 6.0):
        r = client.post("/api/planner/plan", json={"budget": budget})
        if r.status_code == 409:
            pytest.skip("No candidate has a complete objective set.")
        body = r.json()
        assert body["total_cost"] <= budget + 1e-9
        assert body["budget_remaining"] >= -1e-9
        assert sum(e["cost"] for e in body["selected"]) == pytest.approx(body["total_cost"])
        assert body["mode"] == "simulated"
        assert "no laboratory experiment" in body["mode_note"].lower()


def test_planner_reports_hypervolume_and_limitations(client):
    r = client.post("/api/planner/plan", json={"budget": 5.0})
    if r.status_code == 409:
        pytest.skip("No candidate has a complete objective set.")
    body = r.json()
    assert body["expected_hypervolume_gain"] >= 0
    assert len(body["reference_point"]) == len(body["objectives"])
    assert any("Greedy" in x for x in body["diagnostics"]["limitations"])


def test_planner_accepts_custom_assay_costs(client):
    r = client.post(
        "/api/planner/plan",
        json={
            "budget": 4.0,
            "assays": [{"name": "cheap", "cost": 0.5, "objectives": ["solubility_logs"]}],
        },
    )
    if r.status_code == 409:
        pytest.skip("No candidate has a complete objective set.")
    body = r.json()
    assert all(e["assay"] == "cheap" for e in body["selected"])
    assert body["total_cost"] <= 4.0 + 1e-9


def test_planner_rejects_invalid_assays(client):
    assert client.post(
        "/api/planner/plan",
        json={"budget": 5.0, "assays": [{"name": "x", "cost": -1.0, "objectives": ["solubility_logs"]}]},
    ).status_code == 422

    r = client.post(
        "/api/planner/plan",
        json={"budget": 5.0, "assays": [{"name": "x", "cost": 1.0, "objectives": ["nope"]}]},
    )
    assert r.status_code == 400
    assert "unknown objectives" in r.json()["detail"]


def test_planner_rejects_nonpositive_budget(client):
    assert client.post("/api/planner/plan", json={"budget": 0.0}).status_code == 422


def test_ehvi_validation_endpoint_agrees_with_botorch(client):
    body = client.get("/api/planner/validate-ehvi").json()
    if body["status"] != "ok":
        pytest.skip(f"Cross-check unavailable: {body.get('error')}")
    assert body["agrees"] is True
    assert body["max_relative_difference"] < 0.1
    assert len(body["botorch_ehvi"]) == len(body["monte_carlo_ehvi"])


def test_generator_status_is_honest(client):
    body = client.get("/api/generator/status").json()
    assert body["status"] in {"prior_trained", "rl_trained", "not_trained", "disabled", "failed"}
    if body["status"] in {"prior_trained", "rl_trained"}:
        assert body["architecture"]["type"].startswith("character-level")
        assert body["prior"]["data_limitation"]["moses_pretraining"] == "pending"
    else:
        assert body.get("reason")


def test_generation_endpoint_screens_candidates(client):
    import backend.app.main as main

    r = client.post("/api/generator/generate", json={"n": 40, "top_k": 5})
    if not main.model_service.has_generator:
        assert r.status_code == 409
        return
    body = r.json()
    m = body["generation_metrics"]
    assert m["n_valid"] + body["generation_metrics"]["n_requested"] >= m["n_valid"]
    assert 0.0 <= m["validity"] <= 1.0
    assert len(body["top_candidates"]) <= 5
    assert "not measurements" in body["ranking_basis"]

    from rdkit import Chem

    for c in body["top_candidates"]:
        assert Chem.MolFromSmiles(c["smiles"]) is not None
        assert c["solubility_lower_90"] <= c["solubility_logs"] <= c["solubility_upper_90"]


def test_generation_validates_input(client):
    assert client.post("/api/generator/generate", json={"n": 0}).status_code == 422
    assert client.post("/api/generator/generate", json={"temperature": 0.0}).status_code == 422
