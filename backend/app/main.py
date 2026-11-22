"""
MoleculeMind FastAPI Application.
Uncertainty-Aware, Human-in-the-Loop Platform for Multi-Objective Molecular Discovery.
"""

import os
import io
import csv
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from backend.app.core.config import settings

from backend.app.core.mol_analyzer import (
    validate_and_canonicalize,
    generate_2d_svg,
    generate_3d_sdf,
    compute_molecular_descriptors,
)
from backend.app.core.models import ModelService
from backend.app.core.pareto import ParetoLaboratory
from backend.app.core.active_learning import ActiveLearningEngine
from backend.app.core.probabilistic_pareto import (
    conformal_to_gaussian_scale,
    probability_pareto_optimal,
    rank_with_risk,
    residual_normality_report,
)
from backend.app.core.experiment_planner import (
    DEFAULT_ASSAYS,
    Assay,
    ExperimentPlanner,
    compute_hypervolume,
    infer_reference_point,
    validate_against_botorch_qehvi,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("moleculemind")

# Services are constructed empty at import and populated in the lifespan
# startup hook. Training at import time made every `uvicorn --reload`, test
# collection and plain `import main` pay the full training cost.
model_service: Optional[ModelService] = None
pareto_lab = ParetoLaboratory()
al_engine: Optional[ActiveLearningEngine] = None
_pareto_cache: Dict[str, Any] = {}


def _startup():
    """Load cached models if available, otherwise train, then build the AL campaign."""
    global model_service, al_engine
    if model_service is not None:
        return
    svc = ModelService()
    svc.initialize()
    logger.info(
        "Models ready (version %s, %s)",
        svc.model_version,
        "loaded from cache" if svc.loaded_from_cache else "freshly trained",
    )
    model_service = svc
    al_engine = ActiveLearningEngine(
        dataset_manager=svc.dataset_manager,
        pool_payload=svc.al_pool_payload,
        model_service=svc,
    )
    _pareto_cache.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup()
    yield


app = FastAPI(
    title="MoleculeMind API",
    description="Backend for Uncertainty-Aware Multi-Objective Molecular Discovery",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def services() -> tuple:
    """Fetch initialised services, starting them if the lifespan hook has not run."""
    if model_service is None or al_engine is None:
        _startup()
    return model_service, al_engine



# --- Request models ---------------------------------------------------------

class AnalyzeRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=1000)

class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=1000)

class RankRequest(BaseModel):
    strategy: Optional[str] = "ucb"
    beta: Optional[float] = Field(2.0, ge=0.0, le=100.0)

class AssayRequest(BaseModel):
    candidate_ids: List[str] = Field(..., min_length=1)

class ProbabilisticParetoRequest(BaseModel):
    n_samples: Optional[int] = Field(None, ge=10, le=5000)
    alpha: Optional[float] = Field(0.1, gt=0.0, lt=1.0)
    correlated: Optional[bool] = False

class RiskSliderRequest(BaseModel):
    lam: float = Field(..., ge=0.0, le=1.0, description="Risk appetite lambda in [0,1]")
    kappa_max: Optional[float] = Field(None, ge=0.0, le=10.0)
    alpha: Optional[float] = Field(0.1, gt=0.0, lt=1.0)

class AssaySpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    cost: float = Field(..., gt=0.0)
    objectives: List[str] = Field(..., min_length=1)

class PlannerRequest(BaseModel):
    budget: float = Field(..., gt=0.0, le=1e6)
    assays: Optional[List[AssaySpec]] = None
    imputation: Optional[str] = Field("mean", pattern="^(mean|sample)$")
    allow_repeat_molecule: Optional[bool] = False
    alpha: Optional[float] = Field(0.1, gt=0.0, lt=1.0)

class GenerateRequest(BaseModel):
    n: Optional[int] = Field(100, ge=1, le=1000)
    temperature: Optional[float] = Field(1.0, gt=0.0, le=3.0)
    top_k: Optional[int] = Field(20, ge=1, le=200)

class RLCampaignRequest(BaseModel):
    steps: Optional[int] = Field(20, ge=1, le=200)
    batch_size: Optional[int] = Field(24, ge=4, le=128)
    sigma: Optional[float] = Field(None, gt=0.0, le=500.0)
    n_final_samples: Optional[int] = Field(150, ge=10, le=500)


# --- Routes -----------------------------------------------------------------

@app.get("/api/health")
def health_check():
    ms, al = services()
    return {
        "status": "healthy",
        "platform": "MoleculeMind",
        "api_version": app.version,
        "model_version": ms.model_version,
        "models_initialized": ms.is_initialized,
        "dmpnn_available": ms.has_dmpnn,
        "generator_available": ms.has_generator,
        "latent_applicability_domain": ms.ad_engine.has_latent_space,
        "esol_metrics": ms.esol_metrics,
    }


@app.post("/api/analyze")
def analyze_molecule(req: AnalyzeRequest):
    """Molecular Analyzer: validation, 2D SVG, 3D conformer, descriptors."""
    ms, al = services()
    is_valid, canonical_smiles, mol, err_msg = validate_and_canonicalize(req.smiles)
    if not is_valid or mol is None:
        raise HTTPException(status_code=400, detail=err_msg or "Invalid SMILES structure.")

    sdf_3d, sdf_err = generate_3d_sdf(mol)
    return {
        "smiles": canonical_smiles,
        "svg_2d": generate_2d_svg(mol, width=400, height=300),
        "sdf_3d": sdf_3d,
        "sdf_error": sdf_err,
        "sdf_note": (
            "3D coordinates are an ETKDGv3-embedded conformer optimised with the "
            "MMFF94 force field. This is a computed low-energy conformer, not an "
            "experimentally determined structure."
        ),
        "descriptors": compute_molecular_descriptors(mol),
    }


@app.post("/api/predict")
def predict_properties(req: PredictRequest):
    """Property prediction with conformal intervals, applicability domain and provenance."""
    ms, al = services()
    is_valid, canonical_smiles, mol, err_msg = validate_and_canonicalize(req.smiles)
    if not is_valid or mol is None:
        raise HTTPException(status_code=400, detail=err_msg or "Invalid SMILES structure.")

    try:
        result = ms.predict_molecule(canonical_smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Prediction failed for %s", canonical_smiles)
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    sdf_3d, _ = generate_3d_sdf(mol)
    result["sdf_3d"] = sdf_3d
    return result


@app.get("/api/benchmark")
def get_benchmark_metrics():
    """
    Benchmark metrics computed from held-out evaluation splits at startup.
    No value on this route is hardcoded.
    """
    ms, al = services()
    return {
        "model_version": ms.model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loaded_from_cache": ms.loaded_from_cache,
        "metrics": ms.esol_metrics,
        "scaffold_split_metrics": ms.scaffold_metrics,
        "clintox_metrics": ms.clintox_metrics,
        "applicability_domain": ms.ad_engine.diagnostics(),
        "calibration_history": ms.calibration_history,
        "dmpnn": ms.dmpnn_service.summary() if ms.dmpnn_service else {"status": "disabled"},
        "parity_plot_data": ms.test_parity_data,
    }


@app.get("/api/config")
def get_config():
    """
    The configuration this run used, for reproducibility.

    Every value here can be overridden with a `MOLECULEMIND_`-prefixed
    environment variable.
    """
    ms, al = services()
    return {
        "model_version": ms.model_version,
        "api_version": app.version,
        "loaded_from_cache": ms.loaded_from_cache,
        "cache_key": ms.cache_key(),
        "settings": settings.as_dict(),
        "dataset_provenance": ms.esol_metrics.get("provenance", {}),
    }


@app.get("/api/applicability-domain/diagnostics")
def get_ad_diagnostics():
    """
    How the applicability-domain thresholds were derived from the training set.

    Thresholds are percentiles of the training set's own leave-one-out
    diagnostics, not literature constants.
    """
    ms, al = services()
    return ms.ad_engine.diagnostics()


@app.get("/api/benchmark/report")
def download_benchmark_report(format: str = "json"):
    """
    Downloadable benchmark report.

    `format=json` returns the full record including provenance and
    configuration; `format=csv` returns the test-set parity table.
    Every number is computed at startup from a held-out split.
    """
    ms, al = services()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if format == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "compound_id", "smiles", "actual", "predicted",
                "std", "lower_90", "upper_90", "error",
            ],
        )
        writer.writeheader()
        for row in ms.test_parity_data:
            writer.writerow(row)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="moleculemind-parity-{stamp}.csv"'
                )
            },
        )

    if format != "json":
        raise HTTPException(
            status_code=400, detail="format must be 'json' or 'csv'."
        )

    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "MoleculeMind",
        "api_version": app.version,
        "model_version": ms.model_version,
        "configuration": settings.as_dict(),
        "esol_random_split": ms.esol_metrics,
        "esol_scaffold_split": ms.scaffold_metrics,
        "clintox": ms.clintox_metrics,
        "applicability_domain": ms.ad_engine.diagnostics(),
        "calibration_history": ms.calibration_history,
        "dmpnn": ms.dmpnn_service.summary() if ms.dmpnn_service else {"status": "disabled"},
        "objectives_available": ms.available_objectives(),
        "test_set_parity": ms.test_parity_data,
        "honesty_notes": [
            "All metrics are computed at startup from held-out splits; none are hardcoded.",
            "Conformal coverage is marginal over the test distribution, not a "
            "per-molecule probability of correctness.",
            "The scaffold split on ESOL is weakened by the dataset's many acyclic "
            "compounds, each of which forms a singleton scaffold group.",
            "ClinTox CT_TOX marks trials terminated for toxicity, not intrinsic "
            "molecular toxicity, and its probabilities are uncalibrated.",
            "Properties listed as unavailable have no trained model and are excluded "
            "from multi-objective optimisation.",
        ],
    }
    payload = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        iter([payload]),
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="moleculemind-benchmark-{stamp}.json"'
            )
        },
    )


@app.get("/api/models/dmpnn")
def get_dmpnn_status():
    """
    D-MPNN ensemble status, architecture, metrics and independent validation.

    Reports `status: "not_trained"` with a reason when the ensemble is absent,
    rather than implying a model exists.
    """
    ms, al = services()
    if ms.dmpnn_service is None:
        return {
            "status": "disabled",
            "reason": (
                "The D-MPNN ensemble is disabled for this process "
                "(MOLECULEMIND_ENABLE_DMPNN=false)."
            ),
        }
    return ms.dmpnn_service.summary()


@app.get("/api/models/comparison")
def get_model_comparison():
    """
    Independent model comparison: D-MPNN vs a Random Forest on fingerprints.

    Two different model families on two different molecular representations,
    evaluated on the same held-out compounds.
    """
    ms, al = services()
    if not ms.has_dmpnn:
        raise HTTPException(
            status_code=409,
            detail=(
                "Model comparison requires a trained D-MPNN ensemble. "
                "See /api/models/dmpnn for its status."
            ),
        )
    return {
        "baseline": {
            "model": ms.esol_metrics.get("model"),
            "representation": "1024-bit Morgan fingerprint + 8 descriptors",
            "r2_score": ms.esol_metrics.get("r2_score"),
            "rmse": ms.esol_metrics.get("rmse"),
            "mae": ms.esol_metrics.get("mae"),
            "test_samples": ms.esol_metrics.get("test_samples"),
        },
        "dmpnn": ms.dmpnn_service.metrics.get("solubility", {}),
        "independent_validation": ms.dmpnn_service.comparison,
        "note": (
            "The D-MPNN's ESOL test partition is pinned to the baseline's, so both "
            "are measured on identical held-out compounds. Agreement between models "
            "trained on the same data is weak evidence of reliability, not proof of "
            "correctness."
        ),
    }


@app.post("/api/predict/dmpnn")
def predict_with_dmpnn(req: PredictRequest):
    """Graph-neural-network prediction with conformal intervals and prediction sets."""
    ms, al = services()
    if not ms.has_dmpnn:
        raise HTTPException(
            status_code=409,
            detail=(
                "The D-MPNN ensemble is not available. See /api/models/dmpnn."
            ),
        )
    is_valid, canonical_smiles, mol, err_msg = validate_and_canonicalize(req.smiles)
    if not is_valid or mol is None:
        raise HTTPException(status_code=400, detail=err_msg or "Invalid SMILES structure.")
    try:
        return ms.dmpnn_service.predict(canonical_smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("D-MPNN prediction failed for %s", canonical_smiles)
        raise HTTPException(status_code=500, detail="Internal prediction error.")


# --- Stage 4 helpers --------------------------------------------------------

def _candidate_objective_matrix(ms, al, alpha: float = 0.1):
    """
    Build (ids, mu, sigma_tilde, objective_specs) for the current candidate pool.

    sigma_tilde is the conformal interval converted to a Gaussian scale. Only
    objectives with a validated model or a deterministic calculation are
    included; a candidate missing any of them is excluded and reported, never
    imputed.
    """
    specs = ms.available_objectives()
    keys = [o["key"] for o in specs]
    maximize = [o["maximize"] for o in specs]

    ids, mu_rows, sigma_rows, excluded, records = [], [], [], [], []

    for cand in al.public_pool():
        pred = ms.predict_molecule(cand["smiles"])
        props = pred["properties"]
        missing = [k for k in keys if props.get(k, {}).get("status") != "available"]
        if missing:
            excluded.append({"id": cand["id"], "missing_objectives": missing})
            continue

        mu_row, sigma_row = [], []
        for key in keys:
            prop = props[key]
            mu_row.append(float(prop["value"]))
            conformal = prop.get("conformal")
            if conformal is not None:
                iv = conformal["intervals"][f"{int((1 - alpha) * 100)}%"]
                half_width = iv["margin_error"]
                # sigma_tilde = half_width / z, the width-matching conversion.
                import math as _math
                from scipy import stats as _stats
                z = _stats.norm.ppf(1.0 - alpha / 2.0)
                sigma_row.append(float(half_width) / z)
            else:
                # A deterministic calculation (QED) has no predictive spread.
                sigma_row.append(0.0)

        ids.append(cand["id"])
        mu_rows.append(mu_row)
        sigma_rows.append(sigma_row)
        records.append(
            {
                "id": cand["id"],
                "name": cand["name"],
                "smiles": cand["smiles"],
                "svg_2d": cand["svg_2d"],
                "assayed": cand["assayed"],
                "ad_status": pred["applicability_domain"]["status"],
                "ad_index": pred["applicability_domain"]["domain_index"],
                **{k: mu_row[j] for j, k in enumerate(keys)},
                **{f"{k}_sigma_tilde": round(sigma_row[j], 4) for j, k in enumerate(keys)},
            }
        )

    import numpy as _np
    mu = _np.array(mu_rows, dtype=float) if mu_rows else _np.zeros((0, len(keys)))
    sigma_tilde = _np.array(sigma_rows, dtype=float) if sigma_rows else _np.zeros((0, len(keys)))
    return ids, mu, sigma_tilde, keys, maximize, specs, excluded, records


@app.post("/api/pareto/probabilistic")
def get_probabilistic_pareto(req: ProbabilisticParetoRequest):
    """
    Probability that each candidate is Pareto non-dominated.

    Monte Carlo over the calibrated predictive distribution: a conformal
    interval is converted to a Gaussian scale by width matching, candidates are
    sampled jointly, and the non-dominated set is recomputed per draw.
    """
    ms, al = services()
    alpha = req.alpha if req.alpha is not None else 0.1
    ids, mu, sigma_tilde, keys, maximize, specs, excluded, records = (
        _candidate_objective_matrix(ms, al, alpha)
    )
    if not ids:
        raise HTTPException(
            status_code=409,
            detail="No candidate has a complete set of validated objectives.",
        )

    n_samples = req.n_samples or settings.pareto_mc_samples
    correlation = None
    if req.correlated:
        # Residual correlation is not estimable here without joint residuals on
        # all objectives, so correlated sampling is refused rather than faked
        # with an invented matrix.
        raise HTTPException(
            status_code=501,
            detail=(
                "Correlated sampling needs a residual correlation matrix estimated "
                "from joint calibration residuals across all objectives. This "
                "platform has joint labels for too few molecules (64 with both "
                "solubility and toxicity) to estimate it honestly, so the option "
                "is refused rather than filled with an invented matrix."
            ),
        )

    result = probability_pareto_optimal(
        mu, sigma_tilde, maximize,
        n_samples=n_samples,
        correlation=correlation,
        random_state=settings.random_seed,
        prefilter_k=settings.pareto_prefilter_k,
    )

    for rec, extra in zip(records, result.as_records()):
        rec.update(extra)
    records.sort(
        key=lambda r: (-r["probability_pareto_optimal"], r["deterministic_pareto_rank"])
    )

    normality = None
    if ms.conformal_engine.cal_residuals is not None:
        normality = residual_normality_report(ms.conformal_engine.cal_residuals)

    return {
        "objectives": keys,
        "objective_specs": specs,
        "alpha": alpha,
        "n_samples": n_samples,
        "candidates": records,
        "excluded_candidates": excluded,
        "deterministic_front_size": int(result.deterministic_front.sum()),
        "expected_front_size": round(float(result.probability.sum()), 3),
        "diagnostics": result.diagnostics,
        "residual_normality": normality,
        "interpretation": (
            "probability_pareto_optimal is the fraction of sampled joint outcomes "
            "in which a candidate is non-dominated by the others IN THIS POOL under "
            "THIS model. It is not a probability that the molecule is a good drug, "
            "and adding or removing candidates changes every value."
        ),
    }


@app.post("/api/pareto/risk")
def get_risk_adjusted_ranking(req: RiskSliderRequest):
    """
    Risk-slider ranking: f = mu + kappa*sigma_tilde, kappa = kappa_max * lambda.

    lambda = 0 is pure exploitation; lambda = 1 gives uncertainty maximum weight.
    Minimised objectives have their sign flipped, so optimism means a lower value.
    """
    ms, al = services()
    alpha = req.alpha if req.alpha is not None else 0.1
    ids, mu, sigma_tilde, keys, maximize, specs, excluded, records = (
        _candidate_objective_matrix(ms, al, alpha)
    )
    if not ids:
        raise HTTPException(
            status_code=409,
            detail="No candidate has a complete set of validated objectives.",
        )

    kappa_max = req.kappa_max if req.kappa_max is not None else settings.risk_kappa_max
    ranked = rank_with_risk(mu, sigma_tilde, maximize, req.lam, kappa_max)

    for i, rec in enumerate(records):
        rec["risk_adjusted_scores"] = {
            k: round(float(ranked["scores"][i, j]), 4) for j, k in enumerate(keys)
        }
        rec["risk_pareto_rank"] = int(ranked["pareto_rank"][i])
        rec["risk_composite_score"] = round(float(ranked["composite_score"][i] * 100), 1)
    records.sort(key=lambda r: (r["risk_pareto_rank"], -r["risk_composite_score"]))

    return {
        "lambda": ranked["lambda"],
        "kappa": ranked["kappa"],
        "kappa_max": ranked["kappa_max"],
        "objectives": keys,
        "objective_specs": specs,
        "candidates": records,
        "excluded_candidates": excluded,
        "risk_pareto_front_size": int((ranked["pareto_rank"] == 1).sum()),
        "note": ranked["note"],
    }


@app.get("/api/planner/assays")
def get_default_assays():
    """Assay catalogue with costs. All assays here are simulated."""
    ms, al = services()
    return {
        "assays": [
            {
                "name": a.name,
                "cost": a.cost,
                "objectives": a.objectives,
                "description": a.description,
            }
            for a in DEFAULT_ASSAYS
        ],
        "available_objectives": [o["key"] for o in ms.available_objectives()],
        "note": (
            "Costs are in arbitrary budget units. Every assay here is SIMULATED; "
            "no laboratory measurement is performed."
        ),
    }


@app.post("/api/planner/plan")
def plan_experiments(req: PlannerRequest):
    """
    Cost-aware experiment plan maximising Expected Hypervolume Improvement.

    Greedy sequential selection of (molecule, assay) pairs subject to a total
    budget. EHVI is computed by Monte Carlo over the calibrated predictive
    distribution with each hypervolume term computed exactly.
    """
    ms, al = services()
    alpha = req.alpha if req.alpha is not None else 0.1
    ids, mu, sigma_tilde, keys, maximize, specs, excluded, records = (
        _candidate_objective_matrix(ms, al, alpha)
    )
    if not ids:
        raise HTTPException(
            status_code=409,
            detail="No candidate has a complete set of validated objectives.",
        )

    if req.assays:
        try:
            assays = [
                Assay(name=a.name, cost=a.cost, objectives=a.objectives)
                for a in req.assays
            ]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        assays = [a for a in DEFAULT_ASSAYS if set(a.objectives) <= set(keys)]
        if not assays:
            raise HTTPException(
                status_code=409,
                detail=(
                    "No default assay matches the currently available objectives "
                    f"{keys}. Supply assays explicitly."
                ),
            )

    try:
        planner = ExperimentPlanner(
            objectives=keys,
            maximize=maximize,
            assays=assays,
            n_ehvi_samples=settings.ehvi_samples,
            random_state=settings.random_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Already-assayed candidates form the observed front the plan improves on.
    import numpy as _np
    assayed_rows = [i for i, r in enumerate(records) if r["assayed"]]
    observed_front = mu[assayed_rows] if assayed_rows else None

    try:
        plan = planner.plan(
            candidate_ids=ids,
            mu=mu,
            sigma=sigma_tilde,
            budget=req.budget,
            observed_front=observed_front,
            imputation=req.imputation or "mean",
            allow_repeat_molecule=bool(req.allow_repeat_molecule),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    by_id = {r["id"]: r for r in records}
    enriched = []
    for item in plan.as_dict()["selected"]:
        rec = by_id.get(item["candidate_id"], {})
        enriched.append(
            {
                **item,
                "name": rec.get("name"),
                "smiles": rec.get("smiles"),
                "ad_status": rec.get("ad_status"),
            }
        )

    payload = plan.as_dict()
    payload["selected"] = enriched
    payload["objectives"] = keys
    payload["objective_specs"] = specs
    payload["excluded_candidates"] = excluded
    payload["mode"] = "simulated"
    payload["mode_note"] = (
        "The plan recommends SIMULATED assays. Executing one reveals a withheld "
        "measurement; no laboratory experiment is performed."
    )
    return payload


@app.get("/api/planner/validate-ehvi")
def validate_ehvi():
    """
    Cross-check the Monte Carlo EHVI estimator against BoTorch's `qEHVI`.

    Both integrate over the same fitted GP posterior, so agreement confirms the
    estimator computes EHVI rather than something EHVI-shaped.
    """
    return validate_against_botorch_qehvi(n_points=8, seed=0)


@app.get("/api/generator/status")
def get_generator_status():
    """Generator status, architecture and the MOSES pre-training limitation."""
    ms, al = services()
    if ms.generator_service is None:
        return {
            "status": "disabled",
            "reason": "The generator is disabled (MOLECULEMIND_ENABLE_GENERATOR=false).",
        }
    return ms.generator_service.summary()


@app.post("/api/generator/generate")
def generate_molecules(req: GenerateRequest):
    """Sample molecules, validate them with RDKit, and screen through the models."""
    ms, al = services()
    if not ms.has_generator:
        raise HTTPException(
            status_code=409,
            detail="The generator is not trained. See /api/generator/status.",
        )
    try:
        return ms.generator_service.generate_and_screen(
            n=req.n or 100,
            temperature=req.temperature or 1.0,
            top_k=req.top_k or 20,
            random_state=settings.random_seed,
        )
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Internal generation error.")


@app.post("/api/generator/reinforce")
def run_rl_campaign(req: RLCampaignRequest):
    """
    Run a REINVENT-style RL campaign with the uncertainty-aware reward.

    Reward uses the LOWER conformal bound, so the agent is rewarded for
    confident predictions rather than optimistic ones. This is a blocking call;
    keep `steps` small for interactive use.
    """
    ms, al = services()
    if not ms.has_generator:
        raise HTTPException(
            status_code=409,
            detail="The generator is not trained. See /api/generator/status.",
        )
    try:
        return ms.generator_service.run_campaign(
            steps=req.steps,
            batch_size=req.batch_size,
            sigma=req.sigma,
            n_final_samples=req.n_final_samples or 150,
        )
    except Exception:
        logger.exception("RL campaign failed")
        raise HTTPException(status_code=500, detail="Internal RL campaign error.")


@app.get("/api/objectives")
def get_objectives():
    """Objectives currently backed by a model or a deterministic calculation."""
    ms, al = services()
    from backend.app.core.models import UNAVAILABLE_PROPERTIES

    return {
        "available": ms.available_objectives(),
        "unavailable": [
            {"key": k, **v} for k, v in UNAVAILABLE_PROPERTIES.items()
        ],
    }


@app.get("/api/pareto")
def get_pareto_laboratory():
    """
    Pareto Laboratory: multi-objective trade-offs over the candidate pool.

    Only objectives with a validated model or a deterministic calculation are
    optimised. Candidates missing any active objective are reported separately
    rather than imputed.
    """
    ms, al = services()
    cache_key = ms.model_version
    if _pareto_cache.get("key") == cache_key:
        return _pareto_cache["value"]

    objective_specs = ms.available_objectives()
    objectives = [o["key"] for o in objective_specs]
    maximize = [o["maximize"] for o in objective_specs]

    candidates: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []

    for cand in al.public_pool():
        pred = ms.predict_molecule(cand["smiles"])
        props = pred["properties"]

        record: Dict[str, Any] = {
            "id": cand["id"],
            "name": cand["name"],
            "smiles": cand["smiles"],
            "svg_2d": cand["svg_2d"],
            "molecular_weight": pred["descriptors"]["molecular_weight"],
            "logp": pred["descriptors"]["logp"],
            "solubility_std": props["solubility_logs"]["ensemble_spread"],
            "solubility_lower_90": props["solubility_logs"]["conformal"]["intervals"]["90%"]["lower_bound"],
            "solubility_upper_90": props["solubility_logs"]["conformal"]["intervals"]["90%"]["upper_bound"],
            "ad_status": pred["applicability_domain"]["status"],
            "ad_score": pred["applicability_domain"]["reliability_score"],
            "assayed": cand["assayed"],
        }

        missing = [k for k in objectives if props.get(k, {}).get("status") != "available"]
        if missing:
            excluded.append({"id": cand["id"], "missing_objectives": missing})
            continue
        for k in objectives:
            record[k] = props[k]["value"]
        candidates.append(record)

    pareto_candidates = pareto_lab.compute_pareto_front(candidates, objectives, maximize)
    frontier = [c for c in pareto_candidates if c["is_pareto_optimal"]]

    payload = {
        "total_candidates": len(pareto_candidates),
        "pareto_optimal_count": len(frontier),
        "objectives": objectives,
        "objective_specs": objective_specs,
        "excluded_candidates": excluded,
        "note": (
            "Objectives without a trained, evaluated model are excluded from this "
            "optimisation rather than estimated. See /api/objectives."
        ),
        "candidates": pareto_candidates,
        "pareto_frontier": frontier,
    }
    _pareto_cache.update({"key": cache_key, "value": payload})
    return payload


@app.get("/api/active-learning/pool")
def get_al_pool(strategy: str = "ucb", beta: float = 2.0):
    """
    Active Learning pool, ranked by the chosen acquisition function.

    Hidden measurements are never included: a candidate carries a value only
    after its simulated assay has been executed.
    """
    ms, al = services()
    ranked = al.rank_candidates(strategy=strategy, beta=beta)
    return {**al.campaign_state(strategy, beta), "candidates": ranked}


@app.post("/api/active-learning/rank")
def rank_al_pool(req: RankRequest):
    ms, al = services()
    strategy = req.strategy or "ucb"
    beta = req.beta if req.beta is not None else 2.0
    ranked = al.rank_candidates(strategy=strategy, beta=beta)
    return {**al.campaign_state(strategy, beta), "candidates": ranked}


@app.post("/api/active-learning/assay")
def run_al_assay(req: AssayRequest):
    """Execute a SIMULATED assay, revealing withheld ESOL measurements."""
    ms, al = services()
    result = al.run_assay(req.candidate_ids)
    if result["status"] == "no_op":
        raise HTTPException(
            status_code=400,
            detail=(
                "No assayable candidates. Unknown ids: "
                f"{result['rejected_unknown']}; already assayed: "
                f"{result['rejected_already_assayed']}."
            ),
        )
    return result


@app.post("/api/active-learning/reset")
def reset_al():
    ms, al = services()
    return al.reset()


# --- Static frontend --------------------------------------------------------

frontend_dist = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist"
)
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
