"""
MoleculeMind FastAPI Application.
Uncertainty-Aware, Human-in-the-Loop Platform for Multi-Objective Molecular Discovery.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.app.core.mol_analyzer import (
    validate_and_canonicalize,
    generate_2d_svg,
    generate_3d_sdf,
    compute_molecular_descriptors
)
from backend.app.core.models import ModelService
from backend.app.core.pareto import ParetoLaboratory
from backend.app.core.active_learning import ActiveLearningEngine


app = FastAPI(
    title="MoleculeMind API",
    description="Backend for Uncertainty-Aware Multi-Objective Molecular Discovery",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services on startup
model_service = ModelService.get_instance()
pareto_lab = ParetoLaboratory()
al_engine = ActiveLearningEngine(dataset_manager=model_service.dataset_manager)


# --- Request/Response Models ---
class AnalyzeRequest(BaseModel):
    smiles: str

class PredictRequest(BaseModel):
    smiles: str

class ParetoRequest(BaseModel):
    objectives: Optional[List[str]] = ["bioactivity_pic50", "solubility_logs", "toxicity_risk_probability", "qed_drug_likeness"]
    maximize: Optional[List[bool]] = [True, True, False, True]

class RankRequest(BaseModel):
    strategy: Optional[str] = "ucb"
    beta: Optional[float] = 2.0

class AssayRequest(BaseModel):
    candidate_ids: List[str]


# --- API Routes ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "MoleculeMind",
        "models_initialized": model_service.is_initialized,
        "esol_metrics": model_service.esol_metrics
    }


@app.post("/api/analyze")
def analyze_molecule(req: AnalyzeRequest):
    """
    1. Molecular Analyzer: Validates SMILES, computes 2D SVG, 3D conformer, and physicochemical descriptors.
    """
    is_valid, canonical_smiles, mol, err_msg = validate_and_canonicalize(req.smiles)
    if not is_valid or mol is None:
        raise HTTPException(status_code=400, detail=err_msg or "Invalid SMILES structure.")

    # 2D SVG
    svg_2d = generate_2d_svg(mol, width=400, height=300)
    
    # 3D Conformer for 3Dmol.js
    sdf_3d, sdf_err = generate_3d_sdf(mol)

    # Descriptors & Lipinski
    descriptors = compute_molecular_descriptors(mol)

    return {
        "smiles": canonical_smiles,
        "svg_2d": svg_2d,
        "sdf_3d": sdf_3d,
        "sdf_error": sdf_err,
        "descriptors": descriptors
    }


@app.post("/api/predict")
def predict_properties(req: PredictRequest):
    """
    2. Property Prediction, 3. Uncertainty Engine (Conformal Intervals), and 4. Applicability Domain.
    """
    is_valid, canonical_smiles, mol, err_msg = validate_and_canonicalize(req.smiles)
    if not is_valid or mol is None:
        raise HTTPException(status_code=400, detail=err_msg or "Invalid SMILES structure.")

    try:
        prediction_result = model_service.predict_molecule(canonical_smiles)
        # Also include 3D conformer for seamless visualization in prediction tab
        sdf_3d, _ = generate_3d_sdf(mol)
        prediction_result["sdf_3d"] = sdf_3d
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/benchmark")
def get_benchmark_metrics():
    """
    Returns authentic training and test set metrics on the Delaney ESOL dataset,
    including R2, RMSE, MAE, conformal calibration coverage, and test set parity points.
    """
    return {
        "metrics": model_service.esol_metrics,
        "parity_plot_data": model_service.test_parity_data
    }


@app.get("/api/pareto")
def get_pareto_laboratory():
    """
    5. Pareto Laboratory: Returns candidate pool with multi-objective trade-offs,
    computed non-dominated Pareto frontier, and composite scores.
    """
    # Build candidate pool from active learning library + test set samples
    candidates = []
    for cand in al_engine.candidate_pool:
        # Predict genuine properties for each candidate
        pred = model_service.predict_molecule(cand["smiles"])
        cand_record = {
            "id": cand["id"],
            "name": cand["name"],
            "smiles": cand["smiles"],
            "svg_2d": cand["svg_2d"],
            "bioactivity_pic50": pred["multi_objective_profile"]["bioactivity_pic50"],
            "solubility_logs": pred["solubility"]["predicted_value"],
            "solubility_std": pred["solubility"]["std_deviation"],
            "solubility_lower_90": pred["solubility"]["conformal"]["intervals"]["90%"]["lower_bound"],
            "solubility_upper_90": pred["solubility"]["conformal"]["intervals"]["90%"]["upper_bound"],
            "toxicity_risk_probability": pred["multi_objective_profile"]["toxicity_risk_probability"],
            "metabolic_stability_half_life_min": pred["multi_objective_profile"]["metabolic_stability_half_life_min"],
            "qed_drug_likeness": pred["descriptors"]["qed"],
            "molecular_weight": pred["descriptors"]["molecular_weight"],
            "logp": pred["descriptors"]["logp"],
            "ad_status": pred["applicability_domain"]["status"],
            "ad_score": pred["applicability_domain"]["reliability_score"],
            "assayed": cand["assayed"]
        }
        candidates.append(cand_record)

    # Standard Discovery Objectives:
    # 1. bioactivity_pic50 (max)
    # 2. solubility_logs (max)
    # 3. toxicity_risk_probability (min)
    # 4. qed_drug_likeness (max)
    objectives = ["bioactivity_pic50", "solubility_logs", "toxicity_risk_probability", "qed_drug_likeness"]
    maximize = [True, True, False, True]

    pareto_candidates = pareto_lab.compute_pareto_front(candidates, objectives, maximize)
    pareto_frontier = [c for c in pareto_candidates if c["is_pareto_optimal"]]

    return {
        "total_candidates": len(pareto_candidates),
        "pareto_optimal_count": len(pareto_frontier),
        "objectives": objectives,
        "candidates": pareto_candidates,
        "pareto_frontier": pareto_frontier
    }


@app.get("/api/active-learning/pool")
def get_al_pool(strategy: str = "ucb", beta: float = 2.0):
    """
    6. Active Learning: Candidate discovery pool scored with BoTorch acquisition function.
    """
    ranked_pool = al_engine.rank_candidates(strategy=strategy, beta=beta)
    return {
        "iteration": al_engine.iteration,
        "strategy": strategy,
        "beta": beta,
        "history": al_engine.history,
        "candidates": ranked_pool
    }


@app.post("/api/active-learning/rank")
def rank_al_pool(req: RankRequest):
    ranked_pool = al_engine.rank_candidates(strategy=req.strategy or "ucb", beta=req.beta or 2.0)
    return {
        "iteration": al_engine.iteration,
        "strategy": req.strategy,
        "beta": req.beta,
        "candidates": ranked_pool
    }


@app.post("/api/active-learning/assay")
def run_al_assay(req: AssayRequest):
    if not req.candidate_ids:
        raise HTTPException(status_code=400, detail="No candidate IDs provided.")
    result = al_engine.run_assay(req.candidate_ids)
    return result


@app.post("/api/active-learning/reset")
def reset_al():
    result = al_engine.reset()
    return result


# --- Static Files / Production Frontend Serving ---
import os
from fastapi.staticfiles import StaticFiles

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

