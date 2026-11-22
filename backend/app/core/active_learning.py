"""
Active Learning / Bayesian Optimisation engine (simulated laboratory mode).

SIMULATED LABORATORY MODE
-------------------------
This module runs a *retrospective simulation*, not a real laboratory workflow.
The "assay" reveals an existing experimental measurement from the Delaney ESOL
dataset that was withheld from the model. No new experiment is performed.

ORACLE SEPARATION (see PROJECT_AUDIT.md section 10.3)
-----------------------------------------------------
The previous implementation returned `true_pic50`, `true_solubility` and
`true_toxicity` for every candidate - assayed or not - directly over the API,
so the full answer key was visible in the browser. It also attributed invented
activity and toxicity numbers to named real drugs.

This implementation enforces four separate stores:

  1. `self._oracle`          - measured values, SERVER-SIDE ONLY, never serialised
  2. `self._revealed`        - ids whose measurement has been released by an assay
  3. `self.candidates`       - public records; carry a measurement only once revealed
  4. the model's test set    - held separately in ModelService, never in this pool

`public_pool()` is the only serialisation path and asserts that no oracle key
can escape.

GAUSSIAN PROCESS (see PROJECT_AUDIT.md section 10.2)
----------------------------------------------------
The previous GP received PCA components computed from the *unscaled* 1032-dim
feature matrix, where molecular weight carried 91% of the variance and inputs
spanned [-113, +178]. BoTorch's default lengthscale prior assumes inputs in
roughly [0,1]^d, so every train/test pair was effectively uncorrelated, the
posterior collapsed to the prior, and all candidates received an identical
acquisition score (UCB = 0 + sqrt(2)*1 = 1.414 for all 24).

The fix: standardise features before PCA, map components to the unit cube, and
apply BoTorch's `Normalize` input transform and `Standardize` outcome transform.
`test_acquisition_is_discriminative` guards against a regression.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound

try:  # BoTorch >= 0.9 recommends the log-space analytic EI
    from botorch.acquisition import LogExpectedImprovement as _EIClass
    _EI_NAME = "LogExpectedImprovement"
except ImportError:  # pragma: no cover
    from botorch.acquisition import ExpectedImprovement as _EIClass
    _EI_NAME = "ExpectedImprovement"

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

from backend.app.core.config import settings
from backend.app.core.mol_analyzer import compute_molecular_descriptors, generate_2d_svg
from backend.app.core.dataset import DatasetManager


STRATEGIES = {
    "ucb": "Upper Confidence Bound: mu + sqrt(beta) * sigma. Balances exploitation and exploration.",
    "ei": f"{_EI_NAME}: expected gain over the best measurement observed so far.",
    "uncertainty": "Pure uncertainty sampling: rank by posterior sigma alone (maximum exploration).",
    "greedy": "Greedy exploitation: rank by posterior mean alone (no exploration).",
}

# Keys that must never cross the API boundary.
_ORACLE_KEYS = {"measured_logs", "true_solubility", "true_pic50", "true_toxicity"}


class ActiveLearningEngine:
    """
    Human-in-the-loop candidate selection over a pool of real, held-out
    ESOL compounds whose measured solubility the surrogate has never seen.

    Objective: maximise measured aqueous solubility, logS (a single real,
    traceable endpoint). Objectives without a validated model are deliberately
    excluded rather than simulated.
    """

    N_PCA_COMPONENTS = settings.al_pca_components
    N_SEED = settings.al_seed_size

    def __init__(
        self,
        dataset_manager: DatasetManager,
        pool_payload: Optional[Dict[str, Any]] = None,
        model_service: Optional[Any] = None,
    ):
        self.dm = dataset_manager
        self.pool_payload = pool_payload or {}
        # Optional back-reference used to refit conformal calibration once new
        # labels arrive. Without it the campaign still runs, but the intervals
        # would be based on a calibration set that predates the new labels.
        self.model_service = model_service
        self.last_recalibration: Optional[Dict[str, Any]] = None

        self._oracle: Dict[str, float] = {}
        self._revealed: set = set()
        self.candidates: List[Dict[str, Any]] = []

        self.features: Optional[np.ndarray] = None  # scaled, unit-cube GP inputs
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

        self.iteration: int = 0
        self.history: List[Dict[str, Any]] = []
        self.last_fit_ok: Optional[bool] = None
        self.last_fit_error: Optional[str] = None

        self.initialize_candidate_library()

    # ------------------------------------------------------------- library

    def initialize_candidate_library(self):
        compounds = self.pool_payload.get("compounds", [])
        if not compounds:
            self.candidates = []
            self._oracle = {}
            self._revealed = set()
            self.history = []
            return

        self._oracle = {}
        self._revealed = set()
        self.candidates = []
        raw_features: List[np.ndarray] = []

        for item in compounds:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is None:
                continue
            desc = compute_molecular_descriptors(mol)
            raw_features.append(self.dm.extract_features_from_mol(mol))

            # Measured value goes to the private oracle store only.
            self._oracle[item["id"]] = float(item["measured_logs"])

            self.candidates.append(
                {
                    "id": item["id"],
                    "name": item.get("name", item["id"]),
                    "smiles": item["smiles"],
                    "svg_2d": generate_2d_svg(mol, width=280, height=200),
                    "mw": desc["molecular_weight"],
                    "logp": desc["logp"],
                    "tpsa": desc["tpsa"],
                    "qed": desc["qed"],
                    "assayed": False,
                    "measured_value": None,
                    "assay_round": None,
                    "predicted_value": None,
                    "uncertainty_sigma": None,
                    "acquisition_score": None,
                }
            )

        # ---- Feature pipeline: standardise -> PCA -> unit cube -------------
        X = np.array(raw_features, dtype=np.float64)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        n_comp = min(self.N_PCA_COMPONENTS, Xs.shape[0] - 1, Xs.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=0)
        Z = self.pca.fit_transform(Xs)
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        self.features = (Z - lo) / np.maximum(hi - lo, 1e-9)

        # ---- Seed set: a diverse starting batch, revealed up front ---------
        for cand in self.candidates[: self.N_SEED]:
            self._reveal(cand, round_number=0)

        self.iteration = 0
        self.history = [self._history_entry(round_number=0, newly_assayed=[])]

    def _reveal(self, cand: Dict[str, Any], round_number: int):
        """Release a measurement from the oracle into the public record."""
        cand["assayed"] = True
        cand["assay_round"] = round_number
        cand["measured_value"] = round(self._oracle[cand["id"]], 3)
        self._revealed.add(cand["id"])

    def _history_entry(
        self, round_number: int, newly_assayed: List[str]
    ) -> Dict[str, Any]:
        measured = [
            c["measured_value"] for c in self.candidates if c["assayed"]
        ]
        best = max(measured) if measured else None
        best_cmpd = None
        if best is not None:
            best_cmpd = next(
                c["name"] for c in self.candidates if c["measured_value"] == best
            )
        return {
            "round": round_number,
            "assayed_count": len(measured),
            "newly_assayed": newly_assayed,
            "best_revealed_logs": best,
            "best_compound": best_cmpd,
        }

    # ------------------------------------------------------------- ranking

    def rank_candidates(
        self, strategy: str = "ucb", beta: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Fit a GP on revealed measurements and score the unlabelled pool.

        Returns the public pool. Candidates already labelled receive no
        acquisition score and are excluded from selection.
        """
        if strategy not in STRATEGIES:
            strategy = "ucb"

        labelled = [i for i, c in enumerate(self.candidates) if c["assayed"]]
        unlabelled = [i for i, c in enumerate(self.candidates) if not c["assayed"]]

        if len(labelled) < 2 or not unlabelled or self.features is None:
            return self.public_pool()

        X_train = torch.tensor(self.features[labelled], dtype=torch.float64)
        y_train = torch.tensor(
            [[self.candidates[i]["measured_value"]] for i in labelled],
            dtype=torch.float64,
        )

        d = X_train.shape[-1]
        gp = SingleTaskGP(
            X_train,
            y_train,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
            self.last_fit_ok, self.last_fit_error = True, None
        except Exception as e:
            # Surfaced rather than swallowed: the previous `except: pass` hid
            # every optimiser failure behind a silently degenerate posterior.
            self.last_fit_ok, self.last_fit_error = False, str(e)
            print(f"[active_learning] GP fit failed, using prior hyperparameters: {e}")

        X_cand = torch.tensor(self.features[unlabelled], dtype=torch.float64)

        with torch.no_grad():
            posterior = gp.posterior(X_cand)
            mean = posterior.mean.squeeze(-1).numpy()
            sigma = np.sqrt(
                np.clip(posterior.variance.squeeze(-1).numpy(), 1e-12, None)
            )

            best_f = float(y_train.max().item())
            q_cand = X_cand.unsqueeze(1)  # q-batch shape (n, 1, d)

            if strategy == "ucb":
                acq = UpperConfidenceBound(gp, beta=beta)(q_cand).numpy()
            elif strategy == "ei":
                acq = _EIClass(gp, best_f=best_f)(q_cand).numpy()
            elif strategy == "uncertainty":
                acq = sigma
            else:  # greedy
                acq = mean

        for k, idx in enumerate(unlabelled):
            c = self.candidates[idx]
            c["predicted_value"] = round(float(mean[k]), 3)
            c["uncertainty_sigma"] = round(float(sigma[k]), 3)
            c["acquisition_score"] = round(float(acq[k]), 4)

        for idx in labelled:
            c = self.candidates[idx]
            c["predicted_value"] = None
            c["uncertainty_sigma"] = None
            c["acquisition_score"] = None  # already labelled: never reselected

        return self.public_pool()

    def acquisition_spread(self) -> Dict[str, Any]:
        """Diagnostic: how discriminative the current acquisition scores are."""
        scores = [
            c["acquisition_score"]
            for c in self.candidates
            if not c["assayed"] and c["acquisition_score"] is not None
        ]
        if not scores:
            return {"n": 0, "n_unique": 0, "degenerate": None}
        return {
            "n": len(scores),
            "n_unique": len(set(scores)),
            "min": min(scores),
            "max": max(scores),
            "degenerate": len(set(scores)) <= 1,
        }

    # --------------------------------------------------------------- assay

    def run_assay(self, candidate_ids: List[str]) -> Dict[str, Any]:
        """
        SIMULATED ASSAY. Releases the withheld Delaney ESOL measurement for the
        approved candidates. Candidates already labelled are rejected rather
        than silently re-measured.
        """
        requested = set(candidate_ids)
        known = {c["id"] for c in self.candidates}

        unknown = sorted(requested - known)
        already = sorted(
            c["id"] for c in self.candidates if c["id"] in requested and c["assayed"]
        )
        to_assay = [
            c for c in self.candidates if c["id"] in requested and not c["assayed"]
        ]

        if not to_assay:
            return {
                "status": "no_op",
                "iteration": self.iteration,
                "rejected_unknown": unknown,
                "rejected_already_assayed": already,
                "newly_assayed": [],
                "history": self.history,
            }

        self.iteration += 1
        for c in to_assay:
            self._reveal(c, round_number=self.iteration)

        # The surrogate must be refitted on the enlarged labelled set.
        self.rank_candidates()

        # Conformal intervals must not silently reuse a stale calibration once
        # the labelled set has changed.
        recalibration = self._recalibrate_conformal(to_assay)
        entry = self._history_entry(
            round_number=self.iteration, newly_assayed=[c["name"] for c in to_assay]
        )
        self.history.append(entry)

        return {
            "status": "ok",
            "mode": "simulated",
            "mode_note": (
                "Simulated assay: reveals a withheld Delaney ESOL measurement. "
                "No laboratory experiment was performed."
            ),
            "iteration": self.iteration,
            "newly_assayed": [
                {"id": c["id"], "name": c["name"], "measured_value": c["measured_value"]}
                for c in to_assay
            ],
            "rejected_unknown": unknown,
            "rejected_already_assayed": already,
            "total_assayed": len(self._revealed),
            "remaining_candidates": len(self.candidates) - len(self._revealed),
            "recalibration": recalibration,
            "history": self.history,
        }

    def _recalibrate_conformal(
        self, newly_assayed: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Fold the newly revealed measurements into the conformal calibration set."""
        if self.model_service is None:
            return {
                "status": "skipped",
                "reason": "No model service attached to this engine.",
            }
        if self.iteration % max(1, settings.al_recalibration_interval) != 0:
            return {
                "status": "deferred",
                "reason": (
                    f"Recalibration runs every {settings.al_recalibration_interval} "
                    f"round(s); next at round "
                    f"{self.iteration + settings.al_recalibration_interval - (self.iteration % settings.al_recalibration_interval)}."
                ),
            }
        try:
            result = self.model_service.recalibrate_conformal(
                extra_smiles=[c["smiles"] for c in newly_assayed],
                extra_targets=[self._oracle[c["id"]] for c in newly_assayed],
                trigger=f"active learning round {self.iteration}",
            )
            self.last_recalibration = {"status": "ok", **result}
        except Exception as e:
            # Reported, never swallowed: stale calibration is a correctness
            # issue the user needs to know about.
            print(f"[active_learning] conformal recalibration failed: {e}")
            self.last_recalibration = {"status": "failed", "error": str(e)}
        return self.last_recalibration

    def reset(self) -> Dict[str, Any]:
        self.initialize_candidate_library()
        self.rank_candidates()
        return {"status": "reset", "iteration": 0, "history": self.history}

    # ---------------------------------------------------------- public view

    def public_pool(self) -> List[Dict[str, Any]]:
        """
        The ONLY serialisation path for candidates.

        Strips every oracle key and emits `measured_value` solely for candidates
        whose assay has actually been executed.
        """
        out: List[Dict[str, Any]] = []
        for c in self.candidates:
            record = {k: v for k, v in c.items() if k not in _ORACLE_KEYS}
            if not c["assayed"]:
                record["measured_value"] = None
                record["assay_round"] = None
            out.append(record)

        # Defensive assertion: an oracle key must never reach a client.
        for record in out:
            leaked = _ORACLE_KEYS & set(record.keys())
            if leaked:  # pragma: no cover - guard
                raise RuntimeError(f"Oracle leak prevented: {leaked}")
        return out

    def campaign_state(self, strategy: str, beta: float) -> Dict[str, Any]:
        return {
            "mode": "simulated",
            "mode_note": (
                "Simulated laboratory mode. Assays reveal withheld Delaney ESOL "
                "measurements; no new experiment is performed."
            ),
            "objective": {
                # Deliberately not named after an oracle field: `_ORACLE_KEYS`
                # is scanned for as a substring across serialised payloads.
                "key": "solubility_logs",
                "label": self.pool_payload.get(
                    "target_name", "Measured aqueous solubility, logS"
                ),
                "maximize": True,
                "source": self.pool_payload.get("source", "unknown"),
            },
            "iteration": self.iteration,
            "strategy": strategy,
            "strategy_description": STRATEGIES.get(strategy, ""),
            "beta": beta,
            "available_strategies": STRATEGIES,
            "gp_fit_ok": self.last_fit_ok,
            "last_recalibration": self.last_recalibration,
            "gp_fit_error": self.last_fit_error,
            "acquisition_diagnostics": self.acquisition_spread(),
            "history": self.history,
        }
