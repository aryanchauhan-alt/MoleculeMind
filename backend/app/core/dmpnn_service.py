"""
D-MPNN ensemble service: multi-task training, evaluation, calibration and
independent-model comparison.

Multi-task construction
-----------------------
ESOL (regression, logS) and ClinTox (classification, CT_TOX) are different
datasets with different molecules. They are joined on canonical SMILES into one
multi-task table with **NaN where a label is absent**, and the masked loss in
`dmpnn.py` handles the gaps. This is the point of multi-task learning here: a
molecule with only a solubility label still contributes gradient to the shared
encoder, which is what the classification head benefits from.

Splitting
---------
Partitions are assigned on the **union** of molecules, so a molecule never
appears in two partitions for either task. The ESOL evaluation partition is kept
identical to the one `ModelService` uses, so the D-MPNN and the ExtraTrees
baseline are compared on the same held-out compounds rather than on two
different random splits.

Calibration
-----------
Normalised split conformal using the ensemble's own predicted sigma as the
normaliser:

    s = |y - mu| / sigma,    interval = mu +- q_hat * sigma

This is the methodology's intended form. Stage 2's conformal layer used the
inter-tree spread of an ExtraTrees model as sigma, which is a heuristic
dispersion measure; here sigma comes from a model trained by Gaussian NLL to
predict its own error scale, with the epistemic component added from ensemble
disagreement.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from backend.app.core.config import settings
from backend.app.core.conformal import (
    ClassificationConformalEngine,
    ConformalPredictionEngine,
)
from backend.app.core.dataset import DatasetManager
from backend.app.core.dmpnn import DMPNNConfig, DMPNNEnsemble, TaskSpec
from backend.app.core.molecular_graph import smiles_to_graph

logger = logging.getLogger("moleculemind.dmpnn")


def to_native(obj: Any) -> Any:
    """
    Recursively convert numpy scalars and arrays to plain Python types.

    FastAPI's JSON encoder cannot serialise `numpy.float32`, and the failure
    surfaces as an opaque 500 at request time rather than at training time. The
    metrics payload is therefore sanitised at the point it is built.
    """
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

TASKS = [
    TaskSpec(name="solubility_logs", kind="regression"),
    TaskSpec(name="clintox_toxicity", kind="classification"),
]
REG_TASK = 0
CLF_TASK = 0  # index within the classification head outputs


class DMPNNService:
    """Owns the D-MPNN ensemble, its calibration and its evaluation record."""

    def __init__(
        self,
        dataset_manager: DatasetManager,
        quick: bool = False,
        n_members: Optional[int] = None,
    ):
        self.dm = dataset_manager
        self.quick = quick
        self.n_members = n_members if n_members is not None else (2 if quick else 5)

        self.ensemble: Optional[DMPNNEnsemble] = None
        self.conformal = ConformalPredictionEngine()
        self.clf_conformal = ClassificationConformalEngine(
            class_names=["not flagged", "trial terminated for toxicity"]
        )

        self.metrics: Dict[str, Any] = {}
        self.comparison: Dict[str, Any] = {}
        self.training_report: Dict[str, Any] = {}
        self.train_smiles: List[str] = []
        self.is_trained: bool = False
        self.status: str = "not_trained"
        self.failure_reason: Optional[str] = None

    # ------------------------------------------------------------- dataset

    def build_multitask_table(self) -> Dict[str, Any]:
        """Join ESOL and ClinTox on canonical SMILES into a 2-task label matrix."""
        esol = self.dm.get_esol_dataset()
        clintox = self.dm.get_clintox_dataset()

        labels: Dict[str, List[Optional[float]]] = {}
        for smi, y in zip(esol.smiles, esol.y):
            labels.setdefault(smi, [None, None])[0] = float(y)
        n_clintox = 0
        if clintox is not None:
            for smi, y in zip(clintox.smiles, clintox.y):
                labels.setdefault(smi, [None, None])[1] = float(y)
                n_clintox += 1

        smiles = sorted(labels)  # deterministic order
        targets = np.full((len(smiles), 2), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            sol, tox = labels[smi]
            if sol is not None:
                targets[i, 0] = sol
            if tox is not None:
                targets[i, 1] = tox

        both = int(np.sum(~np.isnan(targets).any(axis=1)))
        return {
            "smiles": smiles,
            "targets": targets,  # numpy; stripped before entering `metrics`

            "n_molecules": len(smiles),
            "n_solubility_labels": int(np.sum(~np.isnan(targets[:, 0]))),
            "n_toxicity_labels": int(np.sum(~np.isnan(targets[:, 1]))),
            "n_with_both_labels": both,
            "esol_data_hash": esol.data_hash,
            "clintox_data_hash": clintox.data_hash if clintox else None,
        }

    # ------------------------------------------------------------ training

    def train(
        self,
        esol_eval_smiles: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the ensemble, calibrate conformal prediction and evaluate.

        `esol_eval_smiles` pins the ESOL test partition to the one the
        ExtraTrees baseline used, so the two models are compared on identical
        held-out compounds.
        """
        started = time.time()
        table = self.build_multitask_table()
        smiles = table["smiles"]
        targets = table["targets"]

        graphs = []
        keep = []
        for i, smi in enumerate(smiles):
            g = smiles_to_graph(smi)
            if g is not None and g.n_atoms > 0:
                graphs.append(g)
                keep.append(i)
        targets = targets[keep]
        smiles = [smiles[i] for i in keep]
        logger.info(
            "Multi-task table: %d molecules (%d logS, %d CT_TOX, %d with both)",
            len(smiles), table["n_solubility_labels"],
            table["n_toxicity_labels"], table["n_with_both_labels"],
        )

        # ---- partition on the union, pinned to the baseline's ESOL test set --
        index_of = {s: i for i, s in enumerate(smiles)}
        pinned_test = (
            [index_of[s] for s in esol_eval_smiles if s in index_of]
            if esol_eval_smiles
            else []
        )
        remaining = np.array([i for i in range(len(smiles)) if i not in set(pinned_test)])

        rng_state = settings.random_seed
        if pinned_test:
            # Pinning the test partition to the baseline's ESOL compounds gives a
            # fair regression comparison, but those molecules carry almost no
            # CT_TOX labels (only 64 molecules have both), which would leave the
            # classification head with no held-out evaluation at all. A
            # stratified slice of ClinTox-only molecules is therefore added to
            # the test partition. It cannot affect the regression comparison,
            # because those molecules have no logS label to score.
            tox_only = np.array(
                [
                    i
                    for i in remaining
                    if not np.isnan(targets[i, 1]) and np.isnan(targets[i, 0])
                ]
            )
            extra_test = np.array([], dtype=int)
            if len(tox_only) >= 50:
                labels = targets[tox_only, 1]
                stratify = labels if len(np.unique(labels)) > 1 else None
                tox_keep, tox_test = train_test_split(
                    tox_only,
                    test_size=settings.test_fraction,
                    random_state=rng_state,
                    stratify=stratify,
                )
                extra_test = tox_test
                remaining = np.array(
                    [i for i in remaining if i not in set(tox_test.tolist())]
                )

            test_idx = np.concatenate([np.array(pinned_test), extra_test]).astype(int)
            train_idx, cal_idx = train_test_split(
                remaining, test_size=settings.cal_fraction, random_state=rng_state
            )
        else:
            train_val, test_idx = train_test_split(
                remaining, test_size=settings.test_fraction, random_state=rng_state
            )
            train_idx, cal_idx = train_test_split(
                train_val,
                test_size=settings.cal_fraction / (1.0 - settings.test_fraction),
                random_state=rng_state,
            )

        # An inner validation split drives early stopping; it must not be the
        # calibration set (which conformal needs untouched) or the test set.
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=rng_state
        )

        n_test_sol = int(np.sum(~np.isnan(targets[test_idx][:, 0])))
        n_test_tox = int(np.sum(~np.isnan(targets[test_idx][:, 1])))
        logger.info(
            "D-MPNN split -> train %d, val %d, cal %d, test %d (%d logS, %d CT_TOX)",
            len(train_idx), len(val_idx), len(cal_idx), len(test_idx),
            n_test_sol, n_test_tox,
        )
        self.train_smiles = [smiles[i] for i in train_idx]

        # ---- train the ensemble ---------------------------------------------
        config = DMPNNConfig(
            hidden_size=64 if self.quick else 300,
            depth=3,
            ffn_hidden_size=64 if self.quick else 300,
            dropout=0.1,
            lr=1e-3,
            warmup_epochs=2,
            max_epochs=8 if self.quick else 80,
            patience=3 if self.quick else 12,
            batch_size=64,
        )
        self.ensemble = DMPNNEnsemble(
            TASKS, config=config, n_members=self.n_members, base_seed=settings.random_seed
        )
        self.training_report = self.ensemble.fit(
            [graphs[i] for i in train_idx],
            targets[train_idx],
            [graphs[i] for i in val_idx],
            targets[val_idx],
            verbose=verbose,
        )

        # ---- calibrate -------------------------------------------------------
        cal_out = self.ensemble.predict([graphs[i] for i in cal_idx])
        cal_sol_mask = ~np.isnan(targets[cal_idx][:, 0])
        cal_tox_mask = ~np.isnan(targets[cal_idx][:, 1])

        if cal_sol_mask.sum() >= 20:
            self.conformal.calibrate(
                targets[cal_idx][cal_sol_mask, 0],
                cal_out["mu"][cal_sol_mask, REG_TASK],
                cal_out["std"][cal_sol_mask, REG_TASK],
            )
        if cal_tox_mask.sum() >= 20:
            self.clf_conformal.calibrate(
                targets[cal_idx][cal_tox_mask, 1],
                cal_out["probability"][cal_tox_mask, CLF_TASK],
                random_state=settings.random_seed,
            )

        # ---- evaluate --------------------------------------------------------
        test_out = self.ensemble.predict([graphs[i] for i in test_idx])
        self.metrics = self._evaluate(targets[test_idx], test_out, len(train_idx), len(cal_idx))
        # `table` carries the raw label matrix and the full SMILES list; only the
        # summary counts belong in a served payload.
        self.metrics["dataset"] = {
            k: v for k, v in table.items() if k not in ("targets", "smiles")
        }
        self.metrics["config"] = config.as_dict()
        self.metrics["n_members"] = self.n_members
        self.metrics["total_seconds"] = round(time.time() - started, 1)
        self.metrics = to_native(self.metrics)

        # ---- independent model comparison ------------------------------------
        self.comparison = to_native(
            self._compare_independent_model(
                smiles, targets, train_idx, test_idx, test_out
            )
        )

        self.is_trained = True
        self.status = "trained"
        logger.info(
            "D-MPNN trained in %.1fs -> logS R2 %.3f, coverage@90 %.1f%%",
            self.metrics["total_seconds"],
            self.metrics["solubility"]["r2_score"],
            self.metrics["solubility"]["conformal_coverage"]["90%"]["empirical_coverage"],
        )
        return self.metrics

    def _evaluate(
        self,
        y_true: np.ndarray,
        out: Dict[str, np.ndarray],
        n_train: int,
        n_cal: int,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "n_train": int(n_train),
            "n_cal": int(n_cal),
            "n_test": int(y_true.shape[0]),
        }

        sol_mask = ~np.isnan(y_true[:, 0])
        if sol_mask.sum():
            y = y_true[sol_mask, 0]
            mu = out["mu"][sol_mask, REG_TASK]
            sigma = out["std"][sol_mask, REG_TASK]
            metrics["solubility"] = {
                "task": "Aqueous solubility, logS (regression)",
                "n_test": int(sol_mask.sum()),
                "r2_score": round(float(r2_score(y, mu)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y, mu))), 3),
                "mae": round(float(mean_absolute_error(y, mu)), 3),
                "mean_sigma": round(float(sigma.mean()), 3),
                "mean_aleatoric_variance": round(
                    float(out["aleatoric_variance"][sol_mask, REG_TASK].mean()), 4
                ),
                "mean_epistemic_variance": round(
                    float(out["epistemic_variance"][sol_mask, REG_TASK].mean()), 4
                ),
                "error_sigma_correlation": round(
                    float(np.corrcoef(np.abs(y - mu), sigma)[0, 1]), 3
                ),
                "conformal_coverage": self.conformal.evaluate_test_coverage(y, mu, sigma),
                "conformal_method": (
                    "normalised split conformal, s = |y - mu| / sigma, with sigma from "
                    "the ensemble's own Gaussian-NLL head plus epistemic spread"
                ),
            }

        tox_mask = ~np.isnan(y_true[:, 1])
        if not tox_mask.sum() or len(np.unique(y_true[tox_mask, 1])) < 2:
            metrics["toxicity"] = {
                "task": "ClinTox CT_TOX (classification)",
                "status": "not_evaluated",
                "reason": (
                    f"The test partition contains {int(tox_mask.sum())} CT_TOX labels "
                    "with fewer than two classes present, so no classification metric "
                    "can be computed on it."
                ),
            }
        if tox_mask.sum() and len(np.unique(y_true[tox_mask, 1])) > 1:
            y = y_true[tox_mask, 1].astype(int)
            p = out["probability"][tox_mask, CLF_TASK]
            entry: Dict[str, Any] = {
                "task": "ClinTox CT_TOX (classification)",
                "status": "evaluated",
                "n_test": int(tox_mask.sum()),
                "n_positive": int(y.sum()),
                "roc_auc": round(float(roc_auc_score(y, p)), 3),
                "pr_auc": round(float(average_precision_score(y, p)), 3),
                "baseline_pr_auc": round(float(y.mean()), 4),
            }
            if self.clf_conformal.lac_scores is not None:
                entry["conformal_sets"] = self.clf_conformal.evaluate_coverage(y, p)
            metrics["toxicity"] = entry

        return metrics

    def _compare_independent_model(
        self,
        smiles: List[str],
        targets: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        dmpnn_out: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Independent validation: a Random Forest on Morgan fingerprints.

        This is deliberately a *different model family on a different
        representation* (bagged trees on hashed substructure counts, versus a
        learned graph network). Agreement between them is weak evidence that a
        prediction is representation-independent; disagreement flags molecules
        where the choice of representation is doing the work.

        Agreement is NOT a correctness guarantee. Two models trained on the same
        data share its biases, so they can agree and both be wrong.
        """
        try:
            sol_train = train_idx[~np.isnan(targets[train_idx][:, 0])]
            sol_test = test_idx[~np.isnan(targets[test_idx][:, 0])]
            if len(sol_train) < 50 or len(sol_test) < 10:
                return {"status": "skipped", "reason": "Too few solubility labels."}

            def featurise(idx):
                feats = [
                    self.dm.extract_features_from_smiles(smiles[i]) for i in idx
                ]
                return np.array([f for f in feats if f is not None], dtype=np.float32)

            X_train, X_test = featurise(sol_train), featurise(sol_test)
            y_train = targets[sol_train][:, 0]
            y_test = targets[sol_test][:, 0]

            rf = RandomForestRegressor(
                n_estimators=50 if self.quick else 200,
                random_state=settings.random_seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            # Align the D-MPNN predictions to the same molecules.
            test_pos = {int(v): k for k, v in enumerate(test_idx)}
            rows = [test_pos[int(i)] for i in sol_test]
            dm_pred = dmpnn_out["mu"][rows, REG_TASK]
            dm_sigma = dmpnn_out["std"][rows, REG_TASK]

            disagreement = np.abs(dm_pred - rf_pred)
            # "Material" disagreement: the two models differ by more than the
            # D-MPNN's own 1-sigma uncertainty, i.e. more than it expects to be wrong by.
            flagged = disagreement > dm_sigma

            worst = np.argsort(-disagreement)[:5]
            return {
                "status": "ok",
                "independent_model": (
                    f"RandomForestRegressor(n_estimators={rf.n_estimators}) on "
                    "1024-bit Morgan fingerprints + 8 descriptors"
                ),
                "n_test": int(len(y_test)),
                "dmpnn": {
                    "r2_score": round(float(r2_score(y_test, dm_pred)), 3),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, dm_pred))), 3),
                },
                "independent": {
                    "r2_score": round(float(r2_score(y_test, rf_pred)), 3),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, rf_pred))), 3),
                },
                "prediction_agreement": {
                    "pearson_r": round(float(np.corrcoef(dm_pred, rf_pred)[0, 1]), 3),
                    "mean_absolute_difference": round(float(disagreement.mean()), 3),
                    "max_absolute_difference": round(float(disagreement.max()), 3),
                    "n_exceeding_dmpnn_sigma": int(flagged.sum()),
                    "fraction_exceeding_dmpnn_sigma": round(float(flagged.mean()), 3),
                },
                "largest_disagreements": [
                    {
                        "smiles": smiles[int(sol_test[i])],
                        "measured": round(float(y_test[i]), 3),
                        "dmpnn": round(float(dm_pred[i]), 3),
                        "independent": round(float(rf_pred[i]), 3),
                        "difference": round(float(disagreement[i]), 3),
                        "dmpnn_sigma": round(float(dm_sigma[i]), 3),
                    }
                    for i in worst
                ],
                "caveat": (
                    "Agreement between two models trained on the same data is weak "
                    "evidence of reliability, not a correctness guarantee: they share "
                    "the dataset's biases and can agree while both being wrong. "
                    "Disagreement is the more informative signal."
                ),
            }
        except Exception as e:
            logger.error("Independent model comparison failed: %s", e)
            return {"status": "failed", "error": str(e)}

    # ----------------------------------------------------------- inference

    def predict(self, smiles: str) -> Dict[str, Any]:
        """Ensemble prediction for one molecule, with conformal interval and set."""
        if not self.is_trained or self.ensemble is None:
            raise RuntimeError("DMPNNService.predict() called before train().")

        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        out = self.ensemble.predict([graph])
        mu = float(out["mu"][0, REG_TASK])
        sigma = float(out["std"][0, REG_TASK])
        prob = float(out["probability"][0, CLF_TASK])

        result: Dict[str, Any] = {
            "model": "D-MPNN ensemble",
            "n_members": self.n_members,
            "solubility_logs": {
                "property_name": "Aqueous Solubility (logS)",
                "status": "available",
                "provenance": "model",
                "value": round(mu, 3),
                "unit": "log(mol/L)",
                "sigma": round(sigma, 3),
                "aleatoric_variance": round(float(out["aleatoric_variance"][0, REG_TASK]), 4),
                "epistemic_variance": round(float(out["epistemic_variance"][0, REG_TASK]), 4),
                "member_predictions": [
                    round(float(v), 3) for v in out["member_means"][:, 0, REG_TASK]
                ],
                "conformal": self.conformal.predict_intervals(mu, sigma),
            },
            "toxicity": {
                "property_name": "Clinical Toxicity Risk (ClinTox CT_TOX)",
                "status": "available",
                "provenance": "model",
                "value": round(prob, 3),
                "unit": "probability, uncalibrated point estimate",
                "ensemble_spread": round(float(out["probability_std"][0, CLF_TASK]), 3),
            },
        }

        if self.clf_conformal.lac_scores is not None:
            probs = np.array([1.0 - prob, prob])
            result["toxicity"]["conformal_sets"] = {
                method: self.clf_conformal.describe_set(probs, 0.90, method)
                for method in ("lac", "aps_randomized")
            }
        return result

    def embed_smiles(self, smiles_list: Sequence[str]) -> Optional[np.ndarray]:
        """Latent embeddings for the applicability-domain latent space."""
        if not self.is_trained or self.ensemble is None:
            return None
        graphs, ok = [], []
        for s in smiles_list:
            g = smiles_to_graph(s)
            if g is not None and g.n_atoms > 0:
                graphs.append(g)
                ok.append(s)
        if not graphs:
            return None
        if len(ok) != len(smiles_list):
            logger.warning(
                "Embedding %d of %d molecules; the rest could not be parsed.",
                len(ok), len(smiles_list),
            )
            return None
        return self.ensemble.embed(graphs)

    # --------------------------------------------------------- persistence

    def save(self, directory: str, key: str):
        """
        Persist the weights and the evaluation record together.

        The metrics are saved alongside the checkpoint, not recomputed on load:
        they were measured on a held-out split at training time, and a warm
        start has no access to that split. Reporting a model as "trained" with
        no metrics, or recomputing them against a different split, would both be
        misleading.
        """
        if self.ensemble is None:
            return
        os.makedirs(directory, exist_ok=True)
        self.ensemble.save(os.path.join(directory, f"dmpnn-{key}.pt"))

        import json as _json

        sidecar = os.path.join(directory, f"dmpnn-{key}.metrics.json")
        try:
            with open(sidecar, "w") as fh:
                _json.dump(
                    {
                        "metrics": self.metrics,
                        "comparison": self.comparison,
                        "train_smiles": self.train_smiles,
                        "conformal": {
                            "cal_residuals": (
                                self.conformal.cal_residuals.tolist()
                                if self.conformal.cal_residuals is not None
                                else None
                            ),
                            "cal_norm_residuals": (
                                self.conformal.cal_norm_residuals.tolist()
                                if self.conformal.cal_norm_residuals is not None
                                else None
                            ),
                            "quantiles": {
                                str(k): v for k, v in self.conformal.quantiles.items()
                            },
                            "norm_quantiles": {
                                str(k): v for k, v in self.conformal.norm_quantiles.items()
                            },
                            "cal_size": self.conformal.cal_size,
                        },
                        "clf_conformal": {
                            "lac_scores": (
                                self.clf_conformal.lac_scores.tolist()
                                if self.clf_conformal.lac_scores is not None
                                else None
                            ),
                            "aps_scores": (
                                self.clf_conformal.aps_scores.tolist()
                                if self.clf_conformal.aps_scores is not None
                                else None
                            ),
                            "aps_rand_scores": (
                                self.clf_conformal.aps_rand_scores.tolist()
                                if self.clf_conformal.aps_rand_scores is not None
                                else None
                            ),
                            "lac_thresholds": {
                                str(k): v for k, v in self.clf_conformal.lac_thresholds.items()
                            },
                            "aps_thresholds": {
                                str(k): v for k, v in self.clf_conformal.aps_thresholds.items()
                            },
                            "aps_rand_thresholds": {
                                str(k): v
                                for k, v in self.clf_conformal.aps_rand_thresholds.items()
                            },
                            "cal_size": self.clf_conformal.cal_size,
                            "n_classes": self.clf_conformal.n_classes,
                        },
                    },
                    fh,
                    default=str,
                )
        except Exception as e:
            logger.warning("Could not write D-MPNN metrics sidecar: %s", e)

    def load(self, directory: str, key: str) -> bool:
        path = os.path.join(directory, f"dmpnn-{key}.pt")
        if not os.path.exists(path):
            return False
        try:
            self.ensemble = DMPNNEnsemble.load(path)
            self.training_report = self.ensemble.training_report
        except Exception as e:
            logger.warning("Could not load D-MPNN checkpoint %s: %s", path, e)
            return False

        # Without the evaluation record and the calibration state the model
        # cannot be served honestly, so a missing sidecar invalidates the load.
        import json as _json

        sidecar = os.path.join(directory, f"dmpnn-{key}.metrics.json")
        if not os.path.exists(sidecar):
            logger.warning(
                "D-MPNN checkpoint %s has no metrics sidecar; retraining.", path
            )
            self.ensemble = None
            return False
        try:
            with open(sidecar) as fh:
                blob = _json.load(fh)
            self.metrics = blob["metrics"]
            self.comparison = blob["comparison"]
            self.train_smiles = blob.get("train_smiles", [])

            c = blob["conformal"]
            self.conformal.cal_residuals = (
                np.array(c["cal_residuals"]) if c["cal_residuals"] is not None else None
            )
            self.conformal.cal_norm_residuals = (
                np.array(c["cal_norm_residuals"])
                if c["cal_norm_residuals"] is not None
                else None
            )
            self.conformal.quantiles = {float(k): v for k, v in c["quantiles"].items()}
            self.conformal.norm_quantiles = {
                float(k): v for k, v in c["norm_quantiles"].items()
            }
            self.conformal.cal_size = c["cal_size"]

            k = blob["clf_conformal"]
            for attr in ("lac_scores", "aps_scores", "aps_rand_scores"):
                setattr(
                    self.clf_conformal,
                    attr,
                    np.array(k[attr]) if k[attr] is not None else None,
                )
            self.clf_conformal.lac_thresholds = {
                float(a): b for a, b in k["lac_thresholds"].items()
            }
            self.clf_conformal.aps_thresholds = {
                float(a): b for a, b in k["aps_thresholds"].items()
            }
            self.clf_conformal.aps_rand_thresholds = {
                float(a): b for a, b in k["aps_rand_thresholds"].items()
            }
            self.clf_conformal.cal_size = k["cal_size"]
            self.clf_conformal.n_classes = k["n_classes"]
        except Exception as e:
            logger.warning("D-MPNN metrics sidecar at %s is unusable: %s", sidecar, e)
            self.ensemble = None
            return False

        self.is_trained = True
        self.status = "trained"
        return True

    def summary(self) -> Dict[str, Any]:
        """Status payload for the API, honest about whether training happened."""
        if not self.is_trained:
            return {
                "status": self.status,
                "reason": self.failure_reason
                or (
                    "The D-MPNN ensemble has not been trained in this process. Set "
                    "MOLECULEMIND_ENABLE_DMPNN=true to train it at startup."
                ),
                "architecture": {
                    "type": "Directed Message Passing Neural Network ensemble",
                    "tasks": [t.name for t in TASKS],
                },
            }
        return {
            "status": "trained",
            "architecture": {
                "type": "Directed Message Passing Neural Network ensemble",
                "message_passing_steps": self.metrics.get("config", {}).get("depth"),
                "hidden_size": self.metrics.get("config", {}).get("hidden_size"),
                "readout": "mean over atoms",
                "head": "two-layer feed-forward, ReLU, dropout 0.1",
                "regression_loss": "Gaussian negative log-likelihood",
                "classification_loss": "binary cross-entropy with logits, masked",
                "n_members": self.n_members,
            },
            "metrics": self.metrics,
            "training_report": self.training_report,
            "independent_validation": self.comparison,
        }
