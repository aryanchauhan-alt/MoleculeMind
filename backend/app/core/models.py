"""
Machine Learning Model Service.

Trains and serves models with explicit provenance for every value returned:

  * ESOL aqueous solubility (logS) - ExtraTrees regressor, conformal intervals
  * ClinTox clinical toxicity      - RandomForest classifier, properly evaluated
  * QED / descriptors              - deterministic RDKit computation

Properties that this platform does NOT have a validated model for are returned
with `status: "unavailable"` and a reason. They are never given a numeric value
and never enter multi-objective optimisation.

Scientific honesty notes
------------------------
Earlier revisions of this module returned two properties computed by closed-form
arithmetic on descriptors and labelled them as model predictions:

    bioactivity_pic50  = clip(5.5 + 2.5*QED + 1.0*(max_tanimoto - 0.5), 4.0, 9.8)
    metabolic_t_half   = clip(60 - 5*logP - 2*rot_bonds + 10*QED, 5.0, 150.0)

Neither was fitted to any data. The first is not target-specific at all and its
similarity term measured closeness to an aqueous-solubility training set, which
carries no information about binding affinity. Both have been removed; see
PROJECT_AUDIT.md section 10.4. Reinstating them requires a curated bioactivity
dataset (e.g. ChEMBL EGFR) and a trained, evaluated QSAR model.
"""

import os
import json
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from rdkit import Chem

from backend.app.core.config import settings
from backend.app.core.dataset import DatasetManager, MolecularDataset, scaffold_split
from backend.app.core.conformal import ConformalPredictionEngine
from backend.app.core.applicability_domain import ApplicabilityDomainEngine
from backend.app.core.mol_analyzer import compute_molecular_descriptors, generate_2d_svg


logger = logging.getLogger("moleculemind.models")

MODEL_VERSION = "3.0.0"
# Bump when the trained-artifact layout changes so stale caches are rejected.
CACHE_FORMAT = 3
RANDOM_SEED = settings.random_seed

# Properties the platform is asked about but has no validated model for.
UNAVAILABLE_PROPERTIES: Dict[str, Dict[str, str]] = {
    "bioactivity_pic50": {
        "property_name": "Target Bioactivity (pIC50)",
        "reason": (
            "No curated target-specific bioactivity dataset (e.g. ChEMBL EGFR) is "
            "bundled with this platform, so no QSAR model has been trained or "
            "evaluated. A descriptor heuristic was previously returned here; it has "
            "been removed because it was not fitted to any experimental data."
        ),
        "required_to_enable": "Curated ChEMBL assay data + trained and evaluated QSAR model",
    },
    "metabolic_stability": {
        "property_name": "Metabolic Stability (microsomal t1/2)",
        "reason": (
            "No metabolic-stability dataset is bundled with this platform. The "
            "previous value came from an arbitrary descriptor formula with invented "
            "coefficients and invented units, and has been removed."
        ),
        "required_to_enable": "Curated microsomal clearance / half-life dataset + trained model",
    },
}


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(
        self,
        data_dir: Optional[str] = None,
        quick: bool = False,
        artifact_dir: Optional[str] = None,
        use_cache: Optional[bool] = None,
        write_cache: Optional[bool] = None,
    ):
        data_dir = data_dir or settings.data_dir
        self.artifact_dir = artifact_dir or settings.artifact_dir
        self.use_cache = settings.use_model_cache if use_cache is None else use_cache
        self.write_cache = settings.write_model_cache if write_cache is None else write_cache
        self.loaded_from_cache: bool = False
        self.dataset_manager = DatasetManager(data_dir=data_dir)
        self.conformal_engine = ConformalPredictionEngine()
        self.ad_engine = ApplicabilityDomainEngine()
        self.quick = quick

        self.esol_model: Optional[ExtraTreesRegressor] = None
        self.esol_metrics: Dict[str, Any] = {}
        self.scaffold_metrics: Dict[str, Any] = {}
        self.test_parity_data: List[Dict[str, Any]] = []

        self.clintox_model: Optional[RandomForestClassifier] = None
        self.clintox_metrics: Dict[str, Any] = {}
        # Canonical SMILES used to FIT the ClinTox model. A prediction for a
        # molecule in this set is partly memorisation, not generalisation, so
        # the API says so rather than presenting it as a fresh prediction.
        self.clintox_train_smiles: set = set()

        # Stage 3: D-MPNN ensemble. Optional and separately checkpointed, so a
        # deployment that cannot afford the training cost still gets the full
        # Stage 2 platform rather than a broken one.
        self.dmpnn_service: Optional[Any] = None
        # Stage 4: SMILES generator, separately checkpointed and optional.
        self.generator_service: Optional[Any] = None

        # Held-out ESOL compounds reserved for the active-learning campaign.
        # These are never used to train or calibrate the solubility model, so
        # their measured values are genuinely unknown to it.
        self.al_pool_payload: Dict[str, Any] = {}

        self.model_version: str = MODEL_VERSION
        self.is_initialized: bool = False

        # Conformal calibration inputs are retained so the engine can be
        # refitted after active learning adds labels, instead of silently
        # reusing a stale calibration (see recalibrate_conformal).
        self._cal_y: Optional[np.ndarray] = None
        self._cal_pred: Optional[np.ndarray] = None
        self._cal_std: Optional[np.ndarray] = None
        self.calibration_generation: int = 0
        self.calibration_history: List[Dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    # ------------------------------------------------------------- training

    # ------------------------------------------------------- persistence

    def cache_key(self) -> str:
        """
        Identity of a trained artifact set.

        Hashes the raw dataset FILES (not the parsed datasets) together with
        every setting that affects training, the model version and the cache
        format. Using file bytes keeps the key cheap to compute, so a cache hit
        skips RDKit featurisation of ~2,600 molecules entirely rather than
        paying for it just to discover the cache is valid.
        """
        h = hashlib.sha256()
        for filename in ("delaney-processed.csv", "clintox.csv.gz"):
            path = os.path.join(self.dataset_manager.data_dir, filename)
            h.update(filename.encode())
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            else:
                h.update(b"<missing>")

        payload = {
            "cache_format": CACHE_FORMAT,
            "model_version": MODEL_VERSION,
            "quick": self.quick,
            "data_files": h.hexdigest(),
            "settings": settings.training_fingerprint(),
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:20]

    def cache_path(self) -> str:
        return os.path.join(self.artifact_dir, f"moleculemind-{self.cache_key()}.pkl")

    def _save_cache(self):
        path = self.cache_path()
        try:
            os.makedirs(self.artifact_dir, exist_ok=True)
            artifact = {
                "cache_format": CACHE_FORMAT,
                "model_version": MODEL_VERSION,
                "cache_key": self.cache_key(),
                "settings": settings.training_fingerprint(),
                "esol_model": self.esol_model,
                "esol_metrics": self.esol_metrics,
                "scaffold_metrics": self.scaffold_metrics,
                "test_parity_data": self.test_parity_data,
                "clintox_model": self.clintox_model,
                "clintox_metrics": self.clintox_metrics,
                "clintox_train_smiles": self.clintox_train_smiles,
                "al_pool_payload": self.al_pool_payload,
                "ad_engine": self.ad_engine,
                "conformal_engine": self.conformal_engine,
                "cal_y": self._cal_y,
                "cal_pred": self._cal_pred,
                "cal_std": self._cal_std,
                "calibration_history": self.calibration_history,
            }
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)  # atomic: never leaves a half-written cache
            logger.info("Saved model cache to %s", path)
        except Exception as e:
            # A cache write failure must never break a working service, but it
            # must not leave a partial file behind either.
            try:
                if os.path.exists(path + ".tmp"):
                    os.remove(path + ".tmp")
            except OSError:
                pass
            logger.warning("Could not write model cache (%s); continuing.", e)

    def _load_cache(self) -> bool:
        path = self.cache_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                artifact = pickle.load(fh)
        except Exception as e:
            logger.warning("Model cache at %s is unreadable (%s); retraining.", path, e)
            return False

        if (
            artifact.get("cache_format") != CACHE_FORMAT
            or artifact.get("model_version") != MODEL_VERSION
            or artifact.get("cache_key") != self.cache_key()
        ):
            logger.info("Model cache at %s does not match current config; retraining.", path)
            return False

        try:
            self.esol_model = artifact["esol_model"]
            self.esol_metrics = artifact["esol_metrics"]
            self.scaffold_metrics = artifact["scaffold_metrics"]
            self.test_parity_data = artifact["test_parity_data"]
            self.clintox_model = artifact["clintox_model"]
            self.clintox_metrics = artifact["clintox_metrics"]
            self.clintox_train_smiles = artifact["clintox_train_smiles"]
            self.al_pool_payload = artifact["al_pool_payload"]
            self.ad_engine = artifact["ad_engine"]
            self.conformal_engine = artifact["conformal_engine"]
            self._cal_y = artifact["cal_y"]
            self._cal_pred = artifact["cal_pred"]
            self._cal_std = artifact["cal_std"]
            self.calibration_history = artifact["calibration_history"]
            self.calibration_generation = len(self.calibration_history)
        except KeyError as e:
            logger.warning("Model cache is missing %s; retraining.", e)
            return False

        self.loaded_from_cache = True
        logger.info("Loaded trained models from cache %s", path)
        return True

    # ---------------------------------------------------------- training

    def initialize(self):
        """Load a matching cached artifact set if one exists, otherwise train."""
        if self.is_initialized:
            return

        if self.use_cache and self._load_cache():
            # The D-MPNN has its own checkpoint; the latent AD state travels
            # inside the pickled AD engine, so only the network must be reloaded.
            self._load_or_train_dmpnn()
            self._load_or_train_generator()
            self.is_initialized = True
            return

        self._train_from_scratch()
        if self.write_cache:
            self._save_cache()

    def _train_from_scratch(self):
        logger.info("=== Training MoleculeMind models ===")
        ds = self.dataset_manager.get_esol_dataset()
        logger.info(
            "ESOL: %d unique compounds (%d duplicate structures removed, "
            "%d unparseable) hash=%s",
            len(ds), ds.n_duplicates_removed, ds.n_unparseable, ds.data_hash,
        )

        # ---- Partition: train 70% / calibration 15% / test 15% -------------
        # Deduplication above guarantees no canonical structure spans partitions.
        idx = np.arange(len(ds))
        test_frac = settings.test_fraction
        cal_frac = settings.cal_fraction
        idx_tr_cal, idx_test = train_test_split(
            idx, test_size=test_frac, random_state=RANDOM_SEED
        )
        idx_train, idx_cal = train_test_split(
            idx_tr_cal, test_size=cal_frac / (1.0 - test_frac), random_state=RANDOM_SEED
        )

        # Reserve a slice of the test partition for the active-learning campaign
        # so the AL oracle consists of real measurements the model has never seen.
        n_al = min(settings.al_pool_size, len(idx_test) // 3)
        rng = np.random.default_rng(RANDOM_SEED)
        idx_al = rng.choice(idx_test, size=n_al, replace=False)
        idx_test_eval = np.array([i for i in idx_test if i not in set(idx_al.tolist())])

        logger.info(
            "Split -> train %d, cal %d, test %d, AL pool %d",
            len(idx_train), len(idx_cal), len(idx_test_eval), len(idx_al),
        )

        X_train, y_train = ds.X[idx_train], ds.y[idx_train]
        X_cal, y_cal = ds.X[idx_cal], ds.y[idx_cal]
        X_test, y_test = ds.X[idx_test_eval], ds.y[idx_test_eval]

        n_estimators = 50 if self.quick else settings.esol_n_estimators
        self.esol_model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=settings.esol_max_depth,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.esol_model.fit(X_train, y_train)

        # ---- Conformal calibration on the held-out calibration partition ----
        y_cal_pred, y_cal_std = self._ensemble_predict(X_cal)
        self._cal_y, self._cal_pred, self._cal_std = y_cal, y_cal_pred, y_cal_std
        self.conformal_engine.calibrate(y_cal, y_cal_pred, y_cal_std)
        self.calibration_generation = 1
        self.calibration_history = [
            {
                "generation": 1,
                "trigger": "initial fit",
                "calibration_set_size": int(len(y_cal)),
                "extra_labels": 0,
            }
        ]

        # ---- Test evaluation ------------------------------------------------
        y_test_pred, y_test_std = self._ensemble_predict(X_test)
        r2 = float(r2_score(y_test, y_test_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        mae = float(mean_absolute_error(y_test, y_test_pred))
        coverage_eval = self.conformal_engine.evaluate_test_coverage(
            y_test, y_test_pred, y_test_std
        )

        self.esol_metrics = {
            "dataset": "Delaney ESOL (Aqueous Solubility)",
            "target": ds.target_name,
            "model": f"ExtraTreesRegressor(n_estimators={n_estimators}, max_depth=20)",
            "model_version": self.model_version,
            "random_seed": RANDOM_SEED,
            "split": "random (deduplicated on canonical SMILES)",
            "provenance": ds.provenance(),
            "n_samples": len(ds),
            "train_samples": len(idx_train),
            "cal_samples": len(idx_cal),
            "test_samples": len(idx_test_eval),
            "al_reserved_samples": len(idx_al),
            "r2_score": round(r2, 3),
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "conformal_test_coverage": coverage_eval,
            "settings": settings.training_fingerprint(),
        }
        logger.info("ESOL random split -> R2 %.3f  RMSE %.3f  MAE %.3f", r2, rmse, mae)

        # ---- Scaffold split: the harder, chemically honest evaluation -------
        self.scaffold_metrics = self._evaluate_scaffold_split(ds, n_estimators)

        # ---- Parity points for the benchmark dashboard ----------------------
        self.test_parity_data = []
        for k, i in enumerate(idx_test_eval[: min(100, len(idx_test_eval))]):
            pos = int(np.where(idx_test_eval == i)[0][0])
            conf_int = self.conformal_engine.predict_intervals(
                float(y_test_pred[pos]), float(y_test_std[pos])
            )
            self.test_parity_data.append(
                {
                    "compound_id": ds.ids[i],
                    "smiles": ds.smiles[i],
                    "actual": round(float(y_test[pos]), 3),
                    "predicted": round(float(y_test_pred[pos]), 3),
                    "std": round(float(y_test_std[pos]), 3),
                    "lower_90": conf_int["intervals"]["90%"]["lower_bound"],
                    "upper_90": conf_int["intervals"]["90%"]["upper_bound"],
                    "error": round(float(abs(y_test[pos] - y_test_pred[pos])), 3),
                }
            )

        # ---- Applicability domain, fitted on the training partition only ----
        self.ad_engine.fit(
            smiles_list=[ds.smiles[i] for i in idx_train],
            compound_ids=[ds.ids[i] for i in idx_train],
            targets=list(y_train),
            feature_matrix=X_train,
        )
        logger.info(
            "Applicability domain fitted: similarity threshold %.3f, leverage threshold %.4f "
            "(%.1f%% of training molecules flagged)",
            self.ad_engine.similarity_threshold,
            self.ad_engine.leverage_threshold,
            100.0 * self.ad_engine.threshold_diagnostics["training_flag_rate"],
        )

        # ---- Active-learning pool: real held-out measurements ---------------
        self.al_pool_payload = {
            "source": "Delaney ESOL held-out partition (never used for training or calibration)",
            "target_name": "Measured aqueous solubility, logS",
            "data_hash": ds.data_hash,
            "compounds": [
                {
                    "id": ds.ids[i],
                    "name": ds.ids[i],
                    "smiles": ds.smiles[i],
                    "measured_logs": float(ds.y[i]),
                }
                for i in sorted(idx_al.tolist())
            ],
        }

        self._train_clintox()
        self._load_or_train_dmpnn(
            esol_eval_smiles=[ds.smiles[i] for i in idx_test_eval]
        )
        self._load_or_train_generator()
        self.is_initialized = True

    def _load_or_train_dmpnn(self, esol_eval_smiles: Optional[List[str]] = None):
        """Train (or load) the D-MPNN ensemble and fit the latent AD space."""
        if not settings.enable_dmpnn:
            logger.info("D-MPNN disabled (MOLECULEMIND_ENABLE_DMPNN=false); skipping.")
            return

        from backend.app.core.dmpnn_service import DMPNNService

        service = DMPNNService(
            self.dataset_manager, quick=self.quick, n_members=settings.dmpnn_members
        )
        key = self.cache_key()

        if self.use_cache and service.load(self.artifact_dir, key):
            logger.info("Loaded D-MPNN checkpoint from cache.")
        elif esol_eval_smiles is None:
            # Reached only on a warm start whose D-MPNN checkpoint is missing.
            # Retraining here would silently cost ~70 s on a supposedly fast
            # path, so the service reports itself untrained instead.
            logger.warning(
                "No D-MPNN checkpoint for cache key %s; the ensemble is unavailable "
                "this run. Delete %s to force a full retrain.",
                key, self.artifact_dir,
            )
            service.status = "not_trained"
            service.failure_reason = (
                "No checkpoint matched the current cache key and this process started "
                "from a cached baseline, so the ensemble was not retrained. Clear "
                "backend/artifacts/ to retrain."
            )
            self.dmpnn_service = service
            return
        else:
            try:
                # Pin the D-MPNN's test partition to the baseline's, so the two
                # models are compared on identical held-out compounds.
                service.train(esol_eval_smiles=esol_eval_smiles)
                if self.write_cache:
                    service.save(self.artifact_dir, key)
            except Exception as e:
                logger.error("D-MPNN training failed: %s", e)
                service.status = "failed"
                service.failure_reason = str(e)
                self.dmpnn_service = service
                return

        self.dmpnn_service = service

        # Fit the applicability-domain latent space on the SAME training rows the
        # AD engine was fitted with, in the same order.
        if settings.use_latent_ad and service.is_trained:
            try:
                embeddings = service.embed_smiles(self.ad_engine.train_smiles)
                if embeddings is not None:
                    diag = self.ad_engine.fit_latent(embeddings)
                    logger.info(
                        "Latent applicability domain fitted: dim %d, threshold %.3f",
                        diag["embedding_dim"], diag["latent_threshold"],
                    )
            except Exception as e:
                logger.warning("Could not fit the latent applicability domain: %s", e)

    # ------------------------------------------------- recalibration

    def recalibrate_conformal(
        self,
        extra_smiles: List[str],
        extra_targets: List[float],
        trigger: str = "active learning",
    ) -> Dict[str, Any]:
        """
        Refit conformal calibration with newly revealed labels folded in.

        Required by the methodology: once active learning reveals measurements,
        the calibration set is no longer the one the stale quantile was computed
        from. Reusing it would quietly misstate coverage. The new labels are
        appended to the original calibration residuals and the quantiles are
        recomputed from the enlarged set.

        Caveat recorded honestly: the base model itself is NOT retrained here,
        so this corrects the calibration set, not the predictor. Labels chosen
        by an acquisition function are also not an i.i.d. sample of the
        candidate distribution, so strict exchangeability with the original
        calibration set does not hold and coverage after recalibration is
        approximate. The returned payload states this.
        """
        if self._cal_y is None or self.esol_model is None:
            raise RuntimeError("recalibrate_conformal() called before initialize().")
        if len(extra_smiles) != len(extra_targets):
            raise ValueError("extra_smiles and extra_targets must be the same length.")

        new_feats, new_y = [], []
        skipped = []
        for smi, target in zip(extra_smiles, extra_targets):
            feat = self.dataset_manager.extract_features_from_smiles(smi)
            if feat is None:
                skipped.append(smi)
                continue
            new_feats.append(feat)
            new_y.append(float(target))

        if new_feats:
            X_new = np.array(new_feats, dtype=np.float32)
            pred_new, std_new = self._ensemble_predict(X_new)
            cal_y = np.concatenate([self._cal_y, np.array(new_y, dtype=np.float32)])
            cal_pred = np.concatenate([self._cal_pred, pred_new])
            cal_std = np.concatenate([self._cal_std, std_new])
        else:
            cal_y, cal_pred, cal_std = self._cal_y, self._cal_pred, self._cal_std

        self.conformal_engine.calibrate(cal_y, cal_pred, cal_std)
        self.calibration_generation += 1
        entry = {
            "generation": self.calibration_generation,
            "trigger": trigger,
            "calibration_set_size": int(len(cal_y)),
            "extra_labels": int(len(new_feats)),
            "skipped_unparseable": skipped,
        }
        self.calibration_history.append(entry)
        logger.info(
            "Conformal recalibrated (generation %d): %d calibration points (+%d new)",
            self.calibration_generation, len(cal_y), len(new_feats),
        )
        return {
            **entry,
            "quantiles": {
                f"{int(c * 100)}%": round(float(q), 4)
                for c, q in sorted(self.conformal_engine.norm_quantiles.items())
            }
            or None,
            "caveat": (
                "The base model was not retrained; only the calibration set was "
                "extended. Actively selected labels are not an i.i.d. sample, so "
                "exchangeability is approximate and coverage after recalibration "
                "should be treated as indicative rather than guaranteed."
            ),
        }

    def _ensemble_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and inter-tree spread across the ExtraTrees ensemble.

        The spread is a heuristic dispersion measure, not a calibrated predictive
        standard deviation. It is only used as the normalising term sigma in the
        conformal nonconformity score s = |y - mu| / sigma, where calibration
        supplies the actual coverage guarantee.
        """
        preds = np.array([tree.predict(X) for tree in self.esol_model.estimators_])
        return preds.mean(axis=0), preds.std(axis=0)

    def _evaluate_scaffold_split(
        self, ds: MolecularDataset, n_estimators: int
    ) -> Dict[str, Any]:
        """Train a parallel model on a Bemis-Murcko scaffold split and report it separately."""
        try:
            tr, cal, te = scaffold_split(ds.smiles)
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=20,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(ds.X[tr], ds.y[tr])

            cal_preds = np.array([t.predict(ds.X[cal]) for t in model.estimators_])
            te_preds = np.array([t.predict(ds.X[te]) for t in model.estimators_])

            engine = ConformalPredictionEngine()
            engine.calibrate(ds.y[cal], cal_preds.mean(axis=0), cal_preds.std(axis=0))

            y_hat = te_preds.mean(axis=0)
            coverage = engine.evaluate_test_coverage(
                ds.y[te], y_hat, te_preds.std(axis=0)
            )
            return {
                "split": "Bemis-Murcko scaffold (deterministic, largest-group-first)",
                "train_samples": len(tr),
                "cal_samples": len(cal),
                "test_samples": len(te),
                "r2_score": round(float(r2_score(ds.y[te], y_hat)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(ds.y[te], y_hat))), 3),
                "mae": round(float(mean_absolute_error(ds.y[te], y_hat)), 3),
                "conformal_test_coverage": coverage,
                "note": (
                    "Scaffold splitting places whole chemical series on one side of the "
                    "split. Lower R2 than the random split is expected and is the more "
                    "realistic estimate of prospective performance on new chemotypes."
                ),
            }
        except Exception as e:  # pragma: no cover - diagnostic path
            return {"status": "failed", "error": str(e)}

    def _train_clintox(self):
        """Train and *evaluate* the ClinTox classifier on a stratified split."""
        ds = self.dataset_manager.get_clintox_dataset()
        if ds is None or len(ds) < 200:
            self.clintox_metrics = {
                "status": "unavailable",
                "reason": "ClinTox dataset missing or too small.",
            }
            return

        n_pos = int(ds.y.sum())
        try:
            idx = np.arange(len(ds))
            idx_tr, idx_te = train_test_split(
                idx, test_size=0.2, random_state=RANDOM_SEED, stratify=ds.y
            )
            X_tr, y_tr = ds.X[idx_tr], ds.y[idx_tr]
            X_te, y_te = ds.X[idx_te], ds.y[idx_te]
            self.clintox_train_smiles = {ds.smiles[i] for i in idx_tr}
            model = RandomForestClassifier(
                n_estimators=100 if self.quick else settings.clintox_n_estimators,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                class_weight="balanced",  # 7.5% positive rate
            )
            model.fit(X_tr, y_tr)
            p_te = model.predict_proba(X_te)[:, 1]

            self.clintox_model = model
            self.clintox_metrics = {
                "status": "available",
                "dataset": "ClinTox (CT_TOX, clinical trial toxicity)",
                "model": "RandomForestClassifier(class_weight='balanced')",
                "model_version": self.model_version,
                "random_seed": RANDOM_SEED,
                "provenance": ds.provenance(),
                "n_samples": len(ds),
                "n_positive": n_pos,
                "positive_rate": round(n_pos / len(ds), 4),
                "train_samples": len(y_tr),
                "test_samples": len(y_te),
                "roc_auc": round(float(roc_auc_score(y_te, p_te)), 3),
                "pr_auc": round(float(average_precision_score(y_te, p_te)), 3),
                "brier_score": round(float(brier_score_loss(y_te, p_te)), 4),
                "baseline_pr_auc": round(float(y_te.mean()), 4),
                "caveat": (
                    "Severe class imbalance (7.5% positive). PR-AUC should be read "
                    "against the baseline positive rate, not against ROC-AUC. These "
                    "probabilities are not calibrated and must not be used as clinical "
                    "toxicity assessments."
                ),
                "label_semantics": (
                    "CT_TOX marks a compound whose clinical trial was terminated for "
                    "toxicity reasons in the source records. It is not a general "
                    "statement that the molecule is toxic: aspirin, for example, is "
                    "labelled CT_TOX = 1 in this dataset. Read the output as 'resembles "
                    "compounds that failed trials on toxicity grounds'."
                ),
            }
            logger.info(
                "ClinTox -> ROC-AUC %s, PR-AUC %s (baseline %s)",
                self.clintox_metrics["roc_auc"],
                self.clintox_metrics["pr_auc"],
                self.clintox_metrics["baseline_pr_auc"],
            )
        except Exception as e:
            self.clintox_model = None
            self.clintox_metrics = {"status": "failed", "error": str(e)}
            logger.error("ClinTox training failed: %s", e)

    # ------------------------------------------------------------ inference

    def predict_molecule(self, smiles: str) -> Dict[str, Any]:
        """
        End-to-end prediction with explicit provenance on every property.

        Every entry under `properties` carries a `provenance` field, one of:
          * "model"       - output of a trained, evaluated model
          * "computed"    - deterministic cheminformatics calculation (RDKit)
          * "unavailable" - no validated model exists; no value is returned
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES structure: {smiles}")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        feat = self.dataset_manager.extract_features_from_mol(mol)
        feat_2d = feat.reshape(1, -1)

        # Solubility (trained model) ------------------------------------------
        sol_pred, sol_std = self._ensemble_predict(feat_2d)
        solubility_pred = float(sol_pred[0])
        solubility_std = float(sol_std[0])
        conformal_info = self.conformal_engine.predict_intervals(
            solubility_pred, solubility_std
        )

        # Latent vector for the applicability domain, when a D-MPNN exists.
        latent_vector = None
        if (
            self.dmpnn_service is not None
            and self.dmpnn_service.is_trained
            and self.ad_engine.has_latent_space
        ):
            try:
                emb = self.dmpnn_service.embed_smiles([canonical_smiles])
                if emb is not None:
                    latent_vector = emb[0]
            except Exception as e:
                logger.warning("Latent embedding failed for %s: %s", canonical_smiles, e)

        ad_assessment = self.ad_engine.assess(mol, feat, latent_vector=latent_vector)
        descriptors = compute_molecular_descriptors(mol)

        properties: Dict[str, Any] = {
            "solubility_logs": {
                "property_name": "Aqueous Solubility (logS)",
                "status": "available",
                "provenance": "model",
                "model": "ExtraTrees on Delaney ESOL",
                "value": round(solubility_pred, 3),
                "unit": "log(mol/L)",
                "ensemble_spread": round(solubility_std, 3),
                "classification": (
                    "Soluble"
                    if solubility_pred > -2.0
                    else ("Moderately Soluble" if solubility_pred > -4.0 else "Poorly Soluble")
                ),
                "conformal": conformal_info,
            },
            "qed_drug_likeness": {
                "property_name": "QED Drug-Likeness",
                "status": "available",
                "provenance": "computed",
                "model": "RDKit QED (Bickerton et al. 2012)",
                "value": descriptors["qed"],
                "unit": "dimensionless, 0-1",
            },
        }

        # Toxicity (trained model, only if it was successfully evaluated) ------
        if self.clintox_model is not None:
            tox_prob = float(self.clintox_model.predict_proba(feat_2d)[0, 1])
            in_train = canonical_smiles in self.clintox_train_smiles
            properties["toxicity_risk_probability"] = {
                "in_training_set": in_train,
                "memorisation_warning": (
                    "This molecule is in the ClinTox training partition, so the value "
                    "reflects its stored label rather than a generalising prediction."
                ) if in_train else None,
                "property_name": "Clinical Toxicity Risk (ClinTox CT_TOX)",
                "status": "available",
                "provenance": "model",
                "model": "RandomForest on ClinTox",
                "value": round(tox_prob, 3),
                "unit": "probability, uncalibrated",
                "category": (
                    "Low Risk" if tox_prob < 0.25
                    else ("Moderate Risk" if tox_prob < 0.50 else "High Risk")
                ),
                "caveat": (
                    "Uncalibrated probability from an imbalanced dataset. CT_TOX means "
                    "'clinical trial terminated for toxicity', not 'molecule is toxic'. "
                    "Not a clinical toxicity assessment."
                ),
                "metrics": {
                    "roc_auc": self.clintox_metrics.get("roc_auc"),
                    "pr_auc": self.clintox_metrics.get("pr_auc"),
                },
            }
        else:
            properties["toxicity_risk_probability"] = {
                "property_name": "Clinical Toxicity Risk (ClinTox CT_TOX)",
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": "ClinTox model unavailable: "
                + str(self.clintox_metrics.get("reason") or self.clintox_metrics.get("error")),
            }

        # Properties with no validated model -----------------------------------
        for key, info in UNAVAILABLE_PROPERTIES.items():
            properties[key] = {
                "property_name": info["property_name"],
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": info["reason"],
                "required_to_enable": info["required_to_enable"],
            }

        # D-MPNN prediction alongside the baseline, for independent comparison.
        dmpnn_block: Optional[Dict[str, Any]] = None
        if self.dmpnn_service is not None and self.dmpnn_service.is_trained:
            try:
                dmpnn_pred = self.dmpnn_service.predict(canonical_smiles)
                baseline = solubility_pred
                graph_value = dmpnn_pred["solubility_logs"]["value"]
                graph_sigma = dmpnn_pred["solubility_logs"]["sigma"]
                difference = abs(graph_value - baseline)
                dmpnn_block = {
                    **dmpnn_pred,
                    "agreement_with_baseline": {
                        "baseline_model": "ExtraTrees on Morgan fingerprints",
                        "baseline_value": round(baseline, 3),
                        "dmpnn_value": graph_value,
                        "absolute_difference": round(difference, 3),
                        "exceeds_dmpnn_sigma": bool(difference > graph_sigma),
                        "interpretation": (
                            "The two models use different representations. Disagreement "
                            "larger than the D-MPNN's own sigma means the prediction is "
                            "representation-sensitive and deserves scrutiny. Agreement "
                            "is weak evidence of reliability, not proof of correctness."
                        ),
                    },
                }
            except Exception as e:
                logger.warning("D-MPNN prediction failed for %s: %s", canonical_smiles, e)
                dmpnn_block = {"status": "failed", "error": str(e)}

        return {
            "smiles": canonical_smiles,
            "svg_2d": generate_2d_svg(mol, width=380, height=260),
            "descriptors": descriptors,
            "model_version": self.model_version,
            "properties": properties,
            "dmpnn": dmpnn_block,
            "applicability_domain": ad_assessment,
            # Legacy key retained so the existing Prediction tab keeps rendering.
            "solubility": {
                "property_name": "Aqueous Solubility (logS)",
                "predicted_value": round(solubility_pred, 3),
                "std_deviation": round(solubility_std, 3),
                "unit": "log(mol/L)",
                "classification": properties["solubility_logs"]["classification"],
                "conformal": conformal_info,
            },
        }

    def _load_or_train_generator(self, allow_training: bool = True):
        """Load (or train) the SMILES generator prior."""
        if not settings.enable_generator:
            logger.info("Generator disabled (MOLECULEMIND_ENABLE_GENERATOR=false).")
            return

        from backend.app.core.generator_service import GeneratorService

        service = GeneratorService(self, quick=self.quick)
        key = self.cache_key()
        if self.use_cache and service.load(self.artifact_dir, key):
            logger.info("Loaded generator checkpoint from cache.")
        elif allow_training:
            report = service.train_prior()
            if report.get("status") == "trained" and self.write_cache:
                service.save(self.artifact_dir, key)
        self.generator_service = service

    @property
    def has_generator(self) -> bool:
        return (
            self.generator_service is not None
            and self.generator_service.generator is not None
        )

    @property
    def has_dmpnn(self) -> bool:
        return self.dmpnn_service is not None and self.dmpnn_service.is_trained

    def available_objectives(self) -> List[Dict[str, Any]]:
        """Objectives backed by a model or a deterministic calculation, with direction."""
        objectives = [
            {
                "key": "solubility_logs",
                "label": "Aqueous Solubility (logS)",
                "maximize": True,
                "provenance": "model",
            },
            {
                "key": "qed_drug_likeness",
                "label": "QED Drug-Likeness",
                "maximize": True,
                "provenance": "computed",
            },
        ]
        if self.clintox_model is not None:
            objectives.append(
                {
                    "key": "toxicity_risk_probability",
                    "label": "Clinical Toxicity Risk",
                    "maximize": False,
                    "provenance": "model",
                }
            )
        return objectives
