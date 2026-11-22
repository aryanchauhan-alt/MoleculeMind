"""
Applicability Domain (AD) Detection Engine.

Two independent diagnostics are combined:

  1. **Structural similarity** - maximum ECFP4 Tanimoto similarity to the
     training set, plus the mean of the top-5 neighbours.
  2. **Descriptor-space extrapolation** - leverage h = z (ZᵀZ)⁻¹ zᵀ computed in
     a PCA space fitted on *standardised* features.

Thresholds are DERIVED FROM THE TRAINING SET rather than asserted
--------------------------------------------------------------------
Earlier revisions hardcoded Tanimoto cut-offs of 0.55 and 0.35. Those are
plausible literature values but they were never checked against this dataset,
and the right cut-off depends entirely on how diverse the training set is.

The thresholds are now computed at fit time from leave-one-out training
diagnostics:

  * `similarity_threshold` = the Nth percentile (default 5th) of each training
    compound's leave-one-out nearest-neighbour similarity. By construction,
    ~95% of training compounds sit at or above it, so a query below it is
    less typical of the training set than almost every training compound is.
  * `leverage_threshold` = the Nth percentile (default 95th) of the training
    leverage distribution. The OECD rule-of-thumb h* = 3(p+1)/n is computed
    alongside it for reference and reported, but the empirical percentile is
    what the classification uses.

Classification follows the two-condition rule in the project methodology: a
candidate is flagged **out of domain only when both** diagnostics fail, and
**borderline** when exactly one does. Failing one of two independent tests is
weaker evidence than failing both, and collapsing them into a single verdict
would discard that distinction.

What this is NOT
----------------
`domain_index` is a bounded composite of the two diagnostics, useful for
ranking. It is **not** a calibrated probability that a prediction is correct,
and it must not be read as one. Conformal intervals are the calibrated
uncertainty statement; the applicability domain tells you whether the
exchangeability assumption those intervals rest on is plausible for this query.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.app.core.config import settings


class ApplicabilityDomainEngine:
    def __init__(
        self,
        fp_radius: int = 2,
        fp_size: int = 1024,
        similarity_percentile: Optional[float] = None,
        leverage_percentile: Optional[float] = None,
        n_components: Optional[int] = None,
    ):
        self.fp_radius = fp_radius
        self.fp_size = fp_size
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_size
        )
        self.train_fps: List[Any] = []
        self.train_smiles: List[str] = []
        self.train_ids: List[str] = []
        self.train_targets: List[float] = []

        self.similarity_percentile = (
            similarity_percentile
            if similarity_percentile is not None
            else settings.ad_similarity_percentile
        )
        self.leverage_percentile = (
            leverage_percentile
            if leverage_percentile is not None
            else settings.ad_distance_percentile
        )
        self.n_components = (
            n_components if n_components is not None else settings.ad_pca_components
        )

        # Fitted descriptor-space model. Features MUST be standardised before
        # PCA: on the raw 1032-dim matrix molecular weight carries ~91% of the
        # variance and leverage degenerates into a molecular-weight test.
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.inv_cov: Optional[np.ndarray] = None

        # Derived thresholds.
        self.similarity_threshold: float = 0.0
        self.leverage_threshold: float = 0.0
        self.oecd_leverage_threshold: float = 0.0
        self.n_samples: int = 0
        self.threshold_diagnostics: Dict[str, Any] = {}

        # Optional learned latent space (D-MPNN molecule embeddings). When
        # fitted, nearest-neighbour latent distance replaces descriptor-space
        # leverage as the second condition, which is the diagnostic pair the
        # project methodology specifies. Leverage is still computed and
        # reported, but it is weak on a sparse fingerprint space.
        self.train_latent: Optional[np.ndarray] = None
        self.latent_threshold: float = 0.0
        self.latent_diagnostics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        smiles_list: List[str],
        compound_ids: List[str],
        targets: List[float],
        feature_matrix: np.ndarray,
    ):
        """Index the training set and derive both thresholds from it."""
        self.train_fps = []
        self.train_smiles = []
        self.train_ids = []
        self.train_targets = []
        keep_rows: List[int] = []

        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            self.train_fps.append(self.fp_generator.GetFingerprint(mol))
            self.train_smiles.append(s)
            self.train_ids.append(
                compound_ids[i] if i < len(compound_ids) else f"CMPD_{i}"
            )
            self.train_targets.append(targets[i] if i < len(targets) else 0.0)
            keep_rows.append(i)

        self.n_samples = len(self.train_fps)
        if self.n_samples < 3:
            raise ValueError(
                f"Applicability domain needs at least 3 training molecules, got {self.n_samples}."
            )

        X = np.asarray(feature_matrix, dtype=np.float64)[keep_rows]

        # ---- descriptor space: standardise, then PCA --------------------
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        n_comp = int(min(self.n_components, Xs.shape[1], self.n_samples - 1))
        self.pca = PCA(n_components=n_comp, random_state=settings.random_seed)
        Z = self.pca.fit_transform(Xs)
        self.inv_cov = np.linalg.pinv(Z.T @ Z)

        # ---- leverage threshold from the empirical training distribution --
        train_leverage = np.einsum("ij,jk,ik->i", Z, self.inv_cov, Z)
        self.leverage_threshold = float(
            np.percentile(train_leverage, self.leverage_percentile)
        )
        p = int(self.pca.n_components_)
        self.oecd_leverage_threshold = float(3.0 * (p + 1) / self.n_samples)

        # ---- similarity threshold from leave-one-out nearest neighbours ---
        loo_max_sim = np.empty(self.n_samples, dtype=np.float64)
        for i, fp in enumerate(self.train_fps):
            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fp, self.train_fps), dtype=np.float64
            )
            sims[i] = -1.0  # exclude the compound itself
            loo_max_sim[i] = sims.max()
        self.similarity_threshold = float(
            np.percentile(loo_max_sim, self.similarity_percentile)
        )

        # ---- record how the thresholds behave on the training set itself --
        flagged = int(
            np.sum(
                (loo_max_sim < self.similarity_threshold)
                & (train_leverage > self.leverage_threshold)
            )
        )
        self.threshold_diagnostics = {
            "n_training_molecules": self.n_samples,
            "pca_components": p,
            "pca_explained_variance": round(
                float(self.pca.explained_variance_ratio_.sum()), 4
            ),
            "similarity_percentile": self.similarity_percentile,
            "similarity_threshold": round(self.similarity_threshold, 4),
            "training_loo_similarity": {
                "min": round(float(loo_max_sim.min()), 4),
                "median": round(float(np.median(loo_max_sim)), 4),
                "max": round(float(loo_max_sim.max()), 4),
            },
            "leverage_percentile": self.leverage_percentile,
            "leverage_threshold": round(self.leverage_threshold, 6),
            "oecd_leverage_threshold": round(self.oecd_leverage_threshold, 6),
            "training_leverage": {
                "min": round(float(train_leverage.min()), 6),
                "median": round(float(np.median(train_leverage)), 6),
                "max": round(float(train_leverage.max()), 6),
            },
            "training_molecules_flagged_out_of_domain": flagged,
            "training_flag_rate": round(flagged / self.n_samples, 4),
            "note": (
                "Thresholds are percentiles of the training set's own leave-one-out "
                "diagnostics, not literature constants. They describe how typical a "
                "query is of this training set; they are not a guarantee of "
                "generalisation."
            ),
        }

    # --------------------------------------------------------------- assess

    def assess(
        self,
        mol: Chem.Mol,
        feature_vector: np.ndarray,
        latent_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Assess a single query molecule against the fitted training domain.

        When `latent_vector` is supplied and a latent space has been fitted, the
        second condition is nearest-neighbour latent distance (the methodology's
        choice). Otherwise it falls back to descriptor-space leverage.
        """
        if not self.train_fps:
            raise RuntimeError("ApplicabilityDomainEngine.assess() called before fit().")

        query_fp = self.fp_generator.GetFingerprint(mol)

        # 1. Structural similarity to the training set.
        tanimoto = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query_fp, self.train_fps),
            dtype=np.float64,
        )
        max_idx = int(np.argmax(tanimoto))
        max_tanimoto = float(tanimoto[max_idx])

        k = min(5, len(tanimoto))
        top_k_idx = np.argsort(tanimoto)[-k:][::-1]
        mean_top5 = float(np.mean(tanimoto[top_k_idx]))

        nearest_analogs = [
            {
                "compound_id": self.train_ids[i],
                "smiles": self.train_smiles[i],
                "measured_value": round(float(self.train_targets[i]), 3),
                "tanimoto_similarity": round(float(tanimoto[i]), 3),
            }
            for i in top_k_idx
        ]

        # 2. Descriptor-space leverage.
        z = self.pca.transform(
            self.scaler.transform(np.asarray(feature_vector, dtype=np.float64).reshape(1, -1))
        )
        leverage = float(np.squeeze(z @ self.inv_cov @ z.T))

        similarity_fails = max_tanimoto < self.similarity_threshold
        leverage_fails = leverage > self.leverage_threshold

        # 3. Second condition: latent distance when available, else leverage.
        latent_dist: Optional[float] = None
        latent_fails: Optional[bool] = None
        if latent_vector is not None and self.has_latent_space:
            latent_dist = self.latent_distance(latent_vector)
            latent_fails = latent_dist > self.latent_threshold
            second_condition_fails = latent_fails
            second_condition_name = "latent-space nearest-neighbour distance"
        else:
            second_condition_fails = leverage_fails
            second_condition_name = "descriptor-space leverage"

        # 4. Two-condition classification.
        if similarity_fails and second_condition_fails:
            status = "OUT_OF_DOMAIN"
            confidence_level = "Low (extrapolation)"
            color = "red"
            explanation = (
                "Both diagnostics fail: the molecule has no close structural analogue "
                "in the training set AND sits outside its descriptor-space spread. "
                "Predictions are extrapolations and the conformal coverage guarantee, "
                "which assumes exchangeability with the calibration set, should not be "
                "relied on here."
            )
        elif similarity_fails or second_condition_fails:
            status = "BORDERLINE"
            confidence_level = "Moderate"
            color = "amber"
            failed = "structural similarity" if similarity_fails else second_condition_name
            explanation = (
                f"One of two diagnostics fails ({failed}). The molecule is on the "
                "perimeter of the training domain; the prediction is usable but carries "
                "elevated epistemic uncertainty."
            )
        else:
            status = "IN_DOMAIN"
            confidence_level = "High"
            color = "green"
            explanation = (
                "Both diagnostics pass: the molecule has a close structural analogue in "
                "the training set and lies within its descriptor-space spread."
            )

        # 5. Bounded composite index for ranking. NOT a probability.
        sim_component = float(np.clip(max_tanimoto, 0.0, 1.0))
        if latent_dist is not None:
            excess = max(0.0, latent_dist - self.latent_threshold)
            second_component = float(np.exp(-excess / max(self.latent_threshold, 1e-9)))
        else:
            excess = max(0.0, leverage - self.leverage_threshold)
            second_component = float(np.exp(-excess / max(self.leverage_threshold, 1e-9)))
        domain_index = float(
            np.clip(0.6 * sim_component + 0.4 * second_component, 0.0, 1.0)
        )

        return {
            "status": status,
            "confidence_level": confidence_level,
            "color": color,
            "explanation": explanation,
            "domain_index": round(domain_index, 3),
            # Legacy key kept so existing frontend fields keep resolving.
            "reliability_score": round(domain_index, 3),
            "index_caveat": (
                "domain_index is a bounded composite of structural similarity and "
                "descriptor-space leverage, intended for ranking only. It is not a "
                "calibrated probability that the prediction is correct."
            ),
            "conditions": {
                "similarity_fails": bool(similarity_fails),
                "leverage_fails": bool(leverage_fails),
                "latent_distance_fails": None if latent_fails is None else bool(latent_fails),
                "second_condition": second_condition_name,
                "rule": "OUT_OF_DOMAIN requires both conditions to fail; one failure is BORDERLINE.",
            },
            "metrics": {
                "max_tanimoto_similarity": round(max_tanimoto, 3),
                "mean_top5_tanimoto": round(mean_top5, 3),
                "similarity_threshold": round(self.similarity_threshold, 3),
                "leverage": round(leverage, 4),
                "leverage_threshold": round(self.leverage_threshold, 4),
                "oecd_leverage_threshold": round(self.oecd_leverage_threshold, 4),
                "leverage_warning": bool(leverage_fails),
                "latent_distance": None if latent_dist is None else round(latent_dist, 4),
                "latent_threshold": (
                    round(self.latent_threshold, 4) if self.has_latent_space else None
                ),
            },
            "nearest_analog": nearest_analogs[0],
            "nearest_analogs": nearest_analogs,
            "threshold_provenance": {
                "method": "percentiles of training-set leave-one-out diagnostics",
                "similarity_percentile": self.similarity_percentile,
                "leverage_percentile": self.leverage_percentile,
                "latent_space_fitted": self.has_latent_space,
                "n_training_molecules": self.n_samples,
            },
        }

    # ------------------------------------------------------- latent space

    def fit_latent(
        self,
        train_embeddings: np.ndarray,
        percentile: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Index a learned latent space and derive its distance threshold.

        `train_embeddings` must be in the SAME ROW ORDER as the SMILES passed to
        `fit()`. The threshold is the Nth percentile (default 95th) of each
        training molecule's leave-one-out nearest-neighbour Euclidean distance,
        so a query further than it is more isolated in latent space than ~95% of
        the training set.
        """
        emb = np.asarray(train_embeddings, dtype=np.float64)
        if emb.shape[0] != self.n_samples:
            raise ValueError(
                f"Expected {self.n_samples} embeddings to match the fitted training "
                f"set, got {emb.shape[0]}."
            )
        pct = percentile if percentile is not None else self.leverage_percentile
        self.train_latent = emb

        # Leave-one-out nearest-neighbour distance: the diagonal of the pairwise
        # distance matrix is masked out so a molecule is not its own neighbour.
        sq = np.sum(emb ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (emb @ emb.T)
        np.fill_diagonal(d2, np.inf)
        loo = np.sqrt(np.maximum(d2.min(axis=1), 0.0))

        self.latent_threshold = float(np.percentile(loo, pct))
        flagged = int(np.sum(loo > self.latent_threshold))
        self.latent_diagnostics = {
            "enabled": True,
            "embedding_dim": int(emb.shape[1]),
            "percentile": pct,
            "latent_threshold": round(self.latent_threshold, 4),
            "training_loo_distance": {
                "min": round(float(loo.min()), 4),
                "median": round(float(np.median(loo)), 4),
                "max": round(float(loo.max()), 4),
            },
            "training_molecules_above_threshold": flagged,
            "source": "D-MPNN ensemble molecule embeddings (mean over members)",
            "note": (
                "Latent distance replaces descriptor-space leverage as the second "
                "applicability-domain condition when a learned representation is "
                "available. Leverage on a standardised 1032-dim sparse fingerprint "
                "space explains little variance and is a weak diagnostic."
            ),
        }
        self.threshold_diagnostics["latent_space"] = self.latent_diagnostics
        return self.latent_diagnostics

    def latent_distance(self, latent_vector: np.ndarray) -> Optional[float]:
        """Euclidean distance from a query embedding to its nearest training neighbour."""
        if self.train_latent is None:
            return None
        v = np.asarray(latent_vector, dtype=np.float64).reshape(1, -1)
        if v.shape[1] != self.train_latent.shape[1]:
            raise ValueError(
                f"Latent dimension mismatch: query {v.shape[1]} vs "
                f"training {self.train_latent.shape[1]}."
            )
        d = np.linalg.norm(self.train_latent - v, axis=1)
        return float(d.min())

    @property
    def has_latent_space(self) -> bool:
        return self.train_latent is not None

    def diagnostics(self) -> Dict[str, Any]:
        d = dict(self.threshold_diagnostics)
        d["latent_space"] = self.latent_diagnostics or {"enabled": False}
        return d

    # ------------------------------------------------------------ pickling

    # RDKit's FingerprintGenerator64 is a Boost.Python object and cannot be
    # pickled. It is fully determined by (radius, fpSize), so it is dropped on
    # the way out and rebuilt on the way in.
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("fp_generator", None)
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=state.get("fp_radius", 2), fpSize=state.get("fp_size", 1024)
        )
        # Defaults for attributes added after an artifact may have been written.
        # The cache key covers the model version, so a stale artifact is normally
        # rejected before reaching here; this keeps an older pickle degrading to
        # "no latent space" instead of raising AttributeError.
        for attr, default in (
            ("train_latent", None),
            ("latent_threshold", 0.0),
            ("latent_diagnostics", {}),
        ):
            if not hasattr(self, attr):
                setattr(self, attr, default)
