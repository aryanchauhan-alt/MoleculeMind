"""
Applicability Domain (AD) Detection Engine.
Evaluates chemical space novelty and prediction reliability using:
1. Morgan Fingerprint (ECFP4) Tanimoto similarity k-NN
2. Leverage (Hat matrix diagonal) in PCA descriptor space with OECD cutoff h* = 3(p+1)/n
3. Chemical structural analog identification
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.decomposition import PCA


class ApplicabilityDomainEngine:
    def __init__(self, fp_radius: int = 2, fp_size: int = 1024):
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_size)
        self.train_fps: List[Any] = []
        self.train_smiles: List[str] = []
        self.train_ids: List[str] = []
        self.train_targets: List[float] = []
        
        # PCA & Leverage setup
        self.pca: Optional[PCA] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.leverage_threshold: float = 0.0
        self.n_samples: int = 0
        self.n_components: int = 10

    def fit(
        self,
        smiles_list: List[str],
        compound_ids: List[str],
        targets: List[float],
        feature_matrix: np.ndarray
    ):
        """
        Index training set molecules for Tanimoto search and compute PCA leverage threshold.
        """
        self.train_fps = []
        self.train_smiles = []
        self.train_ids = []
        self.train_targets = []

        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = self.fp_generator.GetFingerprint(mol)
                self.train_fps.append(fp)
                self.train_smiles.append(s)
                self.train_ids.append(compound_ids[i] if i < len(compound_ids) else f"CMPD_{i}")
                self.train_targets.append(targets[i] if i < len(targets) else 0.0)

        self.n_samples = len(self.train_fps)

        # Fit PCA on physical descriptor part (last 8 features) + compressed bit vector
        # Using 10 principal components
        self.pca = PCA(n_components=min(self.n_components, feature_matrix.shape[1], self.n_samples - 1))
        Z = self.pca.fit_transform(feature_matrix)
        
        # Compute pseudo-inverse of Z^T Z for leverage calculation: h = z (Z^T Z)^-1 z^T
        Z_pinv = np.linalg.pinv(Z.T @ Z)
        self.inv_cov = Z_pinv
        
        # Standard OECD Guidance Document on QSAR Applicability Domain threshold:
        # h* = 3 * (p + 1) / n, where p = number of variables/components, n = number of compounds
        p = self.pca.n_components_
        self.leverage_threshold = float(3.0 * (p + 1) / self.n_samples)

    def assess(self, mol: Chem.Mol, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Assess applicability domain for a single query molecule.
        """
        query_fp = self.fp_generator.GetFingerprint(mol)
        
        # 1. Compute Tanimoto similarities across training set
        tanimoto_scores = DataStructs.BulkTanimotoSimilarity(query_fp, self.train_fps)
        tanimoto_arr = np.array(tanimoto_scores)
        
        max_idx = int(np.argmax(tanimoto_arr))
        max_tanimoto = float(tanimoto_arr[max_idx])
        
        # Top-5 nearest neighbors
        top5_indices = np.argsort(tanimoto_arr)[-5:][::-1]
        top5_tanimoto = [float(tanimoto_arr[idx]) for idx in top5_indices]
        mean_top5_tanimoto = float(np.mean(top5_tanimoto))
        
        nearest_analog = {
            "compound_id": self.train_ids[max_idx],
            "smiles": self.train_smiles[max_idx],
            "measured_value": round(float(self.train_targets[max_idx]), 3),
            "tanimoto_similarity": round(max_tanimoto, 3)
        }

        # 2. Compute Leverage in PCA space
        leverage = 0.0
        leverage_warning = False
        if self.pca is not None and self.inv_cov is not None:
            z = self.pca.transform(feature_vector.reshape(1, -1))
            leverage_val = float(np.squeeze(z @ self.inv_cov @ z.T))
            leverage = leverage_val
            leverage_warning = leverage > self.leverage_threshold

        # 3. Applicability Domain Classification
        # High reliability: Max Tanimoto >= 0.55 and leverage <= threshold
        # Borderline: 0.35 <= Max Tanimoto < 0.55
        # Out of domain: Max Tanimoto < 0.35 or significant leverage extrapolation
        if max_tanimoto >= 0.55 and not leverage_warning:
            status = "IN_DOMAIN"
            confidence_level = "High"
            explanation = "Molecule is structurally well-represented in the ESOL training domain with high fingerprint similarity and safe leverage."
            color = "green"
        elif max_tanimoto >= 0.35 and not leverage_warning:
            status = "BORDERLINE"
            confidence_level = "Moderate"
            explanation = "Molecule is on the perimeter of the training domain. Prediction is usable but has increased epistemic uncertainty."
            color = "amber"
        else:
            status = "OUT_OF_DOMAIN"
            confidence_level = "Low (Extrapolation)"
            explanation = "Molecule is chemically distant from training analogs or exhibits high leverage. Predictions should be treated with caution."
            color = "red"

        # Reliability score in [0, 1]
        tanimoto_norm = np.clip(max_tanimoto, 0.0, 1.0)
        leverage_factor = np.exp(-max(0.0, leverage - self.leverage_threshold) * 2.0)
        reliability_score = float(np.clip(tanimoto_norm * 0.7 + leverage_factor * 0.3, 0.05, 0.99))

        return {
            "status": status,
            "confidence_level": confidence_level,
            "color": color,
            "explanation": explanation,
            "reliability_score": round(reliability_score, 3),
            "metrics": {
                "max_tanimoto_similarity": round(max_tanimoto, 3),
                "mean_top5_tanimoto": round(mean_top5_tanimoto, 3),
                "leverage": round(leverage, 4),
                "leverage_threshold": round(self.leverage_threshold, 4),
                "leverage_warning": leverage_warning
            },
            "nearest_analog": nearest_analog
        }
