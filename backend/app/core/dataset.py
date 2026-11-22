"""
Dataset loader and chemical feature extraction module.
Manages Delaney ESOL, Lipophilicity, and ClinTox datasets.
Extracts Morgan Fingerprints (ECFP4) and physicochemical features using RDKit.

Scientific notes
----------------
* All datasets are deduplicated on **canonical SMILES** before any split is made.
  Duplicate structures straddling a train/test boundary are a direct source of
  optimistic bias; the raw Delaney file contains 11 such duplicates.
* Both random and Bemis-Murcko scaffold splits are provided. Scaffold splitting
  is the chemically harder evaluation and is reported alongside the random split
  rather than replacing it.
* A dataset fingerprint (SHA-256 over the sorted canonical SMILES + targets) is
  recorded so that reported metrics can be traced to an exact data version.
"""

import os
import gzip
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


# Number of physicochemical descriptors appended to the fingerprint.
N_PHYS_DESCRIPTORS = 8
FP_SIZE = 1024
FP_RADIUS = 2
FEATURE_DIM = FP_SIZE + N_PHYS_DESCRIPTORS


@dataclass
class MolecularDataset:
    """A deduplicated, feature-extracted molecular dataset."""

    name: str
    X: np.ndarray
    y: np.ndarray
    smiles: List[str]
    ids: List[str]
    target_name: str
    n_raw_rows: int = 0
    n_unparseable: int = 0
    n_duplicates_removed: int = 0
    data_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def provenance(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "target": self.target_name,
            "n_raw_rows": self.n_raw_rows,
            "n_unparseable": self.n_unparseable,
            "n_duplicates_removed": self.n_duplicates_removed,
            "n_used": len(self.y),
            "data_hash": self.data_hash,
        }


def compute_data_hash(smiles: List[str], targets: np.ndarray) -> str:
    """Stable content hash so reported metrics can be tied to an exact data version."""
    pairs = sorted(f"{s}|{t:.6f}" for s, t in zip(smiles, targets))
    return hashlib.sha256("\n".join(pairs).encode()).hexdigest()[:16]


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES; empty string for acyclic molecules."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_split(
    smiles: List[str],
    frac_train: float = 0.7,
    frac_cal: float = 0.15,
    frac_test: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic Bemis-Murcko scaffold split.

    Scaffold groups are sorted largest-first and assigned greedily to the partition
    that is furthest below its quota. A molecule's scaffold never appears in two
    partitions, which makes this a strictly harder generalisation test than a
    random split. Acyclic molecules (empty scaffold) are each treated as their own
    singleton group so they do not collapse into one artificial cluster.
    """
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles):
        scaf = murcko_scaffold(s)
        key = scaf if scaf else f"__acyclic_{i}__"
        groups.setdefault(key, []).append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

    n = len(smiles)
    quotas = [frac_train * n, frac_cal * n, frac_test * n]
    buckets: List[List[int]] = [[], [], []]

    for group in ordered:
        deficits = [quotas[k] - len(buckets[k]) for k in range(3)]
        target = int(np.argmax(deficits))
        buckets[target].extend(group)

    return buckets[0], buckets[1], buckets[2]


class DatasetManager:
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = data_dir
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_SIZE
        )
        self.esol_df: Optional[pd.DataFrame] = None
        self.lipophilicity_df: Optional[pd.DataFrame] = None
        self.clintox_df: Optional[pd.DataFrame] = None
        self._esol_cache: Optional[MolecularDataset] = None
        self._clintox_cache: Optional[MolecularDataset] = None

    # ------------------------------------------------------------------ raw IO

    def load_esol(self) -> pd.DataFrame:
        if self.esol_df is None:
            path = os.path.join(self.data_dir, "delaney-processed.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"ESOL dataset not found at {path}")
            self.esol_df = pd.read_csv(path)
        return self.esol_df

    def load_lipophilicity(self) -> pd.DataFrame:
        if self.lipophilicity_df is None:
            path = os.path.join(self.data_dir, "lipophilicity.csv")
            if os.path.exists(path):
                self.lipophilicity_df = pd.read_csv(path)
            else:
                self.lipophilicity_df = pd.DataFrame()
        return self.lipophilicity_df

    def load_clintox(self) -> pd.DataFrame:
        if self.clintox_df is None:
            path = os.path.join(self.data_dir, "clintox.csv.gz")
            if os.path.exists(path):
                with gzip.open(path, "rt") as f:
                    self.clintox_df = pd.read_csv(f)
            else:
                self.clintox_df = pd.DataFrame()
        return self.clintox_df

    # ---------------------------------------------------------------- features

    def extract_features_from_mol(self, mol: Chem.Mol) -> np.ndarray:
        """
        1024-bit Morgan fingerprint (ECFP4) + 8 physicochemical descriptors.
        Feature dimension = 1032.

        NOTE: these features are intentionally *unscaled*. Any consumer that is
        scale-sensitive (PCA, Gaussian processes, distance metrics) must scale
        them first - see `fit_feature_scaler`. Feeding the raw matrix to PCA makes
        molecular weight account for ~91% of the variance and destroys all
        structural signal.
        """
        fp = self.fp_generator.GetFingerprint(mol)
        fp_arr = np.array(fp, dtype=np.float32)

        phys_desc = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                Descriptors.HeavyAtomCount(mol),
            ],
            dtype=np.float32,
        )

        return np.concatenate([fp_arr, phys_desc])

    def extract_features_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.extract_features_from_mol(mol)

    # ---------------------------------------------------------------- datasets

    def _build_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        smiles_col: str,
        target_col: str,
        id_col: Optional[str],
        target_cast=float,
    ) -> MolecularDataset:
        """Parse, featurise and deduplicate on canonical SMILES (first occurrence wins)."""
        features: List[np.ndarray] = []
        targets: List[float] = []
        valid_smiles: List[str] = []
        compound_ids: List[str] = []

        seen: Dict[str, int] = {}
        n_unparseable = 0
        n_duplicates = 0

        for _, row in df.iterrows():
            raw_smiles = row[smiles_col]
            if not isinstance(raw_smiles, str):
                n_unparseable += 1
                continue
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                n_unparseable += 1
                continue
            if pd.isna(row[target_col]):
                n_unparseable += 1
                continue

            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in seen:
                n_duplicates += 1
                continue
            seen[canonical] = len(valid_smiles)

            features.append(self.extract_features_from_mol(mol))
            targets.append(target_cast(row[target_col]))
            valid_smiles.append(canonical)
            if id_col and id_col in row and pd.notna(row[id_col]):
                compound_ids.append(str(row[id_col]))
            else:
                compound_ids.append(f"{name.upper()}-{len(valid_smiles):04d}")

        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        return MolecularDataset(
            name=name,
            X=X,
            y=y,
            smiles=valid_smiles,
            ids=compound_ids,
            target_name=target_col,
            n_raw_rows=len(df),
            n_unparseable=n_unparseable,
            n_duplicates_removed=n_duplicates,
            data_hash=compute_data_hash(valid_smiles, y),
        )

    def get_esol_dataset(self) -> MolecularDataset:
        """Delaney ESOL aqueous solubility, deduplicated on canonical SMILES."""
        if self._esol_cache is None:
            self._esol_cache = self._build_dataset(
                self.load_esol(),
                name="Delaney ESOL",
                smiles_col="smiles",
                target_col="measured log solubility in mols per litre",
                id_col="Compound ID",
            )
        return self._esol_cache

    def get_clintox_dataset(self) -> Optional[MolecularDataset]:
        """ClinTox clinical-trial toxicity (CT_TOX), deduplicated on canonical SMILES."""
        if self._clintox_cache is None:
            df = self.load_clintox()
            if df.empty or "CT_TOX" not in df.columns:
                return None
            self._clintox_cache = self._build_dataset(
                df,
                name="ClinTox",
                smiles_col="smiles",
                target_col="CT_TOX",
                id_col=None,
                target_cast=lambda v: float(int(v)),
            )
        return self._clintox_cache

    # RDKit's generator is not picklable; rebuild it from its parameters.
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("fp_generator", None)
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_SIZE
        )

    # Backwards-compatible shim for existing callers.
    def prepare_esol_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        ds = self.get_esol_dataset()
        return ds.X, ds.y, ds.smiles, ds.ids
