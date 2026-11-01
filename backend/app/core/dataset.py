"""
Dataset loader and chemical feature extraction module.
Manages Delaney ESOL, Lipophilicity, and ClinTox datasets.
Extracts Morgan Fingerprints (ECFP4) and physicochemical features using RDKit.
"""

import os
import gzip
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


class DatasetManager:
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = data_dir
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        self.esol_df: Optional[pd.DataFrame] = None
        self.lipophilicity_df: Optional[pd.DataFrame] = None
        self.clintox_df: Optional[pd.DataFrame] = None

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
                with gzip.open(path, 'rt') as f:
                    self.clintox_df = pd.read_csv(f)
            else:
                self.clintox_df = pd.DataFrame()
        return self.clintox_df

    def extract_features_from_mol(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract 1024-bit Morgan Fingerprint + 8 key physicochemical descriptors.
        Total feature dimension = 1032.
        """
        fp = self.fp_generator.GetFingerprint(mol)
        fp_arr = np.array(fp, dtype=np.float32)
        
        phys_desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol)
        ], dtype=np.float32)
        
        return np.concatenate([fp_arr, phys_desc])

    def extract_features_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.extract_features_from_mol(mol)

    def prepare_esol_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Returns feature matrix X, targets y (measured log solubility logS),
        valid SMILES list, and Compound ID list.
        """
        df = self.load_esol()
        features = []
        targets = []
        valid_smiles = []
        compound_ids = []

        target_col = "measured log solubility in mols per litre"
        smiles_col = "smiles"
        id_col = "Compound ID"

        for _, row in df.iterrows():
            s = row[smiles_col]
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                feat = self.extract_features_from_mol(mol)
                features.append(feat)
                targets.append(float(row[target_col]))
                valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                compound_ids.append(str(row[id_col]) if id_col in row and pd.notna(row[id_col]) else "CMPD")

        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        return X, y, valid_smiles, compound_ids
