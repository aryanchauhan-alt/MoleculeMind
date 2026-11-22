"""
Molecular graph featurisation for the D-MPNN.

Builds the directed-edge representation that message passing operates on. The
conventions follow Yang et al. 2019 ("Analyzing Learned Molecular
Representations for Property Prediction", the Chemprop paper), which is the
architecture the project methodology specifies.

Graph representation
--------------------
Each chemical bond becomes **two directed edges**. For a directed edge
`i : a_i -> b_i` we store:

  * `edge_src[i]  = a_i`   the atom the message flows *from*
  * `edge_dest[i] = b_i`   the atom the message flows *into*
  * `edge_rev[i]`          the index of the opposite edge `b_i -> a_i`

`edge_rev` is what makes the network *directed*: when computing the message for
edge `a->b` we sum the messages arriving at `a` but must exclude the one that
came back along `b->a`, or information would immediately flow back where it came
from and the walk would degenerate into a 2-cycle.

Batching
--------
Molecules are concatenated into one flat graph with index offsets, and
`atom_to_mol` records which molecule each atom belongs to so readout can be
scattered per molecule. This avoids padding entirely.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem


# --- feature vocabularies ---------------------------------------------------
# Fixed vocabularies keep the feature dimension stable across datasets, which
# matters because a saved checkpoint encodes this width.

ATOMIC_NUMS = [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]  # B C N O F Si P S Cl As Se Br I
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot(value, choices: Sequence) -> List[float]:
    """One-hot with a trailing 'other' slot, so unseen values never crash."""
    encoding = [0.0] * (len(choices) + 1)
    try:
        encoding[choices.index(value)] = 1.0
    except ValueError:
        encoding[-1] = 1.0
    return encoding


def atom_features(atom: Chem.Atom) -> List[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)
        + _one_hot(atom.GetTotalDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
        + [
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
            # Scaled so every feature sits on a comparable magnitude.
            atom.GetMass() * 0.01,
        ]
    )


def bond_features(bond: Optional[Chem.Bond]) -> List[float]:
    if bond is None:
        # Reserved for a self-loop / absent bond; kept for shape stability.
        return [1.0] + [0.0] * (BOND_FEATURE_DIM - 1)
    return (
        [0.0]  # "no bond" flag
        + _one_hot(bond.GetBondType(), BOND_TYPES)
        + _one_hot(bond.GetStereo(), BOND_STEREO)
        + [
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ]
    )


ATOM_FEATURE_DIM = (
    len(ATOMIC_NUMS) + 1
    + len(DEGREES) + 1
    + len(FORMAL_CHARGES) + 1
    + len(NUM_HS) + 1
    + len(HYBRIDIZATIONS) + 1
    + len(CHIRAL_TAGS) + 1
    + 3
)
BOND_FEATURE_DIM = 1 + (len(BOND_TYPES) + 1) + (len(BOND_STEREO) + 1) + 2


@dataclass
class MolGraph:
    """A single molecule's directed-edge graph."""

    atom_feats: np.ndarray      # (n_atoms, ATOM_FEATURE_DIM)
    bond_feats: np.ndarray      # (n_edges, BOND_FEATURE_DIM)
    edge_src: np.ndarray        # (n_edges,)
    edge_dest: np.ndarray       # (n_edges,)
    edge_rev: np.ndarray        # (n_edges,)
    smiles: str = ""

    @property
    def n_atoms(self) -> int:
        return self.atom_feats.shape[0]

    @property
    def n_edges(self) -> int:
        return self.bond_feats.shape[0]


def mol_to_graph(mol: Chem.Mol, smiles: str = "") -> MolGraph:
    """Convert an RDKit molecule into its directed-edge graph."""
    atom_feats = np.array(
        [atom_features(a) for a in mol.GetAtoms()], dtype=np.float32
    )
    if atom_feats.size == 0:  # a molecule with no atoms
        atom_feats = np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)

    bond_feats: List[List[float]] = []
    edge_src: List[int] = []
    edge_dest: List[int] = []
    edge_rev: List[int] = []

    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)

        # Two directed edges per bond; each records the other as its reverse.
        idx_fwd = len(bond_feats)
        idx_bwd = idx_fwd + 1

        bond_feats.append(feats)
        edge_src.append(a)
        edge_dest.append(b)
        edge_rev.append(idx_bwd)

        bond_feats.append(list(feats))
        edge_src.append(b)
        edge_dest.append(a)
        edge_rev.append(idx_fwd)

    return MolGraph(
        atom_feats=atom_feats,
        bond_feats=np.array(bond_feats, dtype=np.float32).reshape(-1, BOND_FEATURE_DIM),
        edge_src=np.array(edge_src, dtype=np.int64),
        edge_dest=np.array(edge_dest, dtype=np.int64),
        edge_rev=np.array(edge_rev, dtype=np.int64),
        smiles=smiles,
    )


def smiles_to_graph(smiles: str) -> Optional[MolGraph]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_graph(mol, smiles=smiles)


@dataclass
class BatchedMolGraph:
    """Several molecules concatenated into one flat graph with index offsets."""

    atom_feats: torch.Tensor    # (total_atoms, ATOM_FEATURE_DIM)
    bond_feats: torch.Tensor    # (total_edges, BOND_FEATURE_DIM)
    edge_src: torch.Tensor      # (total_edges,)
    edge_dest: torch.Tensor     # (total_edges,)
    edge_rev: torch.Tensor      # (total_edges,)
    atom_to_mol: torch.Tensor   # (total_atoms,) molecule index per atom
    n_mols: int
    smiles: List[str]

    def to(self, device: torch.device) -> "BatchedMolGraph":
        return BatchedMolGraph(
            atom_feats=self.atom_feats.to(device),
            bond_feats=self.bond_feats.to(device),
            edge_src=self.edge_src.to(device),
            edge_dest=self.edge_dest.to(device),
            edge_rev=self.edge_rev.to(device),
            atom_to_mol=self.atom_to_mol.to(device),
            n_mols=self.n_mols,
            smiles=self.smiles,
        )


def batch_graphs(graphs: Sequence[MolGraph]) -> BatchedMolGraph:
    """
    Concatenate graphs, offsetting atom and edge indices so they stay disjoint.

    A molecule with no bonds (e.g. methane, or a lone ion) contributes atoms but
    no edges. That is handled naturally: it simply adds nothing to the edge
    arrays, and its atoms receive a zero message vector at readout.
    """
    atom_feats: List[np.ndarray] = []
    bond_feats: List[np.ndarray] = []
    edge_src: List[np.ndarray] = []
    edge_dest: List[np.ndarray] = []
    edge_rev: List[np.ndarray] = []
    atom_to_mol: List[np.ndarray] = []

    atom_offset = 0
    edge_offset = 0

    for mol_idx, g in enumerate(graphs):
        atom_feats.append(g.atom_feats)
        atom_to_mol.append(np.full(g.n_atoms, mol_idx, dtype=np.int64))

        if g.n_edges:
            bond_feats.append(g.bond_feats)
            edge_src.append(g.edge_src + atom_offset)
            edge_dest.append(g.edge_dest + atom_offset)
            edge_rev.append(g.edge_rev + edge_offset)

        atom_offset += g.n_atoms
        edge_offset += g.n_edges

    def _cat(arrays: List[np.ndarray], dim: int, dtype) -> np.ndarray:
        if not arrays:
            return np.zeros((0, dim), dtype=dtype) if dim else np.zeros((0,), dtype=dtype)
        return np.concatenate(arrays, axis=0)

    return BatchedMolGraph(
        atom_feats=torch.from_numpy(_cat(atom_feats, ATOM_FEATURE_DIM, np.float32)),
        bond_feats=torch.from_numpy(_cat(bond_feats, BOND_FEATURE_DIM, np.float32)),
        edge_src=torch.from_numpy(_cat(edge_src, 0, np.int64)),
        edge_dest=torch.from_numpy(_cat(edge_dest, 0, np.int64)),
        edge_rev=torch.from_numpy(_cat(edge_rev, 0, np.int64)),
        atom_to_mol=torch.from_numpy(_cat(atom_to_mol, 0, np.int64)),
        n_mols=len(graphs),
        smiles=[g.smiles for g in graphs],
    )
