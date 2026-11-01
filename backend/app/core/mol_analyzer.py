"""
Molecular Analyzer module using RDKit.
Handles SMILES validation, 2D SVG rendering, 3D conformer generation,
molecular descriptors, and Lipinski's Rule of 5 evaluation.
"""

from typing import Dict, Any, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def validate_and_canonicalize(smiles: str) -> Tuple[bool, Optional[str], Optional[Chem.Mol], Optional[str]]:
    """
    Validate SMILES string and return canonical SMILES, RDKit Mol object, and error message if any.
    """
    if not smiles or not smiles.strip():
        return False, None, None, "SMILES string is empty."
    
    clean_smiles = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return False, None, None, f"Invalid SMILES structure: unable to parse '{clean_smiles}'."
        
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical_smiles, mol, None
    except Exception as e:
        return False, None, None, f"Parsing error: {str(e)}"


def generate_2d_svg(mol: Chem.Mol, width: int = 400, height: int = 300, dark_mode: bool = True) -> str:
    """
    Generate clean, publication-quality 2D SVG rendering of the molecule.
    """
    mol_copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_copy)
    
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    
    if dark_mode:
        opts.setBackgroundColour((0.08, 0.11, 0.18, 0.0))  # Slate transparent
    
    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def generate_3d_sdf(mol: Chem.Mol, max_attempts: int = 50) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate 3D coordinates using ETKDG and optimize geometry with MMFF94 force field.
    Returns SDF/MOL block string suitable for 3Dmol.js rendering.
    """
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        
        embed_status = AllChem.EmbedMolecule(mol_3d, params)
        if embed_status != 0:
            # Fallback embedding
            embed_status = AllChem.EmbedMolecule(mol_3d, maxAttempts=max_attempts, randomSeed=42)
        
        if embed_status == 0:
            # Optimize conformer using MMFF94 or UFF
            try:
                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
                except Exception:
                    pass
            
            mol_block = Chem.MolToMolBlock(mol_3d)
            return mol_block, None
        else:
            # Could not embed 3D coordinates
            return None, "Failed to generate 3D conformer coordinates (conformation too strained or complex)."
    except Exception as e:
        return None, f"3D generation error: {str(e)}"


def compute_molecular_descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Compute comprehensive physicochemical descriptors and drug-likeness metrics.
    """
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    qed_val = float(QED.qed(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot_bonds = int(Lipinski.NumRotatableBonds(mol))
    rings = int(Lipinski.RingCount(mol))
    aromatic_rings = int(Lipinski.NumAromaticRings(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    formal_charge = int(Chem.GetFormalCharge(mol))
    formula = rdMolDescriptors.CalcMolFormula(mol)
    
    # Lipinski's Rule of 5 evaluation
    # 1. MW <= 500
    # 2. MolLogP <= 5.0
    # 3. HBD <= 5
    # 4. HBA <= 10
    violations = []
    if mw > 500:
        violations.append(f"MW {mw:.1f} Da > 500 Da")
    if logp > 5.0:
        violations.append(f"LogP {logp:.2f} > 5.0")
    if hbd > 5:
        violations.append(f"H-Bond Donors {hbd} > 5")
    if hba > 10:
        violations.append(f"H-Bond Acceptors {hba} > 10")
    
    lipinski_pass = len(violations) <= 1  # Standard drug-likeness allows at most 1 violation
    
    # Veber Rule (oral bioavailability):
    # Rotatable bonds <= 10 and TPSA <= 140 A^2
    veber_pass = (rot_bonds <= 10) and (tpsa <= 140.0)
    
    return {
        "formula": formula,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "qed": round(qed_val, 3),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot_bonds,
        "ring_count": rings,
        "aromatic_ring_count": aromatic_rings,
        "heavy_atom_count": heavy_atoms,
        "fraction_csp3": round(fraction_csp3, 3),
        "formal_charge": formal_charge,
        "lipinski": {
            "passed": lipinski_pass,
            "violations_count": len(violations),
            "violations": violations
        },
        "veber": {
            "passed": veber_pass,
            "tpsa_ok": tpsa <= 140.0,
            "rot_bonds_ok": rot_bonds <= 10
        }
    }
