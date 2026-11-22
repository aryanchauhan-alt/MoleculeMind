"""Molecular Analyzer tests: validation, descriptors, 2D/3D generation."""

import pytest
from rdkit import Chem

from backend.app.core.mol_analyzer import (
    validate_and_canonicalize,
    generate_2d_svg,
    generate_3d_sdf,
    compute_molecular_descriptors,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
BENZENE = "c1ccccc1"


@pytest.mark.parametrize(
    "smiles", [ASPIRIN, CAFFEINE, BENZENE, "CCO", "C[C@H](N)C(=O)O"]
)
def test_valid_smiles_accepted(smiles):
    ok, canonical, mol, err = validate_and_canonicalize(smiles)
    assert ok and mol is not None and err is None
    # Canonicalisation must be idempotent.
    assert Chem.MolToSmiles(Chem.MolFromSmiles(canonical)) == canonical


@pytest.mark.parametrize(
    "smiles", ["", "   ", "not_a_molecule", "C(((", "[Xx]", "c1ccccc"]
)
def test_invalid_smiles_rejected_gracefully(smiles):
    ok, canonical, mol, err = validate_and_canonicalize(smiles)
    assert ok is False and mol is None and canonical is None
    assert isinstance(err, str) and err


def test_stereochemistry_preserved():
    ok, canonical, _, _ = validate_and_canonicalize("C[C@H](N)C(=O)O")
    assert ok and "@" in canonical


def test_descriptor_values_for_aspirin():
    mol = Chem.MolFromSmiles(ASPIRIN)
    d = compute_molecular_descriptors(mol)
    assert d["formula"] == "C9H8O4"
    assert d["molecular_weight"] == pytest.approx(180.16, abs=0.1)
    assert d["hbd"] == 1
    assert d["hba"] == 3
    assert d["ring_count"] == 1
    assert d["aromatic_ring_count"] == 1
    assert d["heavy_atom_count"] == 13
    assert 0.0 <= d["qed"] <= 1.0
    assert d["lipinski"]["passed"] is True
    assert d["veber"]["passed"] is True


def test_descriptor_values_for_benzene():
    d = compute_molecular_descriptors(Chem.MolFromSmiles(BENZENE))
    assert d["formula"] == "C6H6"
    assert d["hbd"] == 0 and d["hba"] == 0
    assert d["tpsa"] == pytest.approx(0.0, abs=0.01)
    assert d["fraction_csp3"] == pytest.approx(0.0, abs=0.001)


def test_lipinski_flags_a_large_lipophilic_molecule():
    # A long-chain ester: high MW and high logP -> at least one violation.
    mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCOC(=O)CCCCCCCCCCCC")
    d = compute_molecular_descriptors(mol)
    assert d["molecular_weight"] > 500
    assert d["logp"] > 5.0
    assert d["lipinski"]["violations_count"] >= 2
    assert d["lipinski"]["passed"] is False


def test_veber_flags_many_rotatable_bonds():
    d = compute_molecular_descriptors(Chem.MolFromSmiles("C" * 25))
    assert d["rotatable_bonds"] > 10
    assert d["veber"]["passed"] is False


def test_2d_svg_is_wellformed():
    import xml.etree.ElementTree as ET

    svg = generate_2d_svg(Chem.MolFromSmiles(ASPIRIN), width=300, height=200)
    assert svg.strip().startswith("<?xml") or svg.strip().startswith("<svg")
    ET.fromstring(svg)  # raises if malformed


def test_3d_conformer_generation():
    sdf, err = generate_3d_sdf(Chem.MolFromSmiles(ASPIRIN))
    assert err is None and sdf is not None
    assert "V2000" in sdf or "V3000" in sdf
    mol = Chem.MolFromMolBlock(sdf, removeHs=False)
    assert mol is not None and mol.GetNumConformers() == 1
    # A real 3D embedding must not be planar at z = 0 everywhere.
    conf = mol.GetConformer()
    zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    assert max(zs) - min(zs) > 1e-6


def test_3d_generation_is_deterministic():
    a, _ = generate_3d_sdf(Chem.MolFromSmiles(CAFFEINE))
    b, _ = generate_3d_sdf(Chem.MolFromSmiles(CAFFEINE))
    assert a == b  # seeded ETKDG
