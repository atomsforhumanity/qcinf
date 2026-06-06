import pytest

from qcinf.algorithms.geometry_kernels import rotate_structure

pytest.importorskip("rdkit")  # tries `import rdkit` skips tests if not installed


from qcinf._backends.rdkit import (
    _rmsd_rdkit,
    _smiles_to_structure_rdkit,
    _structure_to_smiles_rdkit,
)


def test_rmsd_identity(water):
    assert _rmsd_rdkit(water, water) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_alignment_happens_before_rmsd(water):
    water2 = rotate_structure(water, "z", 90.0)
    assert _rmsd_rdkit(water, water2, symmetry=False) == pytest.approx(0.0, abs=1e-6)


def test_smiles_to_structure():
    struct = _smiles_to_structure_rdkit("OCC", force_field="UFF")
    assert struct.symbols == ["O", "C", "C", "H", "H", "H", "H", "H", "H"]
    assert struct.charge == 0
    assert struct.multiplicity == 1
    assert struct.identifiers.smiles == "OCC"
    assert struct.identifiers.canonical_smiles == "CCO"
    assert struct.connectivity == [
        (0, 1, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 4, 1.0),
        (1, 5, 1.0),
        (2, 6, 1.0),
        (2, 7, 1.0),
        (2, 8, 1.0),
    ]


def test_smiles_to_structure_aromaticity_connectivity():
    struct = _smiles_to_structure_rdkit("c1ccccc1")  # benzene
    assert struct.connectivity == [
        (0, 1, 1.5),
        (0, 5, 1.5),
        (0, 6, 1.0),
        (1, 2, 1.5),
        (1, 7, 1.0),
        (2, 3, 1.5),
        (2, 8, 1.0),
        (3, 4, 1.5),
        (3, 9, 1.0),
        (4, 5, 1.5),
        (4, 10, 1.0),
        (5, 11, 1.0),
    ]


def test_smiles_to_structure_charges():
    # Check Charge
    struct = _smiles_to_structure_rdkit("[O-]CC")
    assert struct.charge == -1


def test_smiles_to_structure_multiplicity():
    # Check manual multiplicity
    struct = _smiles_to_structure_rdkit("[O-]CC", multiplicity=3)
    assert struct.charge == -1
    assert struct.multiplicity == 3


def test_smiles_charges_round_trip():
    """Test that SMILES with charges are handled correctly."""
    s = _smiles_to_structure_rdkit("CC[O-]")
    assert s.charge == -1
    # Using robust method
    assert _structure_to_smiles_rdkit(s) == "CC[O-]"


def test_structure_to_smiles_hydrogens(water):
    smiles = _structure_to_smiles_rdkit(water)
    assert smiles == "O"
    smiles = _structure_to_smiles_rdkit(water, hydrogens=True)
    assert smiles == "[H]O[H]"
