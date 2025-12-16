import pytest

pytest.importorskip("openbabel")  # tries `import openbabel`

from qcinf._backends.openbabel import (
    _smiles_to_structure_ob,
    _structure_to_smiles_ob,
)


def test_smiles_to_structure():
    struct = _smiles_to_structure_ob("OCC")
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
    struct = _smiles_to_structure_ob("c1ccccc1")  # benzene
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
    struct = _smiles_to_structure_ob("[O-]CC")
    assert struct.charge == -1


def test_smiles_to_structure_multiplicity():
    # Check manual multiplicity
    struct = _smiles_to_structure_ob("[O-]CC", multiplicity=3)
    assert struct.charge == -1
    assert struct.multiplicity == 3


def test_smiles_charges_round_trip():
    """Test that SMILES with charges are handled correctly."""
    s = _smiles_to_structure_ob("CC[O-]")
    assert s.charge == -1
    # Using robust method
    assert _structure_to_smiles_ob(s) == "[O-]CC"


def test_structure_to_smiles_hydrogens(water):
    smiles = _structure_to_smiles_ob(water)
    assert smiles == "O"
    smiles = _structure_to_smiles_ob(water, hydrogens=True)
    assert smiles == "[H]O[H]"
