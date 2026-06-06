import numpy as np
import pytest
from qcconst import constants
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata import ConformerSearchResults, Structure

from qcinf import align, filter_conformers_indices, rmsd


def test_rmsd_identical_structures():
    """Test that RMSD between identical structures is zero."""
    symbols = ["C", "H", "H", "H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.19052746, 1.19052746, 1.19052746],
            [-1.19052746, -1.19052746, 1.19052746],
            [-1.19052746, 1.19052746, -1.19052746],
            [1.19052746, -1.19052746, -1.19052746],
        ]
    )

    struct1 = Structure(symbols=symbols, geometry=geometry)
    struct2 = Structure(symbols=symbols, geometry=geometry)

    calculated_rmsd = rmsd(struct1, struct2, symmetry=False, backend="rdkit")
    assert np.isclose(calculated_rmsd, 0.0, atol=1e-6), (
        "RMSD should be zero for identical structures"
    )


def test_rmsd_rotated_structures_rdkit():
    """Test that RMSD between rotated structures is zero when aligned."""
    symbols = ["O", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9584, 0.0, 0.0],
                [-0.2396, 0.9270, 0.0],
            ]
        )
        * ANGSTROM_TO_BOHR
    )  # Water molecule in Bohr

    # Rotate struct2 by 180 degrees around the z-axis
    theta = np.pi  # 180 degrees
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_geometry = geometry @ rotation_matrix.T

    struct1 = Structure(symbols=symbols, geometry=geometry)
    struct2 = Structure(symbols=symbols, geometry=rotated_geometry)

    # Aligned by atom index (which match in this case)
    calculated_rmsd_no_align = rmsd(struct1, struct2, symmetry=False, backend="rdkit")
    assert np.isclose(calculated_rmsd_no_align, 0.0, atol=1e-6), (
        "RMSD should be zero after alignment"
    )

    # With alignment
    calculated_rmsd_align = rmsd(struct1, struct2, symmetry=True, backend="rdkit")
    assert np.isclose(calculated_rmsd_align, 0.0, atol=1e-6), (
        "RMSD should be zero after alignment"
    )


def test_align_identical_structures():
    """Test that aligning identical structures results in no change."""
    symbols = ["N", "H", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9377, 0.0, 0.0],
                [-0.3126, 0.8892, 0.0],
                [-0.3126, -0.4446, 0.7697],
            ]
        )
        * ANGSTROM_TO_BOHR
    )  # Ammonia molecule

    struct = Structure(symbols=symbols, geometry=geometry)
    refstruct = Structure(symbols=symbols, geometry=geometry)

    aligned_struct, rmsd_val = align(struct, refstruct, symmetry=False)

    assert np.allclose(aligned_struct.geometry, geometry), (
        "Aligned geometry should be the same as original"
    )


def test_align_rotated_structure():
    """Test aligning a rotated structure to the reference structure."""
    symbols = ["O", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9584, 0.0, 0.0],
                [-0.2396, 0.9270, 0.0],
            ]
        )
        * ANGSTROM_TO_BOHR
    )

    # Rotate struct by 90 degrees around the y-axis
    theta = np.pi / 2  # 90 degrees
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    rotated_geometry = geometry @ rotation_matrix.T

    struct = Structure(symbols=symbols, geometry=rotated_geometry)
    refstruct = Structure(symbols=symbols, geometry=geometry)

    aligned_struct, rmsd = align(struct, refstruct, symmetry=False)

    assert np.allclose(aligned_struct.geometry, geometry, atol=1e-6), (
        "Aligned geometry should match the reference"
    )


def test_align_with_atom_reordering_rdkit():
    """Test aligning structures with atom reordering allowed."""
    symbols1 = ["C", "H", "H", "H", "Cl"]
    geometry1 = (
        np.array(
            [
                [0.0, 0.0, 0.0],  # Carbon
                [0.0, 0.0, 1.09],  # Hydrogen
                [1.0267, 0.0, -0.3633],  # Hydrogen
                [-0.5133, -0.8892, -0.3633],  # Hydrogen
                [-0.5133, 0.8892, -0.3633],  # Chlorine
            ]
        )
        * ANGSTROM_TO_BOHR
    )

    symbols2 = ["C", "Cl", "H", "H", "H"]
    geometry2 = geometry1.copy()

    struct1 = Structure(symbols=symbols1, geometry=geometry1)
    struct2 = Structure(symbols=symbols2, geometry=geometry2)

    # Without atom reordering
    aligned_struct_no_reorder, rmsd_no_reorder = align(
        struct1, struct2, symmetry=False, backend="rdkit"
    )
    assert aligned_struct_no_reorder.symbols == struct1.symbols, (
        "Symbols should not be reordered"
    )

    assert rmsd_no_reorder > 0.1, "RMSD should be high without atom reordering"

    # With atom reordering
    aligned_struct_reorder, rmsd_reorder = align(
        struct1, struct2, symmetry=True, backend="rdkit"
    )
    rmsd_reorder = rmsd(
        aligned_struct_reorder, struct2, symmetry=False, backend="rdkit"
    )
    assert aligned_struct_reorder.symbols == struct2.symbols, (
        "Symbols should be reordered"
    )

    assert np.isclose(rmsd_reorder, 0.0, atol=1e-2), (
        "RMSD should be zero with atom reordering"
    )


def test_align_raises_value_error_reorder_different_atom_count_if_symmetry():
    """Test that an error is raised when structures have different atom counts."""
    symbols1 = ["C", "H", "H", "H", "H"]
    geometry1 = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ]
        )
        * ANGSTROM_TO_BOHR
    )

    symbols2 = ["C", "H", "H", "H"]
    geometry2 = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
            ]
        )
        * ANGSTROM_TO_BOHR
    )

    struct1 = Structure(symbols=symbols1, geometry=geometry1)
    struct2 = Structure(symbols=symbols2, geometry=geometry2)
    with pytest.raises(ValueError):
        align(struct2, struct1, symmetry=True)


def test_rmsd_with_numthreads_rdkit():
    """Test RMSD calculation with multiple threads."""
    symbols = ["O", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.7586, 0.0, 0.5043],
                [-0.7586, 0.0, 0.5043],
            ]
        )
        * ANGSTROM_TO_BOHR
    )  # Water molecule

    struct1 = Structure(symbols=symbols, geometry=geometry)
    struct2 = Structure(symbols=symbols, geometry=geometry)

    rmsd_single_thread = rmsd(
        struct1, struct2, symmetry=True, numthreads=1, backend="rdkit"
    )
    rmsd_multi_thread = rmsd(
        struct1, struct2, symmetry=True, numthreads=4, backend="rdkit"
    )

    assert np.isclose(rmsd_single_thread, 0.0, atol=1e-6), (
        "RMSD should be zero with single thread"
    )
    assert np.isclose(rmsd_multi_thread, 0.0, atol=1e-6), (
        "RMSD should be zero with multiple threads"
    )


def test_align_incorrect_atom_mapping():
    """Test that an error is raised when atom mapping fails."""
    symbols1 = ["C", "H", "H", "H", "H"]
    symbols2 = ["O", "H", "H", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],  # Central atom
                [0.63, 0.63, 0.63],  # Hydrogen
                [-0.63, -0.63, 0.63],  # Hydrogen
                [-0.63, 0.63, -0.63],  # Hydrogen
                [0.63, -0.63, -0.63],  # Hydrogen
            ]
        )
        * ANGSTROM_TO_BOHR
    )

    struct1 = Structure(symbols=symbols1, geometry=geometry)
    struct2 = Structure(symbols=symbols2, geometry=geometry)

    with pytest.raises(RuntimeError):  # Raised by RDKit
        align(struct1, struct2, symmetry=False)


def test_align_large_molecule():
    """Test aligning larger molecules."""
    symbols = ["C"] * 10
    geometry = np.random.rand(10, 3) * ANGSTROM_TO_BOHR

    struct1 = Structure(symbols=symbols, geometry=geometry)
    struct2 = Structure(symbols=symbols, geometry=geometry + 0.1)  # Slightly shifted

    aligned_struct, calculated_rmsd = align(struct1, struct2, symmetry=False)
    assert calculated_rmsd < 0.2, "RMSD should be low for slightly shifted structures"


# Note: This test is slow since it uses RDKit's Hueckel method to determine connectivity
# and then uses the Hungarian algorithm to consider symmetries. If needed perhaps we can
# modify this test to run more quickly by changing the connectivity determination method
# and/or not considering symmetries.
def test_conformers_filtered(test_data_dir):
    # Catalyst/Na+ conformer search
    csr = ConformerSearchResults.open(test_data_dir / "conf_search.json")
    keep_indices = filter_conformers_indices(
        csr.conformers, backend="rdkit", threshold=0.47 * constants.ANGSTROM_TO_BOHR
    )
    assert len(keep_indices) == 6
    assert keep_indices == [0, 5, 7, 8, 9, 10]
    # for i, conf in enumerate(keep_indices.conformers):
    #     assert conf == csr.conformers[selected[i]]
    #     assert (
    #         keep_indices.conformer_energies[i]
    #         == csr.conformer_energies_relative[selected[i]]
    #     )
