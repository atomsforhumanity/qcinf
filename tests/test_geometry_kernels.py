import numpy as np
import pytest
from qcdata import Structure  # if you need it explicitly

from qcinf import smiles_to_structure
from qcinf.algorithms.geometry_kernels import (
    _compute_rmsd,
    kabsch,
    qcp_rmsd,
    qcp_rotation,
    qcp_rotation_and_rmsd,
    rotate_structure,
    rotation_matrix,
)


def test_rotation_matrix_z_90():
    R = rotation_matrix("z", 90.0)
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(R, expected, atol=1e-12)


def test_rotate_structure_applies_rotation():
    s = smiles_to_structure("O")
    s_rot = rotate_structure(s, "z", 90.0)

    R = rotation_matrix("z", 90.0)
    manual = s.geometry @ R.T

    assert s_rot.geometry.shape == s.geometry.shape
    assert np.allclose(s_rot.geometry, manual, atol=1e-12)


@pytest.mark.parametrize("axis, angle", [("x", 33.0), ("y", 57.0), ("z", 123.0)])
def test_kabsch_recovers_known_rotation(axis, angle):
    s = smiles_to_structure("CCO")  # small but nontrivial
    s_rot = rotate_structure(s, axis, angle)

    R_k, cP_k, cQ_k = kabsch(s_rot.geometry, s.geometry, allow_inversions=True)
    aligned = (s_rot.geometry - cP_k) @ R_k.T + cQ_k

    # Kabsch should align rotated structure back onto the original
    assert _compute_rmsd(aligned, s.geometry, align=False) < 1e-7


# --------- QCP vs Kabsch: RMSD and transformation equivalence ---------


@pytest.mark.parametrize(
    "smiles",
    [
        "O",  # trivial
        "CCO",  # small molecule
        "c1ccccc1",  # benzene
        # Taxol
        "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)"
        "(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C",
    ],
)
def test_qcp_rmsd_matches_kabsch(smiles):
    s1 = smiles_to_structure(smiles)
    # Generate a rotated copy so the alignment problem is nontrivial
    s2 = rotate_structure(s1, "z", 37.5)

    P = s2.geometry
    Q = s1.geometry

    # Reference: Kabsch RMSD
    R_k, cP_k, cQ_k = kabsch(P, Q, allow_inversions=False)
    P_aligned_k = (P - cP_k) @ R_k.T + cQ_k
    rmsd_k = _compute_rmsd(P_aligned_k, Q, align=False)

    # QCP: rotation + RMSD
    R_q, cP_q, cQ_q, rmsd_q = qcp_rotation_and_rmsd(P, Q)
    # RMSD values should agree closely; but extreme precision is not expected
    assert abs(rmsd_q - rmsd_k) < 1e-5

    # Apply both transforms to P and ensure the *transformed coordinates*
    # are essentially identical (up to numerical noise)
    P_aligned_q = (P - cP_q) @ R_q.T + cQ_q
    rmsd_between_transformed = _compute_rmsd(P_aligned_k, P_aligned_q, align=False)
    assert rmsd_between_transformed < 1e-6


def test_qcp_rotation_wrapper_consistency():
    s1 = smiles_to_structure("CCO")
    s2 = rotate_structure(s1, "x", 12.3)

    P = s2.geometry
    Q = s1.geometry

    R_full, cP_full, cQ_full, rmsd_full = qcp_rotation_and_rmsd(P, Q)
    R_only, cP_only, cQ_only = qcp_rotation(P, Q)
    rmsd_only = qcp_rmsd(P, Q)

    assert np.allclose(R_full, R_only, atol=1e-12)
    assert np.allclose(cP_full, cP_only, atol=1e-12)
    assert np.allclose(cQ_full, cQ_only, atol=1e-12)
    assert abs(rmsd_full - rmsd_only) < 1e-12


def test_compute_rmsd_align_vs_no_align():
    """Test _compute_rmsd tests (backends + align flag)."""
    s = smiles_to_structure("CCO")
    # Translate the structure significantly
    translated_d = s.model_dump()
    translated_d["geometry"] = s.geometry + np.array([10.0, -5.0, 3.0])
    translated = Structure(**translated_d)

    A = s.geometry
    B = translated.geometry

    # Without alignment, RMSD should be large
    rmsd_no_align = _compute_rmsd(A, B, align=False)
    assert rmsd_no_align > 1.0

    # With Kabsch alignment, RMSD should be ~0 for pure translation
    rmsd_kabsch = _compute_rmsd(A, B, align=True, backend="kabsch")
    assert rmsd_kabsch < 1e-10

    # With QCP backend, we should get the same
    rmsd_qcp = _compute_rmsd(A, B, align=True, backend="qcp")
    assert rmsd_qcp < 1e-6  # QCP is less precise
    assert abs(rmsd_qcp - rmsd_kabsch) < 1e-6


def test_rotate_structure():
    """Test that the rotate_structure function correctly rotates a structure."""

    water = Structure(
        symbols=["O", "H", "H"],
        geometry=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # fmt: skip
    )
    # Rotate by 90 degrees around the z-axis
    rotated_struct = rotate_structure(water, "z", 90)

    # Check if the rotation is correct
    expected_geometry = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(rotated_struct.geometry, expected_geometry), (
        "Rotated geometry does not match expected geometry"
    )
