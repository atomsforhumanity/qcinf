import warnings

import numpy as np
from qcdata import Structure

from qcinf.exceptions import NumbaUnavailableError

from .kernels._qcp_numba_kernel import _qcp_rotation_rmsd_numba


def kabsch(
    P: np.ndarray, Q: np.ndarray, proper_only: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the optimal rotation matrix that aligns P onto Q using the Kabsch algorithm.

    To rotate P onto Q, use: P_rotated = (P - centroid_P) @ R.T + cQ

    Args:
        P (np.ndarray): An N x 3 array of coordinates (the structure to rotate).
        Q (np.ndarray): An N x 3 array of coordinates (the reference structure).
        proper_only (bool): If True, ensure the rotation is a proper rotation (det(R) = 1).

    Returns:
        R (np.ndarray): The optimal 3x3 rotation matrix.
        cP (np.ndarray): The centroid of P.
        cQ (np.ndarray): The centroid of Q.
    """
    # Ensure the arrays have the same shape and at least 3 points
    if not P.shape == Q.shape and P.shape[0] >= 3:
        raise ValueError(
            "Input coordinate arrays must have the same shape and at least 3 points."
        )
    # Compute centroids
    cP = np.mean(P, axis=0)
    cQ = np.mean(Q, axis=0)

    # Center the coordinates
    P_centered = P - cP
    Q_centered = Q - cQ

    # Compute covariance matrix C = P^T Q
    C = P_centered.T @ Q_centered
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(C)
    # Compute rotation matrix
    R = Vt.T @ U.T

    # Correct for reflection (ensure a proper rotation with det(R)=1)
    if proper_only and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R, cP, cQ


def qcp_rotation_and_rmsd(
    P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute optimal rotation (3x3 matrix), centroids, and RMSD between P and Q
    using the direct QCP method (Theobald + Liu) with numba acceleration.

    To rotate P onto Q, use: P_rotated = (P - centroid_P) @ R.T + cQ

    Parameters:
        P: An N x 3 array of coordinates.
        Q: An N x 3 array of coordinates.
    Returns:
        R: The optimal 3x3 rotation matrix.
        cP: The centroid of P.
        cQ: The centroid of Q.
        rmsd (float): The minimal RMSD between P and Q after optimal alignment.
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape (N, 3)")
    if P.shape[1] != 3:
        raise ValueError("Coordinates must be N x 3")

    try:
        return _qcp_rotation_rmsd_numba(P, Q)
    except (
        NumbaUnavailableError
    ) as e:  # Gracefully fall back to Kabsch if numba is unavailable
        # Fallback path: Kabsch alignment + RMSD
        warnings.warn(
            "The numba-accelerated QCP kernel is unavailable because numba "
            "could not be imported. Install qcinf with the 'fast' extra, e.g.\n"
            "    python -m pip install 'qcinf[fast]'\n"
            "and ensure numba imports correctly on your system.\n"
            "Falling back to Kabsch for qcp_rotation_and_rmsd",
            RuntimeWarning,
            stacklevel=2,
        )
        R, cP, cQ = kabsch(P, Q, proper_only=True)
        diff = (P - cP) @ R.T + cQ - Q
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return R, cP, cQ, rmsd


def qcp_rotation(
    P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    QCP-based optimal rotation and centroids.

    Thin convenience wrapper around :func:`qcp_rotation_and_rmsd` that returns
    only (R, cP, cQ) and discards the RMSD.
    """
    R, cP, cQ, _ = qcp_rotation_and_rmsd(P, Q)
    return R, cP, cQ


def qcp_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """
    QCP-based minimal RMSD.

    Thin convenience wrapper around :func:`qcp_rotation_and_rmsd` that returns
    only the RMSD.
    """
    _, _, _, rmsd = qcp_rotation_and_rmsd(A, B)
    return rmsd


def _compute_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    *,
    align: bool = True,
    proper_only: bool = True,
    backend: str = "kabsch",
) -> float:
    """
    Compute the RMSD between two sets of coordinates, optionally with alignment.

    Args:
        coords1: An N x 3 array of coordinates.
        coords2: An N x 3 array of coordinates.
        align: If True, optimally align coords1 to coords2 before computing RMSD.
        proper_only: If True, ensure the Kabsch rotation is a proper rotation (avoid
            inversions; det(R) = 1). Only used if align is True. Only relevant for
            the 'kabsch' backend.
        backend: The backend to use for alignment. Options are
            'kabsch' (default) or 'qcp' (numba accelerated). Backend only matters if
            align is True. Both will compute the same answer.

    Returns:
        float: The RMSD value.
    """
    # Ensure the arrays have the same shape
    assert coords1.shape == coords2.shape, "Coordinate arrays must be the same shape."
    supported_backends = ["qcp", "kabsch"]
    assert backend in supported_backends, f"Unsupported backend '{backend}'. Supported: {supported_backends}"  # fmt: skip

    if align:
        if backend == "kabsch":
            # Align coords1 to coords2 using the Kabsch algorithm
            R, cP, cQ = kabsch(coords1, coords2, proper_only=proper_only)
            # Rotate coords1 and translate to the centroid of coords2
            coords1 = (coords1 - cP) @ R.T + cQ
        else:  # backend == "qcp"
            # Use QCP to get optimal rotation rmsd
            _, _, _, rmsd = qcp_rotation_and_rmsd(coords1, coords2)
            return rmsd

    # Compute the difference between the two arrays
    diff = coords1 - coords2
    # Sum squared differences for each atom (row) and average over all atoms
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for a rotation about a given axis by angle in degrees.

    Parameters:
        axis (str): 'x', 'y', or 'z' specifying the rotation axis.
        angle_deg (float): Rotation angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.

    Example:
        >>> rotation_matrix('z', 90)
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    if axis.lower() == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis.lower() == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis.lower() == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")


def rotate_structure(struct: Structure, axis: str, angle_deg: float) -> Structure:
    """
    Return a new Structure with its coordinates rotated by angle_deg about the given axis.

    Parameters:
        struct (Structure): Input structure with a .geometry attribute (an N x 3 numpy array).
        axis (str): Axis to rotate about ('x', 'y', or 'z').
        angle_deg (float): Rotation angle in degrees.

    Returns:
        Structure: New structure with rotated coordinates.
    """
    R = rotation_matrix(axis, angle_deg)
    # Dump the structure to a dictionary and modify the geometry.
    new_struct = struct.model_dump()
    # Apply rotation: for each coordinate, multiply with the rotation matrix.
    # We use R.T because our coordinates are row vectors.
    new_struct["geometry"] = struct.geometry @ R.T
    return Structure.model_validate(new_struct)
