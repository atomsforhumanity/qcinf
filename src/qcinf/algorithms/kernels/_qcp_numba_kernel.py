import math

import numpy as np

from qcinf.exceptions import NumbaUnavailableError

try:
    from numba import njit
except Exception as exc:
    _NUMBA_IMPORT_ERROR = exc

    def _qcp_rotation_rmsd_numba(
        P: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Placeholder stub used when numba is not available.

        Called if the fast QCP kernel is requested but numba cannot be imported.
        Mirrors the signature of the numba kernel so callers type-check alike.

        Args:
            P: Unused; N x 3 coordinates of the structure to rotate.
            Q: Unused; N x 3 coordinates of the reference structure.

        Returns:
            Never returns.

        Raises:
            NumbaUnavailableError: Always, chained from the original import error.
        """
        raise NumbaUnavailableError(
            "The numba-accelerated QCP kernel is unavailable because numba "
            "could not be imported. Install qcinf with the 'fast' extra, e.g.\n"
            "    python -m pip install 'qcinf[fast]'\n"
            "and ensure numba imports correctly on your system."
        ) from _NUMBA_IMPORT_ERROR

# ---------- Small linear-algebra helpers (all JIT’d) ----------
else:

    @njit(cache=True)
    def _det3x3(M: np.ndarray) -> float:
        """Determinant of a 3x3 matrix M."""
        return (
            M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
            - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
        )

    @njit(cache=True)
    def _det4x4(K: np.ndarray) -> float:
        """Determinant of a 4x4 matrix K via expansion along the first row."""
        det = 0.0
        for j in range(4):
            # Build 3x3 minor excluding row 0 and column j
            M = np.empty((3, 3), dtype=np.float64)
            mi = 0
            for r in range(1, 4):
                mj = 0
                for c in range(4):
                    if c == j:
                        continue
                    M[mi, mj] = K[r, c]
                    mj += 1
                mi += 1
            cofactor = ((-1) ** j) * _det3x3(M)
            det += K[0, j] * cofactor
        return det

    @njit(cache=True)
    def _adjoint4x4(A: np.ndarray) -> np.ndarray:
        """Compute adj(A) for a 4x4 matrix A via cofactors.

        adj(A) = C^T, where C_ij = (-1)^(i+j) * det(minor_ij).
        """
        adj = np.empty((4, 4), dtype=np.float64)

        M = np.empty((3, 3), dtype=np.float64)  # temporary for minors

        for i in range(4):
            for j in range(4):
                # Build minor M_ij by removing row i and col j
                mi = 0
                for r in range(4):
                    if r == i:
                        continue
                    mj = 0
                    for c in range(4):
                        if c == j:
                            continue
                        M[mi, mj] = A[r, c]
                        mj += 1
                    mi += 1
                cofactor = ((-1) ** (i + j)) * _det3x3(M)
                # adj(A) = C^T, so cofactor at (i,j) -> adj(j,i)
                adj[j, i] = cofactor

        return adj

    @njit(cache=True)
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """Convert unit quaternion q = [q0, qx, qy, qz] to a 3x3 rotation matrix.

        This version returns the transpose of the canonical column-vector
        rotation matrix used before, so that in the public API you can always apply:

            P_aligned = (P - cP) @ R.T + cQ

        to map P -> Q, consistent with the Kabsch implementation.
        """
        q0 = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        R = np.empty((3, 3), dtype=np.float64)

        R[0, 0] = 1.0 - 2.0 * (qy2 + qz2)
        R[0, 1] = 2.0 * (qx * qy + q0 * qz)
        R[0, 2] = 2.0 * (qx * qz - q0 * qy)

        R[1, 0] = 2.0 * (qx * qy - q0 * qz)
        R[1, 1] = 1.0 - 2.0 * (qx2 + qz2)
        R[1, 2] = 2.0 * (qy * qz + q0 * qx)

        R[2, 0] = 2.0 * (qx * qz + q0 * qy)
        R[2, 1] = 2.0 * (qy * qz - q0 * qx)
        R[2, 2] = 1.0 - 2.0 * (qx2 + qy2)

        return R

    # ---------- Main kernel: rotation + RMSD (Theobald + Liu) ----------

    @njit(cache=True)
    def _qcp_rotation_rmsd_numba(
        P: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Numba-optimized QCP kernel: returns optimal rotation matrix and RMSD.

        Implements:
        - Theobald's quartic / Newton–Raphson to get λ_max
        - Liu's adjoint(K - λI) to get quaternion (eigenvector)
        - Quaternion → rotation
        - RMSD from λ_max
        """
        N = P.shape[0]

        # --- 1. Centroids ---
        cP = np.zeros(3, dtype=np.float64)
        cQ = np.zeros(3, dtype=np.float64)
        for i in range(N):
            cP[0] += P[i, 0]
            cP[1] += P[i, 1]
            cP[2] += P[i, 2]
            cQ[0] += Q[i, 0]
            cQ[1] += Q[i, 1]
            cQ[2] += Q[i, 2]
        invN = 1.0 / N
        cP[0] *= invN
        cP[1] *= invN
        cP[2] *= invN
        cQ[0] *= invN
        cQ[1] *= invN
        cQ[2] *= invN

        # --- 2. GA, GB, and 3x3 covariance M = B_c^T A_c ---
        GA = 0.0
        GB = 0.0
        M = np.zeros((3, 3), dtype=np.float64)

        for i in range(N):
            ax = P[i, 0] - cP[0]
            ay = P[i, 1] - cP[1]
            az = P[i, 2] - cP[2]
            bx = Q[i, 0] - cQ[0]
            by = Q[i, 1] - cQ[1]
            bz = Q[i, 2] - cQ[2]

            GA += ax * ax + ay * ay + az * az
            GB += bx * bx + by * by + bz * bz

            # M = B_c^T A_c (Theobald convention; transpose choice doesn't matter for λ_max)
            M[0, 0] += bx * ax
            M[0, 1] += bx * ay
            M[0, 2] += bx * az

            M[1, 0] += by * ax
            M[1, 1] += by * ay
            M[1, 2] += by * az

            M[2, 0] += bz * ax
            M[2, 1] += bz * ay
            M[2, 2] += bz * az

        # --- 3. Build K from M (Horn/Theobald quaternion matrix) ---
        Sxx = M[0, 0]
        Sxy = M[0, 1]
        Sxz = M[0, 2]
        Syx = M[1, 0]
        Syy = M[1, 1]
        Syz = M[1, 2]
        Szx = M[2, 0]
        Szy = M[2, 1]
        Szz = M[2, 2]

        trace = Sxx + Syy + Szz

        K = np.empty((4, 4), dtype=np.float64)
        K[0, 0] = trace
        K[0, 1] = Syz - Szy
        K[0, 2] = Szx - Sxz
        K[0, 3] = Sxy - Syx

        K[1, 0] = Syz - Szy
        K[1, 1] = Sxx - Syy - Szz
        K[1, 2] = Sxy + Syx
        K[1, 3] = Szx + Sxz

        K[2, 0] = Szx - Sxz
        K[2, 1] = Sxy + Syx
        K[2, 2] = -Sxx + Syy - Szz
        K[2, 3] = Syz + Szy

        K[3, 0] = Sxy - Syx
        K[3, 1] = Szx + Sxz
        K[3, 2] = Syz + Szy
        K[3, 3] = -Sxx - Syy + Szz

        # --- 4. Quartic coefficients for characteristic polynomial of K ---
        # P(λ) = λ^4 + C2 λ^2 + C1 λ + C0 = 0 (C4=1, C3=0)
        # C2 = -2 * tr(M^T M) = -2 * sum(M_ij^2)
        sumsq = 0.0
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                sumsq += v * v
        C2 = -2.0 * sumsq

        # C1 = -8 * det(M)
        detM = _det3x3(M)
        C1 = -8.0 * detM

        # C0 = det(K)
        C0 = _det4x4(K)

        # --- 5. Newton–Raphson for largest root λ_max of P(λ) ---
        # Initial guess λ0 = (GA + GB) / 2  (upper bound; RMSD=0 case)
        lam = 0.5 * (GA + GB)
        tol_rel = 1e-10  # relative tolerance on λ
        max_iter = 50  # should converge in 4-10 iterations

        for _ in range(max_iter):
            lam2 = lam * lam
            lam3 = lam2 * lam
            lam4 = lam2 * lam2

            poly = lam4 + C2 * lam2 + C1 * lam + C0
            dpoly = 4.0 * lam3 + 2.0 * C2 * lam + C1

            if abs(dpoly) < 1e-18:
                break

            lam_new = lam - poly / dpoly

            # Clamp to physically meaningful range
            if lam_new < 0.0:
                lam_new = 0.0

            # Relative convergence test on λ
            if abs(lam_new - lam) <= tol_rel * max(1.0, abs(lam_new)):
                lam = lam_new
                break

            lam = lam_new

        if lam < 0.0:  # Final safety clamp
            lam = 0.0

        # --- 6. Liu: adjoint(K - λI) to get quaternion eigenvector ---
        A4 = np.empty((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                A4[i, j] = K[i, j]
            A4[i, i] -= lam

        adjA = _adjoint4x4(A4)

        # Row of adjA with largest norm is a robust eigenvector choice
        best_row = 0
        best_norm2 = 0.0
        for i in range(4):
            n2 = 0.0
            for j in range(4):
                v = adjA[i, j]
                n2 += v * v
            if n2 > best_norm2:
                best_norm2 = n2
                best_row = i

        q = np.empty(4, dtype=np.float64)
        if best_norm2 > 0.0:
            inv_norm = 1.0 / math.sqrt(best_norm2)
            for j in range(4):
                q[j] = adjA[best_row, j] * inv_norm
        else:
            # Extremely degenerate fallback: identity quaternion
            q[0] = 1.0
            q[1] = 0.0
            q[2] = 0.0
            q[3] = 0.0

        # --- 7. Rotation matrix from quaternion ---
        R = _quat_to_rot(q)

        # --- 8. RMSD from λ_max ---
        E_min = GA + GB - 2.0 * lam
        if E_min < 0.0 and E_min > -1e-12:
            E_min = 0.0
        elif E_min < 0.0:
            E_min = 0.0

        rmsd = math.sqrt(E_min / N)

        return R, cP, cQ, rmsd
