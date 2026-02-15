from collections import defaultdict

import numpy as np
import pynauty as pn
import qcconst.periodic_table as pt
from qcio import Structure

from .connectivity_helpers import _canonical_map_pn, _to_pynauty_graph_adj_color


def determine_connectivity(
    structure: Structure, *, cov_factor: float = 1.2
) -> list[tuple[int, int, float]]:
    """Determine connectivity using covalent radii.

    Two atoms are considered bonded when their distance is less than or equal to
    `cov_factor * (r_cov(i) + r_cov(j))`. Bond orders are set to 1.0 since this
    routine only infers connectivity. Numpy vectorization is used for speed (~27x faster
    than nested loops for a 40-atom molecule, 55x faster for a 113-atom molecule).

    Args:
        structure: The Structure containing symbols and geometry in Bohr.
        cov_factor: Scaling factor applied to the sum of covalent radii.

    Returns:
        A list of (atom_index_a, atom_index_b, bond_order) tuples, using 0-based
        indices and a bond order of 1.0.
    """
    if cov_factor <= 0:
        raise ValueError("cov_factor must be positive.")

    geom = structure.geometry
    syms = structure.symbols
    n = geom.shape[0]

    # 1. Covalent radii as an array
    radii = np.array([getattr(pt, sym).cov_radius for sym in syms], dtype=float)

    # 2. All (i, j) with i < j
    i_idx, j_idx = np.triu_indices(n, k=1)

    # 3. Pairwise distances for just those pairs
    deltas = geom[i_idx] - geom[j_idx]  # shape (num_pairs, 3)
    dists = np.linalg.norm(deltas, axis=1)  # shape (num_pairs,)

    # Threshold for each pair
    thresholds = cov_factor * (radii[i_idx] + radii[j_idx])

    # 4. Mask pairs that are bonded
    mask = dists <= thresholds
    bonded_i = i_idx[mask]
    bonded_j = j_idx[mask]

    # Build connectivity list with bond order 1.0
    connectivity = [(int(i), int(j), 1.0) for i, j in zip(bonded_i, bonded_j)]
    return connectivity


def canonical_ranks(structure: "Structure", break_ties: bool = False) -> list[int]:
    """Compute canonical ranks of atoms in a structure using NAUTY.

    Args:
        structure: The Structure for which to compute canonical ranks.
        break_ties: Whether to break ties or leave atoms in the same symmetry class with
            the same rank.
            - break_ties=False (default): Returns a *symmetry-class rank* per original atom
            index (0..k-1), where atoms in the same graph-automorphism orbit share the same
            rank. Ranks are assigned deterministically by sorting orbits primarily by
            (atomic number, -orbit size), and secondarily by a canonical-label-based tie-breaker
            so results are stable across arbitrary input atom orderings.
            - break_ties=True: Returns a *total-order rank* per original atom index (0..n-1), corresponding
          to each atom's position in NAUTY's canonical labeling (orig -> canon).

    Returns:
        List[int]: ranks aligned to original atom indices.
    """
    A = _to_pynauty_graph_adj_color(structure.adjacency_dict, structure.symbols)

    # Canonical labeling: can[i_canon] = i_orig  (canon -> orig)
    can = pn.canon_label(A)
    # Invert to get inv[i_orig] = i_canon  (orig -> canon)
    inv = [0] * len(can)
    for i_canon, i_orig in enumerate(can):
        inv[i_orig] = i_canon

    if break_ties:
        # Total-order rank per original atom index: position in canonical labeling.
        return inv

    # Otherwise: orbit / symmetry-class ranks.
    *_, A_coloring, _ = pn.autgrp(A)

    # Group vertices by orbit label (coloring label).
    orbit_to_verts: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(A_coloring):
        orbit_to_verts[lab].append(i)

    def orbit_key(lab: int) -> tuple[int, int, int]:
        verts = orbit_to_verts[lab]
        sym = structure.symbols[verts[0]]  # valid if initial coloring is by element
        size = len(verts)
        # Canonical tie-breaker: earliest canonical index among members of this orbit.
        tie = min(inv[v] for v in verts)
        # element primary, larger orbit first, then stable canonical tie-break
        return (getattr(pt, sym).number, -size, tie)

    ordered_orbits = sorted(orbit_to_verts.keys(), key=orbit_key)

    # Map orbit label -> compact rank 0..k-1
    lut = {lab: r for r, lab in enumerate(ordered_orbits)}

    # Return per-atom ranks aligned to original indices.
    return [lut[lab] for lab in A_coloring]


def canonical_map(
    struct_a: Structure,
    struct_b: Structure,
) -> list[int]:
    """Compute canonical map of atoms from struct_a to struct_b using NAUTY.

    Args:
        struct_a: The first Structure.
        struct_b: The second Structure.
    Returns:
        A list of indices mapping atoms in struct_a to struct_b.
    """
    A = _to_pynauty_graph_adj_color(struct_a.adjacency_dict, struct_a.symbols)
    B = _to_pynauty_graph_adj_color(struct_b.adjacency_dict, struct_b.symbols)
    return _canonical_map_pn(A, B)
