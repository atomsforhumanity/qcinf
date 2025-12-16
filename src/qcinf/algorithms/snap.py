"""
snap.py — Symmetry-aware RMSD (snapRMSD)

This module implements the snapRMSD algorithm: a symmetry-aware RMSD metric
designed to correctly handle molecular symmetries and conformational diversity while
remaining extremely fast; a trifecta not available via exist graph isomorphism or
Hungarian approaches. The core idea is to *factor* the molecule into a backbone and
pendant components, align on the backbone, and then align components sequentially
consistent with each pendant component's local automorphism group while respecting the
global symmetry of the molecule. This approach allows snapRMSD to efficiently explore
the space of symmetry-equivalent mappings without the combinatorial explosion of the
graph isomorphism approach while still guaranteeing chemically correct mappings that
respect connectivity (which the Hungarian approach regularly violates to produce
spuriously low RMSD values).
"""

import itertools as it
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pynauty as pn
from qcio import Structure
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

from .connectivity_helpers import _canonical_map_pn, _to_pynauty_graph_adj_color
from .geometry_kernels import _compute_rmsd, kabsch, qcp_rotation


@dataclass
class Component:
    """A component with optional parent and attachment indices.

    Attributes:
        parent_idx: The index of the parent component this component is attached to. No
            parent_idx means this is a backbone component.
        pool: Whether this component is a pool pendant (True) or non-pool pendant (False).
        idxs: The indices of the atoms in this component.
    """

    parent_idx: int | None
    pool: bool
    idxs: np.ndarray


Batch = list[Component]


def _component_partition(
    coloring: list[int], P_opt: np.ndarray, C: np.ndarray
) -> list[int]:
    """
    Start from base_colors (e.g., Nauty coloring). Encode:
      - pointwise stabilizer of 'fixed' by promoting to singletons
      - setwise stabilizer of C by splitting each color into (in-C) vs (out-of-C)
    """
    new_coloring = list(coloring)
    next_id = max(new_coloring) + 1

    # Promote fixed to unique colors (singletons)
    for i, is_fixed in enumerate(P_opt != -1):
        if is_fixed:
            new_coloring[i] = next_id
            next_id += 1

    # Place component indices into their own color class by existing color
    for idx in C:
        # Magic number to offset in-C vs out-of-C
        new_coloring[idx] = new_coloring[idx] - 1000000

    return new_coloring


def _restrict_global_generators_to_C(
    gens_global: list[list[int]],
    C: np.ndarray,
) -> list[Permutation]:
    """
    Restrict each global generator (array form on 0..n-1) to a k-point
    Permutation on C, deduplicated.
    Assumes gens_global all preserve C setwise (enforced by the partition).
    """
    g_to_l = {v: i for i, v in enumerate(C)}  # global index -> local index
    seen: set[tuple[int, ...]] = set()
    local: list[Permutation] = []

    for g in gens_global:
        local_map = tuple(g_to_l[g[v]] for v in C)
        if local_map not in seen:
            seen.add(local_map)
            local.append(Permutation(local_map))
    if not local:  # Empty generators (trivial group)
        local.append(Permutation(list(range(len(C)))))  # identity
    return local


def _setwise_stabilizer_on_C(
    adj_dict: dict[int, list[int]],
    base_colors: list[int],
    P_opt: np.ndarray,
    C: np.ndarray,
) -> PermutationGroup:
    """(Aut(G)_(fixed))_{C}↾C as a local SymPy group on |C| points.

    This computes the setwise stabilizer of C within the pointwise stabilizer of
    singletons restricted to the action on C.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        P_opt: Partial mapping of structure indices; -1 means unfixed.
        component: The list of indices in the current component.
    """
    # Fix already assigned indices and stabilize component setwise
    colors = _component_partition(base_colors, P_opt, C)
    G_c = _to_pynauty_graph_adj_color(adj_dict, colors)
    gens, *_ = pn.autgrp(G_c)
    # Restrict to C
    local_gens = _restrict_global_generators_to_C(gens, C)
    return PermutationGroup(*local_gens)


def _iter_component_permutations(
    adj_dict: dict[int, list[int]],
    base_colors: list[int],
    P_opt: np.ndarray,
    component: Component,
) -> Iterable[list[int]]:
    """Aut(G)_{fixed}↾C as a local SymPy group on |C| points.

    This computes the pointwise stabilizer of the fixed vertices restricted to
    the action on C.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        P_opt: Partial mapping of structure indices; -1 means unfixed.
        component: The list of indices in the current component.

    Returns:
        An iterable of permutations (as lists) acting on the component indices
        reindex to 0..|C|-1.
    """
    if component.pool:  # Direct permutations by element type
        # Group by color (proxy for element types)
        groups = defaultdict(list)
        for i, c_idx in enumerate(component.idxs):
            groups[base_colors[c_idx]].append(i)
        # e.g., -CH2F -> [[0,1],[2]] -> permutations of [0,1] cross [2]
        # which yields [[0,1,2], [1,0,2]] for the full component
        return (
            list(it.chain.from_iterable(combo))
            for combo in it.product(
                *(it.permutations(idxs) for idxs in groups.values())
            )
        )
    else:  # Backbone or non-pool pendant: full setwise stabilizer
        return _setwise_stabilizer_on_C(
            adj_dict, base_colors, P_opt, component.idxs
        ).generate_schreier_sims(af=True)


def _factor_backbone_and_pendants(
    s1: Structure, depth: int = 1
) -> dict[int | None, Component]:
    """
    Factor the structure into a backbone + pendant groups for each element.

    For now just one layer and using a dictionary to group pendants by parent atom.

    Parameters:
        s1: The structure to factor.
        depth: The depth of pendant groups to consider (default 1). 0 means no pendant
            factoring (the whole structure will be considered backbone). 1 means singly
            attached pendants with siblings, such as methyl and methylene hydrogens or
            halogens, but atoms without siblings such as hydroxyl hydrogens, or phenyl
            hydrogens will remain in the backbone. Depth 2 includes bi-connected
            substituents such as phenyl rings (which contain internal symmetry) or
            multiple identical substituents attached to the same parent atom (which
            also contains symmetry), such as geminal diols (-C(OH)2) or other geminal
            substituents. Depth > 2 recursively peels more layers of substituents until
            the desired depth is reached or no further pendants exist.

    Returns:
        Dictionary mapping parent_idx (or None for backbone) to Component.
    """
    if depth == 0:
        idxs = np.array(range(len(s1.symbols)))  # Whole structure as backbone
        return {None: Component(parent_idx=None, pool=False, idxs=idxs)}

    # Depth == 1: identify singly attached pendants
    # Identify pendant groups
    pendant_groups: dict[int, list[int]] = defaultdict(list)  # (parent_idx) -> list of atom indices  # fmt: skip
    for atom_idx, nbrs in s1.adjacency_dict.items():
        if len(nbrs) == 1:  # Single attached pendant
            pendant_groups[nbrs[0]].append(atom_idx)

    backbone_indices = set(range(len(s1.symbols)))  # Initialize to all indices
    factored: dict[int | None, Component] = {}

    # Build components: pendants first, then backbone.
    for parent_idx, pendant_idxs in pendant_groups.items():
        if len(pendant_idxs) > 1:  # Multiple atoms in this pendant group
            component = Component(
                parent_idx=parent_idx,
                pool=True,
                idxs=np.array(pendant_idxs),
            )
            factored[parent_idx] = component
            backbone_indices.difference_update(pendant_idxs)

    # Add backbone component
    factored[None] = Component(
        parent_idx=None, pool=False, idxs=np.array(sorted(backbone_indices))
    )
    return factored


def _factored_to_batches(factored: dict[int | None, Component]) -> list[Batch]:
    """
    Convert the factored dictionary to batches.

    Returns:
        List of batches, where the first batch is the backbone and the subsequent batch
            contains the pendant groups.
    """
    batches: list[list[Component]] = [[], []]
    for parent_idx, component in factored.items():
        if parent_idx is None:
            batches[0].append(component)
        else:
            batches[1].append(component)
    return batches


def _align_fixed(
    A: np.ndarray, B: np.ndarray, P_opt: np.ndarray, alignment_backend: str = "kabsch"
) -> np.ndarray:
    """Return a copy of A aligned to B using pairs (i -> P[i]) for i with fixed[i]."""
    fixed = P_opt != -1

    if alignment_backend == "kabsch":
        R, cA, cB = kabsch(A[fixed], B[P_opt[fixed]])
    elif alignment_backend == "qcp":
        R, cA, cB = qcp_rotation(A[fixed], B[P_opt[fixed]])
    else:
        raise ValueError(f"Unknown alignment backend '{alignment_backend}'.")

    return (A - cA) @ R.T + cB  # align all of A


def _aligned_rmsd_for_candidate(
    A_geom: np.ndarray,
    B_geom: np.ndarray,
    P_opt: np.ndarray,
    A_C: np.ndarray,
    B_idx_perm: np.ndarray,
    backend: str = "kabsch",
) -> float:
    """Compute RMSD after aligning A to B using fixed + component indices."""
    # score on fixed ∪ C, with a per-candidate temporary alignment
    A_idx_fixed = np.flatnonzero(P_opt != -1)
    A_fixed_plus_C = np.concatenate([A_idx_fixed, A_C])
    B_fixed_plus_C = np.concatenate([P_opt[A_idx_fixed], B_idx_perm])
    return _compute_rmsd(
        A_geom[A_fixed_plus_C], B_geom[B_fixed_plus_C], align=True, backend=backend
    )


def _match_pool_by_color(
    A_C: np.ndarray,
    B_pool: np.ndarray,
    A_coloring: list[int],
    B_coloring: list[int],
) -> np.ndarray:
    """Return B_pool indices aligned *positionally* to A_C by color."""
    bins: dict[int, deque] = defaultdict(deque)  # color -> deque of indices in B_pool

    for idx in B_pool:
        bins[B_coloring[idx]].append(idx)

    B_C = np.empty_like(A_C)
    for i, a_idx in enumerate(A_C):
        B_C[i] = bins[A_coloring[a_idx]].popleft()  # maintain order for convenience

    return B_C


def snap_rmsd_and_assignment(
    s1: Structure,
    s2: Structure,
    *,
    align: bool = True,
    align_per_component: bool = False,
    factor: str | dict[int | None, Component] | None = "pendant",
    factor_depth: int = 1,
    alignment_backend: str = "kabsch",
) -> tuple[float, list[int]]:
    """
    Compute snapRMSD(s1→s2). Returns (rmsd, P) where P optimally maps s1 indices to s2 indices.

    Strategy:
      - Build graphs; require isomorphism.
      - Initialize global map P from canonical labels (M).
      - For each batch:
          * For each component C:
              · Build local setwise stabilizer (given 'fixed')
              · Score candidates WITHOUT mutating P (zero-copy indices)
              · Lock-in winner: P[C] = base_C[best_perm]
          * After batch, optionally align entire G1 to G2 using (P, fixed)
      - Final RMSD on aligned coordinates (or raw if align=False).

    Parameters:
        s1: First structure.
        s2: Second structure.
        align: Whether to align the structures before computing RMSD. Default is True.
            Generally one should only set this to False if the structures should be
            compared in their absolute orientations, e.g., from a crystal structure
            or docking pose.
        align_per_component: Whether to realign the structures after each factored component
            is processed. Default is False meaning that realignment is only done after
            each batch, except for the first batch which must align for each proposed component
            to establish a common reference frame.
        factor: May be a string specifying the built-in factoring method to use (default
            "pendant"), None which turns off factoring (equivalent to a graph isomorphism
            matching algorithm), or a custom structure factoring. If a factoring is
            provided it is list[Batch] where each Batch is a list of components and
            each component is a list of atom indices corresponding to s1, e.g.,
            [[ [0,1,2], [3,4] ], [ [5,6], [7] ]].
        factor_depth: The depth of pendant groups to consider (default is 1).


    Returns:
        (float, P_opt): The computed snapRMSD value and the optimal permutation as a
            list of indices for s1 onto s2.
    """
    # --- 1. Build graphs, canonical label map, compute automorphism group -----------
    # NOTE: Currently running nauty 3x on A and 2x on B. Can optimize?
    #   - Once per graph for isomorphism check
    #   - Once per graph for canonical labels
    #   - Once for A to get automorphism group

    # --- Graphs & isomorphism check
    A_adj = s1.adjacency_dict  # compute once since used multiple times
    if not A_adj or not s2.adjacency_dict:
        raise ValueError("Structures lack connectivity information.")

    A = _to_pynauty_graph_adj_color(A_adj, s1.symbols)
    B = _to_pynauty_graph_adj_color(s2.adjacency_dict, s2.symbols)

    if not pn.isomorphic(A, B):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    # --- Canonical labels & initial map
    M = np.array(_canonical_map_pn(A, B), dtype=int)  # i_A -> i_B
    P_opt = np.full(len(s1.symbols), -1, dtype=int)  # placeholder for final map

    # --- Compute automorphism group of A
    A_gens, A_mantissa, A_exponent, A_coloring, A_num_orbits = pn.autgrp(A)
    A_geom = s1.geometry  # Will not mutate s1.geometry
    B_geom = s2.geometry  # Will not mutate s2.geometry
    B_coloring = list(np.array(A_coloring)[np.argsort(M)])  # coloring of B via M
    # --- 2. Factor batches ----------------------------------------------------------
    if isinstance(factor, str):
        if factor.lower() == "pendant":
            A_factored = _factor_backbone_and_pendants(s1, depth=factor_depth)
            B_factored = _factor_backbone_and_pendants(s2, depth=factor_depth)
        else:
            raise ValueError(f"Unknown factoring method '{factor}'.")
    elif factor is None:
        A_factored = _factor_backbone_and_pendants(s1, depth=0)
        B_factored = _factor_backbone_and_pendants(s2, depth=0)
    else:
        A_factored = factor
    batches = _factored_to_batches(A_factored)  # ensure Batch format
    # TODO: Need to factor B based on A using M.

    # --- 3. Main control loop -----------------------------------------------------
    for b_idx, batch in enumerate(batches):
        for c_idx, component in enumerate(batch):
            best_rmsd = float("inf")
            best_perm: list[int] | None = None
            A_C = component.idxs

            # Get corresponding B_C for this component
            p_idx = component.parent_idx
            if p_idx is None or P_opt[p_idx] == M[p_idx]:  # Backbone component or parent not remapped  # fmt: skip
                B_C = M[A_C]  # Initial canonical map still valid
            elif component.pool:  # Remapped pool pendant
                B_pool = B_factored[P_opt[p_idx]].idxs
                B_C = _match_pool_by_color(A_C, B_pool, A_coloring, B_coloring)
            else:  # Remapped structured pendant
                # Implement with nauty canonical maps on local subgraphs rooted at parent_idx
                raise NotImplementedError(
                    "Remapped structured pendants not yet implemented."
                )

            for perm in _iter_component_permutations(A_adj, A_coloring, P_opt, component):  # fmt: skip
                B_idx_perm = B_C[perm]

                if align and (align_per_component or (b_idx == 0 and c_idx == 0)):
                    # Align per-candidate; use fixed + component indices
                    score = _aligned_rmsd_for_candidate(A_geom, B_geom, P_opt, A_C,B_idx_perm, backend=alignment_backend)  # fmt: skip
                else:
                    # No alignment; direct RMSD on component given established frame
                    score = _compute_rmsd(A_geom[A_C], B_geom[B_idx_perm], align=False)

                if score < best_rmsd:
                    best_rmsd = score
                    best_perm = perm

            # Lock-in best permutation for this component
            P_opt[A_C] = B_C[best_perm]

            # Per component and/or first component alignment (establishes frame)
            if align and (align_per_component or (b_idx == 0 and c_idx == 0)):
                A_geom = _align_fixed(A_geom, B_geom, P_opt)

        # Realign after each batch; avoid duplicate alignments
        if align and not align_per_component and not (b_idx == 0 and c_idx == 0):
            A_geom = _align_fixed(A_geom, B_geom, P_opt)

    assert np.all(P_opt != -1), "Not all indices got fixed!."  # Sanity check while debugging # fmt: skip
    assert all(v == 1 for v in Counter(P_opt).values()), "P_opt is not a valid permutation!"  # Sanity check while debugging # fmt: skip
    # Align false because already aligned at the end of the last batch if needed
    final_rmsd = _compute_rmsd(A_geom, B_geom[P_opt], align=False)
    return float(final_rmsd), P_opt.tolist()


def snap_rmsd(s1: Structure, s2: Structure, **kwargs) -> float:
    """Thin convenience wrapper around :func:`srmsd_and_opt_assignment` to return only RMSD."""
    rmsd_val, _ = snap_rmsd_and_assignment(s1, s2, **kwargs)
    return rmsd_val


def snap_assignment(
    s1: Structure,
    s2: Structure,
    **kwargs,
) -> list[int]:
    """Thin convenience wrapper around :func:`srmsd_and_opt_assignment` to return only assignment."""
    _, P_opt = snap_rmsd_and_assignment(s1, s2, **kwargs)
    return P_opt
