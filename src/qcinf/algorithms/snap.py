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
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
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
ComponentLike = list[int]
BatchLike = list[ComponentLike]


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


def _reduce_to_backbone(adj: dict[int, set[int]], backbone: set[int]) -> None:
    """Remove all vertices not in backbone from adj in-place."""
    to_remove = [v for v in adj.keys() if v not in backbone]
    for v in to_remove:
        # Remove v from its neighbors
        for nbr in adj[v]:
            adj[nbr].discard(v)
        # Remove v itself
        del adj[v]


def _peel_cyclic_pendants(
    adj: dict[int, set[int]],
) -> tuple[set[int], list[Component]]:
    """Placeholder: extract structured cyclic pendant Components and mutate adj to remove them.

    Future behavior (depth >= 2):
      - compute 2-core on current adj
      - find core components that are pendant-attached to the remainder
      - expand each with attached trees that do not reach another core component
      - emit Component(parent_idx=<attachment>, pool=False, idxs=<subgraph vertices>, attachment_idx=<in-component neighbor>)
      - delete those vertices from adj
    """
    print("Warning: cyclic pendant peeling not yet implemented.")
    return set(adj.keys()), []


def _peel_pool_pendants(
    adj: dict[int, set[int]],
) -> tuple[set[int], list[Component]]:
    """Peel sibling leaf groups (size>1). Returns (batch, removed_vertices).

    Parameters:
        adj: The adjacency dictionary of the structure graph.

    Returns:
        A tuple of (parent_idxs, list[Component]).
    """
    pools: dict[int, list[int]] = defaultdict(list)  # (parent_idx) -> list of atom indices  # fmt: skip

    for atom_idx, nbrs in adj.items():
        if len(nbrs) == 1:  # Single attached pendant
            pools[next(iter(nbrs))].append(atom_idx)

    backbone_indices = set(adj)  # Initialize to all indices
    factored: list[Component] = []

    # Build components
    for parent_idx, pendant_idxs in pools.items():
        if len(pendant_idxs) > 1:  # Multiple atoms in this pendant group
            cmp = Component(
                parent_idx=parent_idx,
                pool=True,
                idxs=np.array(sorted(pendant_idxs)),
            )
            factored.append(cmp)
            backbone_indices.difference_update(pendant_idxs)

    return backbone_indices, factored


def factor_plan_recursive(adj_dict: dict[int, list[int]], depth: int) -> list[Batch]:
    adj = {i: set(nbrs) for i, nbrs in adj_dict.items()}
    reversed_batch: list[Batch] = []

    def _recur(depth: int) -> None:
        nonlocal adj, reversed_batch

        if depth == 0:
            return

        backbone_indices, factored_pools = _peel_pool_pendants(adj)
        if factored_pools:
            reversed_batch.append(factored_pools)
            _reduce_to_backbone(adj, backbone_indices)

        backbone_indices, factored_pendants = _peel_cyclic_pendants(adj)
        if factored_pendants:
            reversed_batch.append(factored_pendants)

        if factored_pools or factored_pendants:
            _recur(depth - 1)

    # Initial factoring of first pendant layer
    backbone_indices, factored_pools = _peel_pool_pendants(adj)
    if factored_pools:
        reversed_batch.append(factored_pools)
        _reduce_to_backbone(adj, backbone_indices)

    _recur(depth - 1)

    # Backbone is whatever remains after factoring
    backbone_idxs = np.array(sorted(adj.keys()), dtype=int)
    backbone = Component(parent_idx=None, pool=False, idxs=backbone_idxs)
    return [[backbone], *reversed(reversed_batch)]


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


def _b_lookup(
    batches: list[Batch], M: npt.NDArray[np.int64]
) -> dict[tuple[int | None, bool], npt.NDArray[np.int64]]:
    """Build lookup of B components by parent_idx using canonical map M."""
    # (B_parent_idx, pool) -> indices in B
    b_lookup: dict[tuple[int | None, bool], npt.NDArray[np.int64]] = {}
    for component in it.chain.from_iterable(batches):
        idxs = M[component.idxs]
        if component.parent_idx is None:
            key = (None, component.pool)
        else:
            key = (M[component.parent_idx], component.pool)
        b_lookup[key] = idxs

    return b_lookup


def induced_local_adj(
    adj: dict[int, list[int]], nodes: npt.NDArray[np.int64]
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """
    Construct the induced subgraph on `nodes` with a local 0..k-1 labeling.

    Important ordering contract:
        The local vertex indices are assigned *in the order of `nodes`*.
        I.e., the first entry in `nodes` maps to local vertex 0, the second
        entry maps to local vertex 1, and so on. This ordering is relied upon by
        downstream logic in _match_structured_component to align the canonical
        labeling maps.

    Parameters:
        adj: Global adjacency dictionary.
        nodes: Array of global vertex IDs defining the induced subgraph.

    Returns:
        local_adj: Adjacency dictionary on local vertex IDs (0..k-1).
        g2l: Mapping from global vertex ID -> local vertex ID.
    """
    nodes_i = [int(v) for v in nodes]  # nodes are np.int64
    g2l = {v: i for i, v in enumerate(nodes_i)}
    local_adj: dict[int, list[int]] = {}
    for gv in nodes_i:
        lv = g2l[gv]
        local_adj[lv] = [g2l[gn] for gn in adj[gv] if gn in g2l]
    return local_adj, g2l


def _match_structured_component(
    A_C: npt.NDArray[np.int64],
    B_idxs: npt.NDArray[np.int64],
    A_adj: dict[int, list[int]],
    B_adj: dict[int, list[int]],
    A_coloring: list[int],
    B_coloring: list[int],
) -> npt.NDArray[np.int64]:
    """Return B indices aligned positionally to A_C using nauty on induced subgraphs.

    Parameters:
        A_C: The indices of the atoms in structure A's component.
        B_idxs: The indices of the atoms in structure B's component.
        A_adj: The adjacency dictionary of structure A.
        B_adj: The adjacency dictionary of structure B.
        A_coloring: The vertex coloring of structure A.
        B_coloring: The vertex coloring of structure B.

    Returns:
        B_C: The indices of the atoms in structure B's component aligned positionally to A_C.
    """
    A_loc_colors = [A_coloring[int(v)] for v in A_C]
    B_loc_colors = [B_coloring[int(v)] for v in B_idxs]

    if Counter(A_loc_colors) != Counter(B_loc_colors):
        raise ValueError("Structured component color histograms do not match. These components cannot be isomorphic.")  # pragma: no cover # fmt: skip

    A_loc_adj, A_g2l = induced_local_adj(A_adj, A_C)
    B_loc_adj, B_g2l = induced_local_adj(B_adj, B_idxs)

    A_sub = _to_pynauty_graph_adj_color(A_loc_adj, A_loc_colors)
    B_sub = _to_pynauty_graph_adj_color(B_loc_adj, B_loc_colors)

    local_map = _canonical_map_pn(A_sub, B_sub)

    B_C = np.empty_like(A_C)
    for i, gvA in enumerate(A_C):
        lvA = A_g2l[int(gvA)]
        lvB = local_map[lvA]
        B_C[i] = B_idxs[lvB]

    return B_C


def snap_rmsd_and_assignment(
    s1: Structure,
    s2: Structure,
    *,
    factor_depth: int = 1,
    factoring: list[Batch] | None = None,
    s1_coloring: list[Any] | None = None,
    s2_coloring: list[Any] | None = None,
    align: bool = True,
    alignment_backend: str = "kabsch",
    align_per_component: bool = False,
) -> tuple[float, list[int]]:
    # TODO: Add compute_connectivity_if_missing parameter?
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

        actor_depth: The depth of pendant groups to factor (default is 1). A depth of
            1 factors only singly attached pendant groups such as methyl and methylene
            hydrogens and halides. A depth of 2 would also factor their parents (e.g.,
            factor the three methyl groups on a tert-butyl substituent) and factor
            cyclic pendants such as phenyl rings. Higher depths are possible but
            rarely useful. A depth of 0 turns off factoring (equivalent to a full graph
            isomorphism approach). Ignored if a custom factoring is passed.
        factoring: A custom structure factoring with indices correspond to s1. If a
            factoring is provided users may provide a list[Batch] where each Batch is a
            list of Components or a simplified list[BatchLike] where each component is a list
            of atom indices. Factoring is grouped into batches to control algorithm
            execution e.g., [ [[0,1,2]], [[3,4]], [[5,6], [7]]]. After each batch the
            structures are realigned.
        s1_coloring: Optional list of colors for each vertex in s1, e.g., [0,0,0,3,3,3]
            indicates atoms 0-2 are of one color and atoms 3-5 are a second color.
            If no value is provided the Structure's symbols array is used as the
            default coloring. Coloring may be any hashable type such as str or int
            and may contain any chemical information deemed relevant by the user.
            E.g., hybridization states or partial charges could be encoded as 'Csp3' for
            an atom's color. If no coloring is provided the Structure's symbols array
            is used as the default coloring.
        s2_coloring: Coloring for s2; same format as s1_coloring.
        align: Whether to align the structures before computing RMSD. Default is True.
            Generally one should only set this to False if the structures should be
            compared in their absolute orientations, e.g., from a crystal structure
            or docking pose.
        alignment_backend: The backend to use for alignment calculations. Options are
            `kabsch` (default) or `qcp`. `qcp` uses a numba accelerated implementation
            of the QCP algorithm. `qcp` requires ~3s to compile on the first call and is
            then cached. It won't need re-compilation across process restarts after that.
            On subsequent process executions the first call will requires ~0.15 seconds
            to load the function from cache. Subsequent calls are much faster than
            `kabsch`. The break-even point where `qcp` becomes faster than `kabsch` is
            usually around ~100-200 RMSD calculations depending on system size.
        align_per_component: Whether to realign the structures after each factored component
            is processed. Default is False meaning that realignment is only done after
            each batch, except for the first component which must align for each proposed
            permutation to establish the initial common reference frame.

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
    A_adj, B_adj = s1.adjacency_dict, s2.adjacency_dict
    if not A_adj or not B_adj:
        raise ValueError("Structures lack connectivity information.")

    A = _to_pynauty_graph_adj_color(A_adj, s1_coloring or s1.symbols)
    B = _to_pynauty_graph_adj_color(B_adj, s2_coloring or s2.symbols)

    if not pn.isomorphic(A, B):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    # --- Canonical labels & initial map
    M = np.array(_canonical_map_pn(A, B), dtype=int)  # i_A -> i_B
    P_opt = np.full(len(s1.symbols), -1, dtype=int)

    # --- Compute automorphism group of A
    A_gens, A_mantissa, A_exponent, A_coloring, A_num_orbits = pn.autgrp(A)
    A_geom = s1.geometry  # Will not mutate s1.geometry
    B_geom = s2.geometry  # Will not mutate s2.geometry
    B_coloring = list(np.array(A_coloring)[np.argsort(M)])  # coloring of B via M
    # --- 2. Factor batches ----------------------------------------------------------
    if factoring is not None:
        # TODO: Possibly implement normalization for the simplified BatchLike format
        batches = factoring
    elif factor_depth == 0:  # Whole structure as backbone
        idxs = np.array(range(len(s1.symbols)))
        batches = [[Component(parent_idx=None, pool=False, idxs=idxs)]]
    else:
        batches = factor_plan_recursive(A_adj, depth=factor_depth)

    B_lookup = _b_lookup(batches, M)

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
                B_idxs = B_lookup[P_opt[p_idx], component.pool]
                B_C = _match_pool_by_color(A_C, B_idxs, A_coloring, B_coloring)
            else:  # Remapped structured pendant
                B_C = _match_structured_component(
                    A_C, B_idxs, A_adj, B_adj, A_coloring, B_coloring
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
    return final_rmsd, [int(x) for x in P_opt]


def snap_rmsd(s1: Structure, s2: Structure, **kwargs) -> float:
    """Thin convenience wrapper around :func:`snap_rmsd_and_opt_assignment` to return only RMSD."""
    rmsd_val, _ = snap_rmsd_and_assignment(s1, s2, **kwargs)
    return rmsd_val


def snap_assignment(
    s1: Structure,
    s2: Structure,
    **kwargs,
) -> list[int]:
    """Thin convenience wrapper around :func:`snap_rmsd_and_opt_assignment` to return only assignment."""
    _, P_opt = snap_rmsd_and_assignment(s1, s2, **kwargs)
    return P_opt
