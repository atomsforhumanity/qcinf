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
    idxs: npt.NDArray[np.int64]


Batch = list[Component]
ComponentLike = list[int]
BatchLike = list[ComponentLike]

## -- Factoring functions -------------------------------------------------------------


def _prune_graph(adj: dict[int, set[int]], keep: set[int]) -> None:
    """Remove all vertices not in keep from adj in-place."""
    to_remove = [v for v in adj.keys() if v not in keep]
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

    backbone_idx_set = set(adj)  # Initialize to all indices
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
            backbone_idx_set.difference_update(pendant_idxs)

    return backbone_idx_set, factored


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
            _prune_graph(adj, backbone_indices)

        backbone_indices, factored_pendants = _peel_cyclic_pendants(adj)
        if factored_pendants:
            reversed_batch.append(factored_pendants)

        if factored_pools or factored_pendants:
            _recur(depth - 1)

    # Initial factoring of first pendant layer
    backbone_idxs, factored_pools = _peel_pool_pendants(adj)
    if factored_pools:
        reversed_batch.append(factored_pools)
        _prune_graph(adj, backbone_idxs)

    _recur(depth - 1)

    # Backbone is whatever remains after factoring
    backbone_idxs = np.array(sorted(adj.keys()), dtype=int)
    backbone = Component(parent_idx=None, pool=False, idxs=backbone_idxs)
    return [[backbone], *reversed(reversed_batch)]


## -- Helper functions -------------------------------------------------------------
def _induced_local_adj(
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


def _build_B_comp_lookup(
    batches: list[Batch], M: npt.NDArray[np.int64]
) -> dict[tuple[int | None, bool], npt.NDArray[np.int64]]:
    """Build lookup of B components by parent_idx using canonical map M."""
    # (B_parent_idx, pool) -> indices in B
    b_lookup: dict[tuple[int | None, bool], npt.NDArray[np.int64]] = {}
    for component in it.chain.from_iterable(batches):
        idxs = M[component.idxs]
        is_pool = component.pool
        if component.parent_idx is None:
            parent = (None, is_pool)
        else:
            parent = (M[component.parent_idx], is_pool)
        b_lookup[parent] = idxs

    return b_lookup


def _match_pool_component(
    A_C: np.ndarray,
    B_pool: np.ndarray,
    A_orbit_ids: list[int],
    B_orbit_ids: list[int],
) -> np.ndarray:
    """Return B_pool indices aligned *positionally* to A_C by orbit id."""
    bins: dict[int, deque] = defaultdict(deque)  # orbit id -> deque of indices in B_pool  # fmt: skip

    for idx in B_pool:
        bins[B_orbit_ids[idx]].append(idx)

    B_C = np.empty_like(A_C)
    for i, a_idx in enumerate(A_C):
        B_C[i] = bins[A_orbit_ids[a_idx]].popleft()  # maintain order for convenience

    return B_C


def _match_structured_component(
    A_C: npt.NDArray[np.int64],
    B_idxs: npt.NDArray[np.int64],
    A_adj: dict[int, list[int]],
    B_adj: dict[int, list[int]],
    A_orbit_ids: list[int],
    B_orbit_ids: list[int],
) -> npt.NDArray[np.int64]:
    """Return B indices aligned positionally to A_C using nauty on induced subgraphs.

    Parameters:
        A_C: The indices of the atoms in structure A's component.
        B_idxs: The indices of the atoms in structure B's component.
        A_adj: The adjacency dictionary of structure A.
        B_adj: The adjacency dictionary of structure B.
        A_orbit_ids: The global orbit ids for structure A.
        B_orbit_ids: The global orbit ids for structure B.

    Returns:
        B_C: The indices of the atoms in structure B's component aligned positionally to A_C.
    """
    A_loc_orbit_ids = [A_orbit_ids[int(v)] for v in A_C]
    B_loc_orbit_ids = [B_orbit_ids[int(v)] for v in B_idxs]

    if Counter(A_loc_orbit_ids) != Counter(B_loc_orbit_ids):
        raise ValueError("Structured component orbit id histograms do not match. These components cannot be isomorphic.")  # pragma: no cover # fmt: skip

    A_loc_adj, A_g2l = _induced_local_adj(A_adj, A_C)
    B_loc_adj, B_g2l = _induced_local_adj(B_adj, B_idxs)

    A_sub = _to_pynauty_graph_adj_color(A_loc_adj, A_loc_orbit_ids)
    B_sub = _to_pynauty_graph_adj_color(B_loc_adj, B_loc_orbit_ids)

    local_map = _canonical_map_pn(A_sub, B_sub)

    B_C = np.empty_like(A_C)
    for i, gvA in enumerate(A_C):
        lvA = A_g2l[int(gvA)]
        lvB = local_map[lvA]
        B_C[i] = B_idxs[lvB]

    return B_C


## -- Local automorphism group computation ------------------------------------------------
def _component_autogroup(
    adj_dict: dict[int, list[int]],
    orbit_ids: list[int],
    C: npt.NDArray[np.int64],
) -> PermutationGroup:
    """Compute the local automorphism group acting on component C (0..|C|-1)"""
    # Build induced subgraph with local 0..k-1 indexing
    local_adj, _ = _induced_local_adj(adj_dict, C)

    # Restrict global orbit ids to this component (in C order)
    loc_orbit_ids = [orbit_ids[int(v)] for v in C]

    G_sub = _to_pynauty_graph_adj_color(local_adj, loc_orbit_ids)
    gens_local, *_ = pn.autgrp(G_sub)  # generators already on 0..k-1
    if not gens_local:
        return PermutationGroup(Permutation(list(range(len(C)))))  # Identity only
    return PermutationGroup(*(Permutation(g) for g in gens_local))


def _iter_component_perms(
    adj_dict: dict[int, list[int]],
    orbit_ids: list[int],
    component: Component,
) -> Iterable[list[int]]:
    """Compute the component's local automorphism group consistent with global orbit labels.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        orbit_ids: The orbit ids for the whole structure.
        component: The list of indices in the current component.

    Returns:
        An iterable of permutations (as lists) acting on the component indices
        reindex to 0..|C|-1.
    """
    if component.pool:  # Direct permutations by element type
        # Group by orbit id (proxy for element types)
        groups = defaultdict(list)
        for i, c_idx in enumerate(component.idxs):
            groups[orbit_ids[c_idx]].append(i)
        # e.g., -CH2F -> [[0,1],[2]] -> permutations of [0,1] cross [2]
        # which yields [[0,1,2], [1,0,2]] for the full component
        return (
            list(it.chain.from_iterable(combo))
            for combo in it.product(
                *(it.permutations(idxs) for idxs in groups.values())
            )
        )
    else:  # Backbone or non-pool pendant
        return _component_autogroup(
            adj_dict, orbit_ids, component.idxs
        ).generate_schreier_sims(af=True)


## -- Alignment helpers -------------------------------------------------------------
def _aligned_rmsd_for_candidate(
    A_geom: npt.NDArray[np.float64],
    B_geom: npt.NDArray[np.float64],
    P_opt: npt.NDArray[np.int64],
    A_C: npt.NDArray[np.int64],
    B_idx_perm: npt.NDArray[np.int64],
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


# -- Public API ---------------------------------------------------------------------
def snap_rmsd_align_and_assign(
    a_struct: Structure,
    b_struct: Structure,
    *,
    factor_depth: int = 1,
    factoring: list[Batch] | None = None,
    a_coloring: list[Any] | None = None,
    b_coloring: list[Any] | None = None,
    align: bool = True,
    alignment_backend: str = "kabsch",
    align_per_component: bool = False,
) -> tuple[float, Structure, list[int]]:
    # TODO: Add compute_connectivity_if_missing parameter?
    """
    Compute snapRMSD(A_struct → B_struct). Returns (rmsd, P) where P optimally maps
    A_struct indices to B_struct indices.

    Strategy:
      - Compute canonical label map, compute automorphism group for global orbit ids
      - For each batch:
          * For each component C:
              · Build build component automorphism group using global orbit ids
              · Score candidate permutations for each component
              · Lock-in winner
          * After batch, optionally align a_struct to b_struct using (P, fixed)
      - Final RMSD on optimally assigned coordinates

    Parameters:
        A_struct: First structure.
        B_struct: Second structure.
        factor_depth: The depth of pendant groups to factor (default is 1). A depth of
            1 factors only singly attached pendant groups such as methyl and methylene
            hydrogens and halides. A depth of 2 would also factor their parents (e.g.,
            factor the three methyl groups on a tert-butyl substituent) and factor
            cyclic pendants such as phenyl rings. Higher depths are possible but
            rarely useful. A depth of 0 turns off factoring (equivalent to a full graph
            isomorphism approach). Ignored if a custom factoring is passed.
        factoring: A custom structure factoring with indices correspond to a_struct. If a
            factoring is provided users may provide a list[Batch] where each Batch is a
            list of Components or a simplified list[BatchLike] where each component is a list
            of atom indices. Factoring is grouped into batches to control algorithm
            execution e.g., [ [[0,1,2]], [[3,4]], [[5,6], [7]]]. After each batch the
            structures are realigned.
        a_coloring: Optional list of colors for each vertex in a, e.g., [0,0,0,3,3,3]
            indicates atoms 0-2 are of one color and atoms 3-5 are a second color.
            If no value is provided the Structure's symbols array is used as the
            default coloring. Coloring may be any hashable type such as str or int
            and may contain any chemical information deemed relevant by the user.
            E.g., hybridization states or partial charges could be encoded as 'Csp3' for
            an atom's color. If no coloring is provided the Structure's symbols array
            is used as the default coloring.
        b_coloring: Coloring for b_struct; same format as a_coloring.
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
        (float, a_aligned, P_opt): The computed snapRMSD value, an aligned a_struct, and the
            optimal permutation as a list of indices for a_struct onto b_struct.
    """
    # --- 1. Build graphs, canonical label map, compute automorphism group -----------
    # NOTE: Currently running nauty 3x on A and 2x on B. Can optimize?
    #   - Once per graph for isomorphism check
    #   - Once per graph for canonical labels
    #   - Once for A to get automorphism group

    # --- Graphs & isomorphism check
    A_adj, B_adj = a_struct.adjacency_dict, b_struct.adjacency_dict
    if not A_adj or not B_adj:
        raise ValueError("Structures lack connectivity information.")

    A = _to_pynauty_graph_adj_color(A_adj, a_coloring or a_struct.symbols)
    B = _to_pynauty_graph_adj_color(B_adj, b_coloring or b_struct.symbols)

    if not pn.isomorphic(A, B):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    # --- Canonical labels & initial map
    M = np.array(_canonical_map_pn(A, B), dtype=int)  # i_A -> i_B
    P_opt = np.full(len(a_struct.symbols), -1, dtype=int)

    # --- Compute automorphism group of A
    A_gens, A_mantissa, A_exponent, A_orbit_ids, A_orbit_count = pn.autgrp(A)
    A_geom = a_struct.geometry  # Will not mutate a.geometry
    B_geom = b_struct.geometry  # Will not mutate b_struct.geometry
    B_orbit_ids = list(np.array(A_orbit_ids)[np.argsort(M)])  # orbit ids for B via M
    # --- 2. Factor batches ----------------------------------------------------------
    if factoring is not None:
        # TODO: Possibly implement normalization for the simplified BatchLike format
        batches = factoring
    elif factor_depth == 0:  # Whole structure as backbone
        idxs = np.array(range(len(a_struct.symbols)))
        batches = [[Component(parent_idx=None, pool=False, idxs=idxs)]]
    else:
        batches = factor_plan_recursive(A_adj, depth=factor_depth)

    B_C_by_parent = _build_B_comp_lookup(batches, M)

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
                B_idxs = B_C_by_parent[P_opt[p_idx], component.pool]
                B_C = _match_pool_component(A_C, B_idxs, A_orbit_ids, B_orbit_ids)
            else:  # Remapped structured pendant
                B_idxs = B_C_by_parent[P_opt[p_idx], component.pool]
                B_C = _match_structured_component(A_C, B_idxs, A_adj, B_adj, A_orbit_ids, B_orbit_ids)  # fmt: skip

            for perm in _iter_component_perms(A_adj, A_orbit_ids, component):
                B_C_perm = B_C[perm]

                if align and (align_per_component or (b_idx == 0 and c_idx == 0)):
                    # Align per-candidate; use fixed + component indices
                    score = _aligned_rmsd_for_candidate(A_geom, B_geom, P_opt, A_C, B_C_perm, backend=alignment_backend)  # fmt: skip
                else:
                    # No alignment; direct RMSD on component given established frame
                    score = _compute_rmsd(A_geom[A_C], B_geom[B_C_perm], align=False)  # fmt: skip

                if score < best_rmsd:
                    best_rmsd, best_perm = score, perm

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
    # Create aligned a Structure
    a_aligned = a_struct.model_copy(update={"geometry": A_geom}, deep=True)
    return final_rmsd, a_aligned, [int(x) for x in P_opt]


def snap_rmsd(a_struct: Structure, b_struct: Structure, **kwargs) -> float:
    """Thin convenience wrapper around :func:`snap_rmsd_and_opt_assignment` to return only RMSD."""
    rmsd_val, _, _ = snap_rmsd_align_and_assign(a_struct, b_struct, **kwargs)
    return rmsd_val


def snap_align(
    a_struct: Structure,
    b_struct: Structure,
    **kwargs,
) -> Structure:
    """Thin convenience wrapper around :func:`snap_rmsd_and_opt_assignment` to return only aligned Structure."""  # fmt: skip
    _, a_aligned, _ = snap_rmsd_align_and_assign(a_struct, b_struct, **kwargs)
    return a_aligned


def snap_assign(
    a_struct: Structure,
    b_struct: Structure,
    **kwargs,
) -> list[int]:
    """Thin convenience wrapper around :func:`snap_rmsd_and_opt_assignment` to return only assignment."""
    _, _, P_opt = snap_rmsd_align_and_assign(a_struct, b_struct, **kwargs)
    return P_opt
