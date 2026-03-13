"""
Symmetry-aware RMSD and assignment for isomorphic molecular graphs.

This module factors a molecule into a core plus peripheral components (leaves, cyclic
components, and repeated branches), then computes a connectivity-preserving atom mapping
between two isomorphic structures by scoring symmetry-allowed component permutations.

Main entry point:
    - `snap_rmsd_align_assign`: returns `(rmsd, a_aligned, p_opt)`.

Public convenience wrappers:
    - `snap_rmsd`: returns `rmsd`.
    - `snap_align`: returns `a_aligned`.
    - `snap_rmsd_align`: returns `(rmsd, a_aligned)`.
    - `snap_assign`: returns `p_opt`.
"""

import copy
import itertools as it
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import pynauty as pn
import rustworkx as rx
from qcconst import constants
from qcio import LengthUnit, Structure
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

from .connectivity import adjacency_dict
from .connectivity_helpers import (
    _canonical_map_pn,
    _get_branch_info,
    _to_pynauty_graph_adj_color,
)
from .geometry_kernels import _compute_rmsd, kabsch, qcp_rotation


@dataclass
class Component:
    """A component of the molecular structure.

    Attributes:
        parent_idx: The index of the parent component this component is attached to. No
            parent_idx means this is the core component.
        leaves: Whether the component contains leaf peripheral components (True) or
            non-leaf peripheral components (False).
        idxs: The indices of the atoms in this component.

    """

    parent_idx: int | None
    leaves: bool
    idxs: npt.NDArray[np.int64]


Batch = list[Component]
ComponentLike = list[int]
BatchLike = list[ComponentLike]

# ----------------------------
# Graph Primitives
# ----------------------------


def _prune_graph(adj: dict[int, set[int]], keep: set[int]) -> None:
    """Remove all vertices not in keep from adj in-place."""
    to_remove = [v for v in adj.keys() if v not in keep]
    for v in to_remove:
        # Remove v from its neighbors
        for nbr in adj[v]:
            adj[nbr].discard(v)
        # Remove v itself
        del adj[v]


def _rx_graph_from_adj(adj: dict[int, set[int]]) -> rx.PyGraph:
    """
    Build an undirected rustworkx PyGraph where:

    - Node payload == original vertex id
    - Node index == rustworkx internal index (opaque)
    - Edges are added once (undirected)

    You retrieve original ids later via g[node_index].
    """
    g = rx.PyGraph(multigraph=False)
    adj2rx = {v: g.add_node(v) for v in adj}  # original id -> rx node index

    for v, nbrs in adj.items():
        rv = adj2rx[v]
        for nbr in nbrs:
            rn = adj2rx[nbr]
            if rv < rn:  # add each undirected edge once
                g.add_edge(rv, rn, None)
    return g


def _cycles(adj: dict[int, set[int]]) -> list[set[int]]:
    """
    Return cycle as sets of vertex ids (keys of adj).

    A "cycle" is a connected component of the subgraph consisting only of
    non-bridge edges (edges that lie on at least one cycle).
    """

    def _edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _has_cycle(g_cycle: rx.PyGraph, comp: set[int]) -> bool:
        """
        Determine if the component has at least one cycle. Required due to rustworkx bug.
        https://github.com/Qiskit/rustworkx/issues/1534
        """
        n = len(comp)
        if n < 2:
            return False

        m = 0
        for u, v in g_cycle.edge_list():
            if u in comp and v in comp:
                m += 1

        return m >= n

    # Build rustworkx graph + find bridges
    g = _rx_graph_from_adj(adj)
    bridges = {_edge_key(*edge) for edge in rx.bridges(g)}

    # Build graph with only non-bridge edges (i.e., cycle edges)
    g_cycle = rx.PyGraph(multigraph=False)
    g_cycle.add_nodes_from(range(g.num_nodes()))  # same indices as g

    for u, v in g.edge_list():  # u,v are rx indices
        if _edge_key(u, v) not in bridges:
            g_cycle.add_edge(u, v, None)

    # Compute connected components of g_cycle (only cycles)
    cycles: list[set[int]] = []
    for cycle in rx.connected_components(g_cycle):
        # >= 2 should work but due to a bug in rustworkx which omits the first edge as a bridge (if it correctly is one) we use >= 3. https://github.com/Qiskit/rustworkx/issues/1534  # fmt: skip
        # if len(cycle) >= 3:
        # Logically the above if statement should be sufficient, but due to the rustworkx bug we add an explicit cycle check
        if _has_cycle(g_cycle, cycle):
            cycles.append({g[i] for i in cycle})  # translate back to original ids

    return cycles


def _outside_components(adj: dict[int, set[int]], excluded: set[int]) -> list[set[int]]:
    """
    Connected components of the induced subgraph on outside vertices (V \\ excluded).
    """
    _adj = copy.deepcopy(adj)
    keep = set(adj) - excluded
    _prune_graph(_adj, keep)
    g = _rx_graph_from_adj(_adj)
    return [{g[i] for i in comp} for comp in rx.connected_components(g)]


# ----------------------------
# Cycle Factoring
# ----------------------------


def _absorb_components(
    adj: dict[int, set[int]],
    cycles: list[set[int]],
    outside_components: list[set[int]],
) -> tuple[list[set[int]], list[set[int]]]:
    """
    Absorb outside connected-components into cycles when they "single-touch" exactly one cycle.

    Definition:
      - An outside component C "touches" a cycle blob S if there exists an edge (u, v)
        with u in C and v in S.
      - If C touches exactly one cycle blob, absorb C into that cycle blob.

    Returns:
      (new_cycles, remaining_outside_components)

    Notes:
      - Assumes cycles are disjoint sets of vertices.
      - All indices are in the original adj indexing.
    """
    # Map vertex -> cycle index for O(1) membership queries
    v2cidx: dict[int, int] = {}
    for cidx, cyc in enumerate(cycles):
        for v in cyc:
            v2cidx[v] = cidx

    # Make a new copy of cycles
    new_cycles = [set(cyc) for cyc in cycles]
    remaining: list[set[int]] = []

    for comp in outside_components:
        touched: set[int] = set()

        # Determine which cycle blobs this component touches
        for u in comp:
            for nbr in adj[u]:
                cidx = v2cidx.get(nbr)  # type: ignore
                if cidx is not None:
                    touched.add(cidx)
                    # Small early exit: if it touches >1, we won't absorb
                    if len(touched) > 1:
                        break
            if len(touched) > 1:
                break

        if len(touched) == 1:
            (cidx,) = tuple(touched)
            # Absorb: union vertices into that cycle blob
            new_cycles[cidx].update(comp)
        else:
            # Keep: touches 0 cycles, or touches >1 cycles
            remaining.append(comp)

    return new_cycles, remaining


def _determine_core_and_factor_cycles(
    adj: dict[int, set[int]],
    cycles: list[set[int]],
    remaining: list[set[int]],
) -> tuple[set[int], list[Component]]:
    """
    Determine a core vertex-set and factor peripheral ring systems ("cycle blobs")
    into Components.

    Definitions
    ----------
    - V = set(adj.keys())
    - cycles: list of disjoint vertex-sets, each a "cycle blob" (fused/spiro ring-system).
              These blobs are assumed to already include any absorbed *non-cyclic* substituents
              you want treated as part of the ring substituent layer (i.e., "cycle + its substituents").
    - remaining: connected components of:
          V \\ union(cycles)
      i.e., ALL vertices not belonging to any cycle blob, partitioned into CCs.

    - Outside-neighbor (boundary) of a blob S:
        ∂(S) = { v in V\\S : exists u in S with (u,v) in E }.

    Goal / heuristic
    ----------------
    We want a clean "rooted" factoring where *factored* cycle blobs attach to the core
    by exactly one outside contact that lies in the final core set.

    Algorithm
    ---------
    Pass 0: Precompute boundary contacts for each cycle blob:
        boundary[i] = ∂(cycles[i])

    Pass 1a (size seed): Let R = union(remaining) (non-cycle remainder).
        If R is non-empty and the largest cycle blob is larger than R, absorb that largest cycle
        into the core seed. (Macrocycle-core case.)

        This is a *single* absorption (largest-by-size). After absorption, no other cycle can be
        larger than the core seed (since it was the largest cycle).

    Pass 1b: Define core seed B as:
        B = R  U (optional absorbed-largest-cycle) U union_{i : |boundary[i]| != 1} cycles[i]

      Intuition:
        - all non-cyclic remainder is core by default
        - plus optional "largest macrocycle" if it dominates the remainder
        - |∂| = 0   : freestanding ring fragment => not peripheral (pendant) => core
        - |∂| > 1   : embedded / connective ring => not peripheral (pendant) => core

    Pass 2: For each cycle blob with |∂| = 1, let its unique outside neighbor be p.
        - If p ∈ B: factor this blob as Component(parent_idx=p)
        - Else: do NOT factor; absorb into core
                (prevents biphenyl-like A—B where each sees only the other as its unique outside contact)

      Factored blobs are merged by shared parent_idx to handle multiple rings off the same parent.

    Returns
    -------
    core_idxs: set[int]
        All vertex ids considered core after absorption decisions.
    components: list[Component]
        Factored peripheral cycle Components (leaf=False), merged by parent_idx.
    """
    if not adj:
        return set(), []

    # ---- Pass 0: boundaries ----
    boundaries: list[set[int]] = []
    for cyc in cycles:
        b: set[int] = set()
        for v in cyc:
            for nbr in adj[v]:
                if nbr not in cyc:
                    b.add(nbr)
        boundaries.append(b)

    # ---- Pass 1: core seed from remainder ----
    core_idxs: set[int] = set()
    for r in remaining:
        core_idxs |= r  # R = union(remaining)

    # Pass 1a: size-seed macrocycle-core case
    # If the largest cycle dominates the remainder, treat it as part of core.
    if cycles:
        max_i = max(range(len(cycles)), key=lambda i: len(cycles[i]))
        largest = cycles[max_i]
        if core_idxs:
            # Normal case: compare against non-cyclic remainder size
            if len(largest) > len(core_idxs):
                core_idxs |= largest
        else:
            # Purely cyclic graph (no remainder): seed only if dominant macrocycle exists
            if len(cycles) == 1:
                core_idxs |= largest
            else:
                # Compute 2nd largest size without sorting whole list
                second = max(len(cycles[i]) for i in range(len(cycles)) if i != max_i)
                if len(largest) > second:
                    core_idxs |= largest

    # Pass 1b: absorb all cycles with |∂| != 1
    for cyc, b in zip(cycles, boundaries):
        if len(b) != 1:
            core_idxs |= cyc

    # Freeze the core anchor AFTER Pass 1 decisions
    core_anchor = set(core_idxs)

    # ---- Pass 2: factor peripheral (pendant) cycles whose unique outside contact is in core ----
    parent2idxs: defaultdict[int, set[int]] = defaultdict(set)

    for cyc, b in zip(cycles, boundaries):
        if len(b) != 1:
            continue  # already absorbed in pass 1b (or will be in core anyway)
        if cyc <= core_anchor:
            # Cycle was already absorbed into core in pass 1a/1b; do not refactor it.
            continue

        parent = next(iter(b))
        if parent in core_anchor:
            parent2idxs[parent].update(cyc)
        else:
            # Unique outside contact is not in core => "ring-to-ring" only (biphenyl-like)
            # or other non-rooted situation; absorb it.
            core_idxs |= cyc

    components: list[Component] = [
        Component(
            parent_idx=parent,
            leaves=False,
            idxs=np.array(sorted(idxs), dtype=np.int64),
        )
        for parent, idxs in parent2idxs.items()
    ]

    return core_idxs, components


def _peel_cycles(
    adj: dict[int, set[int]],
) -> tuple[set[int], list[Component]]:
    """
    Peel cyclic peripheral components from the graph.

    Returns:
      core_idxs, list[Components] (cycles with their substituents)
    """
    # 1: cycle blobs
    cycles = _cycles(adj)

    # 2: outside CCs
    outside_ccs = _outside_components(adj, {v for cyc in cycles for v in cyc})

    # 3: Absorb components into cycle blobs
    full_cycles, remaining_outside_ccs = _absorb_components(adj, cycles, outside_ccs)

    # 4: Determine core and factor cycle components
    return _determine_core_and_factor_cycles(adj, full_cycles, remaining_outside_ccs)


# ----------------------------
# Peripheral Factoring
# ----------------------------


def _peel_symmetric_leaves(
    adj: dict[int, set[int]],
    color: list[int],
) -> tuple[set[int], list[Component]]:
    """Peel duplicated leaf siblings, grouping peeled leaves by parent.

    Parameters:
        adj: The adjacency dictionary of the structure graph.
        color: Vertex colors / labels, indexed by atom index.

    Returns:
        A tuple of (core_idxs, list[Component]).
    """
    leaves_by_parent: dict[int, list[int]] = defaultdict(list)

    for atom_idx, nbrs in adj.items():
        if len(nbrs) == 1:  # This vertex is a leaf
            parent_idx = next(iter(nbrs))
            leaves_by_parent[parent_idx].append(atom_idx)

    core_idxs = set(adj)
    factored: list[Component] = []

    for parent_idx, leaf_idxs in leaves_by_parent.items():
        if len(leaf_idxs) < 2:  # Only consider if there are siblings to factor
            continue

        by_color: dict[int, list[int]] = defaultdict(list)  # {H: [1,5,6], F: [4,3]}
        for idx in leaf_idxs:
            by_color[color[idx]].append(idx)

        duplicated_idxs = sorted(
            idx for group in by_color.values() if len(group) > 1 for idx in group
        )

        if not duplicated_idxs:
            continue

        factored.append(
            Component(
                parent_idx=parent_idx,
                leaves=True,
                idxs=np.array(duplicated_idxs),
            )
        )
        core_idxs.difference_update(duplicated_idxs)

    return core_idxs, factored


def _peel_symmetric_branches(
    adj: dict[int, set[int]], *, color: list[int]
) -> tuple[set[int], list[Component]]:
    """
    Peel repeated rooted isomorphic substituent branches off high-degree (>=4) vertices.

    Assumptions:
      - This is called on the "remaining" graph after cycle-related factoring has happened.
      - Branches considered here are acyclic (tree-like) in practice.

    Guardrails:
      - Only consider parents with degree >= 4.
      - Only consider branches with > 1 vertex (so we don't steal leaf leafing work).
      - Parent-local rollback: if peeling ALL factorable groups at a parent would leave the
        parent with <2 remaining neighbors, peel NOTHING at that parent (avoid arbitrary
        core choice on ambiguous small graphs like CCC(O)(O)CC).

    Returns:
      core_idxs: vertices left after peeling.
      components: list of non-leaf Components representing peeled unions.
    """
    core_idxs = set(adj.keys())
    components: list[Component] = []

    for parent, nbrs in adj.items():
        if len(nbrs) < 4:
            continue

        # Map signature -> list of (vertex_set)
        sig2branches = defaultdict(list)

        for root in nbrs:
            # Single pass to get both the set of nodes and the tree signature
            verts, sig = _get_branch_info(adj, parent, root, color)
            if len(verts) > 1:  # Single-node branches are handled by leaf peripheral peeling
                sig2branches[sig].append(verts)

        # Filter for groups that actually have duplicates
        peel_candidates = [
            v_list for v_list in sig2branches.values() if len(v_list) >= 2
        ]
        if not peel_candidates:
            continue

        # If peeling all groups would leave parent with <2 remaining neighbors,
        # peel NOTHING at this parent (avoid arbitrary core choice).
        all_peeled_nodes = {
            v for group in peel_candidates for v_set in group for v in v_set
        }
        remaining_deg = sum(1 for nbr in nbrs if nbr not in all_peeled_nodes)

        if remaining_deg >= 2:
            for v_list in peel_candidates:
                union = set().union(*v_list)
                components.append(
                    Component(
                        parent_idx=parent,
                        leaves=False,
                        idxs=np.array(sorted(union), dtype=np.int64),
                    )
                )
                core_idxs -= union
    return core_idxs, components


# ----------------------------
# Factor Planning
# ----------------------------


def _factor_plan_recursive(
    adj_dict: dict[int, list[int]], *, depth: int, color: list[int]
) -> list[Batch]:
    adj = {i: set(nbrs) for i, nbrs in adj_dict.items()}
    reversed_batch: list[Batch] = []

    def _recur(depth: int) -> None:
        nonlocal adj, reversed_batch

        if depth == 0:
            return

        to_append = []
        core_idxs, leaves = _peel_symmetric_leaves(adj, color=color)
        if leaves:
            to_append.extend(leaves)
            _prune_graph(adj, core_idxs)

        core_idxs, cycles = _peel_cycles(adj)
        if cycles:
            to_append.extend(cycles)
            _prune_graph(adj, core_idxs)

        core_idxs, branches = _peel_symmetric_branches(adj, color=color)
        if branches:
            to_append.extend(branches)
            _prune_graph(adj, core_idxs)

        if to_append:
            to_append.reverse()  # Branches, cycles, then leaves
            reversed_batch.append(to_append)
            _recur(depth - 1)

    # Initial factoring of first peripheral layer
    core_idxs, leaves = _peel_symmetric_leaves(adj, color=color)
    if leaves:
        reversed_batch.append(leaves)
        _prune_graph(adj, core_idxs)

    _recur(depth - 1)

    # Core is whatever remains after factoring
    core_idxs = np.array(sorted(adj.keys()), dtype=int)
    core = Component(parent_idx=None, leaves=False, idxs=core_idxs)
    return [[core], *reversed(reversed_batch)]


def _validate_factoring(batches: list[Batch]) -> None:
    """Validate that the provided factoring is well-formed."""
    seen_idxs = set(batches[0][0].idxs)  # Start with core indices
    # Ensure parent is fixed before child and no duplicate indices
    for batch in batches:
        for component in batch:
            if component.parent_idx is None:
                continue  # Core component, no parent
            if component.parent_idx not in seen_idxs:
                raise ValueError(f"Component with parent_idx={component.parent_idx} appears before its parent in the factored batches.")  # fmt: skip
            for idx in component.idxs:
                if idx in seen_idxs:
                    raise ValueError(f"Duplicate index {idx} found in factored components.")  # fmt: skip
                seen_idxs.add(idx)


# ----------------------------
# Component Mapping Helpers
# ----------------------------


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


def _build_b_component_lookup(
    batches: list[Batch], canonical_map_ab: npt.NDArray[np.int64]
) -> dict[tuple[int | None, bool], npt.NDArray[np.int64]]:
    """Build lookup of B components by parent_idx using canonical map A->B."""
    # (b_parent_idx, is_leaf) -> indices in b
    b_lookup: dict[tuple[int | None, bool], npt.NDArray[np.int64]] = {}
    for component in it.chain.from_iterable(batches):
        idxs = canonical_map_ab[component.idxs]
        is_leaf = component.leaves
        if component.parent_idx is None:
            parent = (None, is_leaf)
        else:
            parent = (canonical_map_ab[component.parent_idx], is_leaf)
        b_lookup[parent] = idxs

    return b_lookup


def _match_leaf_component(
    a_comp_idxs: np.ndarray,
    b_leaf_idxs: np.ndarray,
    a_orbit_ids: list[int],
    b_orbit_ids: list[int],
) -> np.ndarray:
    """Return b_leaf indices aligned *positionally* to a_comp_idxs by orbit id."""
    bins: dict[int, deque] = defaultdict(deque)  # orbit id -> deque of indices in b_leaf_idxs  # fmt: skip

    for idx in b_leaf_idxs:
        bins[b_orbit_ids[idx]].append(idx)

    b_comp_idxs = np.empty_like(a_comp_idxs)
    for i, a_idx in enumerate(a_comp_idxs):
        b_comp_idxs[i] = bins[
            a_orbit_ids[a_idx]
        ].popleft()  # maintain order for convenience

    return b_comp_idxs


def _match_structured_component(
    a_comp_idxs: npt.NDArray[np.int64],
    b_comp_idxs: npt.NDArray[np.int64],
    a_adj: dict[int, list[int]],
    b_adj: dict[int, list[int]],
    a_orbit_ids: list[int],
    b_orbit_ids: list[int],
) -> npt.NDArray[np.int64]:
    """Return B indices aligned positionally to a_comp_idxs using nauty on induced subgraphs.

    Parameters:
        a_comp_idxs: The indices of the atoms in structure A's component.
        b_comp_idxs: The indices of the atoms in structure B's component.
        a_adj: The adjacency dictionary of structure A.
        b_adj: The adjacency dictionary of structure B.
        a_orbit_ids: The global orbit ids for structure A.
        b_orbit_ids: The global orbit ids for structure B.

    Returns:
        The indices of the atoms in structure B's component aligned positionally
        to a_comp_idxs.
    """
    a_local_orbit_ids = [a_orbit_ids[int(v)] for v in a_comp_idxs]
    b_local_orbit_ids = [b_orbit_ids[int(v)] for v in b_comp_idxs]

    if Counter(a_local_orbit_ids) != Counter(b_local_orbit_ids):
        raise ValueError("Structured component orbit id histograms do not match. These components cannot be isomorphic.")  # pragma: no cover # fmt: skip

    a_local_adj, a_global_to_local = _induced_local_adj(a_adj, a_comp_idxs)
    b_local_adj, _ = _induced_local_adj(b_adj, b_comp_idxs)

    a_subgraph = _to_pynauty_graph_adj_color(a_local_adj, a_local_orbit_ids)
    b_subgraph = _to_pynauty_graph_adj_color(b_local_adj, b_local_orbit_ids)

    local_map = _canonical_map_pn(a_subgraph, b_subgraph)

    aligned_b_comp_idxs = np.empty_like(a_comp_idxs)
    for i, a_global_idx in enumerate(a_comp_idxs):
        a_local_idx = a_global_to_local[int(a_global_idx)]
        b_local_idx = local_map[a_local_idx]
        aligned_b_comp_idxs[i] = b_comp_idxs[b_local_idx]

    return aligned_b_comp_idxs


# ----------------------------
# Permutation Helpers
# ----------------------------
def _component_autogroup(
    adj: dict[int, list[int]],
    orbit_ids: list[int],
    comp_idxs: npt.NDArray[np.int64],
) -> PermutationGroup:
    """Compute the local automorphism group acting on `comp_idxs` (0..|C|-1)."""
    # Build induced subgraph with local 0..k-1 indexing
    local_adj, _ = _induced_local_adj(adj, comp_idxs)

    # Restrict global orbit ids to this component (in comp_idxs order)
    loc_orbit_ids = [orbit_ids[int(v)] for v in comp_idxs]

    G_sub = _to_pynauty_graph_adj_color(local_adj, loc_orbit_ids)
    gens_local, *_ = pn.autgrp(G_sub)  # generators already on 0..k-1
    if not gens_local:
        return PermutationGroup(
            Permutation(list(range(len(comp_idxs))))
        )  # Identity only
    return PermutationGroup(*(Permutation(g) for g in gens_local))


def _iter_component_perms(
    adj: dict[int, list[int]],
    orbit_ids: list[int],
    component: Component,
) -> Iterable[list[int]]:
    """Compute the component's local automorphism group consistent with global orbit labels.

    Parameters:
        adj: The adjacency dictionary of the structure graph.
        orbit_ids: The orbit ids for the whole structure.
        component: The current component.

    Returns:
        An iterable of permutations (as lists) acting on the component indices
        reindex to 0..|C|-1.
    """
    if component.leaves:  # Direct permutations by element type
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
    else:  # Core or non-leaf peripheral component
        return _component_autogroup(
            adj, orbit_ids, component.idxs
        ).generate_schreier_sims(af=True)


# ----------------------------
# Scoring Helpers
# ----------------------------


def _aligned_rmsd_for_candidate(
    a_geom: npt.NDArray[np.float64],
    b_geom: npt.NDArray[np.float64],
    p_opt: npt.NDArray[np.int64],
    a_comp_idxs: npt.NDArray[np.int64],
    b_comp_perm: npt.NDArray[np.int64],
    backend: str = "kabsch",
) -> float:
    """Compute RMSD after aligning A to B using fixed + component indices."""
    # score on fixed ∪ C, with a per-candidate temporary alignment
    a_fixed_idxs = np.flatnonzero(p_opt != -1)
    a_fixed_plus_component = np.concatenate([a_fixed_idxs, a_comp_idxs])
    b_fixed_plus_component = np.concatenate([p_opt[a_fixed_idxs], b_comp_perm])
    return _compute_rmsd(
        a_geom[a_fixed_plus_component],
        b_geom[b_fixed_plus_component],
        align=True,
        backend=backend,
    )


def _align_fixed(
    a_coords: np.ndarray,
    b_coords: np.ndarray,
    p_opt: np.ndarray,
    alignment_backend: str = "kabsch",
) -> np.ndarray:
    """Return a copy of `a_coords` aligned to `b_coords` using fixed pairs."""
    fixed = p_opt != -1

    if alignment_backend == "kabsch":
        R, cA, cB = kabsch(a_coords[fixed], b_coords[p_opt[fixed]])
    elif alignment_backend == "qcp":
        R, cA, cB = qcp_rotation(a_coords[fixed], b_coords[p_opt[fixed]])
    else:
        raise ValueError(f"Unknown alignment backend '{alignment_backend}'.")

    return (a_coords - cA) @ R.T + cB  # align all of A


# ----------------------------
# Public API
# ----------------------------


def snap_rmsd_align_assign(
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
    units: str | LengthUnit = "bohr",
) -> tuple[float, Structure, list[int]]:
    # TODO: Add compute_connectivity_if_missing parameter?
    """
    Compute snapRMSD(a_struct -> b_struct). Returns (rmsd, a_aligned, p_opt).

    If connectivity for the structures is not already present, it will be computed using covalent radii heuristics.
    The structures must have the same connectivity to be compared.

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
        a_struct: First structure.
        b_struct: Second structure.
        factor_depth: The depth of peripheral groups to factor (default is 1). A depth of
            1 factors only singly attached peripheral groups such as methyl and methylene
            hydrogens and halides. A depth of 2 would also factor their parents (e.g.,
            factor the three methyl groups on a tert-butyl substituent) and factor
            cyclic peripheral groups such as phenyl rings. Higher depths are possible but
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
        units: The units of the input structures' geometry and the output RMSD. Options are
            'bohr' (default) or 'angstrom'.


    Returns:
        (float, a_aligned, p_opt): The computed snapRMSD value, an aligned a_struct,
            and the optimal permutation as a list of indices for a_struct onto b_struct.
    """
    # --- 1. Build graphs, canonical label map, compute automorphism group -----------
    # NOTE: Currently running nauty 3x on A and 2x on B. Can optimize?
    #   - Once per graph for isomorphism check
    #   - Once per graph for canonical labels
    #   - Once for A to get automorphism group

    try:
        units = LengthUnit(units)
    except ValueError:
        raise ValueError(
            f"Invalid units '{units}'. Valid options are 'bohr' or 'angstrom'."
        )
    # --- Graphs & isomorphism check
    a_adj, b_adj = adjacency_dict(a_struct), adjacency_dict(b_struct)
    if not a_adj or not b_adj:
        raise ValueError("Structures have no bonded connectivity.")

    a_graph = _to_pynauty_graph_adj_color(a_adj, a_coloring or a_struct.symbols)
    b_graph = _to_pynauty_graph_adj_color(b_adj, b_coloring or b_struct.symbols)

    if not pn.isomorphic(a_graph, b_graph):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    # --- Canonical labels & initial map
    canonical_map_ab = np.array(
        _canonical_map_pn(a_graph, b_graph), dtype=int
    )  # i_A -> i_B
    p_opt = np.full(len(a_struct.symbols), -1, dtype=int)

    # --- Compute automorphism group of A
    _a_gens, _a_mantissa, _a_exponent, a_orbit_ids, _a_orbit_count = pn.autgrp(a_graph)
    a_geom = a_struct.geometry  # Will not mutate a_struct.geometry
    b_geom = b_struct.geometry  # Will not mutate b_struct.geometry
    # Orbit ids for B via canonical A->B map
    b_orbit_ids = [0] * len(a_orbit_ids)
    for i_a, i_b in enumerate(canonical_map_ab):
        b_orbit_ids[i_b] = a_orbit_ids[i_a]

    # --- 2. Factor batches ----------------------------------------------------------
    if factoring is not None:
        # TODO: Possibly implement normalization for the simplified BatchLike format
        batches = factoring
    elif factor_depth == 0:  # Whole structure as core
        idxs = np.array(range(len(a_struct.symbols)))
        batches = [[Component(parent_idx=None, leaves=False, idxs=idxs)]]
    else:
        batches = _factor_plan_recursive(a_adj, depth=factor_depth, color=a_orbit_ids)

    _validate_factoring(batches)
    b_comp_by_parent = _build_b_component_lookup(batches, canonical_map_ab)

    # --- 3. Main control loop -----------------------------------------------------
    for b_idx, batch in enumerate(batches):
        for c_idx, component in enumerate(batch):
            best_rmsd = float("inf")
            best_perm: list[int] | None = None
            a_comp_idxs = component.idxs

            # Get corresponding component indices in B for this component
            p_idx = component.parent_idx
            if p_idx is None or p_opt[p_idx] == canonical_map_ab[p_idx]:  # Core component or parent not remapped  # fmt: skip
                b_comp_idxs = canonical_map_ab[a_comp_idxs]  # Initial canonical map still valid # fmt: skip
            elif component.leaves:  # Remapped leaf component
                b_idxs = b_comp_by_parent[p_opt[p_idx], component.leaves]
                b_comp_idxs = _match_leaf_component(
                    a_comp_idxs, b_idxs, a_orbit_ids, b_orbit_ids
                )
            else:  # Remapped branch/cycle peripheral component
                b_idxs = b_comp_by_parent[p_opt[p_idx], component.leaves]
                b_comp_idxs = _match_structured_component(a_comp_idxs, b_idxs, a_adj, b_adj, a_orbit_ids, b_orbit_ids)  # fmt: skip

            count = 0
            for perm in _iter_component_perms(a_adj, a_orbit_ids, component):
                count += 1
                b_comp_perm = b_comp_idxs[perm]

                if align and (align_per_component or (b_idx == 0 and c_idx == 0)):
                    # Align per-candidate; use fixed + component indices
                    score = _aligned_rmsd_for_candidate(a_geom, b_geom, p_opt, a_comp_idxs, b_comp_perm, backend=alignment_backend)  # fmt: skip
                else:
                    # No alignment; direct RMSD on component given established frame
                    score = _compute_rmsd(a_geom[a_comp_idxs], b_geom[b_comp_perm], align=False)  # fmt: skip

                if score < best_rmsd:
                    best_rmsd, best_perm = score, perm

            # Lock-in best permutation for this component
            p_opt[a_comp_idxs] = b_comp_idxs[best_perm]
            count = 0  # Reset count for next component

            # Per component and/or first component alignment (establishes frame)
            if align and (align_per_component or (b_idx == 0 and c_idx == 0)):
                a_geom = _align_fixed(a_geom, b_geom, p_opt)

        # Realign after each batch; avoid duplicate alignments
        if align and not align_per_component and not (b_idx == 0 and c_idx == 0):
            a_geom = _align_fixed(a_geom, b_geom, p_opt)

    assert np.all(p_opt != -1), "Not all indices got fixed!."  # Sanity check while debugging # fmt: skip
    assert all(v == 1 for v in Counter(p_opt).values()), "p_opt is not a valid permutation!"  # Sanity check while debugging # fmt: skip
    # Align false because already aligned at the end of the last batch if needed
    final_rmsd = _compute_rmsd(a_geom, b_geom[p_opt], align=False)
    if units == LengthUnit.ANGSTROM:
        final_rmsd = final_rmsd * constants.BOHR_TO_ANGSTROM
        
    # Create aligned a Structure
    a_aligned = a_struct.model_copy(update={"geometry": a_geom}, deep=True)
    return final_rmsd, a_aligned, [int(x) for x in p_opt]


def snap_rmsd(a_struct: Structure, b_struct: Structure, **kwargs) -> float:
    """Thin convenience wrapper around :func:`snap_rmsd_align_assign` to return only RMSD."""
    rmsd_val, _, _ = snap_rmsd_align_assign(a_struct, b_struct, **kwargs)
    return rmsd_val


def snap_align(
    a_struct: Structure,
    b_struct: Structure,
    **kwargs,
) -> Structure:
    """Thin convenience wrapper around :func:`snap_rmsd_align_assign` to return only aligned structure."""  # fmt: skip
    _, a_aligned, _ = snap_rmsd_align_assign(a_struct, b_struct, **kwargs)
    return a_aligned


def snap_rmsd_align(
    a_struct: Structure,
    b_struct: Structure,
    **kwargs,
) -> tuple[float, Structure]:
    """Thin convenience wrapper around :func:`snap_rmsd_align_assign` to return RMSD and aligned structure."""  # fmt: skip
    rmsd, a_aligned, _ = snap_rmsd_align_assign(a_struct, b_struct, **kwargs)
    return rmsd, a_aligned


def snap_assign(
    a_struct: Structure,
    b_struct: Structure,
    **kwargs,
) -> list[int]:
    """Thin convenience wrapper around :func:`snap_rmsd_align_assign` to return only assignment."""
    _, _, p_opt = snap_rmsd_align_assign(a_struct, b_struct, **kwargs)
    return p_opt
