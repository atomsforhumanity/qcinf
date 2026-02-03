import hashlib
import struct
from collections import defaultdict
from typing import Any

import pynauty as pn
from qcio import Structure


def _list_of_sets(coloring: list[Any]) -> list[set[Any]]:
    """Convert coloring list to pynauty list of sets format.

    !!! Important
        The order of the sets in the output list MUST be consistent between graphs.
        For this reason we sort the color keys before constructing the output list.


    >>> _list_of_sets(["H","O","C","H","C","C"])
    [{2, 4, 5}, {0, 3}, {1}]  # In alphabetical order: C, H, O

    >>> _list_of_sets([2,1,0,1,0,0])
    [{2, 4, 5}, {1, 3}, {0}]  # In numerical order: 0, 1, 2
    """
    buckets: dict[int, set[int]] = defaultdict(set)
    for vertex_idx, color in enumerate(coloring):
        buckets[color].add(vertex_idx)
    ordered_keys = sorted(buckets.keys())
    return [buckets[key] for key in ordered_keys]


def _to_pynauty_graph(struct: Structure, coloring: list[Any] | None = None) -> pn.Graph:
    """
    Convert a Structure's and optional coloring list to a pynauty Graph.

    Parameters:
        struct: The Structure to convert.
        coloring: Optional list of color IDs for each vertex, e.g., [0,0,0,3,3,3]
            indicates atoms 0-2 are of one color and atoms 3-5 are a second color.
            If no value is provided the Structure's symbols array is used as the
            default coloring.
    Returns:
        The corresponding pynauty Graph.
    """
    if coloring is None:
        coloring = struct.symbols
    adj_dict = struct.adjacency_dict
    if not adj_dict:
        # TODO: Add parameter `compute_connectivity_if_missing: bool = True`
        raise ValueError("Structure has no connectivity defined.")

    return _to_pynauty_graph_adj_color(adj_dict, coloring)


def _to_pynauty_graph_adj_color(
    adj_dict: dict[int, list[int]], coloring: list[Any]
) -> pn.Graph:
    """
    Convert an adjacency dictionary and coloring list to a pynauty Graph.

    Parameters:
        adj_dict: Adjacency dictionary of the graph, e.g. {0: [1,2], 1: [0,3], ...}.
            Numbering of vertices must be 0 to N-1 for N vertices.
        coloring: List of color IDs for each vertex, e.g., [0,0,0,3,3,3]
            indicates vertices 0-2 are of one color and vertices 3-5 are a second color.
            Using the structure's symbols array can be used as a default coloring.
    Returns:
        The corresponding pynauty Graph.
    """
    return pn.Graph(
        number_of_vertices=len(adj_dict.keys()),
        adjacency_dict=adj_dict,
        vertex_coloring=_list_of_sets(coloring),
    )


def _canonical_map_pn(
    A: pn.Graph,
    B: pn.Graph,
) -> list[int]:
    """Generate a mapping of vertices from A -> B using canonical labels.
    The final mapping M satisfies M[i_A] = i_B.

    Note:
        pynauty's canonical labels map canon -> original index, meaning that
        canX[i_canon] = i_orig.
    """
    canA, canB = pn.canon_label(A), pn.canon_label(B)
    invA = [0] * len(canA)  # inv(can1): orig1 -> canon
    for i_canon, i_orig1 in enumerate(canA):
        invA[i_orig1] = i_canon
    return [canB[invA[i_orig1]] for i_orig1 in range(len(canA))]


def _get_branch_info(
    adj: dict[int, set[int]],
    parent: int,
    root: int,
    color: list[int],
) -> tuple[set[int], bytes]:
    """
    Compute (branch_vertices, rooted AHU signature) for the branch hanging off `parent`
    through neighbor `root`.

    This is intended to be called on the *full remaining graph* adjacency (not just the branch).
    We restrict traversal to the branch component by forbidding traversal into `parent`.

    The rooted signature is AHU-style for rooted colored trees:
        sig(v) = H( color[v], multiset(sig(children)) )
    with the tree rooted at `root` (away from `parent`).

    Notes:
      - Uses global `color` labels directly (e.g., orbit IDs or packed labels).
      - Safe for deep trees: iterative traversal, no recursion.
      - Robust to accidental non-tree structure: uses `visited` to avoid revisits.
        (If cycles exist, the signature is no longer a canonical GI-invariant, but the
         traversal will not blow up.)

    Parameters:
        adj: Adjacency dict for the current remaining graph.
        parent: The attachment vertex to forbid (the high-degree parent off which this branch hangs).
        root: The neighbor of `parent` that is the entrypoint into this branch.
        color: Global per-vertex labels used as node colors.

    Returns:
        branch_vertices, root_signature
            branch_vertices is the vertex set of the connected component containing `root` in
        G \\ {parent}. root_signature is a deterministic 16-byte digest.
    """
    # Discovery: build parent pointers + DFS order, restricted to the branch component.
    visited: set[int] = set()
    parent_of: dict[int, int] = {root: -1}
    order: list[int] = []

    stack: list[tuple[int, int]] = [(root, -1)]  # (node, parent_in_rooted_tree)

    while stack:
        v, p = stack.pop()
        if v == parent:
            continue
        if v in visited:
            continue
        visited.add(v)
        parent_of[v] = p
        order.append(v)

        for nbr in adj[v]:
            if nbr == parent:
                continue
            if nbr == p:
                continue
            if nbr in visited:
                continue
            stack.append((nbr, v))

    # Build children lists for bottom-up AHU hashing.
    children_map: dict[int, list[int]] = {v: [] for v in visited}
    for v, p in parent_of.items():
        if p != -1:
            children_map[p].append(v)

    # Bottom-up hashing in reverse DFS order.
    node_to_sig: dict[int, bytes] = {}

    for v in reversed(order):
        child_sigs = [node_to_sig[c] for c in children_map[v]]
        child_sigs.sort()

        h = hashlib.blake2b(digest_size=16)
        h.update(struct.pack("<q", int(color[v])))
        for cs in child_sigs:
            h.update(cs)

        node_to_sig[v] = h.digest()

    return visited, node_to_sig[root]
