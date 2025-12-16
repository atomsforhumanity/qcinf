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

    return _to_pynauty_graph_adj_color(adj_dict, struct.symbols)


def _to_pynauty_graph_adj_color(
    adj_dict: dict[int, list[int]], coloring: list[Any]
) -> pn.Graph:
    """
    Convert an adjacency dictionary and coloring list to a pynauty Graph.

    Parameters:
        adj_dict: Adjacency dictionary of the graph, e.g. {0: [1,2], 1: [0,3], ...}.
        coloring: List of color IDs for each vertex, e.g., [0,0,0,3,3,3]
            indicates atoms 0-2 are of one color and atoms 3-5 are a second color.
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
