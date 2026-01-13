import numpy as np
from qcio import Structure

from qcinf import determine_connectivity
from qcinf.algorithms import snap
from qcinf.algorithms.connectivity_helpers import (
    _canonical_map_pn,
    _to_pynauty_graph,
    _to_pynauty_graph_adj_color,
)


def test_determine_connectivity_water():
    """Test connectivity determination on a water molecule."""
    struct = Structure(symbols=["O", "H", "H"], geometry=[[0.0, 0.0, 0.0], [1.83, 0.0, 0.0], [0.0, 1.83, 0.0]])  # fmt: skip
    connectivity = determine_connectivity(struct)

    expected_connectivity = [(0, 1, 1.0), (0, 2, 1.0)]  # O-H bonds
    assert connectivity == expected_connectivity, (
        f"Expected {expected_connectivity}, got {connectivity}"
    )


def test_determine_connectivity_single_atom():
    """Test connectivity determination on a single atom (no bonds)."""
    struct = Structure(symbols=["He"], geometry=[[0.0, 0.0, 0.0]])
    connectivity = determine_connectivity(struct)
    assert connectivity == []


def test_determine_connectivity_unconnected():
    """Test connectivity determination on unconnected atoms."""
    struct = Structure(
        symbols=["He", "Ne"], geometry=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    )
    connectivity = determine_connectivity(struct)
    assert connectivity == []


def test_canonical_map_pn_identity_on_identical_graphs() -> None:
    """
    _canonical_map_pn(A, A) should return identity when graphs are identical
    (same labeling + same coloring).
    """
    # simple path 0-1-2-3 with unique colors
    adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    colors = [10, 11, 12, 13]

    A = _to_pynauty_graph_adj_color(adj, colors)
    M = _canonical_map_pn(A, A)

    assert M == [0, 1, 2, 3]


def test_canonical_map_pn():
    """Test canonical mapping between two isomorphic graphs."""
    s1 = Structure(
        symbols=["C", "C", "O"],
        geometry=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]],
        connectivity=[(0, 1, 1.0), (1, 2, 2.0)],
    )
    s2 = Structure(
        symbols=["C", "O", "C"],
        geometry=[[2.8, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        connectivity=[(2, 1, 1.0), (0, 1, 2.0)],
    )

    A = _to_pynauty_graph(s1)
    B = _to_pynauty_graph(s2)

    mapping = _canonical_map_pn(A, B)
    expected_mapping = [2, 0, 1]  # C(0)->C(2), C(1)->C(0), O(2)->O(1)
    assert mapping == expected_mapping, f"Expected {expected_mapping}, got {mapping}"


def test_induced_local_adj_respects_nodes_ordering_contract() -> None:
    """
    induced_local_adj must assign local indices in the order of `nodes`.
    """
    # Global graph:
    adj: dict[int, list[int]] = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2, 4],
        4: [3],
    }

    nodes = np.array([3, 1, 4], dtype=np.int64)  # local order: 3->0, 1->1, 4->2
    local_adj, g2l = snap._induced_local_adj(adj, nodes)

    assert g2l == {3: 0, 1: 1, 4: 2}

    # Induced edges among {3,1,4}: 3-1, 3-4
    # local 0 (gv 3) -> [local of 1, local of 4] = [1,2]
    # local 1 (gv 1) -> [local of 3] = [0]
    # local 2 (gv 4) -> [local of 3] = [0]
    assert local_adj[0] == [1, 2]
    assert local_adj[1] == [0]
    assert local_adj[2] == [0]


def test_match_structured_component_aligns_B_to_A_order() -> None:
    """
    _match_structured_component should return B_C aligned positionally to A_C.

    We build two isomorphic induced subgraphs with different global labels.
    Colors are unique per vertex to remove automorphism ambiguity.
    """
    # Component in A: a 4-node path 0-1-2-3 (embedded in a larger graph, but irrelevant)
    A_adj: dict[int, list[int]] = {
        0: [1, 9],  # extra neighbor 9 outside component
        1: [0, 2],
        2: [1, 3],
        3: [2, 8],  # extra neighbor 8 outside component
        8: [3],
        9: [0],
    }

    # Same component structure in B but with different global labels: 10-11-12-13
    B_adj: dict[int, list[int]] = {
        10: [11, 99],
        11: [10, 12],
        12: [11, 13],
        13: [12, 98],
        98: [13],
        99: [10],
    }

    # The A component is ordered as [0,1,2,3]
    A_C = np.array([0, 1, 2, 3], dtype=np.int64)

    # The corresponding B nodes are the same subgraph, but we hand them in a scrambled order
    B_idxs = np.array([12, 10, 13, 11], dtype=np.int64)

    # Give each component vertex a unique color class, consistent across A and B by role.
    # Here we encode "position along the path" as the color.
    # A: 0->100, 1->101, 2->102, 3->103
    # B: 10->100, 11->101, 12->102, 13->103
    A_coloring = [0] * (max(A_adj) + 1)
    B_coloring = [0] * (max(B_adj) + 1)

    A_coloring[0] = 100
    A_coloring[1] = 101
    A_coloring[2] = 102
    A_coloring[3] = 103

    B_coloring[10] = 100
    B_coloring[11] = 101
    B_coloring[12] = 102
    B_coloring[13] = 103

    B_C = snap._match_structured_component(
        A_C, B_idxs, A_adj, B_adj, A_coloring, B_coloring
    )
    # Expected: aligned by role/color, i.e. [10,11,12,13] in the order of A_C=[0,1,2,3]
    assert np.array_equal(B_C, np.array([10, 11, 12, 13], dtype=np.int64))
