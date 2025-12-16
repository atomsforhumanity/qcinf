from qcio import Structure

from qcinf import determine_connectivity
from qcinf.algorithms.connectivity_helpers import _canonical_map_pn, _to_pynauty_graph


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
