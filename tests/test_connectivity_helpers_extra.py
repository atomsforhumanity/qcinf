from qcinf.algorithms.connectivity_helpers import _get_branch_info, _list_of_sets


def test_list_of_sets_sorts_color_keys_for_stable_order() -> None:
    coloring = ["H", "O", "C", "H", "C", "C"]
    out = _list_of_sets(coloring)
    # Sorted key order: C, H, O
    assert out == [{2, 4, 5}, {0, 3}, {1}]


def test_get_branch_info_returns_branch_vertices_and_stable_signature() -> None:
    # Parent 0 with two equivalent branches rooted at 1 and 3.
    adj = {
        0: {1, 3, 9},
        1: {0, 2},
        2: {1},
        3: {0, 4},
        4: {3},
        9: {0},
    }
    color = [0] * 10

    verts1, sig1 = _get_branch_info(adj, parent=0, root=1, color=color)
    verts2, sig2 = _get_branch_info(adj, parent=0, root=3, color=color)

    assert verts1 == {1, 2}
    assert verts2 == {3, 4}
    assert isinstance(sig1, bytes)
    assert len(sig1) == 16
    assert sig1 == sig2


def test_get_branch_info_changes_signature_when_colors_differ() -> None:
    adj = {
        0: {1, 3},
        1: {0, 2},
        2: {1},
        3: {0, 4},
        4: {3},
    }
    color: list[int] = [0] * 5
    color[2] = 7  # only one branch leaf differs

    _, sig1 = _get_branch_info(adj, parent=0, root=1, color=color)
    _, sig2 = _get_branch_info(adj, parent=0, root=3, color=color)

    assert sig1 != sig2
