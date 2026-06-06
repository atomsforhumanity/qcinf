import numpy as np
import pytest
from qcdata import Structure

from qcinf.algorithms import snap


def _component_sets(components: list[snap.Component]) -> set[tuple[int | None, bool, tuple[int, ...]]]:
    """Normalize Component objects for order-independent assertions."""
    return {
        (
            c.parent_idx,
            c.leaves,
            tuple(int(i) for i in np.asarray(c.idxs, dtype=int)),
        )
        for c in components
    }


def test_prune_graph_removes_nodes_and_backrefs() -> None:
    adj = {
        0: {1, 2},
        1: {0, 3},
        2: {0},
        3: {1},
    }
    snap._prune_graph(adj, keep={0, 2})
    assert adj == {0: {2}, 2: {0}}


def test_rx_graph_from_adj_preserves_original_vertex_ids_in_payload() -> None:
    # Non-contiguous ids to ensure payload translation is actually used.
    adj = {
        10: {20},
        20: {10, 30},
        30: {20},
    }
    g = snap._rx_graph_from_adj(adj)

    assert g.num_nodes() == 3
    assert g.num_edges() == 2

    payload_ids = {g[i] for i in range(g.num_nodes())}
    assert payload_ids == {10, 20, 30}

    # edge_list returns rustworkx indices; translate back using payloads.
    translated_edges = {
        tuple(sorted((g[u], g[v]))) for u, v in g.edge_list()
    }
    assert translated_edges == {(10, 20), (20, 30)}


def test_cycles_returns_original_vertex_ids_only_for_cycle_subgraph() -> None:
    # Triangle 10-20-30-10 plus a tail 30-40 and acyclic path 50-60.
    adj = {
        10: {20, 30},
        20: {10, 30},
        30: {10, 20, 40},
        40: {30},
        50: {60},
        60: {50},
    }
    cycles = snap._cycles(adj)
    assert cycles == [{10, 20, 30}]


def test_cycles_empty_for_tree() -> None:
    adj = {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2},
    }
    assert snap._cycles(adj) == []


def test_outside_components_uses_original_ids() -> None:
    adj = {
        10: {20, 30},
        20: {10, 30},
        30: {10, 20, 40},
        40: {30, 50},
        50: {40},
        90: {91},
        91: {90},
    }
    outside = snap._outside_components(adj, excluded={10, 20, 30})
    assert {frozenset(c) for c in outside} == {frozenset({40, 50}), frozenset({90, 91})}


def test_absorb_components_absorbs_single_touch_only() -> None:
    # Two cycle blobs.
    cycles = [{0, 1, 2}, {10, 11, 12}]
    # Outside CCs: one touches first cycle only, one touches none, one touches both.
    outside = [{3, 4}, {6, 7}, {5}]
    adj = {
        0: {1, 2, 3},
        1: {0, 2},
        2: {0, 1},
        10: {11, 12, 5},
        11: {10, 12},
        12: {10, 11},
        3: {0, 4},
        4: {3},
        5: {10, 2},  # touches both cycle blobs
        6: {7},
        7: {6},
    }

    new_cycles, remaining = snap._absorb_components(adj, cycles, outside)
    assert {frozenset(c) for c in new_cycles} == {
        frozenset({0, 1, 2, 3, 4}),
        frozenset({10, 11, 12}),
    }
    assert {frozenset(c) for c in remaining} == {frozenset({5}), frozenset({6, 7})}


def test_determine_core_and_factor_cycles_factors_when_parent_in_core() -> None:
    # One 3-cycle attached to a larger non-cyclic remainder through node 3.
    adj = {
        0: {1, 2, 3},
        1: {0, 2},
        2: {0, 1},
        3: {0, 4},
        4: {3, 5},
        5: {4, 6},
        6: {5},
    }
    cycles = [{0, 1, 2}]
    remaining = [{3, 4, 5, 6}]  # larger than cycle, so cycle not absorbed in pass 1a

    core_idxs, components = snap._determine_core_and_factor_cycles(adj, cycles, remaining)
    assert core_idxs == {3, 4, 5, 6}
    assert _component_sets(components) == {(3, False, (0, 1, 2))}


def test_determine_core_and_factor_cycles_ring_to_ring_absorbs_to_core() -> None:
    # Two same-size rings connected by one edge; no non-cyclic remainder.
    adj = {
        0: {1, 2, 10},
        1: {0, 2},
        2: {0, 1},
        10: {11, 12, 0},
        11: {10, 12},
        12: {10, 11},
    }
    cycles = [{0, 1, 2}, {10, 11, 12}]
    remaining: list[set[int]] = []

    core_idxs, components = snap._determine_core_and_factor_cycles(adj, cycles, remaining)
    assert core_idxs == {0, 1, 2, 10, 11, 12}
    assert components == []


def test_peel_cycles_integration() -> None:
    # Same graph as factorable case above.
    adj = {
        0: {1, 2, 3},
        1: {0, 2},
        2: {0, 1},
        3: {0, 4},
        4: {3, 5},
        5: {4, 6},
        6: {5},
    }
    core_idxs, components = snap._peel_cycles(adj)
    # Outside chain single-touches the cycle, so it is absorbed into the cycle blob.
    # With one resulting cycle blob and no remainder, the implementation keeps all in core.
    assert core_idxs == set(adj)
    assert components == []


def test_peel_symmetric_leaves_groups_only_duplicated_colors() -> None:
    # Parent 0 has leaves 1/2 (same color) and 3 (different color).
    adj = {
        0: {1, 2, 3, 4},
        1: {0},
        2: {0},
        3: {0},
        4: {0, 5},
        5: {4},
    }
    color = [9, 1, 1, 2, 7, 8]

    core_idxs, components = snap._peel_symmetric_leaves(adj, color)
    assert core_idxs == {0, 3, 4, 5}
    assert _component_sets(components) == {(0, True, (1, 2))}


def test_peel_symmetric_branches_factors_duplicates_when_parent_retains_degree() -> None:
    # Parent 0 has two identical 2-node branches: 0-1-2 and 0-3-4.
    # Leaves 5 and 7 stay, so remaining degree after peel is 2 (allowed).
    adj = {
        0: {1, 3, 5, 7},
        1: {0, 2},
        2: {1},
        3: {0, 4},
        4: {3},
        5: {0},
        7: {0},
    }
    color = [0] * 8

    core_idxs, components = snap._peel_symmetric_branches(adj, color=color)
    assert core_idxs == {0, 5, 7}
    assert _component_sets(components) == {(0, False, (1, 2, 3, 4))}


def test_peel_symmetric_branches_parent_local_rollback() -> None:
    # Parent 0 has four factorable branches; peeling all leaves no neighbors => rollback.
    adj = {
        0: {1, 3, 5, 7},
        1: {0, 2},
        2: {1},
        3: {0, 4},
        4: {3},
        5: {0, 6},
        6: {5},
        7: {0, 8},
        8: {7},
    }
    color = [0] * 9

    core_idxs, components = snap._peel_symmetric_branches(adj, color=color)
    assert core_idxs == set(adj)
    assert components == []


def test_factor_plan_recursive_depth_zero_keeps_initial_leaf_peel() -> None:
    adj_dict = {
        0: [1],
        1: [0, 2],
        2: [1],
    }
    batches = snap._factor_plan_recursive(adj_dict, depth=0, color=[0, 0, 0])

    # Internal behavior: the initial leaf peel runs before recursion, even at depth=0.
    assert len(batches) == 2
    core = batches[0][0]
    np.testing.assert_array_equal(core.idxs, np.array([1]))
    assert _component_sets(batches[1]) == {(1, True, (0, 2))}


def test_factor_plan_recursive_depth_one_peels_initial_leaf_layer() -> None:
    # 0 connected to leaves 1,2 (duplicate color) and to node 3 with leaves 4,5 (duplicate color).
    adj_dict = {
        0: [1, 2, 3],
        1: [0],
        2: [0],
        3: [0, 4, 5],
        4: [3],
        5: [3],
    }
    color = [9, 1, 1, 8, 2, 2]

    batches = snap._factor_plan_recursive(adj_dict, depth=1, color=color)
    assert len(batches) == 2

    core = batches[0][0]
    np.testing.assert_array_equal(core.idxs, np.array([0, 3]))

    leaf_components = batches[1]
    assert _component_sets(leaf_components) == {
        (0, True, (1, 2)),
        (3, True, (4, 5)),
    }


def test_validate_factoring_raises_on_missing_parent() -> None:
    batches = [
        [snap.Component(parent_idx=None, leaves=False, idxs=np.array([0]))],
        [snap.Component(parent_idx=99, leaves=True, idxs=np.array([1]))],
    ]
    with pytest.raises(ValueError, match="appears before its parent"):
        snap._validate_factoring(batches)


def test_validate_factoring_raises_on_duplicate_idx() -> None:
    batches = [
        [snap.Component(parent_idx=None, leaves=False, idxs=np.array([0]))],
        [snap.Component(parent_idx=0, leaves=True, idxs=np.array([0]))],
    ]
    with pytest.raises(ValueError, match="Duplicate index"):
        snap._validate_factoring(batches)


def test_build_b_component_lookup_uses_parent_mapping_and_leaf_flag() -> None:
    batches = [
        [snap.Component(parent_idx=None, leaves=False, idxs=np.array([0, 1]))],
        [
            snap.Component(parent_idx=0, leaves=True, idxs=np.array([2, 3])),
            snap.Component(parent_idx=1, leaves=False, idxs=np.array([4])),
        ],
    ]
    canonical_map_ab = np.array([10, 11, 12, 13, 14])

    lookup = snap._build_b_component_lookup(batches, canonical_map_ab)

    assert set(lookup.keys()) == {(None, False), (10, True), (11, False)}
    np.testing.assert_array_equal(lookup[(None, False)], np.array([10, 11]))
    np.testing.assert_array_equal(lookup[(10, True)], np.array([12, 13]))
    np.testing.assert_array_equal(lookup[(11, False)], np.array([14]))


def test_match_leaf_component_aligns_positionally_by_orbit() -> None:
    a_comp = np.array([0, 1, 2, 3], dtype=np.int64)
    b_leaf = np.array([7, 6, 5, 4], dtype=np.int64)

    # A orbit ids in component order: [1,2,1,2]
    a_orbit_ids = [1, 2, 1, 2]
    # B orbit ids for global indices 4..7: 4->2, 5->1, 6->2, 7->1
    b_orbit_ids = [0, 0, 0, 0, 2, 1, 2, 1]

    out = snap._match_leaf_component(a_comp, b_leaf, a_orbit_ids, b_orbit_ids)
    np.testing.assert_array_equal(out, np.array([7, 6, 5, 4], dtype=np.int64))


def test_match_structured_component_raises_on_orbit_histogram_mismatch() -> None:
    a_comp = np.array([0, 1], dtype=np.int64)
    b_comp = np.array([10, 11], dtype=np.int64)
    a_adj = {0: [1], 1: [0]}
    b_adj = {10: [11], 11: [10]}
    a_orbit_ids = [1, 1]
    b_orbit_ids = [0] * 12
    b_orbit_ids[10] = 1
    b_orbit_ids[11] = 2

    with pytest.raises(ValueError, match="orbit id histograms do not match"):
        snap._match_structured_component(
            a_comp,
            b_comp,
            a_adj,
            b_adj,
            a_orbit_ids,
            b_orbit_ids,
        )


def test_component_autogroup_identity_when_no_symmetry() -> None:
    adj = {
        0: [1],
        1: [0, 2],
        2: [1],
    }
    comp = np.array([0, 1, 2], dtype=np.int64)
    orbit_ids = [10, 11, 12]  # unique labels remove automorphism

    G = snap._component_autogroup(adj, orbit_ids, comp)
    assert G.order() == 1


def test_component_autogroup_detects_reflection_symmetry() -> None:
    # Path 0-1-2 with endpoint colors equal -> reflection symmetry of size 2.
    adj = {
        0: [1],
        1: [0, 2],
        2: [1],
    }
    comp = np.array([0, 1, 2], dtype=np.int64)
    orbit_ids = [1, 2, 1]

    G = snap._component_autogroup(adj, orbit_ids, comp)
    assert G.order() == 2


def test_iter_component_perms_for_leaf_component_groups_by_orbit() -> None:
    component = snap.Component(parent_idx=0, leaves=True, idxs=np.array([10, 11, 12]))
    orbit_ids = [0] * 13
    orbit_ids[10] = 7
    orbit_ids[11] = 7
    orbit_ids[12] = 9

    perms = list(snap._iter_component_perms(adj={0: [10, 11, 12]}, orbit_ids=orbit_ids, component=component))
    assert {tuple(p) for p in perms} == {(0, 1, 2), (1, 0, 2)}


def test_iter_component_perms_for_non_leaf_uses_automorphism_group() -> None:
    adj = {
        0: [1],
        1: [0, 2],
        2: [1],
    }
    component = snap.Component(parent_idx=None, leaves=False, idxs=np.array([0, 1, 2]))
    orbit_ids = [1, 2, 1]

    perms = list(snap._iter_component_perms(adj=adj, orbit_ids=orbit_ids, component=component))
    assert {tuple(p) for p in perms} == {(0, 1, 2), (2, 1, 0)}


def test_aligned_rmsd_for_candidate_zero_for_consistent_mapping() -> None:
    b_geom = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 1.3, 0.1],
        ]
    )
    a_geom = b_geom.copy()

    p_opt = np.array([-1, 1, -1], dtype=np.int64)  # index 1 is already fixed
    a_comp = np.array([0, 2], dtype=np.int64)
    b_comp_perm = np.array([0, 2], dtype=np.int64)

    rmsd = snap._aligned_rmsd_for_candidate(a_geom, b_geom, p_opt, a_comp, b_comp_perm)
    assert rmsd == pytest.approx(0.0, abs=1e-12)


def test_align_fixed_aligns_rotated_and_translated_coordinates() -> None:
    b_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # Rotate +90 deg around z then translate by [3, -2, 4]
    R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = np.array([3.0, -2.0, 4.0])
    a_coords = b_coords @ R.T + t

    p_opt = np.array([0, 1, 2], dtype=np.int64)
    aligned = snap._align_fixed(a_coords, b_coords, p_opt, alignment_backend="kabsch")
    np.testing.assert_allclose(aligned, b_coords, atol=1e-10)


def test_align_fixed_raises_on_unknown_backend() -> None:
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    p = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="Unknown alignment backend"):
        snap._align_fixed(a, b, p, alignment_backend="nope")


def test_snap_api_wrappers_consistent_on_identity_case(water) -> None:
    rmsd, aligned, p_opt = snap.snap_rmsd_align_assign(water, water, factor_depth=1)

    assert rmsd == pytest.approx(0.0, abs=1e-8)
    assert sorted(p_opt) == list(range(len(water.symbols)))
    np.testing.assert_allclose(aligned.geometry, water.geometry, atol=1e-6)

    assert snap.snap_rmsd(water, water, factor_depth=1) == pytest.approx(rmsd)
    np.testing.assert_allclose(
        snap.snap_align(water, water, factor_depth=1).geometry,
        aligned.geometry,
        atol=1e-6,
    )

    aligned2, rmsd2 = snap.snap_align_rmsd(water, water, factor_depth=1)
    assert rmsd2 == pytest.approx(rmsd)
    np.testing.assert_allclose(aligned2.geometry, aligned.geometry, atol=1e-6)

    assert snap.snap_assign(water, water, factor_depth=1) == p_opt


def test_snap_rmsd_align_assign_handles_peripheral_symmetry_methane() -> None:
    # CH4 has a core atom (C) with four equivalent peripheral leaves (H).
    methane_a = Structure(
        symbols=["C", "H", "H", "H", "H"],
        geometry=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        connectivity=[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
    )
    # Same geometry but with hydrogens permuted in index order.
    methane_b = Structure(
        symbols=["C", "H", "H", "H", "H"],
        geometry=[
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
        connectivity=[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
    )

    # Factoring plan should isolate the core and one peripheral leaf component.
    batches = snap._factor_plan_recursive(
        adj_dict={0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]},
        depth=1,
        color=[0, 1, 1, 1, 1],
    )
    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0][0].idxs, np.array([0]))
    assert _component_sets(batches[1]) == {(0, True, (1, 2, 3, 4))}

    rmsd, aligned, p_opt = snap.snap_rmsd_align_assign(
        methane_a, methane_b, factor_depth=1
    )
    assert rmsd == pytest.approx(0.0, abs=1e-8)
    # Carbon must map to carbon; hydrogens map bijectively among hydrogen indices.
    assert p_opt[0] == 0
    assert set(p_opt[1:]) == {1, 2, 3, 4}
    np.testing.assert_allclose(
        aligned.geometry, methane_b.geometry[p_opt], atol=1e-6
    )


def test_snap_rmsd_align_assign_invalid_units_raises(water) -> None:
    with pytest.raises(ValueError, match="Invalid units"):
        snap.snap_rmsd_align_assign(water, water, units="nanometer")


def test_snap_rmsd_align_assign_non_isomorphic_raises() -> None:
    water = Structure(
        symbols=["O", "H", "H"],
        geometry=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0]],
        connectivity=[(0, 1, 1.0), (0, 2, 1.0)],
    )
    linear = Structure(
        symbols=["O", "H", "H"],
        geometry=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [3.6, 0.0, 0.0]],
        connectivity=[(0, 1, 1.0), (1, 2, 1.0)],
    )

    with pytest.raises(ValueError, match="Structures not isomorphic"):
        snap.snap_rmsd_align_assign(water, linear, factor_depth=1)
