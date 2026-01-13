import numpy as np

from qcinf.algorithms.snap import _match_pool_component


def test_match_pool_by_color():
    pool_a = np.array([0, 1, 2])
    pool_b = np.array([3, 4, 5])

    A_coloring = [0, 1, 0, 2, 3, 4]
    B_coloring = [2, 3, 4, 1, 0, 0]

    B_C = _match_pool_component(pool_a, pool_b, A_coloring, B_coloring)
    np.testing.assert_array_equal(B_C, np.array([4, 3, 5]))
