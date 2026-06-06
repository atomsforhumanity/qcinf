from qcinf import (
    compute_rmsd,
    rmsd,
    snap_align,
    snap_align_rmsd,
    snap_assign,
    snap_rmsd,
    snap_rmsd_align_assign,
)
from qcinf.algorithms.geometry import compute_rmsd as algorithms_compute_rmsd
from qcinf.algorithms.geometry import rmsd as algorithms_rmsd
from qcinf.algorithms.snap import (
    snap_align as algorithms_snap_align,
)
from qcinf.algorithms.snap import (
    snap_align_rmsd as algorithms_snap_align_rmsd,
)
from qcinf.algorithms.snap import (
    snap_assign as algorithms_snap_assign,
)
from qcinf.algorithms.snap import (
    snap_rmsd as algorithms_snap_rmsd,
)
from qcinf.algorithms.snap import (
    snap_rmsd_align_assign as algorithms_snap_rmsd_align_assign,
)


def test_top_level_exports_snap_functions() -> None:
    assert snap_rmsd is algorithms_snap_rmsd
    assert snap_align is algorithms_snap_align
    assert snap_align_rmsd is algorithms_snap_align_rmsd
    assert snap_assign is algorithms_snap_assign
    assert snap_rmsd_align_assign is algorithms_snap_rmsd_align_assign


def test_top_level_exports_compute_rmsd_and_legacy_alias() -> None:
    assert compute_rmsd is algorithms_compute_rmsd
    assert rmsd is algorithms_rmsd
