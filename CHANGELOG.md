# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added

- Native snapRMSD support for symmetry-aware, connectivity-preserving molecular
  RMSD calculations [#11](https://github.com/coltonbh/qcinf/pull/11) by [@coltonbh](https://github.com/coltonbh):

  - Added public APIs `snap_rmsd`, `snap_rmsd_align_assign`, `snap_align`,
    `snap_align_rmsd`, and `snap_assign`.
  - Added graph factoring support for core/peripheral decomposition, including
    symmetric leaves, repeated branches, and peripheral cyclic components.
  - Added canonical graph mapping and automorphism-based assignment helpers for
    matching isomorphic molecular graphs while preserving atom colors and bonded
    connectivity.
  - Added support for inferred connectivity when a `Structure` does not already
    include bonded connectivity, with optional explicit `a_connectivity` and
    `b_connectivity` overrides.
  - Added Kabsch and QCP alignment backend support within snapRMSD scoring and
    alignment workflows.

- Kabsch algorithm for determining the optimal rotation matrix to align two structures and compute an optimally aligned RMSD.
- High-performance Quaternion Characteristic Polynomial (QCP) RMSD/alignment backend as an alternative using a numba-compiled kernel:

  - Implements Theobald’s direct RMSD evaluation and Liu’s adjoint quaternion method for efficient rotation extraction.
  - Provides `qcp_rotation_and_rmsd`, `qcp_rotation`, and `qcp_rmsd` as interfaces to the high performance numba kernel.
  - Achieves ~15x performance improvements over the pure-Python Kabsch implementation while maintaining numerical correctness (the ~15x speedup is basically constant across structure size since the speedup is relative to a constant factor of SVD on a 3x3 matrix).
  - Graceful fallback mechanism:
    - If numba is unavailable or fails to import, the QCP functions automatically fall back to a Kabsch-based implementation with a clear runtime warning.
    - Added optional dependency group `qcinf[fast]` to install numba acceleration.

- New utility functions:

  - `rotation_matrix(axis, angle_deg)` for constructing analytic rotation matrices.
  - `rotate_structure(struct, axis, angle_deg)` for generating rotated structures.

- `compute_connectivity` function that computes `Structure` connectivity using covalent radii and a scaling factor. Bond order is set to 1.0 for all bonds since this routine only infers connectivity.
  ```python
  >>> from qcinf import compute_connectivity, smiles_to_structure
  >>> struct = smiles_to_structure("O")
  >>> compute_connectivity(struct, cov_factor=1.2) # cov_factor is optional and 1.2 by default
  [(0, 1, 1.0), (0, 2, 1.0)]
  ```

### Changed

- Renamed the top-level `rmsd` function to `compute_rmsd` to avoid conflicts
  with common local variable names. The old `rmsd` name remains as a deprecated
  alias that emits a `FutureWarning` and delegates to `compute_rmsd`.
- Updated `filter_conformers_indices` to call `compute_rmsd` internally.
- Updated the README with snapRMSD usage examples and guidance for connectivity,
  assignment, alignment, factoring depth, and the QCP alignment backend.

## [0.2.1] - 2025-10-27

### Added

- `Structure.connectivity` information when creating a `Structure` using `smiles_to_structure`. [#7](https://github.com/coltonbh/qcinf/pull/7)

## [0.2.0] - 2025-10-07

### Fixed

- `RDKit` backend raising `NameError` because `Mol` object was not defined if `rdkit` was not installed. With deferred evaluation it now correctly raises the `ModuleNotFoundError` when top-level functions try to use `rdkit` if it's not installed. [#6](https://github.com/coltonbh/qcinf/pull/6)

### Changed

- 🚨 Dropped python 3.9 support. Minimum supported version is now 3.10.

### Added

- Python 3.10-3.14 test matrix for GitHub Workflows.

## [0.1.1] - 2025-06-01

### Added

- `filter_conformers_indices` function to better serve the existing requirements of `qcdata.view` setup.

## [0.1.0] - 2025-06-01

### Added

- Setup all DevOps workflows and basic package setup.
- Copied over all cheminformatics functions (e.g., `rmsd`, `align`, `filter_conformers` (formerly `ConformerSearchResults.conformers_filtered()`), `Structure.from_smiles()`, `Structure.to_smiles()`, etc.) from `qcdata` into this repo.

[unreleased]: https://github.com/coltonbh/qcinf/compare/0.2.1...HEAD
[0.2.1]: https://github.com/coltonbh/qcinf/releases/tag/0.2.1
[0.2.0]: https://github.com/coltonbh/qcinf/releases/tag/0.2.0
[0.1.1]: https://github.com/coltonbh/qcinf/releases/tag/0.1.1
[0.1.0]: https://github.com/coltonbh/qcinf/releases/tag/0.1.0
