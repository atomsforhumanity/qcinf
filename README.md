# qcinf

[![image](https://img.shields.io/pypi/v/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![image](https://img.shields.io/pypi/l/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![image](https://img.shields.io/pypi/pyversions/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![CI](https://github.com/atomsforhumanity/qcinf/actions/workflows/ci.yaml/badge.svg)](https://github.com/atomsforhumanity/qcinf/actions/workflows/ci.yaml)

Cheminformatics algorithms and structure utilities for [qcdata](https://github.com/atomsforhumanity/qcdata) structures.

## The QC Suite

These packages work together to make quantum chemistry fast, structured, and interoperable.

- [qcref](https://github.com/atomsforhumanity/qcref) - Reference data for computational chemistry: CODATA physical constants, conversion factors, a periodic table, and solvent properties, each value with its source.
- [qcdata](https://github.com/atomsforhumanity/qcdata) - Data structures for quantum chemistry inputs and results, with Jupyter visualizations. [Documentation](https://qcdata.docs.afh.org)
- [qcinf](https://github.com/atomsforhumanity/qcinf) - Cheminformatics algorithms and structure utilities for [qcdata](https://github.com/atomsforhumanity/qcdata) structures.
- [qccodec](https://github.com/atomsforhumanity/qccodec) - Translates [qcdata](https://github.com/atomsforhumanity/qcdata) inputs into native program inputs and parses program outputs back into qcdata objects, one codec per program.
- [qccompute](https://github.com/atomsforhumanity/qccompute) - Runs quantum chemistry programs on [qcdata](https://github.com/atomsforhumanity/qcdata) inputs using [qccodec](https://github.com/atomsforhumanity/qccodec) codecs and returns structured results. [Documentation](https://qccompute.docs.afh.org)
- [BigChem](https://github.com/mtzgroup/bigchem) - Distributed execution of qccompute calculations across clusters of computers or the cloud.
- [ChemCloud](https://github.com/mtzgroup/chemcloud-client) - A [web application](https://github.com/mtzgroup/chemcloud-server) and [Python client](https://github.com/mtzgroup/chemcloud-client) that expose a BigChem cluster securely over the internet.

## Installation

```bash
python -m pip install qcinf
```

To install with numba-accelerated kernels:

```bash
python -m pip install 'qcinf[fast]'
```

Additional backends can be added for conversion and compatibility features, including
`rdkit`, `openbabel`, or install the whole suite with `all`:

```bash
python -m pip install 'qcinf[rdkit,openbabel]'
```

Or

```bash
python -m pip install 'qcinf[all]'
```

## snapRMSD

snapRMSD is the native `qcinf` algorithm for symmetry-aware molecular RMSD. It
uses bonded connectivity to find a connectivity-preserving atom assignment between
two isomorphic structures, optionally aligns the structures, and returns the RMSD
for the best assignment.

```python
from qcdata import Structure
from qcinf import snap_rmsd

reference = Structure(
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

candidate = Structure(
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

rmsd = snap_rmsd(reference, candidate)
```

For the full result, use `snap_rmsd_align_assign`:

```python
from qcinf import snap_rmsd_align_assign

rmsd, aligned_reference, assignment = snap_rmsd_align_assign(reference, candidate)
```

`assignment[i]` is the atom index in `candidate` matched to atom `i` in
`reference`. The returned `aligned_reference` is a copy of `reference` with its
geometry aligned to `candidate`.

Common options:

```python
from qcinf import snap_align, snap_assign, snap_rmsd

# Compare structures without rotational/translational alignment.
rmsd = snap_rmsd(reference, candidate, align=False)

# Increase factoring depth for larger repeated peripheral groups.
rmsd = snap_rmsd(reference, candidate, factor_depth=2)

# Use the QCP alignment backend. Install with `qcinf[fast]` for numba acceleration.
rmsd = snap_rmsd(reference, candidate, alignment_backend="qcp")

# Return only the aligned structure or atom assignment.
aligned = snap_align(reference, candidate)
assignment = snap_assign(reference, candidate)
```

If a `Structure` does not include bonded connectivity, snapRMSD will infer it
from covalent radii. For production workflows, pass validated connectivity on the
structures or through `a_connectivity=` and `b_connectivity=` so the graph used for
assignment is explicit and reproducible.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, the pre-commit workflow, running tests, and the release process.

## Support

If you have any issues with `qcinf` or would like to request a feature, please open an [issue](https://github.com/atomsforhumanity/qcinf/issues).
