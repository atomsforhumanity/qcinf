# Quantum Chemistry Informatics (qcinf)

[![image](https://img.shields.io/pypi/v/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![image](https://img.shields.io/pypi/l/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![image](https://img.shields.io/pypi/pyversions/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![Actions status](https://github.com/coltonbh/qcinf/workflows/Tests/badge.svg)](https://github.com/coltonbh/qcinf/actions)
[![Actions status](https://github.com/coltonbh/qcinf/workflows/Basic%20Code%20Quality/badge.svg)](https://github.com/coltonbh/qcinf/actions)

Cheminformatics algorithms and structure utilities using standardized [qcdata](https://qcdata.coltonhicks.com/) data structures.

## The QC Suite of Programs

`qcinf` works in harmony with a suite of other quantum chemistry tools for fast, structured, and interoperable quantum chemistry.

- [qcconst](https://github.com/coltonbh/qcconst) - Physical constants, conversion factors, and a periodic table with clear source information for every value.
- [qcdata](https://github.com/coltonbh/qcdata) - Elegant and intuitive data structures for quantum chemistry, featuring seamless Jupyter Notebook visualizations. [Documentation](https://qcdata.coltonhicks.com)
- [qcinf](https://github.com/coltonbh/qcinf) - Cheminformatics algorithms and structure utilities using standardized [qcdata](https://qcdata.coltonhicks.com/) data structures.
- [qccodec](https://github.com/coltonbh/qccodec) - A package for translating between standardized [qcdata](https://github.com/coltonbh/qcdata) data structures and native QC program inputs and outputs.
- [qcop](https://github.com/coltonbh/qcop) - A package for operating quantum chemistry programs using standardized [qcdata](https://qcdata.coltonhicks.com/) data structures. Compatible with `TeraChem`, `psi4`, `QChem`, `NWChem`, `ORCA`, `Molpro`, `geomeTRIC` and many more.
- [BigChem](https://github.com/mtzgroup/bigchem) - A distributed application for running quantum chemistry calculations at scale across clusters of computers or the cloud. Bring multi-node scaling to your favorite quantum chemistry program.
- `ChemCloud` - A [web application](https://github.com/mtzgroup/chemcloud-server) and associated [Python client](https://github.com/mtzgroup/chemcloud-client) for exposing a BigChem cluster securely over the internet.

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

## Development

Install the package with all optional dependencies:

```bash
uv sync --all-extras
```

Run tests to ensure your setup is working correctly:

```bash
bash scripts/tests.sh
```

## Support

If you have any issues with `qcinf` or would like to request a feature, please open an [issue](https://github.com/coltonbh/qcinf/issues).
