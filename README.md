# qcinf

[![image](https://img.shields.io/pypi/v/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![image](https://img.shields.io/pypi/l/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![image](https://img.shields.io/pypi/pyversions/qcinf.svg)](https://pypi.python.org/pypi/qcinf)
[![CI](https://github.com/atomsforhumanity/qcinf/actions/workflows/ci.yaml/badge.svg)](https://github.com/atomsforhumanity/qcinf/actions/workflows/ci.yaml)

Cheminformatics algorithms and structure utilities for [qcdata](https://github.com/atomsforhumanity/qcdata) structures.

## The QC Suite

These packages work together to make quantum chemistry fast, structured, and interoperable.

- [qcref](https://github.com/atomsforhumanity/qcref) - Reference data for computational chemistry: CODATA physical constants, conversion factors, a periodic table, and solvent properties, each value with its source.
- [qcdata](https://github.com/atomsforhumanity/qcdata) - Data structures for quantum chemistry inputs and results, with Jupyter visualizations. [Documentation](https://qcdata.docs.afh.org/next/)
- [qcinf](https://github.com/atomsforhumanity/qcinf) - Cheminformatics algorithms and structure utilities for [qcdata](https://github.com/atomsforhumanity/qcdata) structures.
- [qccompute](https://github.com/atomsforhumanity/qccompute) - Runs programs on [qcdata](https://github.com/atomsforhumanity/qcdata) inputs, one calculation or many, in this process or in workers, with an optional store, and returns structured results. [Documentation](https://qccompute.docs.afh.org/next/)
- [qccodec](https://github.com/atomsforhumanity/qccodec) - The optional add-on for external programs (ORCA, Psi4, xtb, CREST, ...): writes their native inputs, runs them and parses their outputs, registering each as a program qccompute runs. [Documentation](https://qccodec.docs.afh.org/next/)
- [qcstore](https://github.com/atomsforhumanity/qcstore) - Keeps results under their calculation key: immutable runs, tags, and an optional index.
- [BigChem](https://github.com/mtzgroup/bigchem) - Distributed execution of qccompute calculations across clusters of computers or the cloud.
- [ChemCloud](https://github.com/mtzgroup/chemcloud-client) - A [web application](https://github.com/mtzgroup/chemcloud-server) and [Python client](https://github.com/mtzgroup/chemcloud-client) that expose a BigChem cluster securely over the internet.

## Installation

```bash
python -m pip install qcinf
```

installs the stable release, which works with qcdata 0.23. This README describes the
next release, which is in alpha and works with qcdata's alpha; install them pinned
(`--prerelease=allow` lets the resolver take pre-releases):

```bash
uv pip install --prerelease=allow "qcdata==0.24.0a6" "qcinf==0.6.0a1"
```

The extras below go on the pinned requirement the same way (`"qcinf[fast]==0.6.0a1"`).
qcinf has no documentation site of its own; the qcdata structures it works on are
documented at [qcdata.docs.afh.org/next](https://qcdata.docs.afh.org/next/) for the alpha.

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

## Molden files

`qcinf.molden` writes a qcdata `Wavefunction` as a Molden file, for orbital viewers
(Molden, Jmol, Avogadro, IboView, Multiwfn) and programs that read the format, and reads
Molden files back. The wavefunction needs its orbital coefficients and basis set, which
programs return on request (the output `orbitals`).

```python
from qcinf import molden

text = molden.write(output.results.wavefunction, output.input_data.structure)
structure, wavefunction = molden.read(text)
```

Files are in atomic units (`[Atoms] AU`, energies in hartree) and qcdata's
`MOLDEN_CONVENTION`: spherical functions as `[5D]`/`[7F]`/`[9G]` when the basis set
has them (`cartesian=True` writes every shell as Cartesian functions), restricted
closed-shell orbitals as one set with occupations 0 to 2, and open shells as alpha and
beta sets with occupations 0 to 1. `read` returns the orbitals in the file's convention;
`wavefunction.with_convention(...)` converts them. Files from xtb, tblite and the g-xTB
program, which write the contraction coefficients of bare primitives, read with
`molden.read(text, normalized_primitives=False)`. The
[module documentation](https://github.com/atomsforhumanity/qcinf/blob/next/src/qcinf/molden.py) lists every convention.

## Contributing

See [CONTRIBUTING.md](https://github.com/atomsforhumanity/qcinf/blob/next/CONTRIBUTING.md) for development setup, the pre-commit workflow, running tests, and the release process.

## Support

If you have any issues with `qcinf` or would like to request a feature, please open an [issue](https://github.com/atomsforhumanity/qcinf/issues).
