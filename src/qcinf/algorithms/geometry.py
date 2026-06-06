"""Top-level functions for geometry-related algorithms."""

from collections.abc import Callable

from qcdata import Structure

from qcinf._backends import rdkit

from . import snap
from .geometry_kernels import _compute_rmsd

_RMSD_BACKEND_MAP: dict[str, Callable[..., float]] = {
    "qcinf": snap.snap_rmsd,
    "rdkit": rdkit._rmsd_rdkit,
}


def rmsd(
    struct1: Structure,
    struct2: Structure,
    *,
    backend: str = "qcinf",
    **kwargs,
) -> float:
    """Calculate the root mean square deviation (RMSD) between two structures.

    Args:
        struct1: The first structure.
        struct2: The second structure.
        backend: The backend to use for the RMSD calculation. Can be 'qcinf' or'rdkit'.
        **kwargs: Backend-specific additional keywords to pass to the RMSD calculation
            function. This can include options like 'symmetry' for symmetry-based RMSD
            calculations. The specific options depend on the backend used. See
            [qcinf._backends.<backend>._rmsd_<backend>] for details.

    Returns:
        The RMSD value between the two structures.

    Raises:
        ValueError: If the backend is not one of the supported options.
    """
    if backend == "qcinf":
        if kwargs.get("symmetry", False):
            # Use snap_rmsd
            return snap.snap_rmsd(struct1, struct2, **kwargs)
        else:
            # Assume atom mapping is known and use basic RMSD calculation
            return _compute_rmsd(struct1.geometry, struct2.geometry, **kwargs)

    try:
        fn = _RMSD_BACKEND_MAP[backend.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{backend}'.  Known: {list(_RMSD_BACKEND_MAP)}"
        ) from e
    return fn(struct1, struct2, **kwargs)


_ALIGN_BACKEND_MAP: dict[str, Callable[..., tuple[Structure, float]]] = {
    "rdkit": rdkit._align_rdkit,
    "qcinf": snap.snap_align_rmsd,
}


def align(
    struct: Structure,
    refstruct: Structure,
    *,
    backend: str = "rdkit",
    symmetry: bool = False,
    **kwargs,
) -> tuple[Structure, float]:
    """Align a structure to a reference structure.

    Args:
        struct: The structure to align.
        refstruct: The reference structure to align against.
        backend: The backend to use for the alignment. Can be 'rdkit'.
        symmetry: Whether to use symmetry in the alignment. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the alignment function.

    Returns:
        A tuple containing the RMSD and the aligned structure.
    """
    try:
        fn = _ALIGN_BACKEND_MAP[backend.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{backend}'.  Known: {list(_ALIGN_BACKEND_MAP)}"
        ) from e
    return fn(struct, refstruct, symmetry=symmetry, **kwargs)


def filter_conformers_indices(
    conformers: list[Structure],
    threshold: float = 1.0,
    backend: str = "rdkit",
    **rmsd_kwargs,
) -> list[int]:
    """Filter conformers based on RMSD threshold.

    Args:
        conformers: A list of Structure objects to filter
        threshold: The RMSD threshold for filtering in Bohr. Defaults to 1.0 Bohr
            (0.53 Angstrom).
        backend: The backend to use for the RMSD calculation. Can be 'rdkit'.
        **rmsd_kwargs: Additional keyword arguments to pass to the RMSD calculation
            function. This can include options like 'symmetry' for symmetry-based RMSD
            calculations. The specific options depend on the backend used. See
            [qcinf._backends.<backend>._rmsd_<backend>] for details.

    Returns:
        List of conformers indices that meet the RMSD threshold.
    """
    filtered = set()
    for i in range(len(conformers)):
        if i not in filtered:
            for j in range(i + 1, len(conformers)):
                if (
                    rmsd(
                        conformers[i],
                        conformers[j],
                        backend=backend,
                        **rmsd_kwargs,
                    )
                    < threshold
                ):
                    filtered.add(j)

    return [i for i in range(len(conformers)) if i not in filtered]


def filter_conformers(
    conformers: list[Structure],
    threshold: float = 1.0,
    backend: str = "rdkit",
    **rmsd_kwargs,
) -> list[Structure]:
    """Filter conformers based on RMSD threshold.

    Args:
        conformers: A list of Structure objects to filter
        threshold: The RMSD threshold for filtering in Bohr. Defaults to 1.0 Bohr
            (0.53 Angstrom).
        backend: The backend to use for the RMSD calculation. Can be 'rdkit'.
        **rmsd_kwargs: Additional keyword arguments to pass to the RMSD calculation
            function. This can include options like 'symmetry' for symmetry-based RMSD
            calculations. The specific options depend on the backend used. See
            [qcinf._backends.<backend>._rmsd_<backend>] for details.

    Returns:
        A list of filtered conformers that meet the RMSD threshold.
    """
    keep_indices = filter_conformers_indices(
        conformers, threshold=threshold, backend=backend, **rmsd_kwargs
    )
    return [conformers[i] for i in keep_indices]
