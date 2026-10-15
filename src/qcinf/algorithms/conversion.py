"""Top-level functions for structure conversion algorithms."""

from collections.abc import Callable
from typing import Any

from qcdata import Structure

from qcinf._backends import openbabel, rdkit

_SMILES_TO_STRUCTURE_BACKENDS: dict[str, Callable[..., Structure]] = {
    "rdkit": rdkit._smiles_to_structure_rdkit,
    "openbabel": openbabel._smiles_to_structure_ob,
}


def smiles_to_structure(
    smiles: str,
    *,
    backend: str = "rdkit",
    force_field: str = "MMFF94",
    **struct_kwargs: Any,
) -> Structure:
    """Convert a SMILES string to a Structure object.

    Args:
        smiles: The SMILES string to convert.
        backend: The backend to use for the conversion. Can be 'rdkit' or 'openbabel'.
        force_field: The force field to use for geometry embedding. Defaults to
            "MMFF94". Options are "UFF", "MMFF94", and "MMFF94s" for rdkit and
            "UFF", "MMFF94" and "GHEMICAL" for openbabel.
        **struct_kwargs: Additional keyword arguments to pass to the Structure
            constructor such as multiplicity, extras, identifiers, etc.

    Returns:
        A Structure object.

    Raises:
        ValueError: If the backend is not one of the supported options.
    """
    try:
        fn = _SMILES_TO_STRUCTURE_BACKENDS[backend.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{backend}'.  Known: {_SMILES_TO_STRUCTURE_BACKENDS.keys()}"
        ) from e
    return fn(smiles, force_field=force_field, **struct_kwargs)


_SMILES_BACKENDS: dict[str, Callable[..., str]] = {
    "rdkit": rdkit._structure_to_smiles_rdkit,
    "openbabel": openbabel._structure_to_smiles_ob,
}


def structure_to_smiles(
    struct: Structure,
    *,
    backend: str = "rdkit",
    hydrogens: bool = False,
    options: dict[str, Any] | None = None,
) -> str:
    """Convert a [`qcdata.Structure`][qcdata.Structure] into a canonical SMILES string.

    Args:
        struct: The Structure object to convert.
        backend: The backend to use for the conversion. Can be 'rdkit' or 'openbabel'.
            Defaults to 'rdkit'.
        hydrogens: If True, include explicit hydrogens in the SMILES string.
        options: A dictionary of backend-specific keywords.  See
            `qcinf._backends.<backend>._structure_to_smiles` for details.

    Returns:
        Canonical SMILES str.
    """
    opts = options or {}
    try:
        fn = _SMILES_BACKENDS[backend.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{backend}'.  Known: {_SMILES_BACKENDS.keys()}"
        ) from e
    return fn(struct, hydrogens=hydrogens, **opts)  # type: ignore[arg-type]
