"""Open Babel backend (wrapper functions) for qcinf."""

import functools
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import qcconst.periodic_table as pt
from qcconst import constants
from qcdata import Structure
from typing_extensions import ParamSpec

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ModuleNotFoundError as _e:
    _OB_ERR: Exception | None = _e
else:
    _OB_ERR = None

P = ParamSpec("P")
R = TypeVar("R")


def requires_openbabel(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that raises a clean error *at call-time* if Open Babel
    isn't installed.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
        if _OB_ERR is not None:
            raise ModuleNotFoundError(
                "Optional dependency 'openbabel' is not installed. "
                "Install with one of:\n"
                "    python -m pip install 'qcinf[openbabel]'\n"
                "    python -m pip install 'qcinf[all]'"
            ) from _OB_ERR
        return func(*args, **kwargs)

    return wrapper


# --- main wrapper functions for Open Babel --- #


@requires_openbabel
def _smiles_to_structure_ob(
    smiles: str, force_field: str = "mmff94", **struct_kwargs
) -> Structure:
    """Convert a SMILES string to a Structure object.

    Args:
        smiles: The SMILES string to convert.
        force_field: The force field to use for geometry embedding. Defaults to
            "mmff94". Options are "uff", "mmff94", and "ghemical".
        **struct_kwargs: Additional keyword arguments to pass to the Structure
            constructor such as multiplicity, extras, identifiers, etc.

    Returns:
        A Structure object.

    Raises:
        ValueError: If the force field is not one of the supported options.
        AssertionError: If the conversion to an Open Babel Mol fails.
    """
    # Validate force field
    force_field = force_field.lower()
    if force_field not in ["uff", "mmff94", "ghemical"]:
        raise ValueError(
            f"Invalid force field '{force_field}'. "
            "Options are 'uff', 'mmff94', and 'ghemical'."
        )

    # Convert SMILES to Open Babel Mol object
    mol = pybel.readstring("smi", smiles)
    if mol is None:
        raise ValueError(
            f"Failed to convert SMILES to Open Babel Mol: {smiles}. "
            "Please check the SMILES string for validity."
        )

    # Add hydrogens
    mol.addh()  # type: ignore

    # Generate 3D coordinates and optimize
    mol.make3D(forcefield=force_field, steps=250)  # type: ignore

    # Get atom symbols
    atoms = [pt.number(atom.atomicnum).symbol for atom in mol.atoms]  # type: ignore

    # Get atom positions
    geometry_angstrom = np.array([atom.coords for atom in mol.atoms])  # type: ignore # noqa: E501
    geometry_bohr = geometry_angstrom * constants.ANGSTROM_TO_BOHR

    # Get canonical SMILES
    canonical_smiles = mol.write("can").strip()  # type: ignore

    # Get charge
    charge = mol.charge  # type: ignore

    # Connectivity
    connectivity = []
    for bond in ob.OBMolBondIter(mol.OBMol):  # type: ignore
        begin_idx = bond.GetBeginAtomIdx() - 1  # Convert to 0-based index
        end_idx = bond.GetEndAtomIdx() - 1  # Convert to 0-based index
        # Treat aromatic bonds explicitly as 1.5; otherwise use the integer order
        order = 1.5 if bond.IsAromatic() else float(bond.GetBondOrder())
        connectivity.append((begin_idx, end_idx, float(order)))

    return Structure(
        symbols=atoms,
        geometry=geometry_bohr,
        charge=charge,
        identifiers={
            "canonical_smiles": canonical_smiles,
            "smiles": smiles,
            "canonical_smiles_program": "openbabel",
        },
        connectivity=connectivity,
        **struct_kwargs,
    )


@requires_openbabel
def _structure_to_smiles_ob(struct: Structure, *, hydrogens: bool = False) -> str:
    """Convert a Structure to a SMILES string.

    Args:
        struct: The Structure object to convert.
        hydrogens: Whether to include explicit hydrogens in the SMILES string.

    Returns:
        A canonical SMILES string.
    """
    # Must remove data in second line for Open Babel
    xyz_lines = struct.to_xyz().splitlines()
    xyz_lines[1] = ""

    # Create Open Babel OBMol object
    mol = pybel.readstring("xyz", "\n".join(xyz_lines))

    # Assign charges
    partial_charges = mol.calccharges()

    # Check if the sum of the partial charges matches the total charge
    if sum(partial_charges) != struct.charge:
        raise ValueError(
            f"Charge mismatch. Open Babel: {sum(partial_charges)} vs Structure: "
            f"{struct.charge}"
        )

    # Set the formal charges on the atoms
    if sum(partial_charges) != 0:
        for atom, charge in zip(mol.atoms, partial_charges):
            atom.OBAtom.SetFormalCharge(int(round(charge)))

    # Ensure the total charge matches the structure
    if mol.charge != struct.charge:
        raise ValueError(
            f"Charge mismatch. Open Babel: {sum(partial_charges)} vs Structure: "
            f"{struct.charge}"
        )

    # Create an OBConversion object to handle output format
    conv = ob.OBConversion()
    conv.SetOutFormat("can")

    if hydrogens:
        conv.AddOption("h", ob.OBConversion.OUTOPTIONS)

    # Generate canonical SMILES with explicit hydrogens
    return conv.WriteString(mol.OBMol).strip()


# --- internal helper functions for Open Babel --- #
