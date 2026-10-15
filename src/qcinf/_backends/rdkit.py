"""RDKit backend (wrapper functions) for qcinf."""

import functools
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from qcdata import LengthUnit, Structure
from qcref import constants
from typing_extensions import ParamSpec

from .utils import mute_c_stderr

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Mol, rdDetermineBonds, rdMolAlign
except ModuleNotFoundError as _e:
    _RDKIT_ERR: Exception | None = _e
else:
    _RDKIT_ERR = None


P = ParamSpec("P")
R = TypeVar("R")


def requires_rdkit(func: Callable[P, R]) -> Callable[P, R]:
    """Raise a clean error *at call-time* if RDKit isn't installed.

    Args:
        func: The function that requires RDKit.

    Returns:
        A wrapped function that raises ``ModuleNotFoundError`` when called if RDKit
        could not be imported, and otherwise calls ``func`` unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
        if _RDKIT_ERR is not None:
            raise ModuleNotFoundError(
                "Optional dependency 'rdkit' is not installed. "
                "Install with: python -m pip install 'qcinf[rdkit]' or "
                "python -m pip install 'qcinf[all]'."
            ) from _RDKIT_ERR
        return func(*args, **kwargs)

    return wrapper


# --- main wrapper functions for rdkit --- #


@requires_rdkit
def _rmsd_rdkit(
    struct1: Structure,
    struct2: Structure,
    *,
    symmetry: bool = True,
    numthreads: int = 1,
    use_hueckel: bool = True,
    use_vdw: bool = False,
    cov_factor: float = 1.3,
    length_unit: LengthUnit = LengthUnit.BOHR,
) -> float:
    """Calculate the root mean square deviation between two structures in Bohr or Angstrom.

    May lead to a 'combinatorial explosion' if symmetry=True and many molecule symmetries
    (e.g., many hydrogens) are present. If this function is taking a long time to
    compute, consider passing `symmetry=False` to disable symmetry consideration. Or
    pass `numthreads=an_integer` to increase the number of threads used for the
    symmetry consideration step.

    Args:
        struct1: The first structure.
        struct2: The second structure.
        symmetry: Whether to consider symmetries in the structures before calculating
            the RMSD, i.e., to allow atom renumbering. This relies on the  RDKit
            `DetermineConnectivity` and `GetBestRMS` functions. If False, the RMSD is
            calculated with alignment but without considering symmetry, i.e., naively
            assuming the atoms are already correctly indexed across structures.
        numthreads: The number of threads to use for the RMSD calculation. Applies only
            to the alignment step if `symmetry=True`.
        use_hueckel: Whether to use Hueckel method when determining connectivity.
            Applies only to `symmetry=True`.
        use_vdw: Whether to use Van der Waals radii when determining connectivity.
            Applies only to `symmetry=True`.
        cov_factor: The scaling factor for the covalent radii when determining
            connectivity. Applies only to `symmetry=True`.
        length_unit: The unit of length to use for the RMSD calculation. Default is
            "bohr". If "angstrom", the RMSD will be in Angstroms.

    Returns:
        The RMSD between the two structures in Angstroms.
    """
    # Create RDKit molecules
    mol1 = _structure_to_rdkit_mol(struct1)
    mol2 = _structure_to_rdkit_mol(struct2)

    # Compute RMSD
    if symmetry:
        # Determine connectivity
        _determine_connectivity_rdkit(
            mol1,
            charge=struct1.charge,
            use_hueckel=use_hueckel,
            use_vdw=use_vdw,
            cov_factor=cov_factor,
        )

        _determine_connectivity_rdkit(
            mol2,
            charge=struct2.charge,
            use_hueckel=use_hueckel,
            use_vdw=use_vdw,
            cov_factor=cov_factor,
        )
        # Take symmetry into account, align the two molecules, compute RMSD
        try:
            rmsd = rdMolAlign.GetBestRMS(mol2, mol1, numThreads=numthreads)
        except RuntimeError as e:  # Possible failure to make substructure match
            try:  # Swap the order of the molecules and try again.
                rmsd = rdMolAlign.GetBestRMS(mol1, mol2, numThreads=numthreads)
            except RuntimeError:  # If it fails again, raise the original error
                raise e from None

    else:  # Do not take symmetry into account. Structs aligned by atom index.
        rmsd = rdMolAlign.AlignMol(mol2, mol1)

    return rmsd * constants.ANGSTROM_TO_BOHR if length_unit == LengthUnit.BOHR else rmsd


@requires_rdkit
def _align_rdkit(
    struct: Structure,
    refstruct: Structure,
    symmetry: bool = True,
    use_hueckel: bool = True,
    use_vdw: bool = False,
    cov_factor: float = 1.3,
    length_unit: LengthUnit = LengthUnit.BOHR,
) -> tuple[Structure, float]:
    """Align a structure to a reference structure and compute their RMSD.

    Returns a new structure that is optimally aligned to the reference structure
    and the RMSD between the two structures in Bohr or Angstroms.

    May lead to a 'combinatorial explosion' especially if many molecule symmetries
    (e.g., many hydrogens) are present.If this function is taking a long time to
    compute, consider passing `symmetry=False` to disable symmetry consideration.

    Args:
        struct: The structure to align.
        refstruct: The reference structure.
        symmetry: Whether to consider symmetries in the structures before aligning and
            calculating the RMSD, i.e., to allow atom renumbering. This relies on the
            RDKit's `GetBestAlignmentTransform` method and `GetBestRMS` functions. If False, the RMSD is
            calculated with alignment but without considering symmetry, i.e., naively
            assuming the atoms are already correctly indexed across structures.
        use_hueckel: Whether to use Hueckel method when determining connectivity.
            Applies only to `best=True`.
        use_vdw: Whether to use Van der Waals radii when determining connectivity.
            Applies only to `best=True`.
        cov_factor: The scaling factor for the covalent radii when determining
            connectivity. Applies only to `best=True`.
        length_unit: The unit of length to use for the RMSD calculation. Default is
            "bohr". If "angstrom", the RMSD will be in Angstroms.

    Returns:
        Tuple of the aligned structure and the RMSD, in Bohr by default or in
        Angstroms if ``length_unit`` is ``LengthUnit.ANGSTROM``.
    """
    # Create RDKit molecules
    mol = _structure_to_rdkit_mol(struct)
    refmol = _structure_to_rdkit_mol(refstruct)

    # Determine connectivity
    _determine_connectivity_rdkit(
        mol,
        charge=struct.charge,
        use_hueckel=use_hueckel,
        use_vdw=use_vdw,
        cov_factor=cov_factor,
    )
    _determine_connectivity_rdkit(
        refmol,
        charge=refstruct.charge,
        use_hueckel=use_hueckel,
        use_vdw=use_vdw,
        cov_factor=cov_factor,
    )

    # Align mol to refmol and compute RMSD
    if symmetry:
        rmsd_val, trnsfm_matrix, atm_map = rdMolAlign.GetBestAlignmentTransform(
            mol, refmol
        )
    else:
        rmsd_val, trnsfm_matrix = rdMolAlign.GetAlignmentTransform(mol, refmol)

    # Convert to homogeneous coordinates in Angstroms
    coords_homogeneous = np.hstack(
        [struct.geometry_angstrom, np.ones((struct.geometry.shape[0], 1))]
    )

    # Apply the transformation matrix
    transformed_coords_homogeneous = coords_homogeneous @ trnsfm_matrix.T

    # Extract the transformed 3D coordinates and convert to Bohr
    transformed_coords = (
        transformed_coords_homogeneous[:, :3] * constants.ANGSTROM_TO_BOHR
    )

    # Reorder the atoms to match the reference structure
    if symmetry:
        if Counter(struct.symbols) != Counter(refstruct.symbols):
            raise ValueError(
                "Structures must have the same number and type of atoms for "
                "`symmetry=True` at this time. Pass "
                "`symmetry=False` to align structures with different atom "
                "counts."
            )
        symbols = refstruct.symbols
        geometry = np.zeros((len(atm_map), 3))

        for probe_idx, ref_idx in atm_map:
            geometry[ref_idx] = transformed_coords[probe_idx]

    # Otherwise, keep the original atom order
    else:
        symbols = struct.symbols
        geometry = transformed_coords

    return (
        Structure(
            symbols=symbols,
            geometry=geometry,
            charge=struct.charge,
            multiplicity=struct.multiplicity,
            connectivity=struct.connectivity,
            identifiers=struct.identifiers,
        ),
        rmsd_val * constants.ANGSTROM_TO_BOHR
        if length_unit == LengthUnit.BOHR
        else rmsd_val,
    )


@requires_rdkit
def _smiles_to_structure_rdkit(
    smiles: str, force_field: str = "MMFF94s", **struct_kwargs: Any
) -> Structure:
    """Convert a SMILES string to a Structure object.

    Args:
        smiles: The SMILES string to convert.
        force_field: The force field to use for geometry embedding. Defaults to
            "MMFF94s". Options are "UFF", "MMFF94", and "MMFF94s".
        **struct_kwargs: Additional keyword arguments to pass to the Structure
            constructor such as multiplicity, extras, identifiers, etc.

    Returns:
        A Structure object.

    Raises:
        ValueError: If the force field is not one of the supported options.
        ValueError: If the SMILES string contains multiple molecules (i.e., the SMILES
            string contains "." characters).
        AssertionError: If the conversion to an RDKit Mol fails.
    """
    # Validate force field
    force_field = force_field.upper()
    if force_field not in ["UFF", "MMFF94", "MMFF94S"]:
        raise ValueError(
            f"Unsupported force field: {force_field}. "
            "Supported force fields are: UFF, MMFF94, MMFF94s."
        )

    # Remove newline characters if present
    smiles = smiles.strip()

    if "." in smiles:
        raise ValueError(
            "Multiple molecules are not supported by RDKit. "
            "Please provide a single molecule or use openbabel instead."
        )

    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        raise ValueError(
            f"Failed to convert SMILES to RDKit Mol: {smiles}. "
            "Please check the SMILES string for validity."
        )

    # Compute smiles
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)  # type: ignore
    mol = Chem.AddHs(mol)  # type: ignore

    # Generate 3D coordinates
    # https://www.rdkit.org/docs/Cookbook.html#conformer-generation-with-etkdg
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())  # type: ignore
    # Optimize the molecule using the specified force field
    if force_field == "UFF":
        AllChem.UFFOptimizeMolecule(mol)  # type: ignore
    elif force_field == "MMFF94":
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")  # type: ignore
    elif force_field == "MMFF94S":
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")  # type: ignore

    # Get atom symbols
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # type: ignore

    # Get atom positions
    conf = mol.GetConformer()
    geometry_angstrom = conf.GetPositions()
    geometry_bohr = geometry_angstrom * constants.ANGSTROM_TO_BOHR

    # Get charge
    charge = Chem.GetFormalCharge(mol)  # type: ignore

    # Connectivity
    connectivity = []
    for bond in mol.GetBonds():  # type: ignore
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_order = bond.GetBondTypeAsDouble()
        connectivity.append((begin_idx, end_idx, bond_order))

    return Structure(
        symbols=atoms,
        geometry=geometry_bohr,
        charge=charge,
        identifiers={
            "canonical_smiles": canonical_smiles,
            "smiles": smiles,
            "canonical_smiles_program": "rdkit",
        },
        connectivity=connectivity,
        **struct_kwargs,
    )


@requires_rdkit
def _structure_to_smiles_rdkit(
    structure: Structure,
    *,
    hydrogens: bool = False,
    robust: bool = True,
    use_hueckel: bool = True,
    use_vdw: bool = False,
    cov_factor: float = 1.3,
    allow_charged_fragments: bool = False,
) -> str:
    """Convert a Structure to a SMILES string.

    Args:
        structure: The Structure object to convert.
        hydrogens: Whether to include hydrogens in the SMILES string.
        robust: Whether to use a robust method for bond determination by trying
            different parameters for the DetermineBonds function automatically.
        use_hueckel: Whether to use the Hueckel method for bond determination.
        use_vdw: Whether to use Van der Waals radii for bond determination.
        cov_factor: The scaling factor for the covalent radii when determining
            connectivity.
        allow_charged_fragments: Whether to allow charged fragments in the bond
            determination step. When allow_charged_fragments=False, RDKit
            avoids assigning formal charges and instead satisfies valence with radicals
            (unpaired electrons) if necessary. Bonding and valence will be reconciled
            without fragments. When True, RDKit will assign formal charges to atoms and
            reconcile bonding and valence with charged fragments.

    Returns:
        A canonical SMILES string.
    """
    if use_hueckel and use_vdw:
        raise ValueError(
            "Cannot use both the Hueckel and Van der Waals methods for bond detection. "
            "Pass use_hueckel=False if you want to use the VdW method. Hueckel method "
            "is used by default."
        )

    # Details: https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html  # noqa: E501
    # Create RDKit molecule and use rdDetermineBonds module to infer bonds
    mol = _determine_bonds_rdkit(
        structure,
        charge=structure.charge,
        robust=robust,
        use_hueckel=use_hueckel,
        use_vdw=use_vdw,
        cov_factor=cov_factor,
        allow_charged_fragments=allow_charged_fragments,
    )

    # Remove hydrogens if necessary
    if not hydrogens:
        mol = Chem.RemoveHs(mol)  # type: ignore

    return Chem.MolToSmiles(mol, canonical=True)  # type: ignore


# --- internal helper functions for rdkit --- #


@requires_rdkit
def _structure_to_rdkit_mol(
    struct: Structure,
) -> "rdkit.Chem.Mol":  # type: ignore # noqa: F821
    """Create an RDKit molecule from a Structure object."""
    # Create RDKit molecule
    mol = Chem.MolFromXYZBlock(struct.to_xyz())  # type: ignore

    if mol is None:
        raise ValueError("Failed create rdkit Molecule from xyz string.")

    # Ensure molecule has conformers
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule lacks 3D coordinates.")

    return mol


@requires_rdkit
def _determine_bonds_rdkit(
    structure: Structure,
    charge: int,
    robust: bool = True,
    use_hueckel: bool = True,
    use_vdw: bool = False,
    cov_factor: float = 1.3,
    allow_charged_fragments: bool = False,
) -> "Mol":
    """Determine the bonds in an RDKit molecule, using robust fallback parameters.

    Hueckel method is most robust; may useVdw=True use VdW radii; default method
    is connect-the-dots (fastest but least robust)

    Developer Note:
        - I am passing a Structure object to this function rather than a Mol object
        because each attempt to determine the bonds modifies the Mol in place and a
        fresh object is required for each attempt as a formerly failed attempt may
        have left the Mol in a state that is not recoverable

    """
    mol = Chem.MolFromXYZBlock(structure.to_xyz())  # type: ignore

    with mute_c_stderr():
        # to suppress superfluous RDKit warnings such as
        # '!!! Warning !!! Distance between atoms ... is suspicious.'
        try:
            # Execute the wrapped code block
            if robust:
                try:  # Original parameters
                    rdDetermineBonds.DetermineBonds(
                        mol,
                        charge=charge,
                        useHueckel=use_hueckel,
                        useVdw=use_vdw,
                        covFactor=cov_factor,
                        allowChargedFragments=allow_charged_fragments,
                    )
                except Exception as e:  # noqa: E722
                    mol = Chem.MolFromXYZBlock(structure.to_xyz())
                    try:  # Swap allow_charged_fragments
                        rdDetermineBonds.DetermineBonds(
                            mol,
                            charge=charge,
                            useHueckel=use_hueckel,
                            useVdw=use_vdw,
                            covFactor=cov_factor,
                            allowChargedFragments=not allow_charged_fragments,
                        )
                    except Exception:  # noqa: E722
                        mol = Chem.MolFromXYZBlock(structure.to_xyz())
                        try:  # Swap method
                            rdDetermineBonds.DetermineBonds(
                                mol,
                                charge=charge,
                                useHueckel=not use_hueckel,
                                useVdw=not use_vdw,
                                covFactor=cov_factor,
                                allowChargedFragments=allow_charged_fragments,
                            )
                        except Exception:  # noqa: E722
                            mol = Chem.MolFromXYZBlock(structure.to_xyz())
                            try:  # Swap method and allow_charged_fragments
                                rdDetermineBonds.DetermineBonds(
                                    mol,
                                    charge=charge,
                                    useHueckel=not use_hueckel,
                                    useVdw=not use_vdw,
                                    covFactor=cov_factor,
                                    allowChargedFragments=not allow_charged_fragments,
                                )
                            except Exception:
                                mol = Chem.MolFromXYZBlock(structure.to_xyz())
                                try:  # Try connect-the-dots method
                                    rdDetermineBonds.DetermineBonds(
                                        mol,
                                        charge=charge,
                                        useHueckel=False,
                                        useVdw=False,
                                        covFactor=cov_factor,
                                        allowChargedFragments=True,
                                    )
                                except Exception:
                                    mol = Chem.MolFromXYZBlock(structure.to_xyz())
                                    try:  # Swap allow_charged_fragments
                                        rdDetermineBonds.DetermineBonds(
                                            mol,
                                            charge=charge,
                                            useHueckel=False,
                                            useVdw=False,
                                            covFactor=cov_factor,
                                            allowChargedFragments=False,
                                        )
                                    except Exception:
                                        raise e from None
            else:
                rdDetermineBonds.DetermineBonds(
                    mol,
                    charge=charge,
                    useHueckel=use_hueckel,
                    useVdw=use_vdw,
                    covFactor=cov_factor,
                    allowChargedFragments=allow_charged_fragments,
                )
        finally:
            # Delete the run.out and nul files created by rdkit
            # Remove run.out and nul files if they exist
            for filename in ["run.out", "nul"]:
                file = Path(filename)
                if file.exists():
                    try:
                        file.unlink()
                    except Exception:
                        pass
        return mol


@requires_rdkit
def _determine_connectivity_rdkit(
    mol: "Mol",
    charge: int,
    use_hueckel: bool = True,
    use_vdw: bool = True,
    cov_factor: float = 1.3,
) -> None:
    """Determine connectivity for an RDKit molecule.

    Args:
        mol: The RDKit molecule.
        charge: The charge of the molecule.
        use_hueckel: Whether to use Hueckel method when determining connectivity.
        use_vdw: Whether to use Van der Waals radii when determining connectivity.
        cov_factor: The scaling factor for the covalent radii when determining
            connectivity.
    """
    try:
        with mute_c_stderr():
            # to suppress superfluous RDKit warnings such as
            # '!!! Warning !!! Distance between atoms ... is suspicious.'
            rdDetermineBonds.DetermineConnectivity(
                mol,
                charge=charge,
                useHueckel=use_hueckel,
                useVdw=use_vdw,
                covFactor=cov_factor,
            )
    finally:
        # Delete the run.out and nul files created by rdkit
        for filename in ["run.out", "nul"]:
            file = Path(filename)
            if file.exists():
                try:
                    file.unlink()
                except Exception:
                    pass
