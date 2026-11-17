"""Molden files: qcinf.molden against an independent reader and the files of axtb and g-xTB.

Every check reads files back with tests/molden_independent.py, which builds the AO
overlap and orbital values from a file's [GTO] block alone, without qcinf or qcdata's
AO conventions.

Fixtures in tests/data/molden (xz-compressed Molden files):

- ``axtb-*``: axtb 0.1.1 (branch next, a62f28e), ``axtb mol.xyz --method METHOD --chrg Q
  --uhf N --threads 1 --molden=FILE``, spherical and, where d shells make them differ,
  ``--molden-cartesian`` files of the same run, for GFN2-xTB (``gfn2``) and g-xTB 2.0.1
  (``gxtb2``): water, triplet O2, [CuCl4]2- (doublet), [ZnCl4]2- and [FeCl4]- (sextet).
  axtb.json has each run's settings, energy and atomic charges.
- ``gxtb-1.1.0-*``: molden.input files of the g-xTB 1.1.0 program (axtb's reference set,
  crates/gxtb/tests/reference), which writes the contraction coefficients of bare
  primitives.
"""

# l is the conventional symbol for angular momentum.
# ruff: noqa: E741
import json
import lzma
import math
from pathlib import Path

import numpy as np
import pytest
from qcdata import (
    DEFAULT_AO_CONVENTION,
    MOLDEN_CONVENTION,
    AOConvention,
    BasisSet,
    OrbitalKind,
    Orbitals,
    OrbitalSpin,
    Shell,
    Structure,
    Wavefunction,
)
from qcdata.utils import water
from qcref import constants as c

import qcinf
from qcinf import molden
from tests import molden_independent as mi

DATA = Path(__file__).parent / "data" / "molden"
AXTB = json.loads((DATA / "axtb.json").read_text())["cases"]
D_CASES = [k for k in AXTB if (DATA / f"axtb-{k}.cart.molden.xz").exists()]
GXTB11 = ["water", "dioxygen_triplet", "iodomethane", "tetrachlorocuprate"]


def fixture(name: str) -> str:
    return lzma.decompress((DATA / f"{name}.molden.xz").read_bytes()).decode()


def numbers(mol: mi.Molden) -> dict[str, np.ndarray]:
    """Every number of a parsed file, by kind."""
    return {
        "positions": np.array([x for _, p in mol.atoms for x in p]),
        "exponents": np.concatenate([e for *_, e, _ in mol.shells]),
        "contraction": np.concatenate([d for *_, d in mol.shells]),
        "energies": np.array([m["energy"] for m in mol.mos]),
        "occupations": np.array([m["occ"] for m in mol.mos]),
        "coefficients": np.concatenate([m["coef"] for m in mol.mos]),
    }


def assert_same_file(a: mi.Molden, b: mi.Molden, rtol: float = 1e-15) -> None:
    """The same atoms, shells, flags and orbitals, every number to round-off."""
    assert [z for z, _ in a.atoms] == [z for z, _ in b.atoms]
    assert [s[:2] for s in a.shells] == [s[:2] for s in b.shells]
    assert a.pure == b.pure
    assert [m["spin"] for m in a.mos] == [m["spin"] for m in b.mos]
    na, nb = numbers(a), numbers(b)
    for key in na:
        np.testing.assert_allclose(nb[key], na[key], rtol=rtol, atol=rtol, err_msg=key)


# ------------------------------------------------------------------ synthetic orbitals

# Water with s to g shells, contracted and not, in an order that is not atom by atom.
SPEC = [
    (0, 0, [5.0, 1.2], [0.4, 0.7]),
    (1, 0, [0.6], [1.0]),
    (0, 1, [1.1, 0.35], [0.5, 0.6]),
    (2, 0, [0.6], [1.0]),
    (0, 2, [0.9, 0.3], [0.6, 0.5]),
    (1, 1, [0.8], [1.0]),
    (0, 3, [0.7], [1.0]),
    (2, 2, [0.55], [1.0]),
    (1, 2, [0.45], [1.0]),
    (0, 4, [0.6], [1.0]),
]


def make_basis(pure: dict[int, bool] | None = None) -> BasisSet:
    """The basis of SPEC; `pure` per l (default: every shell spherical)."""
    pure = pure or {}
    shells = []
    for center, l, exps, raw in SPEC:
        exps_a, raw_a = np.array(exps), np.array(raw)
        xl = mi._Shell(np.zeros(3), l, exps_a, raw_a, np.eye(1))
        norm2 = mi._monomial_block(xl, xl, 0.0)[0, 0]  # <x^l|x^l>
        shells.append(
            Shell(
                center=center,
                angular_momentum=l,
                pure=pure.get(l, True),
                exponents=exps_a,
                coefficients=raw_a / math.sqrt(norm2),
            )
        )
    return BasisSet(name="test", shells=shells)


def default_overlap(basis: BasisSet, positions: np.ndarray) -> np.ndarray:
    """The independent overlap of `basis` in qcdata's default AO convention."""
    specs = [
        (positions[s.center], s.angular_momentum, s.pure, s.exponents, s.coefficients)
        for s in basis.shells
    ]
    return mi.Basis.from_shells(
        specs, mi.PYSCF_SPHERICAL, mi.LEXICOGRAPHIC, "axis"
    ).overlap()


def make_wavefunction(spin: str, basis: BasisSet) -> Wavefunction:
    """Canonical orbitals orthonormal in the independent overlap, default convention."""
    s = default_overlap(basis, water.geometry)
    w, v = np.linalg.eigh(s)
    lowdin = v @ np.diag(w**-0.5) @ v.T
    rng = np.random.default_rng(3)
    nao = basis.nao

    def rotated() -> np.ndarray:
        return lowdin @ np.linalg.qr(rng.normal(size=(nao, nao)))[0]

    n_a, n_b = (5, 5) if spin == "closed" else (6, 4)
    occ_a = np.array([1.0] * n_a + [0.0] * (nao - n_a))
    occ_b = np.array([1.0] * n_b + [0.0] * (nao - n_b))
    energies = np.linspace(-20.0, 3.0, nao)
    fields: dict = {"energies_a": energies, "coefficients_a": rotated()}
    if spin == "unrestricted":
        fields |= {"energies_b": energies + 0.1, "coefficients_b": rotated()}
    orbitals = Orbitals(
        spin=OrbitalSpin.unrestricted
        if spin == "unrestricted"
        else OrbitalSpin.restricted,
        occupations_a=occ_a,
        occupations_b=occ_b,
        **fields,
    )
    return Wavefunction(orbitals=orbitals, basis=basis, overlap=s)


SPINS = ["closed", "restricted_open", "unrestricted"]
BASES = {
    "spherical": {},
    "cartesian": {2: False, 3: False, 4: False},
    "5d10f": {2: True, 3: False, 4: True},
}


@pytest.mark.parametrize("cartesian", [False, True])
@pytest.mark.parametrize("kind", list(BASES))
@pytest.mark.parametrize("spin", SPINS)
def test_written_orbitals_are_orthonormal_in_an_independent_overlap(
    spin, kind, cartesian
):
    wf = make_wavefunction(spin, make_basis(BASES[kind]))
    text = molden.write(wf, water, cartesian=cartesian)
    mol = mi.parse(text)
    basis = mi.Basis.from_molden(mol)
    s = basis.overlap()
    assert mi.orthonormality(mol, s) < 1e-12
    assert mi.electron_count_error(mol, s) < 1e-12
    # the flags declare the shell types the basis has
    expected = {l: not cartesian and BASES[kind].get(l, True) for l in (2, 3, 4)}
    assert mol.pure == expected
    df_flags = {"[5D]", "[7F]", "[6D]", "[10F]", "[5D10F]", "[5D7F]"}
    flags = [line for line in text.splitlines() if line in df_flags]
    assert flags == {
        (False, "spherical"): ["[7F]", "[5D]"],
        (False, "cartesian"): ["[6D]", "[10F]"],
        (False, "5d10f"): ["[5D10F]"],
    }.get((cartesian, kind), ["[6D]", "[10F]"])
    assert ("[9G]" in text) == expected[4]
    # sets and occupations
    spins = [m["spin"] for m in mol.mos]
    occ = numbers(mol)["occupations"]
    nmo = wf.orbitals.nmo
    if spin == "closed":
        assert spins == ["alpha"] * nmo
        assert occ.tolist() == [2.0] * 5 + [0.0] * (nmo - 5)
    else:
        assert spins == ["alpha"] * nmo + ["beta"] * nmo
        assert occ.tolist() == [1.0] * 6 + [0.0] * (nmo - 6) + [1.0] * 4 + [0.0] * (
            nmo - 4
        )
    energies = numbers(mol)["energies"]
    assert np.array_equal(energies[:nmo], wf.orbitals.energies_a)
    if spin == "restricted_open":
        assert np.array_equal(mol.coefficients("alpha"), mol.coefficients("beta"))
        assert np.array_equal(energies[nmo:], wf.orbitals.energies_a)
    # atoms in bohr
    assert text.splitlines()[3] == "[Atoms] AU"
    assert np.array_equal(
        np.array([p for _, p in mol.atoms]), np.asarray(water.geometry)
    )
    assert [z for z, _ in mol.atoms] == [8, 1, 1]


@pytest.mark.parametrize("kind", list(BASES))
@pytest.mark.parametrize("spin", SPINS)
def test_read_gives_back_what_write_wrote(spin, kind):
    wf = make_wavefunction(spin, make_basis(BASES[kind]))
    structure = water.model_copy(update={"charge": -1, "multiplicity": 3})
    structure2, wf2 = molden.read(
        molden.write(wf, structure, title="test"), charge=structure.charge
    )
    assert structure2.symbols == structure.symbols
    assert np.array_equal(structure2.geometry, structure.geometry)
    assert structure2.charge == -1
    assert structure2.multiplicity == (1 if spin == "closed" else 3)
    assert wf2.basis is not None and wf.basis is not None
    assert wf2.basis.ao_convention == MOLDEN_CONVENTION
    assert wf2.overlap is None
    # the file lists the shells atom by atom
    order = sorted(range(len(wf.basis.shells)), key=lambda i: wf.basis.shells[i].center)
    for i, s2 in zip(order, wf2.basis.shells, strict=True):
        s = wf.basis.shells[i]
        assert (s2.center, s2.angular_momentum, s2.pure) == (
            s.center,
            s.angular_momentum,
            s.pure,
        )
        assert np.array_equal(s2.exponents, s.exponents)
        np.testing.assert_allclose(s2.coefficients, s.coefficients, rtol=1e-15)
    back = wf2.with_convention(DEFAULT_AO_CONVENTION).orbitals
    orb = wf.orbitals
    assert back.spin == orb.spin and back.kind == OrbitalKind.canonical
    for name in ("energies_a", "energies_b", "occupations_a", "occupations_b"):
        a, b = getattr(orb, name), getattr(back, name)
        assert (a is None and b is None) or np.array_equal(a, b), name
    for name in ("coefficients_a", "coefficients_b"):
        a = getattr(orb, name)
        if a is None:
            assert getattr(back, name) is None
            continue
        a = wf.basis.transform(
            a, DEFAULT_AO_CONVENTION, kind="coefficients", shell_order=order
        )
        np.testing.assert_allclose(getattr(back, name), a, rtol=1e-14, atol=1e-14)


def test_cartesian_file_holds_the_same_orbitals():
    wf = make_wavefunction("unrestricted", make_basis())
    sph = mi.parse(molden.write(wf, water))
    cart = mi.parse(molden.write(wf, water, cartesian=True))
    assert len(sph.mos[0]["coef"]) == 40 and len(cart.mos[0]["coef"]) == 52
    points = mi.points_around(sph)
    vs = mi.Basis.from_molden(sph).values(points)
    vc = mi.Basis.from_molden(cart).values(points)
    for spin in ("alpha", "beta"):
        a, b = sph.coefficients(spin).T @ vs, cart.coefficients(spin).T @ vc
        assert np.abs(a - b).max() < 1e-13 * np.abs(a).max()


# ------------------------------------------------------------------ axtb's files


@pytest.mark.parametrize("case", list(AXTB))
def test_axtb_files_read_and_write_back(case):
    """Reading and writing an axtb file reproduces it to round-off."""
    info = AXTB[case]
    names = [f"axtb-{case}"] + ([f"axtb-{case}.cart"] if case in D_CASES else [])
    for name in names:
        text = fixture(name)
        structure, wf = molden.read(text, charge=info["charge"])
        assert structure.charge == info["charge"]
        assert structure.multiplicity == info["uhf"] + 1
        # GFN2-xTB's open shells are restricted orbitals, g-xTB's unrestricted
        open_shell = info["uhf"] > 0
        unrestricted = open_shell and info["method"] != "gfn2"
        assert wf.orbitals.spin.value == (
            "unrestricted" if unrestricted else "restricted"
        )
        written = molden.write(wf, structure, title=text.splitlines()[2])
        assert_same_file(mi.parse(text), mi.parse(written))


@pytest.mark.parametrize("case", D_CASES)
def test_cartesian_files_match_axtbs(case):
    """The Cartesian file of axtb's spherical orbitals is axtb's Cartesian file."""
    structure, wf = molden.read(fixture(f"axtb-{case}"))
    ours = mi.parse(molden.write(wf, structure, cartesian=True))
    assert_same_file(mi.parse(fixture(f"axtb-{case}.cart")), ours)


@pytest.mark.parametrize("case", list(AXTB))
def test_written_axtb_orbitals_are_orthonormal(case):
    """With axtb's primitive screening (a b R^2 / (a + b) > 25 skipped), C^T S C = 1."""
    structure, wf = molden.read(fixture(f"axtb-{case}"))
    for cartesian in (False, True):
        mol = mi.parse(molden.write(wf, structure, cartesian=cartesian))
        basis = mi.Basis.from_molden(mol)
        s = basis.overlap(screen=25)
        assert mi.orthonormality(mol, s) < 1e-13
        assert mi.electron_count_error(mol, s) < 1e-12
        # without the screening, as far as axtb's screened overlap is from the exact one
        assert mi.orthonormality(mol, basis.overlap()) < 1e-8


@pytest.mark.parametrize("case", list(AXTB))
def test_mulliken_populations_match_axtbs_charges(case):
    """The populations of a written file and axtb's charges add up to each atom's
    reference electrons (integers, to axtb's 10-decimal occupations in its files)."""
    structure, wf = molden.read(fixture(f"axtb-{case}"))
    mol = mi.parse(molden.write(wf, structure))
    pop = mi.mulliken_populations(mol, mi.Basis.from_molden(mol), screen=25)
    reference = np.asarray(AXTB[case]["charges"]) + pop
    assert np.abs(reference - np.round(reference)).max() < 1e-9
    assert pop.sum() == pytest.approx(numbers(mol)["occupations"].sum(), abs=1e-12)


# ------------------------------------------------------------------ the g-xTB program


@pytest.mark.parametrize("name", GXTB11)
def test_gxtb_program_files(name):
    """g-xTB 1.1.0's files (bare primitives) become Molden-convention files of the same
    orbitals: the same coefficients and, at random points, the same values."""
    text = fixture(f"gxtb-1.1.0-{name}")
    structure, wf = molden.read(text, normalized_primitives=False)
    program = mi.parse(text)
    ours = mi.parse(molden.write(wf, structure))
    assert structure.multiplicity == {
        "dioxygen_triplet": 3,
        "tetrachlorocuprate": 2,
    }.get(name, 1)
    pb = mi.Basis.from_molden(program, primitives="raw")
    ob = mi.Basis.from_molden(ours)
    points = mi.points_around(program)
    vp, vo = pb.values(points), ob.values(points)
    for spin in program.spins:
        assert np.array_equal(ours.coefficients(spin), program.coefficients(spin))
        a, b = program.coefficients(spin).T @ vp, ours.coefficients(spin).T @ vo
        assert np.abs(a - b).max() < 1e-14 * np.abs(a).max()
    s = ob.overlap()
    assert (
        abs(mi.orthonormality(ours, s) - mi.orthonormality(program, pb.overlap()))
        < 1e-14
    )


# ------------------------------------------------------------------ reading


H2O = """[Molden Format]
[Atoms] AU
O 1 8 0.0 0.0 0.2
H 2 1 0.0 1.4 -0.9
H 3 1 0.0 -1.4 -0.9
[GTO]
1 0
sp 2 1.00
 5.0D+00 0.4D+00 0.3D+00
 1.0D+00 0.7D+00 0.8D+00

2 0
s 1 2.00
 0.15 1.0

3 0
s 1 1.00
 0.6 1.0

[MO]
{mos}"""


def mo_block(occupations: list[float], spin: str | None = None, nao: int = 6) -> str:
    out = []
    for k, occ in enumerate(occupations):
        out.append(f"Sym= {k + 1}a\nEne= {k - 3.0}")
        if spin:
            out.append(f"Spin= {spin}")
        out.append(f"Occup= {occ}")
        out += [f"{j + 1} {0.1 * (j + 1) + k}" for j in range(nao)]
    return "\n".join(out) + "\n"


def test_read_details():
    structure, wf = molden.read(H2O.format(mos=mo_block([2, 2, 1, 0, 0, 0])))
    basis = wf.basis
    assert basis is not None
    # an sp shell is an s and a p shell; the scale factor scales exponents by its square
    assert [(s.center, s.angular_momentum) for s in basis.shells] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
    ]
    assert basis.shells[1].exponents.tolist() == [5.0, 1.0]
    assert basis.shells[2].exponents.tolist() == [0.6]
    # contractions are normalized: x^l of each shell has norm 1
    for s in basis.shells:
        xl = mi._Shell(
            np.zeros(3), s.angular_momentum, s.exponents, s.coefficients, np.eye(1)
        )
        assert mi._monomial_block(xl, xl, 0.0)[0, 0] == pytest.approx(1.0, rel=1e-14)
    # integer occupations: aufbau, singly occupied orbitals alpha
    orb = wf.orbitals
    assert orb.spin.value == "restricted"
    assert orb.occupations_a.tolist() == [1, 1, 1, 0, 0, 0]
    assert orb.occupations_b.tolist() == [1, 1, 0, 0, 0, 0]
    assert structure.multiplicity == 2 and structure.charge == 0
    assert orb.coefficients_a is not None
    assert orb.coefficients_a[:, 1].tolist() == pytest.approx(
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    )
    assert orb.energies_a.tolist() == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    # smeared occupations are shared by the spins
    _, wf = molden.read(H2O.format(mos=mo_block([2, 1.9, 1.1, 0, 0, 0])))
    assert wf.orbitals.occupations_a.tolist() == [1, 0.95, 0.55, 0, 0, 0]
    assert np.array_equal(wf.orbitals.occupations_a, wf.orbitals.occupations_b)
    # alpha and beta sets: restricted when they hold the same orbitals
    same = mo_block([1, 1, 1, 0, 0, 0], "Alpha") + mo_block([1, 0, 0, 0, 0, 0], "Beta")
    structure, wf = molden.read(H2O.format(mos=same))
    assert wf.orbitals.spin.value == "restricted" and structure.multiplicity == 3
    other = same.replace("Ene= -3.0\nSpin= Beta", "Ene= -3.5\nSpin= Beta")
    _, wf = molden.read(H2O.format(mos=other))
    assert wf.orbitals.spin.value == "unrestricted"
    assert wf.orbitals.energies_b is not None and wf.orbitals.energies_b[0] == -3.5


def test_read_angstrom_positions():
    bohr = molden.read(H2O.format(mos=mo_block([2, 2, 1, 0, 0, 0])))[0]
    text = H2O.replace("[Atoms] AU", "[Atoms] Angs").replace(
        "0.0 1.4 -0.9",
        f"0.0 {1.4 * c.BOHR_TO_ANGSTROM!r} {-0.9 * c.BOHR_TO_ANGSTROM!r}",
    )
    angs = molden.read(text.format(mos=mo_block([2, 2, 1, 0, 0, 0])))[0]
    np.testing.assert_allclose(angs.geometry[1], bohr.geometry[1], rtol=1e-15)


@pytest.mark.parametrize(
    "flags", ["[5D]\n[7F]\n[9G]", "[5D7F]\n[9G]", "[5d]\n[9g]", "[7F]\n[9G]\n[5D]"]
)
def test_spherical_flags_in_any_order(flags):
    wf = make_wavefunction("closed", make_basis())
    text = molden.write(wf, water)
    assert "[7F]\n[5D]\n[9G]\n" in text
    _, ref = molden.read(text)
    _, other = molden.read(text.replace("[7F]\n[5D]\n[9G]", flags))
    assert other.basis == ref.basis
    assert np.array_equal(other.orbitals.coefficients_a, ref.orbitals.coefficients_a)


def test_read_axis_normalized_cartesian_components():
    """tblite's Cartesian components all carry the norm of x^l."""
    wf = make_wavefunction("closed", make_basis(BASES["cartesian"]))
    text = molden.write(wf, water)
    axis = AOConvention(
        spherical_order="molden",
        cartesian_order="molden",
        cartesian_normalization="axis",
    )
    # the same orbitals in axis-normalized components: a component's coefficient is
    # divided by its norm relative to x^l, sqrt((2lx-1)!! (2ly-1)!! (2lz-1)!! / (2l-1)!!)
    df = lambda n: math.prod(range(n, 0, -2)) if n > 0 else 1  # noqa: E731
    scale = []
    for _, l, *_ in sorted(SPEC, key=lambda s: s[0]):
        for e in mi.MOLDEN_CARTESIAN[l]:
            scale.append(
                math.sqrt(
                    df(2 * e[0] - 1)
                    * df(2 * e[1] - 1)
                    * df(2 * e[2] - 1)
                    / df(2 * l - 1)
                )
            )
    head, mos = text.split("[MO]\n")
    lines = []
    for line in mos.splitlines():
        f = line.split()
        if "=" not in line:
            line = f"{f[0]} {float(f[1]) / scale[int(f[0]) - 1]!r}"
        lines.append(line)
    axis_text = head + "[MO]\n" + "\n".join(lines) + "\n"
    _, ref = molden.read(text)
    _, wf_axis = molden.read(axis_text, ao_convention=axis)
    assert wf_axis.basis is not None and wf_axis.basis.ao_convention == axis
    back = wf_axis.with_convention(MOLDEN_CONVENTION).orbitals.coefficients_a
    np.testing.assert_allclose(
        back, ref.orbitals.coefficients_a, rtol=1e-14, atol=1e-15
    )


def test_read_refuses_what_it_cannot_read():
    mos = mo_block([2, 2, 1, 0, 0, 0])
    good = H2O.format(mos=mos)
    with pytest.raises(ValueError, match=r"no \[MO\]"):
        molden.read(good.split("[MO]")[0])
    with pytest.raises(ValueError, match="STO"):
        molden.read(good + "[STO]\n")
    with pytest.raises(ValueError, match="both spherical and Cartesian"):
        molden.read(good + "[5D]\n[6D]\n")
    with pytest.raises(ValueError, match="function 7"):
        molden.read(H2O.format(mos=mo_block([2, 2, 1, 0, 0, 0], nao=7)))
    with pytest.raises(ValueError, match="0 to 2 electrons"):
        molden.read(H2O.format(mos=mo_block([2.5, 2, 1, 0, 0, 0])))
    two = mo_block([1, 1, 0, 0, 0, 0], "Alpha") + mo_block([2, 1, 0, 0, 0, 0], "Beta")
    with pytest.raises(ValueError, match="0 to 1 electrons"):
        molden.read(H2O.format(mos=two))
    uneven = mo_block([1, 1, 0, 0, 0, 0], "Alpha") + mo_block([1, 0, 0, 0, 0], "Beta")
    with pytest.raises(ValueError, match="6 alpha and 5 beta"):
        molden.read(H2O.format(mos=uneven))
    with pytest.raises(ValueError, match="no Ene"):
        molden.read(good.replace("Ene= -2.0\n", ""))
    with pytest.raises(ValueError, match="atom 4"):
        molden.read(good.replace("3 0\ns 1 1.00", "4 0\ns 1 1.00"))
    with pytest.raises(ValueError, match="both spherical and Cartesian"):
        molden.read(good + "[5D7F]\n[10F]\n")
    with pytest.raises(ValueError, match="not 'h'"):
        molden.read(good.replace("s 1 2.00", "h 1 2.00"))
    with pytest.raises(ValueError, match="element"):
        molden.read(good.replace("H 3 1 ", "Xx 3 0 "))


# ------------------------------------------------------------------ writing


def test_write_refuses_what_molden_cannot_hold():
    wf = make_wavefunction("closed", make_basis())
    levels = Wavefunction(
        orbitals=wf.orbitals.model_copy(update={"coefficients_a": None})
    )
    with pytest.raises(ValueError, match="only orbital levels"):
        molden.write(levels, water)
    two_atoms = Structure(symbols=["O", "H"], geometry=water.geometry[:2])
    with pytest.raises(ValueError, match="2 atoms"):
        molden.write(wf, two_atoms)
    mixed = make_basis()
    mixed.shells[-2] = mixed.shells[-2].model_copy(
        update={"pure": False}
    )  # one d shell
    wf_mixed = make_wavefunction("closed", mixed)
    with pytest.raises(ValueError, match="cartesian=True"):
        molden.write(wf_mixed, water)
    mol = mi.parse(molden.write(wf_mixed, water, cartesian=True))
    assert mi.orthonormality(mol, mi.Basis.from_molden(mol).overlap()) < 1e-12
    h = Shell(
        center=0, angular_momentum=5, pure=True, exponents=[1.0], coefficients=[1.0]
    )
    wf_h = Wavefunction(
        orbitals=Orbitals(
            spin="restricted",
            energies_a=np.zeros(11),
            occupations_a=np.zeros(11),
            occupations_b=np.zeros(11),
            coefficients_a=np.eye(11),
        ),
        basis=BasisSet(shells=[h]),
    )
    with pytest.raises(ValueError, match="up to g"):
        molden.write(wf_h, water)


def test_write_details():
    wf = make_wavefunction("closed", make_basis())
    text = molden.write(wf, water, title="water\nsecond line")
    lines = text.splitlines()
    assert lines[:4] == ["[Molden Format]", "[Title]", "water", "[Atoms] AU"]
    # orbitals without energies (natural or localized) are written with Ene= 0.0
    natural = wf.orbitals.model_copy(update={"kind": "natural", "energies_a": None})
    text = molden.write(Wavefunction(orbitals=natural, basis=wf.basis), water)
    assert {line for line in text.splitlines() if line.startswith("Ene=")} == {
        "Ene= 0.0000000000000000E+00"
    }


def test_module_is_part_of_the_package():
    assert qcinf.molden is molden
    assert molden.__all__ == ["read", "write"]
