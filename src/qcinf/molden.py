"""Molden files of qcdata wavefunctions.

`write` turns a `qcdata.Wavefunction` with orbital coefficients and its `Structure` into
the text of a Molden file, for orbital viewers (Molden, Jmol, Avogadro, IboView,
Multiwfn) and programs that read the format; `read` turns such text back into a
`Structure` and a `Wavefunction`. The format is described at
https://www.theochem.ru.nl/molden/molden_format.html; these are the conventions both
functions use, which are those of the format as standard readers interpret it:

- Units: atomic units throughout. ``[Atoms] AU`` positions are in bohr, ``[GTO]``
  exponents in bohr^-2, ``Ene=`` orbital energies in hartree; ``Occup=`` counts
  electrons. `read` also accepts ``[Atoms] Angs`` and converts it to bohr.
- ``[GTO]``: the shells atom by atom. Each shell's contraction coefficients multiply
  primitives x^l exp(-a r^2) normalized on their own, and the contraction they make is
  normalized (`qcdata.Shell.primitive_coefficients`). `read` normalizes every
  contraction it reads, as Molden and other readers do, so the MO coefficients always
  refer to normalized functions.
- Functions within a shell follow qcdata's `MOLDEN_CONVENTION`, which `write` converts
  to through `Wavefunction.with_convention`:

  - spherical functions in the order m = 0, +1, -1, +2, -2, ... (p as x, y, z), each a
    unit-normalized real solid harmonic without the Condon-Shortley phase (d+1 ~ xz,
    d-1 ~ yz, d+2 ~ x^2 - y^2, d-2 ~ xy);
  - Cartesian components in Molden's order (d: xx, yy, zz, xy, xz, yz; f: xxx, yyy, zzz,
    xyy, xxy, xxz, xzz, yzz, yyz, xyz; and Molden's g order), each normalized to 1.

- Spherical or Cartesian: Molden decides per angular momentum, with ``[5D]``, ``[7F]``
  and ``[9G]`` for spherical d, f and g functions (``[5D10F]`` for spherical d with
  Cartesian f). `write` writes each shell type as the basis set has it (all shells of one
  angular momentum must agree), or every shell as Cartesian functions with
  ``cartesian=True``. Molden's functions stop at g (l = 4).
- ``[MO]``: one entry per orbital with ``Sym=``, ``Ene=``, ``Spin=``, ``Occup=`` and its
  coefficients, in the order of the coefficient matrices' columns:

  - restricted orbitals of a closed shell (equal alpha and beta occupations) are one
    set, ``Spin= Alpha``, with ``Occup`` the electrons in each orbital, 0 to 2;
  - restricted orbitals of an open shell (ROHF; GFN2-xTB's open shells) are written
    twice, an alpha set and a beta set with the same orbitals and energies and each
    spin's occupations, 0 to 1, so no occupation is lost;
  - unrestricted orbitals are an alpha set followed by a beta set, ``Occup`` 0 to 1.

A Molden file does not carry everything a `Wavefunction` holds. `write` leaves out the
AO matrices, effective core potentials, `Orbitals.core_electrons`, the electronic
temperature and Fermi levels, and the orbital kind (orbitals without energies are
written with ``Ene= 0.0``). `read` returns canonical orbitals at 0 K without core
electrons, in the file's AO convention (`MOLDEN_CONVENTION` unless told otherwise);
`Wavefunction.with_convention` converts them to any other.

Example:
    ```python
    from qcinf import molden

    text = molden.write(output.results.wavefunction, output.input_data.structure)
    structure, wavefunction = molden.read(text)
    ```
"""

# l is the conventional symbol for angular momentum.
# ruff: noqa: E741
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from qcdata import (
    MOLDEN_CONVENTION,
    AOConvention,
    BasisSet,
    Orbitals,
    OrbitalSpin,
    Shell,
    Structure,
    Wavefunction,
)
from qcref import constants as c
from qcref import periodic_table as pt

__all__ = ["read", "write"]

_L_LETTERS = "spdfg"

#: Molden's order of the Cartesian components of each shell type, as (lx, ly, lz).
_CARTESIAN_ORDER: dict[int, list[tuple[int, int, int]]] = {
    l: [(s.count("x"), s.count("y"), s.count("z")) for s in labels.split()]
    for l, labels in {
        0: "-",
        1: "x y z",
        2: "xx yy zz xy xz yz",
        3: "xxx yyy zzz xyy xxy xxz xzz yzz yyz xyz",
        4: "xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz zzxy",
    }.items()
}

#: Spherical flag sections and the shell types each makes spherical (True) or
#: Cartesian (False). ``[7F]`` alone leaves d Cartesian, since d is Cartesian by default.
_FLAGS: dict[str, dict[int, bool]] = {
    "5d": {2: True, 3: True},
    "5d7f": {2: True, 3: True},
    "5d10f": {2: True, 3: False},
    "7f": {3: True},
    "9g": {4: True},
    "6d": {2: False},
    "10f": {3: False},
    "15g": {4: False},
}


# --------------------------------------------------------------------------- write


def write(
    wavefunction: Wavefunction,
    structure: Structure,
    *,
    cartesian: bool = False,
    title: str = "",
) -> str:
    """Write a wavefunction's orbitals as the text of a Molden file.

    The orbitals are converted to qcdata's `MOLDEN_CONVENTION` with
    `Wavefunction.with_convention`, and the shells are written atom by atom. Positions
    are written in bohr (``[Atoms] AU``), energies in hartree; every number with 17
    significant digits, so `read` gets back the same values. The module docstring lists
    the conventions.

    Args:
        wavefunction: A wavefunction with orbital coefficients and their basis set
            (programs return them on request, with the output ``orbitals``).
        structure: The structure the basis set's centers refer to.
        cartesian: Write every shell as Cartesian functions (``[6D]``, ``[10F]``), for
            readers without spherical functions. Spherical orbitals are expanded exactly
            in the Cartesian components, so the file holds the same orbitals, with more
            basis functions than orbitals.
        title: The line of the ``[Title]`` section.

    Returns:
        The Molden file's text.

    Raises:
        ValueError: If the wavefunction has no orbital coefficients, the structure has
            fewer atoms than the basis set refers to, a shell has l > 4, or the shells of
            one angular momentum mix spherical and Cartesian functions (without
            ``cartesian``).
    """
    basis = wavefunction.basis
    orbitals = wavefunction.orbitals
    if basis is None or orbitals.coefficients_a is None:
        raise ValueError(
            "a Molden file needs the orbitals' coefficients and their basis set, and "
            "this wavefunction has only orbital levels: programs return them on "
            "request (the output 'orbitals')"
        )
    natoms = len(structure.symbols)
    if basis.shells and max(s.center for s in basis.shells) >= natoms:
        raise ValueError(
            f"the basis set has functions on atom {max(s.center for s in basis.shells)}, "
            f"but the structure has {natoms} atoms"
        )
    if max((s.angular_momentum for s in basis.shells), default=0) > 4:
        raise ValueError(
            "Molden files have functions up to g (l = 4); this basis set has l = "
            f"{max(s.angular_momentum for s in basis.shells)}"
        )
    pure = _pure_flags(basis.shells, cartesian)

    # Molden's AO convention, without the AO matrices (a Molden file has none).
    molden = Wavefunction(orbitals=orbitals, basis=basis).with_convention(
        MOLDEN_CONVENTION
    )
    mbasis = molden.basis
    assert mbasis is not None and molden.orbitals.coefficients_a is not None
    # Molden lists shells atom by atom.
    order = sorted(range(len(mbasis.shells)), key=lambda i: mbasis.shells[i].center)
    shells = [mbasis.shells[i] for i in order]

    def coefficients(c: np.ndarray) -> np.ndarray:
        if order != sorted(order):
            c = mbasis.transform(
                c, MOLDEN_CONVENTION, kind="coefficients", shell_order=order
            )
        if cartesian:
            c = _to_cartesian(shells, c)
        return c

    out = ["[Molden Format]", "[Title]", title.splitlines()[0] if title else ""]
    out.append("[Atoms] AU")
    for i, (symbol, z, xyz) in enumerate(
        zip(
            structure.symbols, structure.atomic_numbers, structure.geometry, strict=True
        )
    ):
        out.append(
            f"{symbol:<2} {i + 1:>6} {z:>4} " + " ".join(f"{_num(x):>24}" for x in xyz)
        )
    out.append("[GTO]")
    for k, s in enumerate(shells):
        if k == 0 or s.center != shells[k - 1].center:
            if k:
                out.append("")
            out.append(f"{s.center + 1:>6} 0")
        out.append(f"{_L_LETTERS[s.angular_momentum]} {len(s.exponents):>4} 1.00")
        for a, d in zip(s.exponents, s.primitive_coefficients(), strict=True):
            out.append(f"{_num(a):>24} {_num(d):>24}")
    out.append("")
    out.extend(_flag_sections(pure))

    morb = molden.orbitals
    assert morb.coefficients_a is not None
    ca = coefficients(morb.coefficients_a)
    out.append("[MO]")
    if morb.spin == OrbitalSpin.unrestricted:
        assert morb.coefficients_b is not None
        cb = coefficients(morb.coefficients_b)
        sets = [
            ("Alpha", morb.energies_a, morb.occupations_a, ca),
            ("Beta", morb.energies_b, morb.occupations_b, cb),
        ]
    elif np.array_equal(morb.occupations_a, morb.occupations_b):
        sets = [("Alpha", morb.energies_a, morb.occupations_a * 2, ca)]
    else:
        sets = [
            ("Alpha", morb.energies_a, morb.occupations_a, ca),
            ("Beta", morb.energies_a, morb.occupations_b, ca),
        ]
    for spin, energies, occupations, coef in sets:
        for k in range(morb.nmo):
            energy = 0.0 if energies is None else float(energies[k])
            out.append(f"Sym= {k + 1}a")
            out.append(f"Ene= {_num(energy)}")
            out.append(f"Spin= {spin}")
            out.append(f"Occup= {_num(float(occupations[k]))}")
            out.extend(f"{j + 1:>6} {_num(x):>24}" for j, x in enumerate(coef[:, k]))
    return "\n".join(out) + "\n"


def _num(x: float) -> str:
    """A number with 17 significant digits, enough to read back the same double."""
    return f"{x:.16E}"


def _pure_flags(shells: list[Shell], cartesian: bool) -> dict[int, bool]:
    """Whether each shell type (l >= 2) of the file is spherical.

    Raises:
        ValueError: If the shells of one angular momentum disagree.
    """
    pure: dict[int, bool] = {}
    for s in shells:
        l = s.angular_momentum
        if l < 2:
            continue
        p = s.pure and not cartesian
        if pure.setdefault(l, p) != p:
            raise ValueError(
                f"Molden files make every {_L_LETTERS[l]} shell spherical or every one "
                "Cartesian, and this basis set has both; write it with cartesian=True"
            )
    return pure


def _flag_sections(pure: dict[int, bool]) -> list[str]:
    """The flag sections that declare the file's spherical and Cartesian shell types.

    Spherical d and f are written ``[7F]`` then ``[5D]``: readers that take ``[7F]`` as
    6D and 7F (as the format description does) and then ``[5D]`` as 5D and 7F, and
    readers that set one shell type per flag, both read 5D and 7F.
    """
    d, f, g = pure.get(2), pure.get(3), pure.get(4)
    out: list[str] = []
    if d and f is False:
        out.append("[5D10F]")
    else:
        if f:
            out.append("[7F]")
        if d:
            out.append("[5D]")
        elif d is False and not f:
            out.append("[6D]")
        if f is False:
            out.append("[10F]")
    if g:
        out.append("[9G]")
    return out


def _to_cartesian(shells: list[Shell], c: np.ndarray) -> np.ndarray:
    """Expand the spherical functions of the coefficients' rows in Cartesian ones.

    Args:
        shells: The shells in AO order, in `MOLDEN_CONVENTION`.
        c: Coefficients, one row per AO of `shells`.

    Returns:
        The coefficients of Molden's unit-normalized Cartesian components.
    """
    blocks: list[np.ndarray] = []
    i = 0
    for s in shells:
        n = s.nfunc
        rows = c[i : i + n]
        if s.pure and s.angular_momentum >= 2:
            rows = _spherical_to_cartesian(s.angular_momentum).T @ rows
        blocks.append(rows)
        i += n
    return np.concatenate(blocks, axis=0)


def _dfact(n: int) -> int:
    """(n)!! with (-1)!! = 1."""
    return math.prod(range(n, 0, -2)) if n > 0 else 1


def _solid_harmonic(l: int, m: int) -> dict[tuple[int, int, int], float]:
    """The real solid harmonic S_lm as a polynomial, up to a positive factor.

    Without the Condon-Shortley phase (Helgaker, Jorgensen and Olsen, Molecular
    Electronic-Structure Theory, eq. 6.4.48): S_l,+l ~ Re (x + i*y)^l and
    S_l,-l ~ Im (x + i*y)^l.

    Returns:
        {(lx, ly, lz): weight} of the monomials x^lx y^ly z^lz.
    """
    am = abs(m)
    poly: dict[tuple[int, int, int], float] = {}
    # 2v runs over even (m >= 0) or odd (m < 0) integers.
    k0 = 0 if m >= 0 else 1
    for t in range((l - am) // 2 + 1):
        for u in range(t + 1):
            for k in range(k0, am + 1, 2):
                sign = (-1) ** (t + (k - k0) // 2)
                weight = (
                    sign
                    * 0.25**t
                    * math.comb(l, t)
                    * math.comb(l - t, am + t)
                    * math.comb(t, u)
                    * math.comb(am, k)
                )
                key = (2 * t + am - 2 * u - k, 2 * u + k, l - 2 * t - am)
                poly[key] = poly.get(key, 0.0) + weight
    return {key: w for key, w in poly.items() if w != 0.0}


def _spherical_to_cartesian(l: int) -> np.ndarray:
    """Spherical functions of a shell in its unit-normalized Cartesian components.

    Returns:
        T, (2l+1) x (l+1)(l+2)/2, rows in Molden's spherical order and columns in its
        Cartesian order: spherical function m is sum_c T[m, c] X_c, with X_c the
        Cartesian component c normalized to 1 (same radial part).
    """
    cart = _CARTESIAN_ORDER[l]
    # Squared norms of the monomials relative to one another; <x^a..|x^b..> needs every
    # a_k + b_k even and is prod (a_k + b_k - 1)!! then.
    gram = np.array(
        [
            [
                math.prod(_dfact(p + q - 1) for p, q in zip(e, f, strict=True))
                if all((p + q) % 2 == 0 for p, q in zip(e, f, strict=True))
                else 0
                for f in cart
            ]
            for e in cart
        ],
        dtype=float,
    )
    norms = np.sqrt(np.diag(gram))
    ms = [0] + [s * m for m in range(1, l + 1) for s in (1, -1)]
    t = np.zeros((2 * l + 1, len(cart)))
    for row, m in enumerate(ms):
        w = np.zeros(len(cart))
        for key, weight in _solid_harmonic(l, m).items():
            w[cart.index(key)] = weight
        t[row] = w * norms / math.sqrt(w @ gram @ w)
    return t


# --------------------------------------------------------------------------- read


@dataclass
class _Mo:
    """One ``[MO]`` entry as read."""

    energy: float | None = None
    spin: str = "alpha"
    occupation: float = 0.0
    coefficients: dict[int, float] = field(default_factory=dict)


def read(
    text: str,
    *,
    charge: int = 0,
    ao_convention: AOConvention = MOLDEN_CONVENTION,
    normalized_primitives: bool = True,
) -> tuple[Structure, Wavefunction]:
    """Read the structure and orbitals of a Molden file.

    The orbitals come back in the file's AO convention, with every contraction
    normalized. Occupations become per-spin occupations:

    - a file with one set of orbitals (no ``Spin= Beta``) holds restricted orbitals and
      their electrons, 0 to 2 each. If every occupation is 0, 1 or 2, the orbitals are
      filled by aufbau with the singly occupied ones holding alpha electrons (a
      high-spin open shell, as ROHF and xtb's GFN2-xTB files have them); otherwise
      (smeared occupations) each orbital's electrons are shared evenly by the two spins;
    - a file with alpha and beta sets holds per-spin occupations, 0 to 1. If the two
      sets have the same energies and coefficients, they are restricted orbitals of an
      open shell (what `write` writes for them); otherwise unrestricted orbitals.

    The structure's multiplicity is that of the occupations, 2S + 1 with
    2S = n_alpha - n_beta rounded.

    Args:
        text: The Molden file's text.
        charge: The molecular charge of the returned structure; Molden files do not
            record it.
        ao_convention: How the file orders and normalizes the functions within a shell.
            Molden files use `MOLDEN_CONVENTION`. The xtb and tblite programs write
            g-xTB 2.0.1 Cartesian components that all carry the norm of x^l
            (``cartesian_normalization="axis"``).
        normalized_primitives: Whether ``[GTO]`` contraction coefficients multiply
            normalized primitives (Molden's convention, and every program following it)
            or bare exp(-a r^2) ones, as xtb, tblite and the g-xTB program write them.

    Returns:
        The structure (positions in bohr) and a wavefunction with the orbitals'
        energies, occupations, coefficients and basis set.

    Raises:
        ValueError: If the text lacks ``[Atoms]``, ``[GTO]`` or ``[MO]``, has Slater
            functions (``[STO]``), contradictory flags, coefficients beyond the basis
            set's functions, alpha and beta sets of different sizes, or occupations out
            of range.
    """
    sections = _sections(text)
    names = [name for name, _, _ in sections]
    if "sto" in names:
        raise ValueError("Slater-type orbitals ([STO]) are not supported, only [GTO]")
    missing = [s for s in ("atoms", "gto", "mo") if s not in names]
    if missing:
        raise ValueError(
            "a Molden file needs [Atoms], [GTO] and [MO]; this one has no "
            + ", ".join(
                f"[{s.upper() if s == 'mo' else s.capitalize()}]" for s in missing
            )
        )
    pure = _read_flags(names)
    body = {name: (rest, lines) for name, rest, lines in sections}
    symbols, geometry = _read_atoms(*body["atoms"])
    shells = _read_gto(body["gto"][1], pure, len(symbols), normalized_primitives)
    basis = BasisSet(shells=shells, ao_convention=ao_convention)
    orbitals = _read_mo(body["mo"][1], basis.nao)
    n_unpaired = float(orbitals.occupations_a.sum() - orbitals.occupations_b.sum())
    structure = Structure(
        symbols=symbols,
        geometry=geometry,
        charge=charge,
        multiplicity=abs(round(n_unpaired)) + 1,
    )
    return structure, Wavefunction(orbitals=orbitals, basis=basis)


_SECTION = re.compile(r"^\s*\[([^\]]*)\](.*)$")


def _sections(text: str) -> list[tuple[str, str, list[str]]]:
    """Split the text into (lower-case name, rest of the header line, lines)."""
    sections: list[tuple[str, str, list[str]]] = []
    for line in text.splitlines():
        m = _SECTION.match(line)
        if m:
            sections.append((m.group(1).strip().lower(), m.group(2).strip(), []))
        elif sections:
            sections[-1][2].append(line)
    return sections


def _read_flags(names: list[str]) -> dict[int, bool]:
    """Whether each shell type is spherical, from the flag sections in any order.

    ``[5D]`` and ``[5D7F]`` make d and f spherical, ``[5D10F]`` spherical d with
    Cartesian f, ``[7F]`` spherical f, ``[9G]`` spherical g; ``[6D]``, ``[10F]`` and
    ``[15G]`` say Cartesian. A flag that names f outright wins over what ``[5D]``
    implies for it.

    Raises:
        ValueError: If two flags contradict each other.
    """
    flags = [n.replace(" ", "") for n in names if n.replace(" ", "") in _FLAGS]
    implied = {3: True} if "5d" in flags else {}
    pure: dict[int, bool] = {}
    for flag in flags:
        for l, p in _FLAGS[flag].items():
            if flag == "5d" and l == 3:
                continue
            if pure.setdefault(l, p) != p:
                raise ValueError(
                    f"the flags {', '.join(f'[{f.upper()}]' for f in flags)} make "
                    f"{_L_LETTERS[l]} functions both spherical and Cartesian"
                )
    return {**implied, **pure}


def _float(s: str) -> float:
    """A number that may use Fortran's D exponent."""
    return float(s.replace("D", "E").replace("d", "e"))


def _read_atoms(rest: str, lines: list[str]) -> tuple[list[str], np.ndarray]:
    """Element symbols and positions (bohr) of the ``[Atoms]`` section."""
    unit = rest.lower()
    scale = 1.0 if re.search(r"\b(au|bohr)\b", unit) else float(c.ANGSTROM_TO_BOHR)
    symbols: list[str] = []
    xyz: list[list[float]] = []
    for line in lines:
        f = line.split()
        if not f:
            continue
        if len(f) < 6:
            raise ValueError(f"cannot read the [Atoms] line {line.strip()!r}")
        symbols.append(_element(f[0], f[2]))
        xyz.append([_float(x) * scale for x in f[3:6]])
    if not symbols:
        raise ValueError("the [Atoms] section lists no atoms")
    return symbols, np.array(xyz)


def _element(label: str, number: str) -> str:
    """The element of an ``[Atoms]`` line: its label without digits, else its number.

    The label comes first: tblite writes the valence charge in the number column.
    """
    letters = re.match(r"[A-Za-z]+", label)
    if letters:
        symbol = letters.group(0).capitalize()
        if symbol in pt.keys():
            return symbol
    try:
        return str(pt.number(int(number)).symbol)
    except (KeyError, ValueError) as err:
        raise ValueError(
            f"cannot tell the element of the [Atoms] label {label!r}"
        ) from err


def _read_gto(
    lines: list[str], pure: dict[int, bool], natoms: int, normalized_primitives: bool
) -> list[Shell]:
    """The shells of the ``[GTO]`` section, with normalized contractions."""
    shells: list[Shell] = []
    atom: int | None = None
    i = 0
    while i < len(lines):
        f = lines[i].split()
        i += 1
        if not f:
            continue
        kind = f[0].lower()
        if kind not in ("s", "p", "d", "f", "g", "sp"):
            if not f[0].isdigit():
                raise ValueError(
                    f"[GTO] shells are s, p, d, f, g or sp, not {f[0]!r}; or an atom "
                    "number was expected"
                )
            atom = int(f[0]) - 1
            if not 0 <= atom < natoms:
                raise ValueError(f"[GTO] lists atom {atom + 1}; [Atoms] has {natoms}")
            continue
        if atom is None:
            raise ValueError("a [GTO] shell comes before the atom it belongs to")
        nprim = int(f[1])
        scale = _float(f[2]) if len(f) > 2 else 1.0
        rows = np.array(
            [[_float(x) for x in lines[i + k].split()] for k in range(nprim)]
        )
        i += nprim
        exponents = rows[:, 0] * scale**2
        parts = (
            [(0, rows[:, 1]), (1, rows[:, 2])]
            if kind == "sp"
            else [(_L_LETTERS.index(kind), rows[:, 1])]
        )
        for l, d in parts:
            raw = d * _xl_norm(l, exponents) if normalized_primitives else d
            shells.append(
                Shell(
                    center=atom,
                    angular_momentum=l,
                    pure=l < 2 or pure.get(l, False),
                    exponents=exponents,
                    coefficients=_normalize(l, exponents, raw),
                )
            )
    return shells


def _xl_norm(l: int, a: np.ndarray) -> np.ndarray:
    """Normalization constant of the primitive x^l exp(-a r^2)."""
    return np.sqrt((2 * a / math.pi) ** 1.5 * (4 * a) ** l / _dfact(2 * l - 1))


def _normalize(l: int, a: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Scale coefficients of bare primitives so that x^l of the shell has norm 1."""
    p = a[:, None] + a[None, :]
    s = float(raw @ ((math.pi / p) ** 1.5 * _dfact(2 * l - 1) / (2 * p) ** l) @ raw)
    return raw / math.sqrt(s)


def _read_mo(lines: list[str], nao: int) -> Orbitals:
    """The orbitals of the ``[MO]`` section."""
    mos: list[_Mo] = []
    cur: _Mo | None = None
    seen: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        if "=" in line:
            key, value = (x.strip() for x in line.split("=", 1))
            key = key.lower()
            if cur is None or cur.coefficients or key in seen:
                cur = _Mo()
                mos.append(cur)
                seen = set()
            seen.add(key)
            if key == "ene":
                cur.energy = _float(value)
            elif key == "spin":
                cur.spin = value.lower()
            elif key == "occup":
                cur.occupation = _float(value)
            continue
        if cur is None:
            raise ValueError(
                f"an [MO] coefficient line comes before its orbital: {line!r}"
            )
        f = line.split()
        cur.coefficients[int(f[0])] = _float(f[1])
    if not mos:
        raise ValueError("the [MO] section has no orbitals")
    alpha = [m for m in mos if m.spin != "beta"]
    beta = [m for m in mos if m.spin == "beta"]
    for k, m in enumerate(mos):
        if m.energy is None:
            raise ValueError(f"orbital {k + 1} of the [MO] section has no Ene=")
        if m.coefficients and max(m.coefficients) > nao:
            raise ValueError(
                f"orbital {k + 1} has a coefficient for function {max(m.coefficients)}, "
                f"but the [GTO] basis set has {nao} functions; check the [5D], [7F] "
                "and [9G] flags"
            )

    def matrix(ms: list[_Mo]) -> np.ndarray:
        out = np.zeros((nao, len(ms)))
        for k, m in enumerate(ms):
            for j, x in m.coefficients.items():
                out[j - 1, k] = x
        return out

    def levels(ms: list[_Mo]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([m.energy for m in ms], dtype=float),
            np.array([m.occupation for m in ms]),
        )

    ea, na = levels(alpha)
    ca = matrix(alpha)
    if not beta:
        if np.any(na < 0) or np.any(na > 2):
            raise ValueError(
                "a Molden file with one set of orbitals has 0 to 2 electrons in each"
            )
        if np.all(np.isin(na, (0.0, 1.0, 2.0))):
            occ_a, occ_b = (na >= 1).astype(float), (na >= 2).astype(float)
        else:
            occ_a = occ_b = na / 2
        return Orbitals(
            spin=OrbitalSpin.restricted,
            energies_a=ea,
            occupations_a=occ_a,
            occupations_b=occ_b,
            coefficients_a=ca,
        )
    if len(alpha) != len(beta):
        raise ValueError(
            f"the alpha and beta sets must have as many orbitals; this file has "
            f"{len(alpha)} alpha and {len(beta)} beta orbitals"
        )
    eb, nb = levels(beta)
    cb = matrix(beta)
    for name, occ in (("alpha", na), ("beta", nb)):
        if np.any(occ < 0) or np.any(occ > 1):
            raise ValueError(f"the {name} orbitals must hold 0 to 1 electrons each")
    if np.array_equal(ea, eb) and np.array_equal(ca, cb):
        return Orbitals(
            spin=OrbitalSpin.restricted,
            energies_a=ea,
            occupations_a=na,
            occupations_b=nb,
            coefficients_a=ca,
        )
    return Orbitals(
        spin=OrbitalSpin.unrestricted,
        energies_a=ea,
        energies_b=eb,
        occupations_a=na,
        occupations_b=nb,
        coefficients_a=ca,
        coefficients_b=cb,
    )
