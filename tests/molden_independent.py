"""An independent reader of Molden files, for checking qcinf.molden.

Nothing here uses qcinf.molden or qcdata's AO conventions. A small parser reads a
file's atoms, shells, flags and orbitals; `Basis` builds every basis function from the
shells alone, as explicit polynomials times contracted Gaussians:

- spherical functions are the real solid harmonics (no Condon-Shortley phase) written
  out below as polynomials, normalized numerically with the overlap;
- Cartesian components are monomials, each normalized on its own (``unit``, Molden's
  convention) or all with the norm of x^l (``axis``);
- overlaps of Cartesian Gaussians come from the Obara-Saika recursion, primitive pair by
  primitive pair, optionally skipping pairs on different atoms with
  a b R^2 / (a + b) above ``screen`` (axtb screens its overlap at 25).

`Basis.from_shells` builds the same functions for shells given directly, in any of the
orders spelled out here, to check qcdata wavefunctions before they are written.
"""

# l is the conventional symbol for angular momentum.
# ruff: noqa: E741
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from qcref import periodic_table as pt

L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

# Cartesian monomials (lx, ly, lz) in Molden's order and in lexicographic order.
MOLDEN_CARTESIAN = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (1, 2, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 0, 2),
        (0, 1, 2),
        (0, 2, 1),
        (1, 1, 1),
    ],
    4: [
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
        (3, 1, 0),
        (3, 0, 1),
        (1, 3, 0),
        (0, 3, 1),
        (1, 0, 3),
        (0, 1, 3),
        (2, 2, 0),
        (2, 0, 2),
        (0, 2, 2),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    ],
}
LEXICOGRAPHIC = {
    l: [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]
    for l in range(5)
}

# Real solid harmonics as unnormalized polynomials {(lx, ly, lz): weight}, by m.
SOLID_HARMONICS: dict[int, dict[int, dict[tuple[int, int, int], float]]] = {
    0: {0: {(0, 0, 0): 1}},
    1: {1: {(1, 0, 0): 1}, -1: {(0, 1, 0): 1}, 0: {(0, 0, 1): 1}},
    2: {
        0: {(0, 0, 2): 2, (2, 0, 0): -1, (0, 2, 0): -1},  # 2z^2 - x^2 - y^2
        1: {(1, 0, 1): 1},  # xz
        -1: {(0, 1, 1): 1},  # yz
        2: {(2, 0, 0): 1, (0, 2, 0): -1},  # x^2 - y^2
        -2: {(1, 1, 0): 1},  # xy
    },
    3: {
        0: {(0, 0, 3): 2, (2, 0, 1): -3, (0, 2, 1): -3},  # z(2z^2 - 3x^2 - 3y^2)
        1: {(1, 0, 2): 4, (3, 0, 0): -1, (1, 2, 0): -1},  # x(4z^2 - x^2 - y^2)
        -1: {(0, 1, 2): 4, (2, 1, 0): -1, (0, 3, 0): -1},  # y(4z^2 - x^2 - y^2)
        2: {(2, 0, 1): 1, (0, 2, 1): -1},  # z(x^2 - y^2)
        -2: {(1, 1, 1): 1},  # xyz
        3: {(3, 0, 0): 1, (1, 2, 0): -3},  # x(x^2 - 3y^2)
        -3: {(2, 1, 0): 3, (0, 3, 0): -1},  # y(3x^2 - y^2)
    },
    4: {
        # 35z^4 - 30z^2 r^2 + 3r^4
        0: {
            (0, 0, 4): 8,
            (2, 0, 2): -24,
            (0, 2, 2): -24,
            (4, 0, 0): 3,
            (2, 2, 0): 6,
            (0, 4, 0): 3,
        },
        1: {(1, 0, 3): 4, (3, 0, 1): -3, (1, 2, 1): -3},  # xz(7z^2 - 3r^2)
        -1: {(0, 1, 3): 4, (2, 1, 1): -3, (0, 3, 1): -3},  # yz(7z^2 - 3r^2)
        # (x^2 - y^2)(7z^2 - r^2)
        2: {(2, 0, 2): 6, (0, 2, 2): -6, (4, 0, 0): -1, (0, 4, 0): 1},
        -2: {(1, 1, 2): 6, (3, 1, 0): -1, (1, 3, 0): -1},  # xy(7z^2 - r^2)
        3: {(3, 0, 1): 1, (1, 2, 1): -3},  # xz(x^2 - 3y^2)
        -3: {(2, 1, 1): 3, (0, 3, 1): -1},  # yz(3x^2 - y^2)
        4: {(4, 0, 0): 1, (2, 2, 0): -6, (0, 4, 0): 1},  # x^4 - 6x^2 y^2 + y^4
        -4: {(3, 1, 0): 1, (1, 3, 0): -1},  # xy(x^2 - y^2)
    },
}
# Orders of the spherical functions, by m.
MOLDEN_SPHERICAL = {
    l: (
        [1, -1, 0]
        if l == 1
        else [0] + [s * m for m in range(1, l + 1) for s in (1, -1)]
    )
    for l in range(5)
}
PYSCF_SPHERICAL = {
    l: ([1, -1, 0] if l == 1 else list(range(-l, l + 1))) for l in range(5)
}


def prim_norm(l: int, a: np.ndarray) -> np.ndarray:
    """Norm of the primitive x^l exp(-a r^2)."""
    return (
        (2 * a / math.pi) ** 0.75
        * (4 * a) ** (l / 2)
        / math.sqrt(math.prod(range(2 * l - 1, 0, -2)))
    )


# ------------------------------------------------------------------- parsing


@dataclass
class Molden:
    """What a Molden file holds; positions in bohr."""

    atoms: list[tuple[int, np.ndarray]]
    shells: list[tuple[int, int, np.ndarray, np.ndarray]]  # atom, l, exponents, coefs
    pure: dict[int, bool]
    mos: list[dict]  # energy, spin, occ, coef (array)
    title: str

    def coefficients(self, spin: str) -> np.ndarray:
        return np.array([m["coef"] for m in self.mos if m["spin"] == spin]).T

    def column(self, key: str, spin: str) -> np.ndarray:
        return np.array([m[key] for m in self.mos if m["spin"] == spin])

    @property
    def spins(self) -> list[str]:
        return sorted({m["spin"] for m in self.mos})


def parse(text: str) -> Molden:
    """Parse a Molden file (a reader of its own, written separately from qcinf's)."""
    atoms: list[tuple[int, np.ndarray]] = []
    shells: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    mos: list[dict[str, Any]] = []
    title = ""
    flags: set[str] = set()
    section = ""
    scale = 1.0
    lines = text.splitlines()
    i, atom = 0, 0
    cur: dict[str, Any] = {}
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        head = re.match(r"^\[([^\]]+)\](.*)$", line)
        if head:
            section = head.group(1).strip().lower()
            flags.add(section)
            if section == "atoms":
                scale = 1.0 if "au" in head.group(2).lower() else 1.8897261246257702
            if section == "title" and i < len(lines) and not lines[i].startswith("["):
                title = lines[i]
            continue
        f = line.split()
        if not f:
            continue
        if section == "atoms":
            label = re.match(r"[A-Za-z]+", f[0])
            element = pt.get(label.group(0).capitalize()) if label else None
            assert element is not None, f[0]
            atoms.append((element.number, np.array([float(x) for x in f[3:6]]) * scale))
        elif section == "gto":
            if f[0].lower() not in L_OF:
                atom = int(f[0]) - 1
                continue
            l, n = L_OF[f[0].lower()], int(f[1])
            p = np.array(
                [
                    [float(x.replace("D", "E")) for x in lines[i + k].split()[:2]]
                    for k in range(n)
                ]
            )
            i += n
            shells.append((atom, l, p[:, 0], p[:, 1]))
        elif section == "mo":
            if "=" in line:
                key, value = (x.strip() for x in line.split("=", 1))
                key = key.lower()
                if key == "sym" or not cur:
                    cur = {"energy": None, "spin": "alpha", "occ": 0.0, "coef": {}}
                    mos.append(cur)
                if key == "ene":
                    cur["energy"] = float(value.replace("D", "E"))
                elif key == "spin":
                    cur["spin"] = value.lower()
                elif key == "occup":
                    cur["occ"] = float(value.replace("D", "E"))
            else:
                cur["coef"][int(f[0]) - 1] = float(f[1].replace("D", "E"))
    pure = {
        2: bool(flags & {"5d", "5d7f", "5d10f"}),
        3: bool(flags & {"5d", "5d7f", "7f"}) and not flags & {"5d10f", "10f"},
        4: "9g" in flags,
    }
    nbf = sum(
        2 * l + 1 if pure.get(l) else len(MOLDEN_CARTESIAN[l]) for _, l, _, _ in shells
    )
    for mo in mos:
        c = np.zeros(nbf)
        for k, v in mo["coef"].items():
            c[k] = v
        mo["coef"] = c
    return Molden(atoms, shells, pure, mos, title)


# ------------------------------------------------------------------- integrals


def _s1d(la: int, lb: int, a: np.ndarray, b: np.ndarray, xa: float, xb: float) -> list:
    """1-D overlaps t[i][j] of x^i and x^j Gaussians about xa and xb, per primitive pair."""
    p = a[:, None] + b[None, :]
    xp = (a[:, None] * xa + b[None, :] * xb) / p
    h = 0.5 / p
    t = [[None] * (lb + 1) for _ in range(la + 1)]
    t[0][0] = np.sqrt(math.pi / p) * np.exp(
        -a[:, None] * b[None, :] / p * (xa - xb) ** 2
    )
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == j == 0:
                continue
            if i > 0:
                v = (xp - xa) * t[i - 1][j]
                if i > 1:
                    v = v + (i - 1) * h * t[i - 2][j]
                if j > 0:
                    v = v + j * h * t[i - 1][j - 1]
            else:
                v = (xp - xb) * t[i][j - 1]
                if j > 1:
                    v = v + (j - 1) * h * t[i][j - 2]
            t[i][j] = v
    return t


@dataclass
class _Shell:
    center: np.ndarray
    l: int
    exponents: np.ndarray
    weights: np.ndarray  # coefficients of bare primitives
    trafo: np.ndarray  # functions x Cartesian monomials (Molden order)


def _monomial_block(sa: _Shell, sb: _Shell, screen: float) -> np.ndarray:
    tx = [
        _s1d(sa.l, sb.l, sa.exponents, sb.exponents, sa.center[k], sb.center[k])
        for k in range(3)
    ]
    w = np.outer(sa.weights, sb.weights)
    r2 = float(np.sum((sa.center - sb.center) ** 2))
    if screen and r2 > 0:
        a, b = sa.exponents[:, None], sb.exponents[None, :]
        w = np.where(a * b / (a + b) * r2 > screen, 0.0, w)
    ca, cb = MOLDEN_CARTESIAN[sa.l], MOLDEN_CARTESIAN[sb.l]
    out = np.empty((len(ca), len(cb)))
    for i, ea in enumerate(ca):
        for j, eb in enumerate(cb):
            out[i, j] = np.sum(
                w * tx[0][ea[0]][eb[0]] * tx[1][ea[1]][eb[1]] * tx[2][ea[2]][eb[2]]
            )
    return out


class Basis:
    """Basis functions as transforms of Cartesian monomials times contracted Gaussians.

    Every contracted function is normalized: spherical functions and ``unit`` Cartesian
    components to 1, ``axis`` components with the norm of x^l.
    """

    def __init__(self, shells: list[_Shell]):
        self.shells = shells
        self.nbf = sum(s.trafo.shape[0] for s in shells)

    @classmethod
    def from_molden(
        cls, mol: Molden, primitives: str = "normalized", cartesian: str = "unit"
    ) -> Basis:
        """The functions of a parsed file; ``primitives="raw"`` for coefficients of bare
        exp(-a r^2) (xtb, tblite, g-xTB)."""
        specs = []
        for atom, l, exps, coefs in mol.shells:
            w = coefs * prim_norm(l, exps) if primitives == "normalized" else coefs
            specs.append((mol.atoms[atom][1], l, l >= 2 and mol.pure.get(l), exps, w))
        return cls.from_shells(specs, MOLDEN_SPHERICAL, MOLDEN_CARTESIAN, cartesian)

    @classmethod
    def from_shells(
        cls, specs, spherical_order, cartesian_order, cartesian: str = "unit"
    ):
        """Shells (center, l, pure, exponents, coefficients of bare primitives) with the
        functions in the given orders ({l: [m]}, {l: [monomials]})."""
        shells = []
        for center, l, pure, exps, w in specs:
            sh = _Shell(
                np.asarray(center, float),
                l,
                np.asarray(exps, float),
                np.asarray(w, float),
                np.eye(1),
            )
            mono = MOLDEN_CARTESIAN[l]
            sh.trafo = np.eye(len(mono))
            self_overlap = _monomial_block(sh, sh, 0.0)
            if pure and l >= 1:
                t = np.zeros((2 * l + 1, len(mono)))
                for row, m in enumerate(spherical_order[l]):
                    for e, v in SOLID_HARMONICS[l][m].items():
                        t[row, mono.index(e)] = v
                    t[row] /= math.sqrt(t[row] @ self_overlap @ t[row])
            else:
                t = np.zeros((len(mono), len(mono)))
                axis = self_overlap[0, 0]
                for row, e in enumerate(cartesian_order[l]):
                    k = mono.index(e)
                    norm2 = self_overlap[k, k] if cartesian == "unit" else axis
                    t[row, k] = 1 / math.sqrt(norm2)
            sh.trafo = t
            shells.append(sh)
        return cls(shells)

    def overlap(self, other: Basis | None = None, screen: float = 0.0) -> np.ndarray:
        other = self if other is None else other
        s = np.zeros((self.nbf, other.nbf))
        i = 0
        for sa in self.shells:
            j = 0
            for sb in other.shells:
                na, nb = sa.trafo.shape[0], sb.trafo.shape[0]
                s[i : i + na, j : j + nb] = (
                    sa.trafo @ _monomial_block(sa, sb, screen) @ sb.trafo.T
                )
                j += nb
            i += na
        return s

    def values(self, points: np.ndarray) -> np.ndarray:
        """Function values, nbf x npoints."""
        out = []
        for sh in self.shells:
            d = points - sh.center
            radial = np.exp(-np.outer((d * d).sum(1), sh.exponents)) @ sh.weights
            mono = np.array(
                [
                    d[:, 0] ** e[0] * d[:, 1] ** e[1] * d[:, 2] ** e[2]
                    for e in MOLDEN_CARTESIAN[sh.l]
                ]
            )
            out.append(sh.trafo @ mono * radial)
        return np.vstack(out)

    def atoms(self, positions: list[np.ndarray]) -> np.ndarray:
        """The atom of each function."""
        out = []
        for sh in self.shells:
            atom = next(
                k for k, p in enumerate(positions) if np.array_equal(p, sh.center)
            )
            out += [atom] * sh.trafo.shape[0]
        return np.array(out)


# ------------------------------------------------------------------- checks


def orthonormality(mol: Molden, s: np.ndarray) -> float:
    """Largest |C^T S C - 1| over the file's orbital sets."""
    worst = 0.0
    for spin in mol.spins:
        c = mol.coefficients(spin)
        worst = max(worst, float(np.abs(c.T @ s @ c - np.eye(c.shape[1])).max()))
    return worst


def electron_count_error(mol: Molden, s: np.ndarray) -> float:
    """|tr(P S) - sum of occupations| with P = sum_i n_i c_i c_i^T over every set."""
    p = sum(m["occ"] * np.outer(m["coef"], m["coef"]) for m in mol.mos)
    return abs(float(np.trace(p @ s)) - sum(m["occ"] for m in mol.mos))


def mulliken_populations(mol: Molden, basis: Basis, screen: float = 0.0) -> np.ndarray:
    """Electrons on each atom, sum over its functions of (P S)_mu,mu."""
    s = basis.overlap(screen=screen)
    p = sum(m["occ"] * np.outer(m["coef"], m["coef"]) for m in mol.mos)
    pop = np.zeros(len(mol.atoms))
    np.add.at(pop, basis.atoms([xyz for _, xyz in mol.atoms]), (p @ s).diagonal())
    return pop


def points_around(mol: Molden, n: int = 400, seed: int = 7) -> np.ndarray:
    """Random points around the atoms (bohr)."""
    xyz = np.array([p for _, p in mol.atoms])
    rng = np.random.default_rng(seed)
    return xyz[rng.integers(0, len(xyz), n)] + rng.normal(scale=1.5, size=(n, 3))
