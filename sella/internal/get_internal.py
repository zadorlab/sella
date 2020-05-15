from typing import List, Collection, Union, Tuple
from ase import Atoms
from ase.data import covalent_radii
from itertools import product
from sella.internal.int_classes import CartToInternal, Constraints
from sella.constraints import cons_to_dict, ConstraintBuilder

import numpy as np

class InternalBuilder:
    # Maximum number of bonds allowed per atom
    _MAX_BONDS = 20

    # threshold number of bonds for an atom to be considered "Bulk-like"
    _BBULK = 6

    # Unit cell stuff
    tvecs = np.empty((0, 3), dtype=np.float64)
    tneg = np.empty(0, dtype=np.int32)

    # Atoms stuff
    nbonds = np.empty(0, dtype=np.int32)
    c10y = np.empty((0, _MAX_BONDS, 2), dtype=np.int32)

    # Internal coordinates
    cart = np.empty((0, 2), dtype=np.int32)
    bonds = np.empty((0, 3), dtype=np.int32)
    angles = np.empty((0, 5), dtype=np.int32)
    dihedrals = np.empty((0, 7), dtype=np.int32)

    # Angle-specific stuff
    bad_angles = None
    atol = None

    def __init__(
        self,
        atoms: Atoms,
        cons: ConstraintBuilder,
        dummies: Atoms = None,
        dinds: np.ndarray = None,
    ) -> None:
        self.atoms = atoms
        self.cons = cons

        self.dinds = dinds
        if dummies is None:
            assert self.dinds is None
            self.dummies = Atoms()
            self.dinds = -np.ones(self.natoms, dtype=np.int32)
        else:
            self.dummies = dummies

        self.nbonds = np.zeros(self.natoms, dtype=np.int32)
        self.c10y = -np.ones((self.natoms, self._MAX_BONDS, 2), dtype=np.int32)

        # set up periodic translation vectors
        tvecs = []
        for ijk in product((0, -1, 1), repeat=3):
            if np.any((np.asarray(ijk) != 0) * np.logical_not(self.atoms.pbc)):
                continue
            tvecs.append(np.asarray(ijk) @ self.atoms.cell)
        self.tvecs = np.array(tvecs, dtype=np.float64)

        # For each translation vector, find its inverse
        self.tneg = -np.ones(self.ntvecs, dtype=np.int32)
        for i in range(self.ntvecs):
            if self.tneg[i] != -1:
                # We've already found the inverse for i
                continue
            for j in range(i, self.ntvecs):
                if self.tneg[j] != -1:
                    # We've already found the inverse for j
                    continue
                if np.all(self.tvecs[i] == -self.tvecs[j]):
                    self.tneg[i] = j
                    self.tneg[j] = i
                    break
            else:
                # We couldn't find an inverse for i, this shouldn't
                # ever happen.
                raise RuntimeError("Couldn't find inverse of tvec!")
        assert np.all(self.tneg >= 0)

    @property
    def ntvecs(self) -> int:
        return len(self.tvecs)

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    @property
    def ndummies(self) -> int:
        return len(self.dummies)

    def flood_fill(self, idx: int, label: int, labels: np.ndarray) -> None:
        for j in self.c10y[idx, :self.nbonds[idx], 0]:
            if labels[j] != label:
                labels[j] = label
                self.flood_fill(j, label, labels)

    def validate_new_bond(self, i: int, j: int, n: int) -> None:
        if self.nbonds[i] >= self._MAX_BONDS:
            raise RuntimeError("Too many bonds to atom {}!".format(i))
        if self.nbonds[j] >= self._MAX_BONDS:
            raise RuntimeError("Too many bonds to atom {}!".format(j))
        if i == j and self.nbonds[i] >= self._MAX_BONDS - 1:
            raise RuntimeError("Too many bonds to atom {}!".format(i))

        for k in range(self.nbonds[i]):
            if np.all(self.c10y[i, k] == [j, n]):
                raise RuntimeError("Duplicate bond found: {} {} {}"
                                   .format(i, j, n))
        for k in range(self.nbonds[j]):
            if np.all(self.c10y[j, k] == [i, self.tneg[n]]):
                raise RuntimeError("Duplicate bond found: {} {} {}"
                                   .format(i, j, n))

    def get_distance(self, i: int, j: int, n: int,
                     vector: bool = False) -> float:
        allatoms = self.atoms + self.dummies
        dx = allatoms.positions[j] - allatoms.positions[i] + self.tvecs[n]
        if vector:
            return dx
        return np.linalg.norm(dx)

    def check_bonds(
        self, check_bonds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        added_bonds = [ijn for ijn in self.cons.user_cons.get('bonds', [])]
        forbidden_bonds = []
        if check_bonds is not None:
            g = -self.atoms.get_forces()
            for i, j, n in check_bonds:
                skip = False
                # skip checking if this bond has already been added, e.g.
                # from user constraints
                for ip, jp, npr in added_bonds:
                    if (
                        (i == ip and j == jp and n == npr)
                        or (i == jp and j == ip and n == self.tneg[npr])
                    ):
                        skip = True
                        break
                if skip:
                    continue
                dx = self.get_distance(i, j, n, vector=True)
                if dx @ (g[j] - g[i]) > 0:
                    added_bonds.append([i, j, n])
                else:
                    forbidden_bonds.append([i, j, n])
        added_bonds = np.array(added_bonds, dtype=np.int32).reshape((-1, 3))
        forbidden_bonds = np.array(forbidden_bonds,
                                   dtype=np.int32).reshape((-1, 3))
        return added_bonds, forbidden_bonds

    def find_bonds(self,
                   check_bonds: np.ndarray = None,
                   initial_scale: float = 1.25) -> None:
        rcov = covalent_radii[self.atoms.numbers].copy()
        labels = -np.ones_like(self.nbonds)

        added_bonds, forbidden_bonds = self.check_bonds(check_bonds)

        for i, j, n in added_bonds:
            c10y[i, nbonds[i]] = [j, n]
            nbonds[i] += 1
            c10y[j, nbonds[j]] = [i, self.tneg[n]]
            nbonds[j] += 1

        scale = initial_scale
        while True:
            labels[:] = -1
            nlabels = 0
            for i, label in enumerate(labels):
                if label == -1:
                    labels[i] = nlabels
                    self.flood_fill(i, nlabels, labels)
                    nlabels += 1
                if self.nbonds[i] == 0:
                    labels[i] = -1
            if nlabels == 1 and np.all(labels >= 0):
                break

            for i in range(self.natoms):
                for n, tvec in enumerate(self.tvecs):
                    for j in range(i, self.natoms):
                        if i == j:
                            if n >= self.tneg[n]:
                                continue
                        elif labels[i] == labels[j] != -1:
                            continue
                        forbidden = False
                        for ip, jp, npr in forbidden_bonds:
                            if (
                                (ip == i and jp == j and npr == n)
                                or (ip == j and jp == i and npr == self.tneg[n])
                            ):
                                forbidden = True
                                break
                        if forbidden:
                            continue
                        dist = self.get_distance(i, j, n)
                        if dist <= scale * (rcov[i] + rcov[j]):
                            self.validate_new_bond(i, j, n)

                            self.c10y[i, self.nbonds[i]] = [j, n]
                            self.nbonds[i] += 1
                            self.c10y[j, self.nbonds[j]] = [i, self.tneg[n]]
                            self.nbonds[j] += 1
            scale *= 1.05

        self.bulklike = np.array((self.nbonds >= self._BBULK), dtype=np.uint8)

        bonds = []
        cart = []
        a = 0
        for i in range(self.natoms):
            if self.bulklike[i]:
                for j in range(3):
                    cart.append([i, j])
            for j, n in self.c10y[i, :self.nbonds[i]]:
                if n > self.tneg[n]:
                    continue
                elif n == self.tneg[n] and i > j:
                    continue
                elif i == j:
                    continue
                if self.bulklike[i] and self.bulklike[j]:
                    continue
                bonds.append([i, j, n])
            if self.dinds[i] > 0:
                bonds.append([i, self.dinds[i], 0])
                self.bond_constraints.append([i, self.dinds[i], 0])
        self.bonds = np.array(bonds, dtype=np.int32).reshape((-1, 3))
        self.cart = np.array(cart, dtype=np.int32).reshape((-1, 2))

    def get_angle(self, i: int, j: int, k: int, n: int, m: int) -> float:
        v1 = self.get_distance(i, j, n, True)
        v2 = self.get_distance(j, k, m, True)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        return 180 * np.arccos(v1 @ v2) / np.pi

    def check_angle(self, i: int, j: int, k: int, n: int, m: int) -> bool:
        if self.bulklike[i] and self.bulklike[j] and self.bulklike[k]:
            return False
        if self.bad_angles is not None:
            for a, b, c, na, nc in self.bad_angles:
                if b != j:
                    continue
                if a == i and c == k and na == n and nc == n:
                    return False
                if (a == k and c == i and na == self.tneg[m]
                        and nc == self.tneg[n]):
                    return False
        angle = self.get_angle(i, j, k, n, m)
        return self.atol < angle < 180 - self.atol

    def is_planar(self, dxs: np.ndarray) -> bool:
        dxs /= np.linalg.norm(dxs, axis=1)[:, np.newaxis]
        U, s, VT = np.linalg.svd(dxs)
        angles = np.arccos(dxs @ VT[:, 2]) * 180 / np.pi
        return np.max(angles) - np.min(angles) < self.atol

    def add_dummy(self, j: int, i: int, k: int, n: int, m: int) -> None:
        self.dinds[j] = self.natoms + self.ndummies
        dx1 = self.get_distance(j, i, n, True)
        dx2 = self.get_distance(j, k, m, True)
        dx1 /= np.linalg.norm(dx1)
        dx2 /= np.linalg.norm(dx2)
        dpos = dx1 + dx2
        dpos -= dx1 * (dpos @ dx1)
        dpos_norm = np.linalg.norm(dpos)
        if dpos_norm < 1e-4:
            dim = np.argmin(np.abs(dx1))
            dpos[:] = 0
            dpos[dim] = 1.
            dpos -= dx1 * (dpos @ dx1)
            dpos /= np.linalg.norm(dpos)
        else:
            dpos /= dpos_norm
        dpos += self.atoms.positions[j]
        self.dummies += Atom('X', dpos)
        self.cons.add_constraint('bonds', j, self.dinds[j], 0)
        self.cons.add_constraint(
            'angles', i, j, self.dinds[j], self.tneg[n], 0
        )
        self.cons.add_constraint(
            'dihedrals', k, j, self.dinds[j], i, self.tneg[m], 0, self.tneg[n]
        )

    def add_dummy_planar(self, j: int, linear: np.ndarray) -> None:
        self.dinds[j] = self.natoms + self.ndummies
        a = linear[0, 0]
        na = linear[0, 2]
        if len(linear) == 1:
            i, k, n, m = linear[0]
            for b, nb in sorted(
                self.c10y[j, :self.nbonds[j]],
                key=lambda x: self.get_distance(j, x[0], x[1])
            ):
                if not ((b == i and nb == self.tneg[n])
                        or (b == k and nb == m)):
                    break
            else:
                raise RuntimeError(
                    "Planar atom center found with only one linear angle, "
                    "but all bonded atom indices belong to linear angles! "
                    "This should be impossible!"
                )
        else:
            b = linear[1, 0]
            nb = linear[1, 2]
        dx1 = self.get_distance(j, a, na, vector=True)
        dx1 /= np.linalg.norm(dx1)
        dx2 = self.get_distance(j, b, nb, vector=True)
        dx2 /= np.linalg.norm(dx2)
        dpos = np.cross(dx1, dx2)
        dpos_norm = np.linalg.norm(dpos)
        if dpos_norm < 1e-4:
            dmin = np.argmin(np.abs(dx1))
            dpos[:] = 0
            dpos[dim] = 1
            dpos -= dx1 * (dpos @ dx1)
        else:
            dpos /= dpos_norm
        self.dummies += Atom('X', dpos + self.atoms.positions[j])
        self.cons.add_constraint('bonds', j, self.dinds[j], 0)
        self.cons.add_constraint(
            'angles', a, j, self.dinds[j], self.tneg[na], 0
        )
        self.cons.add_constraint(
            'angles', b, j, self.dinds[j], self.tneg[nb], 0
        )

    def angle_equiv(self, a1: np.ndarray, a2: np.ndarray) -> bool:
        if np.all(a1 == a2):
            return True
        if (
            np.all(a1[:3] == a2[2::-1])
            and np.all(a1[3:] == self.tneg[a2[:2:-1]])
        ):
            return True
        return False

    def find_angles(self, atol: float, bad_angles: np.ndarray = None) -> None:
        self.atol = atol
        self.bad_angles = bad_angles

        angles = []

        pos = self.atoms.positions

        # Loop over angle centers
        for j, nj in enumerate(self.nbonds):
            #if self.bulklike[j]:
            #    continue
            if nj == 0:
                # We found an atom with no bonds, which isn't allowed.
                raise RuntimeError("Atom {} has no bonds!".format(j))
            elif nj == 2:
                # If an atom has two bonds, then it only has one angle
                (i, nneg), (k, m) = self.c10y[j, :2]
                n = self.tneg[nneg]
                if self.check_angle(i, j, k, n, m):
                    # if the one angle is not close to linear, then add it to
                    # the list of angles
                    angles.append([i, j, k, n, m])
                elif self.dinds[j] < 0 and not self.bulklike[j]:
                    # otherwise if we don't already have a dummy atom, add one
                    self.add_dummy(j, i, k, nneg, m)
            elif nj > 2:
                linear = []  # list of linear angles with j as center
                # double loop over all unique pairs of bonds
                for a, (i, nneg) in enumerate(self.c10y[j, :nj-1]):
                    n = self.tneg[nneg]
                    for k, m in self.c10y[j, a+1:nj]:
                        if self.check_angle(i, j, k, n, m):
                            # if the angle is not close to linear, add it
                            angles.append([i, j, k, n, m])
                        elif not self.bulklike[j]:
                            # otherwise add it to the list of linear angles.
                            # For permutational invariance, order the angle
                            # indices such that the i-j distance is less than
                            # the j-k distance.
                            dx1 = self.get_distance(i, j, n)
                            dx2 = self.get_distance(j, k, m)
                            if dx2 < dx1:
                                i, k = k, i
                                n, m = self.tneg[m], self.tneg[n]
                            linear.append([i, k, n, m])
                if self.dinds[j] < 0 and linear and not self.bulklike[j]:
                    # we have at least one linear angle and no dummy atom yet
                    dxs = pos[self.c10y[j, :nj, 0]] - pos[j]
                    dxs += self.tvecs[self.c10y[j, :nj, 1]]
                    if self.is_planar(dxs):
                        # if the system is (close to) planar, add a dummy atom
                        # in the direction of the plane normal
                        linear = np.array(sorted(
                            linear,
                            key=lambda x: self.get_distance(x[0], j, x[2])
                        ), dtype=np.int32)
                        self.add_dummy_planar(j, linear)
            if self.dinds[j] >= 0:
                for i, n in self.c10y[j, :nj]:
                    if self.check_angle(i, j, self.dinds[j], self.tneg[n], 0):
                        angles.append([i, j, self.dinds[j], self.tneg[n], 0])

        # add constraint angles
        for con_angles in [self.cons.user_cons, self.cons.dummy_cons]:
            for a1 in con_angles['angles']:
                for a2 in angles:
                    if self.angle_equiv(np.asarray(a1), np.asarray(a2)):
                        break
                else:
                    angles.append(a1)

        self.angles = np.array(angles, dtype=np.int32).reshape((-1, 5))

    def validate_dihedral(self, dihedral: List[int]) -> bool:
        if dihedral[0] == dihedral[3]:
            a, b, c = dihedral[4:]
            ttot = self.tvecs[a] + self.tvecs[b] + self.tvecs[c]
            if np.linalg.norm(ttot) < 1e-8:
                return False
        return True

    def stitch_angles(self, a: np.ndarray, b: np.ndarray) -> List[List[int]]:
        # Given angles a and b we can construct a dihedral so long as one end
        # of each angle corresponds with the center of the other.
        # In PBC, it *may* be possible for two angles to match like this in
        # more than one way (at least, I haven't been able to convince myself
        # that this is *not* possible).
        # a + b
        # a0 a1 a2 b2 a3 a4 b4
        # a0 b0 b1 b2 a3 b3 b4
        dihedrals = []
        if a[1] == b[0] and a[2] == b[1] and a[4] == b[3]:
            dihedrals.append([a[0], a[1], a[2], b[2], a[3], a[4], b[4]])
        # a + rev(b)
        # a0 a1 a2 b0 a3 a4 tneg[b3]
        # a0 b2 b1 b0 a3 tneg[b4] tneg[b3]
        if a[1] == b[2] and a[2] == b[1] and a[4] == self.tneg[b[4]]:
            dihedrals.append([a[0], a[1], a[2], b[0], a[3], a[4],
                              self.tneg[b[3]]])
        # rev(a) + b
        # a2 a1 a0 b2 tneg[a4] tneg[a3] b4
        # a2 b0 b1 b2 tneg[a4] b3 b4
        if a[1] == b[0] and a[0] == b[1] and self.tneg[a[3]] == b[3]:
            dihedrals.append([a[2], a[1], a[0], b[2], self.tneg[a[4]],
                              b[3], b[4]])
        # rev(a) + rev(b)
        # a2 a1 a0 b0 tneg[a4] tneg[a3] tneg[b3]
        # a2 b2 b1 b0 tneg[a4] tneg[b4] tneg[b3]
        if a[1] == b[2] and a[0] == b[1] and a[3] == b[4]:
            dihedrals.append([a[2], a[1], a[0], b[0], self.tneg[a[4]],
                              self.tneg[a[3]], self.tneg[b[3]]])
        return dihedrals

    def dihedral_equiv(self, d1: np.ndarray, d2: np.ndarray) -> bool:
        if np.all(d1 == d2):
            return True
        if (
            np.all(d1[:4] == d2[3::-1])
            and np.all(d1[4:] == self.tneg[d2[:3:-1]])
        ):
            return True
        return False


    def find_dihedrals(self) -> None:
        dihedrals = []
        for i, a in enumerate(self.angles):
            for b in self.angles[i+1:]:
                for d in filter(
                    self.validate_dihedral, self.stitch_angles(a, b)
                ):
                    dihedrals.append(d)

        # add constraint angles
        for con_di in [self.cons.user_cons, self.cons.dummy_cons]:
            for d1 in con_di['dihedrals']:
                for d2 in dihedrals:
                    if self.dihedral_equiv(np.asarray(d1), np.asarray(d2)):
                        break
                else:
                    dihedrals.append(d1)

        self.dihedrals = np.array(dihedrals, dtype=np.int32).reshape((-1, 7))

    def to_internal(self) -> CartToInternal:
        return CartToInternal(
            self.atoms,
            cart=self.cart,
            bonds=self.bonds,
            angles=self.angles,
            dihedrals=self.dihedrals,
            bulklike=self.bulklike,
            tneg=self.tneg,
            cellvecs=self.tvecs,
            dummies=self.dummies,
            dinds=self.dinds,
            atol=self.atol
        )

    def to_constraints(self) -> Constraints:
        return self.cons.get_constraints(self.atoms, self.dummies, self.dinds)
