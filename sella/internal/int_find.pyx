# cython: language_level=3

# cimports

cimport cython

from libc.math cimport fabs, acos, pi, HUGE_VAL
from libc.string cimport memset

from scipy.linalg.cython_blas cimport ddot, dcopy

from sella.utilities.math cimport my_daxpy, normalize, vec_sum, cross
from sella.internal.int_eval cimport cart_to_angle

# imports
import numpy as np
from ase import Atoms
from ase.data import covalent_radii

# Maximum number of bonds each atom is allowed to have
cdef int _MAX_BONDS = 20

# Constants for BLAS/LAPACK calls
cdef int UNITY = 1
cdef int THREE = 3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int flood_fill(int idx, int[:] nbonds, int[:, :] c10y, int[:] labels,
                    int label) nogil:
    cdef int i, j, info
    for i in range(nbonds[idx]):
        j = c10y[idx, i]
        if labels[j] != label:
            labels[j] = label
            info = flood_fill(j, nbonds, c10y, labels, label)
            if info != 0:
                return info
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_bonds(atoms):
    cdef int natoms = len(atoms)
    cdef int i, j, k, err
    cdef int nbonds_tot = 0
    cdef double scale = 1.

    #bonds_np = np.zeros((natoms * _MAX_BONDS // 2, 2), dtype=np.int32)
    #cdef int[:, :] bonds = memoryview(bonds_np)

    rcov_np = covalent_radii[atoms.numbers].copy()
    cdef double[:] rcov = memoryview(rcov_np)

    rij_np = atoms.get_all_distances()
    cdef double[:, :] rij = memoryview(rij_np)

    nbonds_np = np.zeros(natoms, dtype=np.int32)
    cdef int[:] nbonds = memoryview(nbonds_np)

    labels_np = np.arange(natoms, dtype=np.int32)
    cdef int[:] labels = memoryview(labels_np)
    cdef int nlabels = natoms

    c10y_np = -np.ones((natoms, _MAX_BONDS), dtype=np.int32)
    cdef int[:, :] c10y = memoryview(c10y_np)

    with nogil:
        # Loop until all atoms share the same label
        while nlabels > 1 and err == 0:
            # Loop over all pairs of atoms s.t. i < j
            for i in range(natoms):
                for j in range(i + 1, natoms):
                    # if i and j have the same label , that means they are
                    # already connected (potentially indirectly), so we will
                    # not add a new bond between them.
                    if labels[i] == labels[j]:
                        continue

                    # Check whether the two atoms are within the scaled
                    # bonding cutoff
                    if rij[i, j] <= scale * (rcov[i] + rcov[j]):
                        # Each atom can only have at most _MAX_BONDS bonding
                        # partners, raise an error if this is violated
                        if (nbonds[i] >= _MAX_BONDS):
                            err = i
                        elif (nbonds[j] >= _MAX_BONDS):
                            err = j
                        if err != 0: break

                        # This is just a sanity check: if atoms i and j have
                        # different labels, then they shouldn't be bonded
                        # to each other already.
                        for k in range(nbonds[i]):
                            if j == c10y[i, k]:
                                err = -1
                                break
                        if err != 0: break

                        # Same explanation as above
                        for k in range(nbonds[j]):
                            if i == c10y[j, k]:
                                err = -1
                                break
                        if err != 0: break

                        # Add a bond between i and j and update all relevant
                        # quantities
                        #bonds[nbonds_tot, 0] = i
                        #bonds[nbonds_tot, 1] = j
                        nbonds_tot += 1
                        c10y[i, nbonds[i]] = j
                        nbonds[i] += 1
                        c10y[j, nbonds[j]] = i
                        nbonds[j] += 1
                if err != 0: break
            if err != 0: break

            # Reset all atom labels
            for i in range(natoms):
                labels[i] = -1

            # Rules for labeling: two atoms must have the same label if
            # they are connected, directly or indirectly. Disconnected
            # subfragments must have different labels.
            #
            # This is done by assigning atom 0 the label 0, then walking
            # the connectivity tree s.t. all atoms connected to 0 also
            # have the same label. Once this is complete, the label is
            # incremented and assigned to the next unlabeled atom. This
            # process is repeated until all atoms are labeled.
            #
            # If all atoms have the same label, that means the bonding
            # network is fully connected, and we can move on.
            nlabels= 0
            for i in range(natoms):
                if labels[i] == -1:
                    labels[i] = nlabels
                    flood_fill(i, nbonds, c10y, labels, nlabels)
                    nlabels += 1
            scale *= 1.05

    bonds_np = np.zeros((nbonds_tot, 2), dtype=np.int32)
    cdef int[:, :] bonds = memoryview(bonds_np)
    cdef int n = 0
    with nogil:
        for i in range(natoms):
            for j in range(nbonds[i]):
                k = c10y[i, j]
                if k <= i:  continue
                if n >= nbonds_tot:
                    err = 1
                    break
                bonds[n, 0] = i
                bonds[n, 1] = k
                n += 1
            if err != 0: break
    return bonds_np, nbonds_np, c10y_np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_angles(atoms, double atol, int[:, :] bonds, int[:] nbonds,
                int[:, :] c10y, dummies=None, dinds_np=None,
                int[:, :] angle_sums_old=None):
    cdef int i, j, k, l, nj, a, b
    cdef int ii, jj, kk, ll, n
    cdef int natoms = len(atoms)
    cdef int nbonds_tot = len(bonds)

    cdef double[:, :] pos = memoryview(atoms.positions)

    cdef double angle

    # Dummy atoms
    if dummies is None:
        dummies = Atoms()
    cdef int ndummies = len(dummies)
    dummypos_np = np.zeros((natoms, 3), dtype=np.float64)
    dummypos_np[:ndummies] = dummies.positions
    cdef double[:, :] dummypos = memoryview(dummypos_np)

    # Bending angles
    cdef int nangles_max = nbonds_tot * _MAX_BONDS // 2
    angles_np = -np.ones((nangles_max, 3), dtype=np.int32)
    cdef int[:, :] angles = memoryview(angles_np)
    cdef int nangles = 0

    # Bond stretch constraints w/ dummies
    bond_constraints_np = -np.ones((natoms, 2), dtype=np.int32)
    cdef int[:, :] bond_constraints = memoryview(bond_constraints_np)
    cdef int nbond_constraints = 0

    # Angle constraints w/ dummies
    angle_constraints_np = -np.ones((natoms, 3), dtype=np.int32)
    cdef int[:, :] angle_constraints = memoryview(angle_constraints_np)
    cdef int nangle_constraints = 0

    # Old angle sums, used to shortcut linear angle detection
    cdef int nangle_sums_old
    if angle_sums_old is None:
        nangle_sums_old = 0
    else:
        nangle_sums_old = angle_sums_old.shape[0]

    # Angle sums w/ dummies
    angle_sums_np = -np.ones((natoms, 4), dtype=np.int32)
    cdef int[:, :] angle_sums = memoryview(angle_sums_np)
    cdef int nangle_sums = 0

    # Angle diffs w/ dummies
    angle_diffs_np = -np.ones((natoms, 4), dtype=np.int32)
    cdef int[:, :] angle_diffs = memoryview(angle_diffs_np)
    cdef int nangle_diffs = 0

    # Dihedral constraints w/ dummies
    dihedral_constraints_np = -np.ones((natoms, 4), dtype=np.int32)
    cdef int[:, :] dihedral_constraints = memoryview(dihedral_constraints_np)
    cdef int ndihedral_constraints = 0

    # Which dummy atom corresponds to each real atom
    # A value of -1 means no dummy atom
    if dinds_np is None:
        dinds_np = -np.ones(natoms, dtype=np.int32)
    cdef int[:] dinds = memoryview(dinds_np)

    # Set of basis vectors for determining whether an atom is planar
    basis_np = np.zeros((3, 3), dtype=np.float64)
    cdef double[:, :] basis = memoryview(basis_np)
    cdef int nbasis

    # temporary arrys for various things below
    work_np = np.zeros((1, 1, 1, 3), dtype=np.float64)
    cdef double[:, :, :, :] work = memoryview(work_np)

    dx1_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx1 = memoryview(dx1_np)

    dx2_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx2 = memoryview(dx2_np)

    # We don't actually use dq or d2q anywhere in this function,
    # but we need to call cart_to_angle to check for linearity, and
    # those functions require dq and d2q.
    #
    # Can we somehow find a way to avoid allocating these arrays?
    # It probably doesn't matter from a performance standpoint, but
    # it's the principle of the thing.
    dq_np = np.zeros((natoms, 3), dtype=np.float64)
    cdef double[:, :] dq = memoryview(dq_np)

    d2q_np = np.zeros((4, 3, 4, 3), dtype=np.float64)
    cdef double[:, :, :, :] d2q = memoryview(d2q_np)

    di_inds_np = np.zeros(4, dtype=np.int32)
    cdef int[:] di_inds = memoryview(di_inds_np)

    dx_orth_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx_orth = memoryview(dx_orth_np)

    linear_angles_np = np.zeros((_MAX_BONDS // 2, 3), dtype=np.int32)
    cdef int[:, :] linear_angles = memoryview(linear_angles_np)
    cdef int nlinear_angles

    cdef int min_dim, angle_constraint
    cdef double min_dim_val, dpos_norm, dx_dot

    # Now we are going to look for angles. We may also need to add dummy
    # atoms and everything that goes along with those (angle sums, bond
    # constraints, angle diff constraints, dihedral constraints).
    with nogil:
        # Angles are sorted by the central atom index
        for j in range(natoms):
            nj = nbonds[j]

            # First check to see whether there's a dummy atom already attached
            # to j
            if dinds[j] >= 0:
                nlinear_angles = 0
                bond_constraints[nbond_constraints, 0] = j
                bond_constraints[nbond_constraints, 1] = dinds[j]
                nbond_constraints += 1
                for a in range(nj):
                    i = c10y[j, a]
                    for b in range(a + 1, nj):
                        k = c10y[j, b]

                        # See if this angle was previously determined to be
                        # nearly linear
                        for n in range(nangle_sums_old):
                            if ((j == angle_sums_old[n, 1]
                                 and dinds[j] == angle_sums_old[n, 3])
                                 and ((i == angle_sums_old[n, 0]
                                       and k == angle_sums_old[n, 2])
                                      or (i == angle_sums_old[n, 2]
                                          and k == angle_sums_old[n, 0]))):
                                angle_sums[nangle_sums, 0] = i
                                angle_sums[nangle_sums, 1] = j
                                angle_sums[nangle_sums, 2] = k
                                angle_sums[nangle_sums, 3] = dinds[j]
                                nangle_sums += 1

                                if nlinear_angles < 2:
                                    angle_diffs[nangle_diffs, 0] = i
                                    angle_diffs[nangle_diffs, 1] = j
                                    angle_diffs[nangle_diffs, 2] = k
                                    angle_diffs[nangle_diffs, 3] = dinds[j]
                                    nangle_diffs += 1
                                nlinear_angles += 1
                                break
                        else:
                            err = vec_sum(pos[j], pos[i], dx1, -1.)
                            if err != 0: break
                            err = vec_sum(pos[k], pos[j], dx2, -1.)
                            if err != 0: break
                            err = cart_to_angle(i, j, k, dx1, dx2, &angle,
                                                dq, d2q, work, False, False)
                            if err != 0: break
                            if atol < angle < pi - atol:
                                angles[nangles, 0] = i
                                angles[nangles, 1] = j
                                angles[nangles, 2] = k
                                nangles += 1
                                continue

                            angle_sums[nangle_sums, 0] = i
                            angle_sums[nangle_sums, 1] = j
                            angle_sums[nangle_sums, 2] = k
                            angle_sums[nangle_sums, 3] = dinds[j]
                            nangle_sums += 1

                            if nlinear_angles < 2:
                                angle_diffs[nangle_diffs, 0] = i
                                angle_diffs[nangle_diffs, 1] = j
                                angle_diffs[nangle_diffs, 2] = k
                                angle_diffs[nangle_diffs, 3] = dinds[j]
                                nangle_diffs += 1
                            nlinear_angles += 1
                        if err != 0: break
                    if err != 0: break
                if err != 0: break
                if nlinear_angles == 0:
                    err = -1
                    break
                elif nlinear_angles == 1:
                    for a in range(nj):
                        i = c10y[j, a]
                        if (i == angle_sums[nangle_sums - 1, 0]
                                or i == angle_sums[nangle_sums - 1, 2]):
                            continue
                        angle_constraints[nangle_constraints, 0] = i
                        angle_constraints[nangle_constraints, 1] = j
                        angle_constraints[nangle_constraints, 2] = dinds[j]
                        nangle_constraints += 1
                        break
                    else:
                        i = angle_sums[nangle_sums - 1, 0]
                        k = angle_sums[nangle_sums - 1, 2]
                        err = vec_sum(pos[i], pos[j], dx2, -1.)
                        if err != 0: break
                        for a in range(nbonds[i]):
                            l = c10y[i, a]
                            if l == j or l == k:
                                continue
                            err = vec_sum(pos[l], pos[i], dx1, -1.)
                            if err != 0: break
                            err = cart_to_angle(j, i, l, dx1, dx2, &angle,
                                                dq, d2q, work, False, False)
                            if err != 0: break
                            if atol < angle < pi - atol:
                                dihedral_constraints[ndihedral_constraints, 0] = l
                                dihedral_constraints[ndihedral_constraints, 1] = i
                                dihedral_constraints[ndihedral_constraints, 2] = j
                                dihedral_constraints[ndihedral_constraints, 3] = dinds[j]
                                ndihedral_constraints += 1
                                nlinear_angles += 1
                                break
                        if err != 0: break
                        if nlinear_angles == 2:
                            continue

                        err = vec_sum(pos[k], pos[j], dx2, -1.)
                        if err != 0: break
                        for a in range(nbonds[k]):
                            l = c10y[k, a]
                            if l == j or l == i:
                                continue
                            err = vec_sum(pos[l], pos[k], dx1, -1.)
                            if err != 0: break
                            err = cart_to_angle(j, k, l, dx1, dx2, &angle,
                                                dq, d2q, work, False, False)
                            if err != 0: break
                            if atol < angle < pi - atol:
                                dihedral_constraints[ndihedral_constraints, 0] = l
                                dihedral_constraints[ndihedral_constraints, 1] = k
                                dihedral_constraints[ndihedral_constraints, 2] = j
                                dihedral_constraints[ndihedral_constraints, 3] = dinds[j]
                                ndihedral_constraints += 1
                                nlinear_angles += 1
                                break
                        if err != 0: break
                        if nlinear_angles == 2:
                            continue
                        dihedral_constraints[ndihedral_constraints, 0] = j
                        dihedral_constraints[ndihedral_constraints, 1] = i
                        dihedral_constraints[ndihedral_constraints, 2] = dinds[j]
                        dihedral_constraints[ndihedral_constraints, 3] = k
                        ndihedral_constraints += 1
                continue

            # If atom j isn't bonded to anything, that means the network isn't
            # fully connected, even though it is supposed to be.
            if nj == 0:
                err = j
                break
            elif nj == 1:
                # Atom j is terminal (likely a Hydrogen atom), so no angles
                # have j as the center
                continue
            elif nj == 2:
                # Only one angle passes through j, so we will check whether
                # it's linear
                i = c10y[j, 0]
                k = c10y[j, 1]
                err = vec_sum(pos[j], pos[i], dx1, -1.)
                if err != 0: break
                err = vec_sum(pos[k], pos[j], dx2, -1.)
                if err != 0: break
                err = cart_to_angle(i, j, k, dx1, dx2, &angle, dq, d2q, work,
                                    False, False)
                if err != 0: break

                if atol < angle < pi - atol:
                    angles[nangles, 0] = i
                    angles[nangles, 1] = j
                    angles[nangles, 2] = k
                    nangles += 1
                    continue

                # The bond is linear. Can we find any atoms to form
                # a dihedral?
                memset(&di_inds[0], 0, 4 * sizeof(int))
                di_inds[0] = -1
                for a in range(nbonds[i]):
                    l = c10y[i, a]
                    if l == j or l == k:
                        continue
                    err = vec_sum(pos[i], pos[l], dx2, -1.)
                    if err != 0: break
                    # Yes, it's supposed to be dx2 then dx1
                    err = cart_to_angle(l, i, j, dx2, dx1, &angle, dq, d2q,
                                        work, False, False)
                    if err != 0: break

                    if atol < angle < pi - atol:
                        di_inds[0] = l
                        di_inds[1] = i
                        di_inds[2] = j
                        di_inds[3] = natoms + ndummies
                        err = vec_sum(pos[l], pos[i], dummypos[ndummies], -1.)
                        break

                if err != 0: break

                if di_inds[0] == -1:
                    err = vec_sum(pos[j], pos[k], dx2, -1.)
                    if err != 0: break
                    for a in range(nbonds[k]):
                        l = c10y[k, a]
                        if l == i or l == j:
                            continue
                        err = vec_sum(pos[k], pos[l], dx1, -1.)
                        if err != 0: break
                        err = cart_to_angle(l, k, j, dx1, dx2, &angle, dq,
                                            d2q, work, False, False)
                        if err != 0: break
                        if atol < angle < pi - atol:
                            di_inds[0] = l
                            di_inds[1] = k
                            di_inds[2] = j
                            di_inds[3] = natoms + ndummies
                            err = vec_sum(pos[l], pos[k], dummypos[ndummies],
                                          -1.)
                            # We don't check err because we are breaking
                            # regardless, and if err is nonzero, then the
                            # check after this loop will catch it.
                            break
                    if err != 0: break

                if di_inds[0] == -1:
                    di_inds[0] = j
                    di_inds[1] = i
                    di_inds[2] = natoms + ndummies
                    di_inds[3] = k
                    min_dim_val = HUGE_VAL
                    min_dim = -1
                    err = vec_sum(pos[k], pos[i], dx1, -1.)
                    if err != 0: break
                    for l in range(3):
                        if fabs(dx1[l]) < min_dim_val:
                            min_dim_val = fabs(dx1[l])
                            min_dim = l
                    if min_dim == -1:
                        err = -10
                        break
                    dummypos[ndummies, min_dim] = 1.
                err = normalize(dummypos[ndummies])
                if err != 0: break
                err = vec_sum(pos[k], pos[i], dx_orth, -1.)
                if err != 0: break
                err = normalize(dx_orth)
                if err != 0: break
                dx_dot = ddot(&THREE, &dummypos[ndummies, 0], &UNITY,
                              &dx_orth[0], &UNITY)
                err = my_daxpy(-dx_dot, dx_orth, dummypos[ndummies])
                if err != 0: break
                err = normalize(dummypos[ndummies])
                if err != 0: break
                err = my_daxpy(1., pos[j], dummypos[ndummies])
                if err != 0: break

                bond_constraints[nbond_constraints, 0] = j
                bond_constraints[nbond_constraints, 1] = natoms + ndummies
                nbond_constraints += 1

                angle_sums[nangle_sums, 0] = i
                angle_sums[nangle_sums, 1] = j
                angle_sums[nangle_sums, 2] = k
                angle_sums[nangle_sums, 3] = natoms + ndummies
                nangle_sums += 1

                angle_diffs[nangle_diffs, 0] = i
                angle_diffs[nangle_diffs, 1] = j
                angle_diffs[nangle_diffs, 2] = k
                angle_diffs[nangle_diffs, 3] = natoms + ndummies
                nangle_diffs += 1

                dihedral_constraints[ndihedral_constraints, 0] = di_inds[0]
                dihedral_constraints[ndihedral_constraints, 1] = di_inds[1]
                dihedral_constraints[ndihedral_constraints, 2] = di_inds[2]
                dihedral_constraints[ndihedral_constraints, 3] = di_inds[3]
                ndihedral_constraints += 1

                dinds[j] = natoms + ndummies
                ndummies += 1
            else:
                nlinear_angles = 0
                for a in range(nj):
                    i = c10y[j, a]
                    for b in range(a+1, nj):
                        k = c10y[j, b]
                        err = vec_sum(pos[j], pos[i], dx1, -1.)
                        if err != 0: break
                        err = vec_sum(pos[k], pos[j], dx2, -1.)
                        if err != 0: break
                        err = cart_to_angle(i, j, k, dx1, dx2, &angle, dq, d2q,
                                            work, False, False)
                        if err != 0: break
                        if atol < angle < pi - atol:
                            angles[nangles, 0] = i
                            angles[nangles, 1] = j
                            angles[nangles, 2] = k
                            nangles += 1
                            continue
                        linear_angles[nlinear_angles, 0] = i
                        linear_angles[nlinear_angles, 1] = k
                        nlinear_angles += 1
                    if err != 0: break
                if err != 0: break

                if nlinear_angles == 0:
                    continue
                nbasis = 0
                for a in range(nlinear_angles):
                    i = linear_angles[a, 0]
                    k = linear_angles[a, 1]
                    err = vec_sum(pos[k], pos[i], dx1, -1.)
                    if err != 0: break
                    err = normalize(dx1)
                    if err != 0: break
                    for b in range(nbasis):
                        dx_dot = ddot(&THREE, &basis[b, 0], &UNITY,
                                      &dx1[0], &UNITY)
                        if acos(fabs(dx_dot)) < atol / 2.:
                            break
                    else:
                        if nbasis >= 3:
                            err = -1
                            break
                        dcopy(&THREE, &dx1[0], &UNITY,
                              &basis[nbasis, 0], &UNITY)
                        nbasis += 1
                    if err != 0: break
                if err != 0: break

                if nbasis == 3:
                    continue
                angle_constraint = -1
                for a in range(nj):
                    l = c10y[j, a]
                    for b in range(nlinear_angles):
                        if (l == linear_angles[b, 0]
                                or l == linear_angles[b, 1]):
                            break
                    else:
                        if angle_constraint == -1:
                            angle_constraint = l
                        err = vec_sum(pos[l], pos[j], dx1, -1.)
                        if err != 0: break
                        err = normalize(dx1)
                        if err != 0: break
                        for b in range(nbasis):
                            dx_dot = ddot(&THREE, &basis[b, 0], &UNITY,
                                          &dx1[0], &UNITY)
                            if acos(fabs(dx_dot)) < atol / 2.:
                                break
                        else:
                            if nbasis >= 3:
                                err = -1
                                break
                            dcopy(&THREE, &dx1[0], &UNITY,
                                  &basis[nbasis, 0], &UNITY)
                            nbasis += 1
                        if err != 0: break
                    if err != 0: break
                if err != 0: break

                if nbasis == 3:
                    continue
                if nbasis == 1:
                    err = -1
                    break

                bond_constraints[nbond_constraints, 0] = j
                bond_constraints[nbond_constraints, 1] = natoms + ndummies
                nbond_constraints += 1
                cross(basis[0], basis[1], dummypos[ndummies])
                err = normalize(dummypos[ndummies])
                if err != 0: break
                for a in range(nlinear_angles):
                    i = linear_angles[a, 0]
                    k = linear_angles[a, 1]
                    angle_sums[nangle_sums, 0] = i
                    angle_sums[nangle_sums, 1] = j
                    angle_sums[nangle_sums, 2] = k
                    angle_sums[nangle_sums, 3] = natoms + ndummies
                    nangle_sums += 1
                    if a < 2:
                        angle_diffs[nangle_diffs, 0] = i
                        angle_diffs[nangle_diffs, 1] = j
                        angle_diffs[nangle_diffs, 2] = k
                        angle_diffs[nangle_diffs, 3] = natoms + ndummies
                        err = vec_sum(pos[k], pos[i], dx1, -1.)
                        if err != 0: break
                        err = normalize(dx1)
                        if err != 0: break
                        dx_dot = ddot(&THREE, &dx1[0], &UNITY,
                                      &dummypos[ndummies, 0], &UNITY)
                        err = my_daxpy(-dx_dot, dx1, dummypos[ndummies])
                        if err != 0: break
                        err = normalize(dummypos[ndummies])
                        if err != 0: break

                if nlinear_angles == 1:
                    if angle_constraint == -1:
                        err = -1
                        break
                    err = vec_sum(pos[angle_constraint], pos[j], dx1, -1.)
                    if err != 0: break
                    err = normalize(dx1)
                    if err != 0: break
                    dx_dot = ddot(&THREE, &dx1[0], &UNITY, &dummypos[ndummies, 0], &UNITY)
                    err = my_daxpy(-dx_dot, dx1, dummypos[ndummies])
                    if err != 0: break
                    err = normalize(dummypos[ndummies])
                    if err != 0: break
                    angle_constraints[nangle_constraints, 0] = angle_constraint
                    angle_constraints[nangle_constraints, 1] = j
                    angle_constraints[nangle_constraints, 2] = (natoms
                                                                + ndummies)
                    nangle_constraints += 1
                err = my_daxpy(1., pos[j], dummypos[ndummies])
                if err != 0: break
                dinds[j] = natoms + ndummies
                ndummies += 1
            if err != 0: break
    if err > 0:
        raise RuntimeError("Atom {} has no bonds! This shouldn't happen."
                           "".format(j))
    elif err == -10:
        raise RuntimeError("Dummy atom position determination failed.")
    elif err != 0:
        raise RuntimeError("An error occurred while identifying angles.")
    dummies = Atoms('X{}'.format(ndummies), dummypos_np[:ndummies])
    return (angles_np[:nangles], dummies, dinds_np, angle_sums[:nangle_sums],
            bond_constraints[:nbond_constraints],
            angle_constraints[:nangle_constraints],
            dihedral_constraints[:ndihedral_constraints],
            angle_diffs[:nangle_diffs])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_dihedrals(atoms, double atol, int[:, :] bonds, int[:, :] angles,
                   int[:] nbonds, int[:, :] c10y, int[:] dinds):
    cdef int i, j, k, l, a, b, err
    cdef int natoms = len(atoms)
    cdef int nbonds_tot = len(bonds)
    cdef int nangles = len(angles)

    cdef double angle

    cdef double[:, :] pos = memoryview(atoms.positions)

    # temporary arrys for various things below
    work_np = np.zeros((1, 1, 1, 3), dtype=np.float64)
    cdef double[:, :, :, :] work = memoryview(work_np)

    dx1_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx1 = memoryview(dx1_np)

    dx2_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx2 = memoryview(dx2_np)

    # We don't actually use dq or d2q anywhere in this function,
    # but we need to call cart_to_angle to check for linearity, and
    # those functions require dq and d2q.
    #
    # Can we somehow find a way to avoid allocating these arrays?
    # It probably doesn't matter from a performance standpoint, but
    # it's the principle of the thing.
    dq_np = np.zeros((natoms, 3), dtype=np.float64)
    cdef double[:, :] dq = memoryview(dq_np)

    d2q_np = np.zeros((4, 3, 4, 3), dtype=np.float64)
    cdef double[:, :, :, :] d2q = memoryview(d2q_np)

    cdef int ndihedrals_max = nbonds_tot * nangles // 2
    dihedrals_np = -np.ones((ndihedrals_max, 4), dtype=np.int32)
    cdef int[:, :] dihedrals = memoryview(dihedrals_np)
    cdef int ndihedrals = 0

    with nogil:
        for a in range(nangles):
            i = angles[a, 0]
            j = angles[a, 1]
            k = angles[a, 2]
            for b in range(nbonds[k]):
                l = c10y[k, b]
                if l > i and l != j:
                    err = vec_sum(pos[k], pos[j], dx1, -1.)
                    if err != 0: break
                    err = vec_sum(pos[l], pos[k], dx2, -1.)
                    if err != 0: break
                    err = cart_to_angle(i, j, k, dx1, dx2, &angle, dq, d2q,
                                        work, False, False)
                    if err != 0: break
                    if atol < angle < pi - atol:
                        dihedrals[ndihedrals, 0] = i
                        dihedrals[ndihedrals, 1] = j
                        dihedrals[ndihedrals, 2] = k
                        dihedrals[ndihedrals, 3] = l
                        ndihedrals += 1
            if err != 0: break

            if dinds[k] >= 0:
                dihedrals[ndihedrals, 0] = i
                dihedrals[ndihedrals, 1] = j
                dihedrals[ndihedrals, 2] = k
                dihedrals[ndihedrals, 3] = dinds[k]
                ndihedrals += 1

            for b in range(nbonds[i]):
                l = c10y[i, b]
                if l > k and l != j:
                    err = vec_sum(pos[i], pos[j], dx1, -1.)
                    if err != 0: break
                    err = vec_sum(pos[l], pos[i], dx2, -1.)
                    if err != 0: break
                    err = cart_to_angle(j, i, l, dx1, dx2, &angle, dq, d2q,
                                        work, False, False)
                    if err != 0: break
                    if atol < angle < pi - atol:
                        dihedrals[ndihedrals, 0] = k
                        dihedrals[ndihedrals, 1] = j
                        dihedrals[ndihedrals, 2] = i
                        dihedrals[ndihedrals, 3] = l
                        ndihedrals += 1
            if err != 0: break

            if dinds[i] >= 0:
                dihedrals[ndihedrals, 0] = k
                dihedrals[ndihedrals, 1] = j
                dihedrals[ndihedrals, 2] = i
                dihedrals[ndihedrals, 3] = dinds[i]
                ndihedrals += 1

        for a in range(nbonds_tot):
            i = bonds[a, 0]
            if dinds[i] == -1:
                continue
            j = bonds[a, 1]
            if dinds[j] == -1:
                continue
            if dinds[j] > dinds[i]:
                dihedrals[ndihedrals, 0] = dinds[i]
                dihedrals[ndihedrals, 1] = i
                dihedrals[ndihedrals, 2] = j
                dihedrals[ndihedrals, 3] = dinds[j]
                ndihedrals += 1

    if err != 0:
        raise RuntimeError("An error occurred while identifing dihedrals")

    return dihedrals_np[:ndihedrals]
