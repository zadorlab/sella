# cimports

from libc.math cimport fabs, acos, pi, HUGE_VAL
from libc.string cimport memset

from scipy.linalg.cython_blas cimport ddot, dcopy, dnrm2
from scipy.linalg.cython_lapack cimport dgesvd

from sella.utilities.blas cimport my_daxpy
from sella.utilities.math cimport normalize, vec_sum, cross
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


def find_bonds(atoms, int[:, :] added_bonds, int[:, :] forbidden_bonds):
    cdef int natoms = len(atoms)
    cdef int i, j, k, err, a, b
    cdef int nbonds_tot = 0
    cdef double scale = 1.25

    cdef int nadded = len(added_bonds)
    cdef int nforbidden = len(forbidden_bonds)

    #bonds_np = np.zeros((natoms * _MAX_BONDS // 2, 2), dtype=np.int32)
    #cdef int[:, :] bonds = memoryview(bonds_np)

    rcov_np = covalent_radii[atoms.numbers].copy()
    cdef double[:] rcov = memoryview(rcov_np)

    rij_np = atoms.get_all_distances()
    cdef double[:, :] rij = memoryview(rij_np)

    nbonds_np = np.zeros(natoms, dtype=np.int32)
    cdef int[:] nbonds = memoryview(nbonds_np)

    labels_np = -np.ones(natoms, dtype=np.int32)
    cdef int[:] labels = memoryview(labels_np)
    cdef int nlabels = natoms

    c10y_np = -np.ones((natoms, _MAX_BONDS), dtype=np.int32)
    cdef int[:, :] c10y = memoryview(c10y_np)

    cdef bint forbidden

    with nogil:
        # Set up initial added bonds
        for i in range(nadded):
            j = added_bonds[i, 0]
            k = added_bonds[i, 1]
            c10y[j, nbonds[j]] = k
            c10y[k, nbonds[k]] = j
            nbonds[j] += 1
            nbonds[k] += 1

        # Loop until all atoms share the same label
        while err == 0:
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
            nlabels = 0
            for i in range(natoms):
                if labels[i] == -1:
                    labels[i] = nlabels
                    flood_fill(i, nbonds, c10y, labels, nlabels)
                    nlabels += 1
            if nlabels == 1:
                break

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
                        # Check whether this bond is forbidden
                        forbidden = False
                        for k in range(nforbidden):
                            a = forbidden_bonds[k, 0]
                            b = forbidden_bonds[k, 1]
                            if (i == a and j == b) or (i == b and j == a):
                                forbidden = True
                                break
                        if forbidden:
                            continue

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


cdef inline (int, bint) check_if_linear(double[:] x1, double[:] x2,
                                        double[:] x3, double[:] dx1,
                                        double[:] dx2, double tol) nogil:
    """Check whether three atoms are (nearly) collinear."""

    cdef int err = 0
    cdef double angle
    cdef bint linear = True
    cdef int sddx1 = dx1.strides[0] >> 3
    cdef int sddx2 = dx2.strides[0] >> 3

    err = vec_sum(x1, x2, dx1, -1.)
    err += vec_sum(x3, x2, dx2, -1.)

    angle = acos(ddot(&THREE, &dx1[0], &sddx1, &dx2[0], &sddx2)
                 / (dnrm2(&THREE, &dx1[0], &sddx1)
                    * dnrm2(&THREE, &dx2[0], &sddx2)))

    if tol < angle < (pi - tol):
        linear = False

    return err, linear

cdef inline (int, bint) check_if_planar(double[:, :] dxs, double[:, :] A,
                                        double[:] s, double[:] work,
                                        double tol) nogil:
    cdef bint planar = True
    cdef int m = dxs.shape[0]
    cdef int n = dxs.shape[1]
    cdef int minnm = min(n, m)
    cdef int ldx = dxs.strides[0] >> 3
    cdef int sdx = dxs.strides[1] >> 3
    cdef int sds = s.strides[0] >> 3
    cdef int lda = A.strides[0] >> 3
    cdef int sda = A.strides[1] >> 3
    cdef double U, VT
    cdef int ldu = 1
    cdef int ldvt = 1
    cdef int lwork = work.shape[0]
    cdef int info
    cdef int i, j
    cdef double norm

    # Normalize the bond vectors and copy into A for SVD
    for i in range(m):
        norm = dnrm2(&n, &dxs[i, 0], &sdx)
        for j in range(n):
            dxs[i, j] /= norm
        dcopy(&n, &dxs[i, 0], &sdx, &A[i, 0], &sda)

    # Get singular vectors of A
    dgesvd('O', 'N', &n, &m, &A[0, 0], &lda, &s[0], &U, &ldu, &VT, &ldvt,
           &work[0], &lwork, &info)

    if info != 0:
        return -1, planar

    # Copy the minor axis into s for safe keeping
    dcopy(&n, &A[2, 0], &sda, &s[0], &sds)

    cdef double angle
    cdef double min_angle = pi / 2
    cdef double max_angle = pi / 2

    # Calculate angle between each bond vector and the minor axis,
    # keep track of maximum/minimum angles
    for i in range(m):
        angle = acos(ddot(&n, &dxs[i, 0], &sdx, &s[0], &sds))
        if angle < min_angle:
            min_angle = angle
        if angle > max_angle:
            max_angle = angle

    # When max_angle - min_angle is close to 0, the atom is almost totally
    # planar. If max_angle - min_angle is large, then the atom is not
    # almost totally planar.
    if max_angle - min_angle > tol:
        planar = False

    return 0, planar


cdef (int, int) find_dummy_dihedral(double[:, :] pos, double[:] dpos,
                                    double[:] dx1, double[:] dx2,
                                    int[:] nbonds, int[:, :] c10y,
                                    int i, int j, int k,
                                    double atol) nogil:
    cdef int a
    cdef int b = -1
    cdef int l
    cdef int info
    cdef bint linear = True
    cdef int sddx1 = dx1.strides[0] >> 3
    cdef int sddx2 = dx2.strides[0] >> 3

    for a in range(nbonds[i]):
        l = c10y[i, a]
        if l == j or l == k:
            continue
        info, linear = check_if_linear(pos[l], pos[i], pos[j], dx1, dx2, atol)
        if info < 0: return info, info

        if not linear:
            b = i
            break

    if b == -1:
        for a in range(nbonds[k]):
            l = c10y[k, a]
            if l == i or l == j:
                continue

            info, linear = check_if_linear(pos[l], pos[k], pos[j], dx1, dx2,
                                           atol)
            if info < 0: return info, info

            if not linear:
                b = k
                break

    if b == -1:
        return 0, 0

    # All l for which angle(i-j-l) == angle(k-j-l) obey the relation:
    # r_jl perpto (r_ij / |r_ij| + r_jl / |r_jl|)
    # Find and normalize the RHS vector
    info = vec_sum(pos[i], pos[j], dx1, -1)
    if info < 0:  return info, info

    info = normalize(dx1)
    if info < 0:  return info, info

    info = vec_sum(pos[j], pos[k], dx2, -1)
    if info < 0:  return info, info

    info = normalize(dx2)
    if info < 0:  return info, info

    info = my_daxpy(1., dx2, dx1)
    if info < 0:  return info, info

    if dnrm2(&THREE, &dx1[0], &sddx1) < 1e-4:
        info = vec_sum(pos[i], pos[k], dx1, -1)
        if info < 0:  return info, info

    info = normalize(dx1)
    if info < 0:  return info, info

    info = vec_sum(pos[b], pos[l], dx2, -1)
    if info < 0:  return info, info

    info = my_daxpy(-ddot(&THREE, &dx1[0], &sddx1, &dx2[0], &sddx2),
                    dx1, dx2)
    if info < 0:  return info, info

    info = normalize(dx2)
    if info < 0:  return info, info

    info = vec_sum(pos[j], dx2, dpos)
    if info < 0:  return info, info

    return l, b


cdef int find_dummy_improper(double[:, :] pos, double[:] dpos, double[:] dx1,
                             double[:] dx2, int i, int j, int k) nogil:
    cdef int err
    cdef int a
    cdef int b = 0
    cdef double dxmin = 1.
    cdef int m = len(dx1)
    cdef int sddx1 = dx1.strides[0] >> 3
    cdef int sddx2 = dx2.strides[0] >> 3

    err = vec_sum(pos[i], pos[k], dx1, -1.)
    if err != 0: return err

    err = normalize(dx1)
    if err != 0: return err

    for a in range(m):
        if fabs(dx1[a]) < dxmin:
            dxmin = fabs(dx1[a])
            b = a
        dx2[a] = 0.
    dx2[b] = 1.

    err = my_daxpy(-ddot(&m, &dx1[0], &sddx1, &dx2[0], &sddx2), dx1, dx2)
    if err != 0: return err

    err = normalize(dx2)
    if err != 0: return err

    err = vec_sum(pos[j], dx2, dpos)
    if err != 0: return err

    return 0


def find_angles(atoms, double atol, int[:, :] bonds, int[:] nbonds,
                int[:, :] c10y, dummies=None, dinds_np=None):
    cdef int i, j, k, l, nj, a, b
    cdef int natoms = len(atoms)
    cdef int nbonds_tot = len(bonds)
    cdef bint linear, planar

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

    dx1_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx1 = memoryview(dx1_np)

    dx2_np = np.zeros(3, dtype=np.float64)
    cdef double[:] dx2 = memoryview(dx2_np)

    linear_angles_np = np.zeros((_MAX_BONDS // 2, 2), dtype=np.int32)
    cdef int[:, :] linear_angles = memoryview(linear_angles_np)
    cdef int nlinear_angles

    cdef int min_dim, angle_constraint
    cdef double min_dim_val, dpos_norm, dx_dot

    dxs_np = np.zeros((_MAX_BONDS, 3), dtype=np.float64)
    cdef double[:, :] dxs = memoryview(dxs_np)

    A_np = dxs_np.copy()
    cdef double[:, :] A = memoryview(A_np)

    axis_np = np.zeros(3, dtype=np.float64)
    cdef double[:] axis = memoryview(axis_np)

    svdwork_np = np.zeros(32, dtype=np.float64)
    cdef double[:] svdwork = memoryview(svdwork_np)

    cdef int err = 0

    with nogil:
        for j in range(natoms):
            nj = nbonds[j]
            if nj == 0:
                # 0 bonding partners: should be impossible.
                err = -1
                break

            elif nj == 1:
                # 1 bonding parter: no angles -> no linear angles
                continue

            elif nj == 2:
                # 2 bonding partners: 1 angle to check for linearity
                i = c10y[j, 0]
                k = c10y[j, 1]
                err, linear = check_if_linear(pos[i], pos[j], pos[k],
                                              dx1, dx2, atol)
                if err != 0:  break

                if not linear:
                    # Not linear, so add the angle and move on to the
                    # next center
                    angles[nangles, 0] = i
                    angles[nangles, 1] = j
                    angles[nangles, 2] = k
                    nangles += 1
                    continue

                if dinds[j] <= 0:
                    dinds[j] = natoms + ndummies

                    # Add dummy bond constraint
                    bond_constraints[nbond_constraints, 0] = j
                    bond_constraints[nbond_constraints, 1] = dinds[j]
                    nbond_constraints += 1

                    angle_constraints[nangle_constraints, 0] = i
                    angle_constraints[nangle_constraints, 1] = j
                    angle_constraints[nangle_constraints, 2] = dinds[j]
                    nangle_constraints += 1

                    ## Angle diff constraint
                    #angle_diffs[nangle_diffs, 0] = i
                    #angle_diffs[nangle_diffs, 1] = j
                    #angle_diffs[nangle_diffs, 2] = k
                    #angle_diffs[nangle_diffs, 3] = dinds[j]
                    #nangle_diffs += 1

                    ## Try to find dummy position and its final constraint
                    ## from dihedrals w/ atoms bonded to i or k
                    #l, b = find_dummy_dihedral(pos, dummypos[ndummies], dx1,
                    #                           dx2, nbonds, c10y, i, j, k,
                    #                           2 * atol)
                    #if l < 0:
                    #    err = l
                    #    break
                    #elif l > 0:
                    #    dihedral_constraints[ndihedral_constraints, 0] = l
                    #    dihedral_constraints[ndihedral_constraints, 1] = b
                    #    dihedral_constraints[ndihedral_constraints, 2] = j
                    #    dihedral_constraints[ndihedral_constraints, 3] = dinds[j]
                    #    ndihedral_constraints += 1
                    #else:
                    #    err = find_dummy_improper(pos, dummypos[ndummies],
                    #                              dx1, dx2, i, j, k)
                    #    if err != 0: break
                    #    dihedral_constraints[ndihedral_constraints, 0] = i
                    #    dihedral_constraints[ndihedral_constraints, 1] = j
                    #    dihedral_constraints[ndihedral_constraints, 2] = dinds[j]
                    #    dihedral_constraints[ndihedral_constraints, 3] = k
                    #    ndihedral_constraints += 1
                    err = find_dummy_improper(pos, dummypos[ndummies],
                                              dx1, dx2, i, j, k)
                    if err != 0: break
                    dihedral_constraints[ndihedral_constraints, 0] = i
                    dihedral_constraints[ndihedral_constraints, 1] = j
                    dihedral_constraints[ndihedral_constraints, 2] = dinds[j]
                    dihedral_constraints[ndihedral_constraints, 3] = k
                    ndihedral_constraints += 1
                    ndummies += 1

                # Add the angle sum
                angle_sums[nangle_sums, 0] = i
                angle_sums[nangle_sums, 1] = j
                angle_sums[nangle_sums, 2] = k
                angle_sums[nangle_sums, 3] = dinds[j]
                nangle_sums += 1

            else:
                # Find all linear angles first, but don't do anything with
                # them yet.
                nlinear_angles = 0
                for a in range(nj):
                    i = c10y[j, a]
                    err = vec_sum(pos[i], pos[j], dxs[a], -1)
                    if err != 0: break
                    for b in range(a + 1, nj):
                        k = c10y[j, b]
                        err, linear = check_if_linear(pos[i], pos[j], pos[k],
                                                      dx1, dx2, atol)
                        if err != 0: break

                        if linear:
                            linear_angles[nlinear_angles, 0] = i
                            linear_angles[nlinear_angles, 1] = k
                            nlinear_angles += 1
                        else:
                            # If this angle is *not* linear, just add it
                            # to the list of angles
                            angles[nangles, 0] = i
                            angles[nangles, 1] = j
                            angles[nangles, 2] = k
                            nangles += 1

                    if err != 0: break
                if err != 0: break

                if nlinear_angles == 0:
                    # If we didn't find any linear angles, then we're done
                    # with this center.
                    continue

                if dinds[j] <= 0:
                    # No dummy atom yet. Check to see if atom center is
                    # planar before adding a dummy atom
                    err, planar = check_if_planar(dxs[:nj], A[:nj], axis, svdwork, atol)
                    if err != 0: break
                    if not planar:
                        continue

                    dinds[j] = natoms + ndummies
                    err = vec_sum(pos[j], axis, dummypos[ndummies], 1.)

                    # Bond constraints
                    bond_constraints[nbond_constraints, 0] = j
                    bond_constraints[nbond_constraints, 1] = dinds[j]
                    nbond_constraints += 1

                    i = linear_angles[0, 0]
                    k = linear_angles[0, 1]

                    err = find_dummy_improper(pos, dummypos[ndummies],
                                              dx1, dx2, i, j, k)
                    if err != 0: break
                    dihedral_constraints[ndihedral_constraints, 0] = i
                    dihedral_constraints[ndihedral_constraints, 1] = j
                    dihedral_constraints[ndihedral_constraints, 2] = dinds[j]
                    dihedral_constraints[ndihedral_constraints, 3] = k
                    ndihedral_constraints += 1

                    angle_constraints[nangle_constraints, 0] = i
                    angle_constraints[nangle_constraints, 1] = j
                    angle_constraints[nangle_constraints, 2] = dinds[j]
                    nangle_constraints += 1
                    #angle_diffs[nangle_diffs, 0] = i
                    #angle_diffs[nangle_diffs, 1] = j
                    #angle_diffs[nangle_diffs, 2] = k
                    #angle_diffs[nangle_diffs, 3] = dinds[j]
                    #nangle_diffs += 1

                    ## angle diff constraint(s)
                    #for a in range(min(nlinear_angles, 2)):
                    #    angle_diffs[nangle_diffs, 0] = linear_angles[a, 0]
                    #    angle_diffs[nangle_diffs, 1] = j
                    #    angle_diffs[nangle_diffs, 2] = linear_angles[a, 1]
                    #    angle_diffs[nangle_diffs, 3] = dinds[j]
                    #    nangle_diffs += 1

                    #if nlinear_angles < 2:
                    #    # regular angle constraint, if necessary
                    #    for a in range(nj):
                    #        i = c10y[j, a]
                    #        for b in range(nlinear_angles):
                    #            if (i == linear_angles[b, 0]
                    #                    or i == linear_angles[b, 1]):
                    #                break
                    #        else:
                    #            angle_constraints[nangle_constraints, 0] = i
                    #            angle_constraints[nangle_constraints, 1] = j
                    #            angle_constraints[nangle_constraints, 2] = dinds[j]
                    #            nangle_constraints += 1
                    #            break
                    #    else:
                    #        err = -1
                    #        break
                    ndummies += 1

                for a in range(nlinear_angles):
                    angle_sums[nangle_sums, 0] = linear_angles[a, 0]
                    angle_sums[nangle_sums, 1] = j
                    angle_sums[nangle_sums, 2] = linear_angles[a, 1]
                    angle_sums[nangle_sums, 3] = dinds[j]
                    nangle_sums += 1
    if err > 0:
        raise RuntimeError("Atom {} has no bonds! This shouldn't happen."
                           "".format(j))
    elif err == -1:
        raise RuntimeError("Dummy atom position determination failed.")
    elif err != 0:
        raise RuntimeError("An error occurred while identifying angles: {}".format(err))
    dummies = Atoms('X{}'.format(ndummies), dummypos_np[:ndummies])
    return (angles_np[:nangles], dummies, dinds_np, angle_sums[:nangle_sums],
            bond_constraints[:nbond_constraints],
            angle_constraints[:nangle_constraints],
            dihedral_constraints[:ndihedral_constraints],
            angle_diffs[:nangle_diffs])


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
