import numpy as np
from ase import Atoms
from ase.data import covalent_radii, vdw_radii

cimport cython
cimport numpy as np
from cython.view cimport array
from cython cimport numeric
from libc.math cimport sqrt, acos, atan2
from libc.string cimport memset

from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dger, dscal, dcopy, dsyr, dsyr2
from scipy.linalg.cython_lapack cimport dlacpy, dlaset, dlascl

cdef int UNITY = 1
cdef int THREE = 3
cdef double DUNITY = 1.
cdef double DNUNITY = -1.
cdef double DZERO = 0.
cdef double DTWO = 2.
cdef double DNTWO = -2.

# 12 is number of contacts in dense packed solids (FCC, HCP).
# For molecules, this could probably be reduced to 4.
cdef size_t _MAX_BONDS = 12

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void cross(double[:] x, double[:] y, double[:] z):
    z[0] = x[1] * y[2] - y[1] * x[2]
    z[1] = x[2] * y[0] - y[2] * x[0]
    z[2] = x[0] * y[1] - y[0] * x[1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void skew(double[:] x, double[:, :] Y, double scale=1.):
    Y[2, 1] = scale * x[0]
    Y[0, 2] = scale * x[1]
    Y[1, 0] = scale * x[2]

    Y[1, 2] = -Y[2, 1]
    Y[2, 0] = -Y[0, 2]
    Y[0, 1] = -Y[1, 0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void symmetrize(double* X, size_t n, size_t lda):
    cdef size_t i, j
    for i in range(n):
        for j in range(i + 1, n):
            X[j * lda + i] = X[i * lda + j]

def get_internal(object atoms, bint use_angles=True, bint use_dihedrals=True):
    """Find bonding (and optionally angle bend/dihedral) internal
    coordinates of a molecule.

    Arguments:
    atoms -- ASE Atoms object representing a molecule
    use_angles -- Include angle bends in internal coordinate search
    use_dihedrals -- Include dihedral angles
    """
    cdef size_t i, j
    cdef size_t natoms = len(atoms)

    cdef double[:, :] rij = memoryview(atoms.get_all_distances())
    cdef double[:] rcov = memoryview((1.5 * np.array([covalent_radii[atom.number] for atom in atoms])))

    # c10y == connectivity
    c10y_np = -np.ones((natoms, _MAX_BONDS), dtype=np.int32)
    cdef int[:, :] c10y = memoryview(c10y_np)

    nbonds_np = np.zeros(natoms, dtype=np.uintp)
    cdef size_t[:] nbonds = memoryview(nbonds_np)

    # actual maximum number of bonds per atom
    cdef size_t max_bonds = 0

    cdef size_t nbonds_tot = 0
    bonds_np = -np.ones((_MAX_BONDS * natoms // 2, 2), dtype=np.int32)
    cdef int[:, :] bonds = memoryview(bonds_np)



    # FIXME: Implement self-bonding, as this can happen with periodic
    # boundary conditions and is relevant for angles/dihedrals.
    for i in range(natoms):
        for j in range(i + 1, natoms):
            if rij[i, j] <= rcov[i] + rcov[j]:
                assert nbonds[i] < _MAX_BONDS
                assert nbonds[j] < _MAX_BONDS

                bonds[nbonds_tot, 0] = i
                bonds[nbonds_tot, 1] = j

                nbonds_tot += 1

                c10y[i, nbonds[i]] = j
                nbonds[i] += 1

                c10y[j, nbonds[j]] = i
                nbonds[j] += 1
        max_bonds = max(max_bonds, nbonds[i])

    bonds_np = np.resize(bonds_np, (nbonds_tot, 2))
    assert np.all(bonds_np >= 0)

    cdef size_t nangles_tot = 0
    # TODO: nbonds_tot**2 is certainly overkill, find a more reasonable
    # upper bound for the number of angles
    if use_angles:
        angles_np = -np.ones((nbonds_tot**2, 3), dtype=np.int32)
    else:
        angles_np = np.empty((0, 3), dtype=np.int32)
    cdef int[:, :] angles = memoryview(angles_np)

    cdef size_t ndihedrals_tot = 0
    # TODO: nbonds_tot**3 is certainly overkill, find a more reasonable
    # upper bound for the number of dihedrals
    if use_dihedrals:
        dihedrals_np = -np.ones((nbonds_tot**3, 4), dtype=np.int32)
    else:
        dihedrals_np = np.empty((0, 4), dtype=np.int32)
    cdef int[:, :] dihedrals = memoryview(dihedrals_np)

    if not (use_angles or use_dihedrals):
        return c10y_np, nbonds_np, bonds_np, angles_np, dihedrals_np

    cdef size_t jj, k, kk, m, mm

    for i in range(natoms):
        for jj in range(nbonds[i]):
            j = c10y[i, jj]
            for kk in range(nbonds[j]):
                k = c10y[j, kk]
                # (i, j, i) is not a valid angle
                if k == i:
                    continue
                # to avoid double-counting angles, only consider triplets
                # (i, j, k) where k > i
                elif use_angles and k>i:
                    angles[nangles_tot, 0] = i
                    angles[nangles_tot, 1] = j
                    angles[nangles_tot, 2] = k
                    nangles_tot += 1
                if not use_dihedrals:
                    continue
                for mm in range(nbonds[k]):
                    m = c10y[k, mm]
                    # similar to before, only consider quadruplets
                    # (i, j, k, m) where m > i
                    if m == j or m <= i:
                        continue
                    dihedrals[ndihedrals_tot, 0] = i
                    dihedrals[ndihedrals_tot, 1] = j
                    dihedrals[ndihedrals_tot, 2] = k
                    dihedrals[ndihedrals_tot, 3] = m
                    ndihedrals_tot += 1
    
    angles_np = np.resize(angles_np, (nangles_tot, 3))
    assert np.all(angles_np >= 0)

#    if not use_dihedrals:
#        return c10y_np, nbonds_np, bonds_np, angles_np, None

    dihedrals_np = np.resize(dihedrals_np, (ndihedrals_tot, 4))
    assert np.all(dihedrals_np >= 0)

#    if not use_angles:
#        return c10y_np, nbonds_np, bonds_np, None, dihedrals_np
    
    return c10y_np, nbonds_np, bonds_np, angles_np, dihedrals_np

def cart_to_internal(np.ndarray[np.float64_t, ndim=2] pos_np,
                     np.ndarray[np.int32_t, ndim=2] bonds_np,
                     np.ndarray[np.int32_t, ndim=2] angles_np,
                     np.ndarray[np.int32_t, ndim=2] dihedrals_np,
                     bint gradient=False,
                     bint curvature=False):
    cdef double[:, :] pos = memoryview(pos_np)
    cdef int[:, :] bonds = memoryview(bonds_np)
    cdef int[:, :] angles = memoryview(angles_np)
    cdef int[:, :] dihedrals = memoryview(dihedrals_np)

    cdef size_t natoms = len(pos_np)
    cdef size_t nbonds = len(bonds_np)
    cdef size_t nangles = len(angles_np)
    cdef size_t ndihedrals = len(dihedrals_np)
    cdef size_t ninternal = nbonds + nangles + ndihedrals

    # Arrays for internal coordinates and their derivatives
    q_np = np.zeros(ninternal, dtype=np.float64)
    if gradient:
        dq_np = np.zeros((ninternal, natoms, 3), dtype=np.float64)
    else:
        dq_np = np.zeros((ninternal, 0, 0), dtype=np.float64)

    if curvature:
        d2q_np = np.zeros((ninternal, natoms, 3, natoms, 3), dtype=np.float64)
    else:
        d2q_np = np.zeros((ninternal, 0, 0, 0, 0), dtype=np.float64)

    # Arrays for displacement vectors between bonded atoms
    dx_bonds_np = np.zeros((nbonds, 3), dtype=np.float64)
    dx_angles_np = np.zeros((nangles, 2, 3), dtype=np.float64)
    dx_dihedrals_np = np.zeros((ndihedrals, 3, 3), dtype=np.float64)

    cdef double[:, :] dx_bonds = memoryview(dx_bonds_np)
    cdef double[:, :, :] dx_angles = memoryview(dx_angles_np)
    cdef double[:, :, :] dx_dihedrals = memoryview(dx_dihedrals_np)

    cdef size_t i

    # Calculate displacement vectors
    for i in range(nbonds):
        dcopy(&THREE, &pos[bonds[i, 1], 0], &UNITY, &dx_bonds[i, 0], &UNITY)
        daxpy(&THREE, &DNUNITY, &pos[bonds[i, 0], 0], &UNITY, &dx_bonds[i, 0], &UNITY)

    for i in range(nangles):
        dcopy(&THREE, &pos[angles[i, 1], 0], &UNITY, &dx_angles[i, 0, 0], &UNITY)
        dcopy(&THREE, &pos[angles[i, 2], 0], &UNITY, &dx_angles[i, 1, 0], &UNITY)

        daxpy(&THREE, &DNUNITY, &pos[angles[i, 0], 0], &UNITY, &dx_angles[i, 0, 0], &UNITY)
        daxpy(&THREE, &DNUNITY, &pos[angles[i, 1], 0], &UNITY, &dx_angles[i, 1, 0], &UNITY)

    for i in range(ndihedrals):
        dcopy(&THREE, &pos[dihedrals[i, 1], 0], &UNITY, &dx_dihedrals[i, 0, 0], &UNITY)
        dcopy(&THREE, &pos[dihedrals[i, 2], 0], &UNITY, &dx_dihedrals[i, 1, 0], &UNITY)
        dcopy(&THREE, &pos[dihedrals[i, 3], 0], &UNITY, &dx_dihedrals[i, 2, 0], &UNITY)

        daxpy(&THREE, &DNUNITY, &pos[dihedrals[i, 0], 0], &UNITY, &dx_dihedrals[i, 0, 0], &UNITY)
        daxpy(&THREE, &DNUNITY, &pos[dihedrals[i, 1], 0], &UNITY, &dx_dihedrals[i, 1, 0], &UNITY)
        daxpy(&THREE, &DNUNITY, &pos[dihedrals[i, 2], 0], &UNITY, &dx_dihedrals[i, 2, 0], &UNITY)

    cdef size_t n = 0

    cdef double[:] q = memoryview(q_np[n : n+nbonds])
    cdef double[:, :, :] dq = memoryview(dq_np[n : n+nbonds])
    cdef double[:, :, :, :, :] d2q = memoryview(d2q_np[n : n+nbonds])

    cart_to_bond(bonds, dx_bonds, q, dq, d2q, gradient, curvature)
    n += nbonds

    q = memoryview(q_np[n : n+nangles])
    dq = memoryview(dq_np[n : n+nangles])
    d2q = memoryview(d2q_np[n : n+nangles])

    cart_to_angle(angles, dx_angles, q, dq, d2q, gradient, curvature)
    n += nangles

    q = memoryview(q_np[n : n+ndihedrals])
    dq = memoryview(dq_np[n : n+ndihedrals])
    d2q = memoryview(d2q_np[n : n+ndihedrals])

    cart_to_dihedral(dihedrals, dx_dihedrals, q, dq, d2q, gradient, curvature)
    n += ndihedrals

    return q_np, dq_np.reshape((-1, 3 * natoms)), d2q_np.reshape((-1, 3 * natoms, 3 * natoms))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void cart_to_bond(int[:, :] bonds,
                       double[:, :] dx,
                       double[:] q,
                       double[:, :, :] dq,
                       double[:, :, :, :, :] d2q,
                       bint gradient=False,
                       bint curvature=False):
    cdef size_t nbonds = len(bonds)

    cdef size_t i, a1, a2
    cdef double tmp
    cdef int sd_dx = dx.strides[1] // 8    # == 1 normally
    cdef int sd_dq = dq.strides[2] // 8   # == 1 normally
    cdef int sd_d2q = d2q.strides[2] // 8 # == 6? normally
    cdef int info

    for i in range(nbonds):
        a1 = bonds[i, 0]
        a2 = bonds[i, 1]
        
        q[i] = dnrm2(&THREE, &dx[i, 0], &sd_dx)

        if not (gradient or curvature):
            continue

        tmp = 1 / q[i]
        daxpy(&THREE, &tmp, &dx[i, 0], &sd_dx, &dq[i, a2, 0], &sd_dq)
        daxpy(&THREE, &DNUNITY, &dq[i, a2, 0], &sd_dq, &dq[i, a1, 0], &sd_dq)

        if not curvature:
            continue

        dlaset('G', &THREE, &THREE, &DZERO, &tmp, &d2q[i, a1, 0, a1, 0], &sd_d2q)

        tmp /= -q[i] * q[i]
        dger(&THREE, &THREE, &tmp, &dx[i, 0], &sd_dx, &dx[i, 0], &sd_dx, &d2q[i, a1, 0, a1, 0], &sd_d2q)
        dlacpy('G', &THREE, &THREE, &d2q[i, a1, 0, a1, 0], &sd_d2q, &d2q[i, a2, 0, a2, 0], &sd_d2q)

        dlacpy('G', &THREE, &THREE, &d2q[i, a1, 0, a1, 0], &sd_d2q, &d2q[i, a1, 0, a2, 0], &sd_d2q)
        dlascl('G', &UNITY, &UNITY, &DUNITY, &DNUNITY, &THREE, &THREE, &d2q[i, a1, 0, a2, 0], &sd_d2q, &info)
        dlacpy('G', &THREE, &THREE, &d2q[i, a1, 0, a2, 0], &sd_d2q, &d2q[i, a2, 0, a1, 0], &sd_d2q)


cdef void cart_to_angle(int[:, :] angles,
                        double[:, :, :] dx,
                        double[:] q,
                        double[:, :, :] dq,
                        double[:, :, :, :, :] d2q,
                        bint gradient=False,
                        bint curvature=False):
    cdef size_t nangles = len(angles)

    cdef double[:] x12, x23
    cdef double[:] x12_x23 = array(shape=(3,), itemsize=sizeof(double), format="d")

    cdef size_t i, j, k
    cdef size_t a1, a2, a3
    cdef double tmp1, tmp2, tmp3, tmp4
    cdef double r12, r23, r122, r232, r12x23, r12d23

    cdef double[:, :] dq_int = array(shape=(2, 3), itemsize=sizeof(double), format="d")

    cdef double[:, :, :, :] d2q_int = array(shape=(2, 3, 2, 3), itemsize=sizeof(double), format="d")

    cdef int sd_dx = dx.strides[2] // 8
    cdef int sd_dq = dq.strides[2] // 8
    cdef int sd_d2q = d2q.strides[2] // 8
    cdef int sd_d2int = d2q_int.strides[1] // 8

    # The standard way of calculating the angle between two vectors
    # a and b is arccos((a.b)/(|a| |b|)). Here, we use a different
    # approach that depends on the relation |sin q| = |a| |b| |axb|.
    # Rather than using arccos (or arcsin), we use arctan2(|axb|, (a.b)).
    # This is primarily because the curvature becomes easier to evaluate
    # (perhaps somewhat counterintuitively). arctan2 is also *somewhat*
    # more accurate than arccos, though this is probably negligable.

    for i in range(nangles):
        a1 = angles[i, 0]
        a2 = angles[i, 1]
        a3 = angles[i, 2]

        x12 = dx[i, 0]
        x23 = dx[i, 1]

        r12d23 = ddot(&THREE, &x12[0], &sd_dx, &x23[0], &sd_dx)

        cross(x12, x23, x12_x23)
        r12x23 = dnrm2(&THREE, &x12_x23[0], &UNITY)

        q[i] = atan2(r12x23, -r12d23)

        if not (gradient or curvature):
            continue

        r12 = dnrm2(&THREE, &x12[0], &sd_dx)
        r23 = dnrm2(&THREE, &x23[0], &sd_dx)

        r122 = r12 * r12
        r232 = r23 * r23
        memset(&dq_int[0, 0,], 0, 6 * sizeof(double))

        tmp1 = -r12d23 / (r122 * r12x23)
        tmp2 = 1 / r12x23
        daxpy(&THREE, &tmp1, &x12[0], &sd_dx, &dq_int[0, 0], &UNITY)
        daxpy(&THREE, &tmp2, &x23[0], &sd_dx, &dq_int[0, 0], &UNITY)

        tmp1 = -r12d23 / (r232 * r12x23)
        daxpy(&THREE, &tmp1, &x23[0], &sd_dx, &dq_int[1, 0], &UNITY)
        daxpy(&THREE, &tmp2, &x12[0], &sd_dx, &dq_int[1, 0], &UNITY)

        dcopy(&THREE, &dq_int[0, 0], &UNITY, &dq[i, a2, 0], &sd_dq)
        dcopy(&THREE, &dq_int[1, 0], &UNITY, &dq[i, a3, 0], &sd_dq)

        daxpy(&THREE, &DNUNITY, &dq_int[0, 0], &UNITY, &dq[i, a1, 0], &sd_dq)
        daxpy(&THREE, &DNUNITY, &dq_int[1, 0], &UNITY, &dq[i, a2, 0], &sd_dq)


        if not curvature:
            continue
        memset(&d2q_int[0, 0, 0, 0], 0, 36 * sizeof(double))


        # 0, 0
        tmp1 = -r12d23 / (r122 * r12x23)
        tmp2 = -r232 / (r12x23**3)
        tmp3 = r12d23 / (r12x23**3)
        tmp4 = r12d23 * (2 / r122 + r232 / r12x23**2) / (r122 * r12x23)

        dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2q_int[0, 0, 0, 0], &sd_d2int)
        dsyr2('L', &THREE, &tmp2, &x12[0], &sd_dx, &x23[0], &sd_dx, &d2q_int[0, 0, 0, 0], &sd_d2int)
        dsyr('L', &THREE, &tmp3, &x23[0], &sd_dx, &d2q_int[0, 0, 0, 0], &sd_d2int)
        dsyr('L', &THREE, &tmp4, &x12[0], &sd_dx, &d2q_int[0, 0, 0, 0], &sd_d2int)

        # 0, 1
        tmp1 = 1 / r12x23**3
        dsyr('L', &THREE, &tmp1, &x12_x23[0], &UNITY, &d2q_int[0, 0, 1, 0], &sd_d2int)
        symmetrize(&d2q_int[0, 0, 1, 0], 3, 6)

        # 1, 1
        tmp1 = -r12d23 / (r232 * r12x23)
        tmp2 = -r122 / (r12x23**3)
        tmp4 = r12d23 * (2 / r232 + r122 / r12x23**2) / (r232 * r12x23)

        dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2q_int[1, 0, 1, 0], &sd_d2int)
        dsyr2('L', &THREE, &tmp2, &x12[0], &sd_dx, &x23[0], &sd_dx, &d2q_int[1, 0, 1, 0], &sd_d2int)
        dsyr('L', &THREE, &tmp3, &x12[0], &sd_dx, &d2q_int[1, 0, 1, 0], &sd_d2int)
        dsyr('L', &THREE, &tmp4, &x23[0], &sd_dx, &d2q_int[1, 0, 1, 0], &sd_d2int)

        symmetrize(&d2q_int[0, 0, 0, 0], 6, 6)

        for j in range(3):
            for k in range(j, 3):
                d2q[i, a1, j, a1, k] = d2q[i, a1, k, a1, j] = d2q_int[0, j, 0, k]
                d2q[i, a2, j, a2, k] = d2q[i, a2, k, a2, j] = d2q_int[1, j, 1, k] + d2q_int[0, j, 0, k] - d2q_int[0, k, 1, j] - d2q_int[0, j, 1, k]
                d2q[i, a3, j, a3, k] = d2q[i, a3, k, a3, j] = d2q_int[1, j, 1, k]
            for k in range(3):
                d2q[i, a1, j, a2, k] = d2q[i, a2, k, a1, j] = d2q_int[0, j, 1, k] - d2q_int[0, j, 0, k]
                d2q[i, a1, j, a3, k] = d2q[i, a3, k, a1, j] = -d2q_int[0, j, 1, k]

                d2q[i, a2, j, a3, k] = d2q[i, a3, k, a2, j] = d2q_int[0, j, 1, k] - d2q_int[1, j, 1, k]


cdef void cart_to_dihedral(int[:, :] dihedrals,
                           double[:, :, :] dx,
                           double[:] q,
                           double[:, :, :] dq,
                           double[:, :, :, :, :] d2q,
                           bint gradient=False,
                           bint curvature=False):
    cdef size_t ndihedrals = len(dihedrals)

    cdef double[:] x12, x23, x34

    cdef int NINE = 9
    cdef int TWELVE = 12
    cdef int EIGHTYONE = 81

    # Arrays needed for q
    cdef double[:] x12_x23 = array(shape=(3,), itemsize=sizeof(double), format="d")
    cdef double[:] x23_x34 = array(shape=(3,), itemsize=sizeof(double), format="d")
    cdef double[:] tmpar1 = array(shape=(3,), itemsize=sizeof(double), format="d")

    # Arrays needed for dq
    cdef double[:] x12_x34 = array(shape=(3,), itemsize=sizeof(double), format="d")
    cdef double[:, :] dnumer = array(shape=(3, 3), itemsize=sizeof(double), format="d")
    cdef double[:, :] ddenom = array(shape=(3, 3), itemsize=sizeof(double), format="d")
    cdef double[:, :] dq_int = array(shape=(3, 3), itemsize=sizeof(double), format="d")

    # Arrays needed for d2q
    cdef double[:, :, :, :] d2numer = array(shape=(3, 3, 3, 3), itemsize=sizeof(double), format="d")
    cdef double[:, :, :, :] d2denom = array(shape=(3, 3, 3, 3), itemsize=sizeof(double), format="d")
    cdef double[:, :, :, :] d2q_int = array(shape=(3, 3, 3, 3), itemsize=sizeof(double), format="d")

    cdef double numer, denom

    cdef size_t i, j, k, m, n
    cdef size_t a1, a2, a3, a4
    cdef double tmp1, tmp2, tmp3, tmp4, tmp5
    cdef double r12, r23, r34

    cdef double r122, r232

    cdef int sd_dx = dx.strides[2] // 8
    cdef int sd_dq = dq.strides[2] // 8
    cdef int sd_d2q = d2q.strides[2] // 8

    cdef int sd_d2int = d2numer.strides[1] // 8

    for i in range(ndihedrals):
        a1 = dihedrals[i, 0]
        a2 = dihedrals[i, 1]
        a3 = dihedrals[i, 2]
        a4 = dihedrals[i, 3]

        x12 = dx[i, 0]
        x23 = dx[i, 1]
        x34 = dx[i, 2]

        r23 = dnrm2(&THREE, &x23[0], &sd_dx)
        cross(x12, x23, x12_x23)
        cross(x23, x34, x23_x34)
        cross(x12_x23, x23_x34, tmpar1)
        numer = ddot(&THREE, &tmpar1[0], &UNITY, &x23[0], &sd_dx) / r23
        denom = ddot(&THREE, &x12_x23[0], &UNITY, &x23_x34[0], &UNITY)
        q[i] = atan2(numer, denom)

        if not (gradient or curvature):
            continue

        cross(x12, x34, x12_x34)

        # Derivative of denominator
        memset(&ddenom[0, 0], 0, 9 * sizeof(double))
        cross(x23_x34, x23, ddenom[0])
        cross(x12, x23_x34, ddenom[1])
        cross(x12_x23, x34, tmpar1)
        daxpy(&THREE, &DUNITY, &tmpar1[0], &UNITY, &ddenom[1, 0], &UNITY)
        cross(x23, x12_x23, ddenom[2])

        # Derivative of numerator
        memset(&dnumer[0, 0], 0, 9 * sizeof(double))
        tmp1 = -r23
        tmp2 = -numer / (r23 * r23)

        daxpy(&THREE, &tmp1, &x23_x34[0], &UNITY, &dnumer[0, 0], &UNITY)
        daxpy(&THREE, &tmp2, &x23[0], &sd_dx, &dnumer[1, 0], &UNITY)
        daxpy(&THREE, &r23, &x12_x34[0], &UNITY, &dnumer[1, 0], &UNITY)
        daxpy(&THREE, &tmp1, &x12_x23[0], &UNITY, &dnumer[2, 0], &UNITY)

        tmp1 = numer * numer + denom * denom
        tmp2 = denom / tmp1
        tmp3 = -numer / tmp1

        memset(&dq_int[0, 0], 0, 9 * sizeof(double))
        for j in range(3):
            daxpy(&THREE, &tmp2, &dnumer[j, 0], &UNITY, &dq_int[j, 0], &UNITY)
            daxpy(&THREE, &tmp3, &ddenom[j, 0], &UNITY, &dq_int[j, 0], &UNITY)

        dcopy(&THREE, &dq_int[0, 0], &UNITY, &dq[i, a1, 0], &sd_dq)
        dcopy(&THREE, &dq_int[1, 0], &UNITY, &dq[i, a2, 0], &sd_dq)
        dcopy(&THREE, &dq_int[2, 0], &UNITY, &dq[i, a3, 0], &sd_dq)

        daxpy(&THREE, &DNUNITY, &dq_int[0, 0], &UNITY, &dq[i, a2, 0], &sd_dq)
        daxpy(&THREE, &DNUNITY, &dq_int[1, 0], &UNITY, &dq[i, a3, 0], &sd_dq)
        daxpy(&THREE, &DNUNITY, &dq_int[2, 0], &UNITY, &dq[i, a4, 0], &sd_dq)

        if not curvature:
            continue

        # Second derivative of denominator
        memset(&d2denom[0, 0, 0, 0], 0, 81 * sizeof(double))

        tmp1 = ddot(&THREE, &x23[0], &sd_dx, &x34[0], &sd_dx)
        dlaset('G', &THREE, &THREE, &DZERO, &tmp1, &d2denom[0, 0, 1, 0], &sd_d2int)
        dger(&THREE, &THREE, &DNTWO, &x23[0], &sd_dx, &x34[0], &sd_dx, &d2denom[0, 0, 1, 0], &sd_d2int)
        dger(&THREE, &THREE, &DUNITY, &x34[0], &sd_dx, &x23[0], &sd_dx, &d2denom[0, 0, 1, 0], &sd_d2int)

        tmp1 = -r23 * r23
        dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2denom[0, 0, 2, 0], &sd_d2int)
        dsyr('L', &THREE, &DUNITY, &x23[0], &sd_dx, &d2denom[0, 0, 2, 0], &sd_d2int)
        symmetrize(&d2denom[0, 0, 2, 0], 3, sd_d2int)

        tmp1 = -2 * ddot(&THREE, &x12[0], &sd_dx, &x34[0], &sd_dx)
        dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2denom[1, 0, 1, 0], &sd_d2int)
        dsyr2('L', &THREE, &DUNITY, &x12[0], &sd_dx, &x34[0], &sd_dx, &d2denom[1, 0, 1, 0], &sd_d2int)

        tmp1 = ddot(&THREE, &x12[0], &sd_dx, &x23[0], &sd_dx)
        dlaset('G', &THREE, &THREE, &DZERO, &tmp1, &d2denom[1, 0, 2, 0], &sd_d2int)
        dger(&THREE, &THREE, &DNTWO, &x12[0], &sd_dx, &x23[0], &sd_dx, &d2denom[1, 0, 2, 0], &sd_d2int)
        dger(&THREE, &THREE, &DUNITY, &x23[0], &sd_dx, &x12[0], &sd_dx, &d2denom[1, 0, 2, 0], &sd_d2int)

        # Second derivative of numerator
        memset(&d2numer[0, 0, 0, 0], 0, 81 * sizeof(double))

        skew(x34, d2numer[0, :, 1, :], -r23)
        tmp1 = 1. / r23
        dger(&THREE, &THREE, &tmp1, &x23[0], &sd_dx, &x23_x34[0], &UNITY, &d2numer[0, 0, 1, 0], &sd_d2int)

        skew(x23, d2numer[0, :, 2, :], r23)

        tmp1 = numer / (r23 * r23)
        dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2numer[1, 0, 1, 0], &sd_d2int)
        tmp2 = -tmp1 / (r23 * r23)
        tmp3 = -1 / r23
        dsyr('L', &THREE, &tmp2, &x23[0], &sd_dx, &d2numer[1, 0, 1, 0], &sd_d2int)
        dsyr2('L', &THREE, &tmp3, &x12_x34[0], &UNITY, &x23[0], &sd_dx, &d2numer[1, 0, 1, 0], &sd_d2int)

        skew(x12, d2numer[1, :, 2, :], -r23)
        tmp1 = 1. / r23
        dger(&THREE, &THREE, &tmp1, &x12_x23[0], &UNITY, &x23[0], &sd_dx, &d2numer[1, 0, 2, 0], &sd_d2int)

        dsyr2('L', &NINE, &DNUNITY, &dq_int[0, 0], &UNITY, &ddenom[0, 0], &UNITY, &d2numer[0, 0, 0, 0], &NINE)
        dsyr2('L', &NINE, &DUNITY, &dq_int[0, 0], &UNITY, &dnumer[0, 0], &UNITY, &d2denom[0, 0, 0, 0], &NINE)

        memset(&d2q_int[0, 0, 0, 0], 0, 81 * sizeof(double))
        tmp1 = numer * numer + denom * denom
        tmp2 = denom / tmp1
        tmp3 = -numer / tmp1

        daxpy(&EIGHTYONE, &tmp2, &d2numer[0, 0, 0, 0], &UNITY, &d2q_int[0, 0, 0, 0], &UNITY)
        daxpy(&EIGHTYONE, &tmp3, &d2denom[0, 0, 0, 0], &UNITY, &d2q_int[0, 0, 0, 0], &UNITY)

        symmetrize(&d2q_int[0, 0, 0, 0], NINE, NINE)

        for j in range(3):
            for k in range(j, 3):
                d2q[i, a1, j, a1, k] = d2q[i, a1, k, a1, j] = d2q_int[0, j, 0, k]
                d2q[i, a2, j, a2, k] = d2q[i, a2, k, a2, j] = d2q_int[1, j, 1, k] + d2q_int[0, j, 0, k] - d2q_int[0, k, 1, j] - d2q_int[0, j, 1, k]
                d2q[i, a3, j, a3, k] = d2q[i, a3, k, a3, j] = d2q_int[2, j, 2, k] + d2q_int[1, j, 1, k] - d2q_int[1, j, 2, k] - d2q_int[1, k, 2, j]
                d2q[i, a4, j, a4, k] = d2q[i, a4, k, a4, j] = d2q_int[2, j, 2, k]
            for k in range(3):
                d2q[i, a1, j, a2, k] = d2q[i, a2, k, a1, j] = d2q_int[0, j, 1, k] - d2q_int[0, j, 0, k]
                d2q[i, a1, j, a3, k] = d2q[i, a3, k, a1, j] = d2q_int[0, j, 2, k] - d2q_int[0, j, 1, k]
                d2q[i, a1, j, a4, k] = d2q[i, a4, k, a1, j] = -d2q_int[0, j, 2, k]

                d2q[i, a2, j, a3, k] = d2q[i, a3, k, a2, j] = d2q_int[1, j, 2, k] + d2q_int[0, j, 1, k] - d2q_int[1, j, 1, k] - d2q_int[0, j, 2, k]
                d2q[i, a2, j, a4, k] = d2q[i, a4, k, a2, j] = d2q_int[0, j, 2, k] - d2q_int[1, j, 2, k]

                d2q[i, a3, j, a4, k] = d2q[i, a4, k, a3, j] = d2q_int[1, j, 2, k] - d2q_int[2, j, 2, k]



