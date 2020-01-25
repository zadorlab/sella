from cython.view cimport array as cvarray

cimport numpy as np
from scipy.linalg.cython_blas cimport dcopy, dgemv
from scipy.linalg.cython_lapack cimport dgels

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp
from libc.string cimport memset

import numpy as np
from ase.data import covalent_radii
from scipy.optimize import minimize, brute

# We are fitting an approximate force field to a given gradient in order to predict
# the initial Hessian matrix for saddle point optimization. The fitted parameters of
# this force field are categorized as being linear or nonlinear. Only the nonlinear
# parameters are explicitly optimized using scipy.optimize -- the linear parameters
# are directly solved for at each iteration of the optimizer using a least squares fit.
#
# dFlin is the gradient of the atomic forces with respect to the LINEAR parameters.
# It is used in the least squares fit.
#
# dFlin[NDOF x NLINEAR]
#
# dFnonlin is the second derivative of the atomic forces with respect to the LINEAR parameters
# *and* the NONLINEAR parameters. Once we know the linear parameters from the least squares
# fit, we calculate from dFnonlin the first order derivative of the atomic forces with respect
# to the NONLINEAR parameters only. This is what goes into the gradient of our cost function.
#
# dFnonlin[NDOF x NNONLINEAR x NLINEAR]

cdef int ONE = 1
cdef int THREE = 3
cdef double D_ZERO = 0.
cdef double D_ONE = 1.
cdef double D_TWO = 2.

cdef double[:, :, :] dFlin
cdef double[:, :, :, :] dFnonlin

cdef struct s_pair_interactions:
    int nint
    int* npairs
    int** indices
    double** coords

ctypedef s_pair_interactions pair_interactions

cdef pair_interactions lj_interactions
cdef pair_interactions buck_interactions
cdef pair_interactions morse_interactions
cdef pair_interactions bond_interactions

cdef void init_pair_interactions(pair_interactions* pi, dict data):
    cdef int i
    cdef int j
    cdef int k
    cdef int npairs

    cdef int nint = len(data)

    pi[0].nint = nint
    pi[0].npairs = <int*> malloc(sizeof(int*) * nint)
    pi[0].indices = <int**> malloc(sizeof(int*) * nint)
    pi[0].coords = <double**> malloc(sizeof(double*) * nint)

    for i, (atom_types, pair_data) in enumerate(data.items()):
        npairs = len(pair_data)
        pi[0].npairs[i] = npairs

        pi[0].indices[i] = <int*> malloc(sizeof(int) * npairs * 2)
        pi[0].coords[i] = <double*>  malloc(sizeof(double) * npairs * 3)

        for j, (indices, coords) in enumerate(pair_data):
            for k in range(3):
                pi[0].coords[i][3 * j + k] = coords[k]
                if k < 2:
                    pi[0].indices[i][2 * j + k] = indices[k]

cdef void free_pair_interactions(pair_interactions* pi):
    cdef int i

    for i in range(pi[0].nint):
        free(pi[0].indices[i])
        free(pi[0].coords[i])

    free(pi[0].npairs)
    free(pi[0].indices)
    free(pi[0].coords)

#cdef struct s_pair_interaction:
#    int i
#    int j
#    double[:, :] xij
#
#ctypedef s_pair_interaction pair_interaction

def force_match(atoms, types=['buck', 'bond']):
    cdef int i
    cdef int j
    cdef int k
    cdef int a
    cdef int natoms = len(atoms)
    cdef int ndof = 3 * natoms
    cdef int nlin = 0
    cdef int nnonlin = 0
    cdef int nbuck = 0
    cdef int nbond = 0
    cdef int nlj = 0
    cdef int nmorse = 0
    cdef int info
    cdef bint do_lj
    cdef bint do_buck
    cdef bint do_morse
    cdef bint do_bond

    global dFlin
    global dFnonlin
    global lj_interactions
    global buck_interactions
    global bond_interactions

    enumbers = atoms.get_atomic_numbers()
    rmax = np.max(atoms.get_all_distances())
    rmin = np.min(atoms.get_all_distances() + rmax * np.eye(natoms))
    rcut = 3 * rmin
    #rcut = 9.5
    pos = atoms.get_positions()


    if np.any(atoms.pbc):
        cell = atoms.get_cell()
        latt_len = np.sqrt((cell**2).sum(1))
        V = abs(np.linalg.det(cell))
        n = atoms.pbc * np.array(np.ceil(rcut * np.prod(latt_len) /
                                         (V * latt_len)), dtype=int)
        tvecs = []
        for i in range(-n[0], n[0] + 1):
            latt_a = i * cell[0]
            for j in range(-n[1], n[1] + 1):
                latt_ab = latt_a + j * cell[1]
                for k in range(-n[2], n[2] + 1):
                    tvecs.append(latt_ab + k * cell[2])
        tvecs = np.array(tvecs)
    else:
        tvecs = np.array([[0., 0., 0.]])

    cdef double[:] rij = cvarray(shape=(3,), itemsize=sizeof(double), format='d')
    cdef double[:, :] pos_mv = memoryview(pos)
    cdef double[:, :] tvecs_mv = memoryview(tvecs)
    cdef int nt = len(tvecs)
    cdef int ti
    cdef double dij2
    cdef double dij
    cdef double rcut2 = rcut * rcut

    rij_t = np.zeros(3, dtype=np.float64)
    cdef double[:] rij_t_mv = memoryview(rij_t)

    do_lj = 'lj' in types
    do_buck = 'buck' in types
    do_morse = 'morse' in types
    do_bond = 'bond' in types

    vdw_data = {}
    cov_data = {}
    ff_data = {'lj': {},
               'buck': {},
               'morse': {},
               'bond': {},
               }
    x0 = []
    brute_range = []
    for i in range(natoms):
        for j in range(i, natoms):
            for k in range(3):
                rij[k] = pos_mv[j, k] - pos_mv[i, k]
            for ti in range(nt):
                if (i == j) and tvecs_mv[ti, 0] == 0. and tvecs_mv[ti, 1] == 0. and tvecs_mv[ti, 2] == 0.:
                    continue
                dij2 = 0.
                for k in range(3):
                    rij_t_mv[k] = rij[k] + tvecs_mv[ti, k]
                    dij2 += rij_t_mv[k] * rij_t_mv[k]
                if dij2 > rcut2:
                    continue
                dij = sqrt(dij2)
                eij = tuple(sorted(enumbers[[i, j]]))

                if do_lj:
                    vdw_x = ff_data['lj'].get(eij)
                    if vdw_x is None:
                        vdw_x = []
                        ff_data['lj'][eij] = vdw_x
                        nlj += 1
                    vdw_x.append(((i, j), rij_t.copy()))

                if do_buck:
                    vdw_x = ff_data['buck'].get(eij)
                    if vdw_x is None:
                        vdw_x = []
                        ff_data['buck'][eij] = vdw_x
                        nbuck += 1
                        x0.append(2.5)
                        brute_range.append((0.1, 10.0))
                    vdw_x.append(((i, j), rij_t.copy()))

                if do_morse:
                    vdw_x = ff_data['morse'].get(eij)
                    if vdw_x is None:
                        vdw_x = []
                        ff_data['morse'][eij] = vdw_x
                        nmorse += 1
                        x0.append(2.5)
                        brute_range.append((0.1, 10.0))
                    vdw_x.append(((i, j), rij_t.copy()))

                if do_bond:
                    rcov = np.sum(covalent_radii[list(eij)])
                    if dij > 1.5 * rcov:
                        continue
                    cov_x = ff_data['bond'].get(eij)
                    if cov_x is None:
                        cov_x = []
                        ff_data['bond'][eij] = cov_x
                        nbond += 1
                        x0.append(rcov)
                        brute_range.append((0.5 * rcov, 2 * rcov))
                    cov_x.append(((i, j), rij_t.copy()))

    nlin = 2 * nlj + 2 * nbuck + 2 * nmorse + nbond
    nnonlin = nbuck + nmorse + nbond

    init_pair_interactions(&lj_interactions, ff_data['lj'])
    init_pair_interactions(&buck_interactions, ff_data['buck'])
    init_pair_interactions(&morse_interactions, ff_data['morse'])
    init_pair_interactions(&bond_interactions, ff_data['bond'])

    dFlin = cvarray(shape=(natoms, 3, nlin), itemsize=sizeof(double), format='d')
    dFnonlin = cvarray(shape=(natoms, 3, nnonlin, nlin), itemsize=sizeof(double), format='d')

    memset(&dFlin[0, 0, 0], 0, sizeof(double) * ndof * nlin)
    memset(&dFnonlin[0, 0, 0, 0], 0, sizeof(double) * ndof * nnonlin * nlin)

    constraints = []
    if atoms.constraints:
        constraints = atoms.constraints
        atoms.constraints = []
    ftrue = atoms.get_forces()
    atoms.constraints = constraints

    bounds = []
    bounds += [(0., None)] * nbuck
    bounds += [(1., None)] * nmorse
    bounds += [(0., None)] * nbond

    if nnonlin < 5:
        x0 = brute(objective, brute_range, args=(ndof, nlin, natoms, ftrue), Ns=10, disp=True)
    else:
        x0 = np.array(x0, dtype=np.float64)

    print(objective(x0, ndof, nlin, natoms, ftrue))

    res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'gtol': 1e-10, 'ftol': 1e-8},
                   args=(ndof, nlin, natoms, ftrue, True), bounds=bounds)
    print(res)
    nonlinpars = res['x']

    linpars = objective(nonlinpars, ndof, nlin, natoms, ftrue, False, True)
    print(linpars, nonlinpars)

    hess = calc_hess(linpars, nonlinpars, natoms)

    free_pair_interactions(&lj_interactions)
    free_pair_interactions(&buck_interactions)
    free_pair_interactions(&morse_interactions)
    free_pair_interactions(&bond_interactions)

    return hess

def objective(pars, int ndof, int nlin, int natoms, np.ndarray[np.float_t, ndim=2] ftrue, bint grad=False, bint ret_linpars=False):
    global dFlin
    global dFnonlin
    global lj_interactions
    global buck_interactions
    global morse_interactions
    global bond_interactions

    cdef int i
    cdef int j
    cdef int k
    cdef int l
    cdef int a
    cdef int b
    cdef int linstart = 0
    cdef int nonlinstart = 0
    cdef int nnonlin = len(pars)
    cdef int info
    cdef int Asize
    cdef int nonlin_size

    cdef double chisq

    cdef double[:] pars_c = memoryview(pars)

    cdef double rho
    cdef double r0

    xij_np = np.zeros(3, dtype=np.float64)
    cdef double* xij = <double*> np.PyArray_DATA(xij_np)

    memset(&dFlin[0, 0, 0], 0, sizeof(double) * ndof * nlin)
    memset(&dFnonlin[0, 0, 0, 0], 0, sizeof(double) * ndof * nnonlin * nlin)

    for j in range(lj_interactions.nint):
        for k in range(lj_interactions.npairs[j]):
            a = lj_interactions.indices[j][2 * k]
            b = lj_interactions.indices[j][2 * k + 1]
            dcopy(&THREE, &lj_interactions.coords[j][3 * k], &ONE, xij, &ONE)
            lj(a, b, linstart, xij)
        linstart += 2

    for j in range(buck_interactions.nint):
        for k in range(buck_interactions.npairs[j]):
            rho = pars[nonlinstart]
            a = buck_interactions.indices[j][2 * k]
            b = buck_interactions.indices[j][2 * k + 1]
            dcopy(&THREE, &buck_interactions.coords[j][3 * k], &ONE, xij, &ONE)
            buck(a, b, linstart, nonlinstart, xij, rho)
        linstart += 2
        nonlinstart += 1

    for j in range(morse_interactions.nint):
        for k in range(morse_interactions.npairs[j]):
            rho = pars[nonlinstart]
            a = morse_interactions.indices[j][2 * k]
            b = morse_interactions.indices[j][2 * k + 1]
            dcopy(&THREE, &morse_interactions.coords[j][3 * k], &ONE, xij, &ONE)
            morse(a, b, linstart, nonlinstart, xij, rho)
        linstart += 2
        nonlinstart += 1

    for j in range(bond_interactions.nint):
        for k in range(bond_interactions.npairs[j]):
            r0 = pars[nonlinstart]
            a = bond_interactions.indices[j][2 * k]
            b = bond_interactions.indices[j][2 * k + 1]
            dcopy(&THREE, &bond_interactions.coords[j][3 * k], &ONE, xij, &ONE)
            bond(a, b, linstart, nonlinstart, xij, r0)
        linstart += 1
        nonlinstart += 1

    linpars, _, _, _ = np.linalg.lstsq(np.asarray(dFlin).reshape((ndof, nlin)), ftrue.ravel(), rcond=None)
    fapprox = np.einsum('ijk,k', dFlin, linpars)
    df = fapprox - ftrue
    chisq = np.sum(df * df)

    if ret_linpars:
        return linpars

    if not grad:
        return chisq

    dchisq = 2 * np.einsum('ij,ijkl,l', df, dFnonlin, linpars)

    return chisq, dchisq

def calc_hess(linpars, nonlinpars, natoms):
    global lj_interactions
    global buck_interactions
    global morse_interactions
    global bond_interactions

    cdef int i
    cdef int j
    cdef int k
    cdef int l
    cdef int a
    cdef int b
    cdef int linstart = 0
    cdef int nonlinstart = 0
    cdef int nlin = len(linpars)
    cdef int nnonlin = len(nonlinpars)

    cdef double C6
    cdef double C12
    cdef double A
    cdef double B
    cdef double D
    cdef double K
    cdef double r0
    cdef double rho

    xij_np = np.zeros(3, dtype=np.float64)
    cdef double* xij = <double*> np.PyArray_DATA(xij_np)

    hess_np = np.zeros((natoms, 3, natoms, 3))
    cdef double[:, :, :, :] hess = memoryview(hess_np)

    for i in range(lj_interactions.nint):
        for j in range(lj_interactions.npairs[i]):
            C6 = linpars[linstart]
            C12 = linpars[linstart + 1]
            a = lj_interactions.indices[i][2 * j]
            b = lj_interactions.indices[i][2 * j + 1]
            dcopy(&THREE, &lj_interactions.coords[i][3 * j], &ONE, xij, &ONE)
            lj_hess(a, b, C6, C12, xij, hess)
        linstart += 2

    for i in range(buck_interactions.nint):
        for j in range(buck_interactions.npairs[i]):
            A = linpars[linstart]
            C6 = linpars[linstart + 1]
            rho = nonlinpars[nonlinstart]

            a = buck_interactions.indices[i][2 * j]
            b = buck_interactions.indices[i][2 * j + 1]
            dcopy(&THREE, &buck_interactions.coords[i][3 * j], &ONE, xij, &ONE)
            buck_hess(a, b, A, C6, rho, xij, hess)
        linstart += 2
        nonlinstart += 1

    for i in range(morse_interactions.nint):
        for j in range(morse_interactions.npairs[i]):
            A = linpars[linstart]
            B = linpars[linstart + 1]
            rho = nonlinpars[nonlinstart]

            a = morse_interactions.indices[i][2 * j]
            b = morse_interactions.indices[i][2 * j + 1]
            dcopy(&THREE, &morse_interactions.coords[i][3 * j], &ONE, xij, &ONE)
            morse_hess(a, b, A, B, rho, xij, hess)
        linstart += 2
        nonlinstart += 1

    for i in range(bond_interactions.nint):
        for j in range(bond_interactions.npairs[i]):
            K = linpars[linstart]
            r0 = nonlinpars[nonlinstart]

            a = bond_interactions.indices[i][2 * j]
            b = bond_interactions.indices[i][2 * j + 1]
            dcopy(&THREE, &bond_interactions.coords[i][3 * j], &ONE, xij, &ONE)
            bond_hess(a, b, K, r0, xij, hess)
        linstart += 1
        nonlinstart += 1

    return hess_np.reshape((3 * natoms, 3 * natoms))

cdef void update_hess(int i, int j, double* xij, double[:, :, :, :] hess, double diag, double rest) nogil:
    cdef int k
    cdef int a
    cdef double hessterm

    for k in range(3):
        hessterm = diag + rest * xij[k] * xij[k]
        hess[i, k, i, k] += hessterm
        hess[i, k, j, k] -= hessterm

        hess[j, k, i, k] -= hessterm
        hess[j, k, j, k] += hessterm
        for a in range(k + 1, 3):
            hessterm = rest * xij[k] * xij[a]
            hess[i, k, i, a] += hessterm
            hess[i, k, j, a] -= hessterm
            hess[j, k, i, a] -= hessterm
            hess[j, k, j, a] += hessterm

            hess[i, a, i, k] += hessterm
            hess[i, a, j, k] -= hessterm
            hess[j, a, i, k] -= hessterm
            hess[j, a, j, k] += hessterm


cdef void lj(int i, int j, int linstart, double* xij) nogil:
    cdef double dij
    cdef double dij2
    cdef double dij8
    cdef double dij14
    cdef double f6
    cdef double f12
    cdef int k

    global dFlin

    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]

    dij = sqrt(dij2)
    dij8 = dij2 * dij2 * dij2 * dij2
    dij14 = dij8 * dij2 * dij2 * dij2

    f6 = 6 / dij8
    f12 = -12 / dij14

    for k in range(3):
        dFlin[i, k, linstart] += f6 * xij[k]
        dFlin[i, k, linstart + 1] += f12 * xij[k]

    for k in range(3):
        dFlin[j, k, linstart] += -f6 * xij[k]
        dFlin[j, k, linstart + 1] += -f12 * xij[k]

cdef void lj_hess(int i, int j, double C6, double C12, double* xij, double[:, :, :, :] hess) nogil:
    cdef double dij2
    cdef double dij8
    cdef double dij14
    cdef double dij10
    cdef double dij16
    cdef double diag
    cdef double rest
    cdef int k
    cdef int a


    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]

    dij8 = dij2 * dij2 * dij2 * dij2
    dij10 = dij8 * dij2
    dij14 = dij10 * dij2 * dij2
    dij16 = dij8 * dij8

    diag = -12 * C12 / dij14 + 6 * C6 / dij8
    rest = 168 * C12 / dij16 - 48 * C6 / dij10

    update_hess(i, j, xij, hess, diag, rest)

cdef void buck(int i, int j, int linstart, int nonlinstart,
               double* xij, double B) nogil:
    cdef double dij
    cdef double dij2
    cdef double dij8
    cdef double expterm
    cdef double fexp
    cdef double f6
    cdef double dB
    cdef int k

    global dFlin
    global dFnonlin

    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]

    dij = sqrt(dij2)
    dij8 = dij2 * dij2 * dij2 * dij2

    expterm = exp(-B * dij)
    fexp = -B * expterm / dij
    dB = B * expterm - expterm / dij

    f6 = 6. / dij8

    for k in range(3):
        dFlin[i, k, linstart] += fexp * xij[k]
        dFlin[i, k, linstart + 1] += f6 * xij[k]

        dFlin[j, k, linstart] -= fexp * xij[k]
        dFlin[j, k, linstart + 1] -= f6 * xij[k]

        dFnonlin[i, k, nonlinstart, linstart] += dB * xij[k]
        dFnonlin[j, k, nonlinstart, linstart] -= dB * xij[k]

cdef void buck_hess(int i, int j, double A, double C6, double B, double* xij, double[:, :, :, :] hess) nogil:
    cdef double dij
    cdef double dij2
    cdef double dij3
    cdef double dij8
    cdef double dij10
    cdef double expterm
    cdef double diag
    cdef double rest

    cdef int k

    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]
    dij = sqrt(dij2)
    dij3 = dij * dij2

    dij8 = dij2 * dij2 * dij2 * dij2
    dij10 = dij8 * dij2

    expterm = exp(-B * dij)

    diag = 6 * C6 / dij8 - A * B * expterm / dij
    rest = -48 * C6 / dij10 + A * B * expterm / dij3 + A * B * B * expterm / dij2

    update_hess(i, j, xij, hess, diag, rest)

cdef void morse(int i, int j, int linstart, int nonlinstart, double* xij, double rho) nogil:
    cdef double dij
    cdef double dij2
    cdef double expterm
    cdef double expterm2
    cdef double f_att
    cdef double f_rep
    cdef double drho_att
    cdef double drho_rep
    cdef int k

    global dFlin
    global dFnonlin

    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]

    dij = sqrt(dij2)

    expterm = exp(-rho * dij)
    expterm2 = expterm * expterm
    f_rep = -2 * rho * expterm2 / dij
    f_att = rho * expterm / dij
    drho_rep = 2 * expterm2 * (2 * rho * dij - 1) / dij
    drho_att = expterm * (1 - rho * dij) / dij

    for k in range(3):
        dFlin[i, k, linstart] += f_rep * xij[k]
        dFlin[i, k, linstart + 1] += f_att * xij[k]

        dFlin[j, k, linstart] -= f_rep * xij[k]
        dFlin[j, k, linstart + 1] -= f_att * xij[k]

        dFnonlin[i, k, nonlinstart, linstart] += drho_rep * xij[k]
        dFnonlin[i, k, nonlinstart, linstart + 1] += drho_att * xij[k]

        dFnonlin[j, k, nonlinstart, linstart] -= drho_rep * xij[k]
        dFnonlin[j, k, nonlinstart, linstart + 1] -= drho_att * xij[k]

cdef void morse_hess(int i, int j, double A, double B, double rho, double* xij, double[:, :, :, :] hess) nogil:
    cdef double dij
    cdef double dij2
    cdef double expterm
    cdef double expterm2
    cdef double diag
    cdef double rest

    cdef int k

    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]
    dij = sqrt(dij2)
    dij3 = dij2 * dij

    expterm = exp(-rho * dij)
    expterm2 = expterm * expterm

    diag = (B * expterm - 2 * A * expterm2) / dij
    rest = rho * ((2 * A * expterm2 - B * expterm) / dij + rho * (4 * A * expterm2 - B * expterm)) / dij2

    update_hess(i, j, xij, hess, diag, rest)

cdef void bond(int i, int j, int linstart, int nonlinstart, double* xij, double r0) nogil:
    cdef double dij
    cdef double dij2
    cdef double fbond
    cdef double dr0
    cdef int k

    global dFlin
    global dFnonlin

    dij2 = 0.
    for k in range(3):
        dij2 += xij[k] * xij[k]

    dij = sqrt(dij2)

    fbond = 2 * (dij - r0) / dij
    dr0 = -2 / dij

    for k in range(3):
        dFlin[i, k, linstart] += fbond * xij[k]
        dFlin[j, k, linstart] -= fbond * xij[k]

        dFnonlin[i, k, nonlinstart, linstart] += dr0 * xij[k]
        dFnonlin[j, k, nonlinstart, linstart] -= dr0 * xij[k]

cdef void bond_hess(int i, int j, double K, double r0, double* xij, double[:, :, :, :] hess) nogil:
    cdef double dij
    cdef double dij2
    cdef double dij3
    cdef double diag
    cdef double rest
    cdef int a

    dij2 = 0
    for a in range(3):
        dij2 += xij[a] * xij[a]
    dij = sqrt(dij2)
    dij3 = dij2 * dij

    diag = 2 * K * (dij - r0) / dij
    rest = 2 * K * r0 / dij3

    update_hess(i, j, xij, hess, diag, rest)
