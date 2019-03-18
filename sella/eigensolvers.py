#!/usr/bin/env python

import numpy as np

from scipy.linalg import eigh, null_space, lstsq, solve
from scipy.sparse.linalg import LinearOperator, bicg, lsqr

from .cython_routines import ortho
from .hessian_update import symmetrize_Y, update_H

def Levi_Civita(n):
    LC = np.zeros((n, n, n))
    for i in range(n):
        even = np.roll(range(n), i)
        LC[tuple(even)] = 1.
        LC[tuple(even[::-1])] = -1.
    return LC

def project_translation(x0):
    d = len(x0)
    if d % 3 != 0:
        raise RuntimeError("Number of degrees of freedom not divisible by 3!")

    natoms = d // 3
    tvecs = np.zeros((3, d))
    tvec = np.array([[1., 0., 0.] for i in range(natoms)]).ravel()
    tvec /= np.linalg.norm(tvec)
    for i in range(3):
        tvecs[i] = np.roll(tvec, i)

    return tvecs.T

LC = Levi_Civita(3)

def project_rotation(x0):
    d = len(x0)
    if d % 3 != 0:
        raise RuntimeError("Number of degrees of freedom not divisible by 3!")

    natoms = d // 3
    rvecs = np.zeros((3, d))
    nrot = 0

    x0_mat = x0.reshape((-1, 3))
    com = np.average(x0_mat, axis=0)
    x0_com = x0_mat - com
    _, I = eigh(np.sum(x0_com**2) * np.eye(3) - x0_com.T @ x0_com)
    rvecs_guess = np.einsum('ij,jk,klm,nm->lin', x0_com, I, LC, I).reshape((3, d))
    for rvec in rvecs_guess:
        rvec_norm = np.linalg.norm(rvec)
        if rvec_norm > 1e-8:
            rvecs[nrot] = rvec / rvec_norm
            nrot += 1

    return rvecs[:nrot].T

def atoms_tr_projection(x0):
    d = len(x0)
    if d % 3 != 0:
        raise RuntimeError("Number of degrees of freedom not divisible by 3!")

    natoms = d // 3
    trvecs = np.zeros((6, d))
    tvec = np.array([[1., 0., 0.] for i in range(natoms)]).ravel()
    tvec /= np.linalg.norm(tvec)
    for i in range(3):
        trvecs[i] = np.roll(tvec, i)
    n_tr = 3
    
    # transform coordinates back into N,3 matrix
    x0_mat = x0.reshape((natoms, 3))
    # Find center of mass in *cartesian* coordinates
    com = np.average(x0_mat, axis=0)
    # offset positions in *mass-weighted cartesian* coordinates
    x0_com = x0_mat - com
    # eigenvectors of the inertia tensor
    _, I = eigh(np.sum(x0_com**2) * np.eye(3) - np.dot(x0_com.T, x0_com))
    # Rotation vectors in mass-weighted cartesian coordinates. It's complicated.
    #rvecs = np.dot(np.dot(x0_com, I), np.dot(LC, I.T)).swapaxes(0, 1).reshape((3, d))
    rvecs = np.einsum('ij,jk,klm,nm->lin', x0_com, I, LC, I).reshape((3, d))
    for rvec in rvecs:
        rvec_norm = np.linalg.norm(rvec)
        # while rvec_norm could be anything, a very small value indicates that the
        # mode may not correspond to a true rotation, e.g. for a linear molecule
        if rvec_norm > 1e-8:
            trvecs[n_tr] = rvec / rvec_norm
            n_tr += 1

    return trvecs[:n_tr].T

class MatrixWrapper(LinearOperator):
    def __init__(self, A):
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = A

    def _matvec(self, v):
        return self.A.dot(v)

    def _rmatvec(self, v):
        return v.dot(self.A)


    def __add__(self, other):
        return MatrixSum(self, other)

    def _matmat(self, other):
        if isinstance(other, MatrixSum):
            raise ValueError
        return MatrixWrapper(self.A.dot(other))
    
    def _rmatmat(self, other):
        if isinstance(other, MatrixSum):
            raise ValueError
        return MatrixWrapper(other.dot(self.A))

    def _transpose(self):
        return MatrixWrapper(self.A.transpose)

    def _adjoint(self):
        return MatrixWrapper(self.A.conj().T)


class NumericalHessian(MatrixWrapper):
    dtype = np.dtype('float64')

    def __init__(self, func, x0, g0, dxL, threepoint):
        self.func = func
        self.x0 = x0
        self.g0 = g0
        self.dxL = dxL
        self.threepoint = threepoint
        self.calls = 0

        n = len(self.x0)
        self.shape = (n, n)

    def _matvec(self, v):
        self.calls += 1
        fplus, gplus = self.func(self.x0 + self.dxL * v.ravel())
        if self.threepoint:
            fminus, gminus = self.func(self.x0 - self.dxL * v.ravel())
            return (gplus - gminus) / (2 * self.dxL)
        return ((gplus - self.g0) / self.dxL).reshape(v.shape)

    def _matmat(self, V):
        W = np.zeros_like(V)
        for i, v in enumerate(V.T):
            W[:, i] = self.matvec(v)
        return W

    def _rmatvec(self, v):
        return self.matvec(v)

    def _transpose(self):
        return self

    def _adjoint(self):
        return self


class ProjectedMatrix(MatrixWrapper):
    dtype = np.dtype('float64')

    def __new__(cls, A, basis, trvecs):
        self = super(ProjectedMatrix, cls).__new__(cls)
        self.A = A
        self.basis = basis
        self.trvecs = trvecs

        if isinstance(A, np.ndarray):
            return basis.T @ A @ basis
        
        return self

    def __init__(self, A, basis, trvecs):
        self.A = A
        self.basis = basis
        self.trvecs = trvecs

        _, self.n = self.basis.shape
        _, self.ntr = self.trvecs.shape
        self.nall = self.n + self.ntr

        self.shape = (self.n, self.n)

    def _matvec(self, v):
        return (self.basis.T @ self.A.dot(self.basis @ v.ravel())).reshape(v.shape)

    def _matmat(self, V):
        if isinstance(V, MatrixSum):
            raise ValueError
        return (self.basis.T @ self.A.dot(self.basis @ V))

    def _adjoint(self):
        return self

    def _transpose(self):
        return self


class MatrixSum(LinearOperator):
    def __init__(self, *args):
        matrices = []
        for arg in args:
            if isinstance(arg, MatrixSum):
                matrices += arg.matrices
            else:
                matrices.append(arg)

        self.dtype = sorted([matrix.dtype for matrix in matrices], reverse=True)[0]
        self.shape = matrices[0].shape
        
        mnum = None
        self.matrices = []
        for matrix in matrices:
            assert matrix.dtype <= self.dtype
            assert matrix.shape == self.shape
            if isinstance(matrix, np.ndarray):
                if mnum is None:
                    mnum = np.zeros(self.shape, dtype=self.dtype)
                mnum += matrix
            else:
                self.matrices.append(matrix)
        if mnum is not None:
            self.matrices.append(mnum)

    def _matvec(self, v):
        w = np.zeros_like(v, dtype=self.dtype)
        for matrix in self.matrices:
            w += matrix.dot(v)
        return w
    
    def _rmatvec(self, v):
        w = np.zeros_like(v, dtype=self.dtype)
        for matrix in self.matrices:
            w += v.dot(matrix)
        return w

    def _matmat(self, V):
        if isinstance(V, np.ndarray):
            return self._matvec(V)
        return MatrixSum(*[matrix.dot(V) for matrix in self.matrices])

    def _rmatmat(self, V):
        return MatrixSum(*[V.dot(matrix) for matrix in self.matrices])

    def _adjoint(self):
        return self

    def _transpose(self):
        return self

    def __add__(self, other):
        return MatrixSum(self, other)


def exact(A, maxres=None, P=None, T=None, shift=None, nlan=None):
    if isinstance(A, np.ndarray):
        lams, vecs = eigh(A)
    else:
        n, _ = A.shape
        if P is None:
            P = np.eye(n)
            vecs_P = np.eye(n)
        else:
            _, vecs_P, _ = exact(P)

        # Construct numerical version of A in case it is a LinearOperator.
        # This should be more or less exact if A is a numpy array already.
        B = np.zeros((n, n))
        for i in range(n):
            v = vecs_P[i]
            B += np.outer(v, A.dot(v))
        B = 0.5 * (B + B.T)
        lams, vecs = eigh(B)
    return lams, vecs, lams[np.newaxis, :] * vecs

def lanczos(A, v0, maxres, P=None, T=None, rightmost=False, niter=None, shift=None):
    d = len(v0)

    if niter is None:
        niter = d

    if maxres <= 0:
        return exact(A, maxres, P)


    if shift is None:
        shift = 1000

    if T is None:
        T = np.empty((d, 0))
        U = np.empty((d, 0))
        W = np.empty((d, 0))
    else:
        U = T.copy()
        W = shift * T.copy()

    _, nt = T.shape

    if rightmost:
        ind = -1 - nt
    else:
        ind = 0

    u0 = ortho(np.array([v0]).T / np.linalg.norm(v0), T)
    U = np.hstack((U, u0))
    W = np.hstack((W, A.dot(u0)))
    Atilde = U.T @ W
    lams, vecs = eigh(Atilde)
    r = W @ vecs[:, ind] - lams[ind] * U @ vecs[:, ind]

    for i in range(1, niter):
        u = ortho(W[:, -1], U)
        w = A @ u
        U = np.hstack([U, u])
        W = np.hstack([W, w])

        L = np.tril(U.T @ W - W.T @ U, -1)
        Wtilde = W + U @ L.T

        Atilde = U.T @ Wtilde
        lams, vecs = eigh(Atilde)
        r = W @ vecs[:, ind] - lams[ind] * U @ vecs[:, ind]
        rnorm = np.linalg.norm(r)
        print(rnorm, lams[ind], rnorm / lams[ind])
        if np.linalg.norm(r) < maxres * abs(lams[ind]):
            print("Lanczos took {} iterations".format(i))
            return lams, U @ vecs, Wtilde @ vecs
    else:
        print("Warning: Lanczos did not converge!")
        return lams, U @ vecs, Wtilde @ vecs


def lobpcg(A, v0, maxres, P=None):

    if maxres <= 0:
        return exact(A, maxres, P)

    # Use identity matrix as preconditioner if none provided
    if P is None:
        N = I.copy()
    else:
        N = P

    N_thetas, N_vecs, _ = exact(N, 0)
    X0 = N_vecs[:, :3]
    N_theta = N_thetas[0]

#    X0 = np.array([v0]).T

    n, nev = X0.shape
    I = np.eye(n)

    # Relative convergence tolerance
    if maxres is None:
        maxres = np.sqrt(1e-15) * n


    # Orthogonalize initial guess vectors
    X = ortho(X0, np.empty((n, 0)))
    U = X.copy()

    # Initial Ritz pairs
    AX = A @ X
    AU = AX.copy()
    Atilde = X.conj().T @ AX
    thetas, Y = eigh(Atilde)

    # Update X and calculate residuals R
    X = X @ Y
    AX = AX @ Y
    Atilde = X.conj().T @ AX
    thetas, Y = eigh(Atilde)
    X = X @ Y[:, :nev]
    AX = AX @ Y[:, :nev]
    R = AX - X @ np.diag(thetas)
    RI = R.copy()

    # P begins empty
    P = np.empty((n, 0))
    AP = np.empty((n, 0))
    for k in range(n):
        # Find new search directions and orthogonalize
        Htilde, _, _, _ = np.linalg.lstsq(N - 1.15 * N_theta * I, RI, rcond=None)
#        Htilde, _ = bicg(N - 1.15 * N_theta * I, RI)
        H = ortho(Htilde, np.hstack((X, P)))
        U = np.hstack((U, H))

        # New set of guess vectors
        S = np.hstack((X, H, P)).copy()

        # Calculate action of A on search directions
        AH = A @ H
        AU = np.hstack((AU, AH))
        AS = np.hstack((AX, AH, AP))

        # Updated Ritz pairs
        Atilde = S.conj().T @ AS
        thetas, Y = eigh(Atilde)

        # Leftmost Ritz vectors becomes new X
        X = S @ Y[:, :nev]
        AX = AS @ Y[:, :nev]

        # Update residuals
        R = AX - X @ np.diag(thetas[:nev])

        # Check which if any vectors are converged
        converged = np.linalg.norm(R, axis=0) < maxres * np.abs(thetas[:nev])
        print(np.linalg.norm(R[:, 0]), thetas[0], np.linalg.norm(R[:, 0]) / thetas[0])

        if all(converged):
            print("LOBPCG converged in {} iterations".format(k))
            return thetas, S @ Y, AS @ Y

        # Indices of unconverged vectors
        iconv = [i for i in range(nev) if not converged[i]]

        RI = R[:, iconv].copy()
        Ytilde = Y[:, iconv].copy()

        # Zero the components belonging to X
        Ytilde[:nev, :] = 0.

        # Strict reorthogonalization with repeated MGS
        YI = ortho(Ytilde, Y[:, :nev])
        P = S @ YI
        AP = AS @ YI
    print('Warning: LOBPCG may not have converged')
    return thetas, S @ Y, AS @ Y

def davidson(A, maxres, P=None, T=None, V0=None, niter=None, shift=None, nlan=0, nrandom=0, nvecs=0):
    n, _ = A.shape

    if niter is None:
        niter = n
    
    if maxres <= 0:
        return exact(A, maxres, P)

    I = np.eye(n)

    if P is None:
        P = np.eye(n)

    if V0 is None:
        V0 = np.empty((n, 0))

    if shift is None:
        shift = 1000

    if T is None:
        T = np.empty((n, 0))
        dA = np.zeros((n, n))
        A_shift = A
        P_shift = P
    else:
        dA = shift * T @ T.T
        A_shift = A# + dA
        #P_shift = P# + dA
        P_shift = P.copy()


    _, nt = T.shape

    AT = shift * T

    P_lams, P_vecs, _ = exact(P_shift, 0)
    nneg = max(2, np.sum(P_lams < 0) + 1, nvecs)

    if nlan <= 0:
        V_lan = np.empty((n, 0))
        AV_lan = np.empty((n, 0))
    else:
        lams, T, AT = lanczos(A_shift, P_vecs[:, -(nt + 1)], maxres=maxres, P=P_shift, rightmost=True, niter=nlan, T=T, shift=shift)
    
    # Adding random vector to search space improves stability
    # of the optimization code by improving the approximate Hessian
    # in directions that are orthogonal to the minimum eigenvector
    # and the step direction.
    V_rand = 2 * np.random.random((n, nrandom)) - 1
    V = ortho(np.hstack((V0, P_vecs[:, :max(2, nneg)], V_rand)), T)

    AV = A_shift.dot(V)
    V = np.hstack((V, T))
    AV = np.hstack((AV, AT))

    method = 2
    seeking = 0
    while True:
        Atilde = V.T @ (symmetrize_Y(V, AV, symm=method))
        lams, vecs = eigh(Atilde)
        nneg = max(2, np.sum(lams < 0) + 1, nvecs)
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        AV = AV @ vecs
        V = V @ vecs
        
        Ytilde = symmetrize_Y(V, AV, symm=method)
        R = Ytilde[:, :nneg] - V[:, :nneg] * lams[np.newaxis, :nneg]
        Rnorm = np.linalg.norm(R, axis=0)
        print(Rnorm, lams[:nneg], Rnorm / lams[:nneg], seeking)

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(zip(Rnorm, lams)):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if rinorm >= maxres * np.abs(thetai):
                ri = R[:, seeking]
                ui = V[:, seeking]
                break
        # If they all seem converged, then we are done
        else:
            return lams, V, AV

        # Find t such that (I - u u^T) (P - theta *I)^-1 t = -r, and t is orthogonal to u,
        # where u is the Ritz vector (not the entire subspace spanned by V!)
        Pproj = P_shift - thetai * I
        Pprojr = solve(Pproj, ri)
        Pproju = solve(Pproj, ui)
        ti = Pproju * (ui @ Pprojr) / (ui @ Pproju) - Pprojr
        t = ortho(ti, V)

        # Davidson failed to find a new search direction
        if t.shape[1] == 0:
            # Do Lanczos instead
            t = ortho(AV[:, -1], V)
            # If Lanczos also fails to find a new search direction,
            # just give up and return the current Ritz pairs
            if t.shape[1] == 0:
                return lams, V, AV

        V = np.hstack([V, t])
        AV = np.hstack([AV, A_shift.dot(t)])
    else:
        return lams, V, AV
