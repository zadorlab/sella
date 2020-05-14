#!/usr/bin/env python

import numpy as np

from sella.hessian_update import update_H

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh


class NumericalHessian(LinearOperator):
    dtype = np.dtype('float64')

    def __init__(self, func, x0, g0, eta, threepoint=False, Uproj=None):
        self.func = func
        self.x0 = x0.copy()
        self.g0 = g0.copy()
        self.eta = eta
        self.threepoint = threepoint
        self.calls = 0
        self.Uproj = Uproj

        self.ntrue = len(self.x0)

        if self.Uproj is not None:
            ntrue, n = self.Uproj.shape
            assert ntrue == self.ntrue
        else:
            n = self.ntrue

        self.shape = (n, n)

        self.Vs = np.empty((self.ntrue, 0), dtype=self.dtype)
        self.AVs = np.empty((self.ntrue, 0), dtype=self.dtype)

    def _matvec(self, v):
        self.calls += 1

        if self.Uproj is not None:
            v = self.Uproj @ v.ravel()

        # Since the sign of v is arbitrary, we choose a "canonical" direction
        # for the finite displacement. Essentially, we always displace in a
        # descent direction, unless the displacement vector is orthogonal
        # to the gradient. In that case, we choose a displacement in the
        # direction which brings the current coordinates projected onto
        # the displacement vector closer to "0". If the displacement
        # vector is orthogonal to both the gradient and the coordinate
        # vector, then choose whatever direction makes the first nonzero
        # element of the displacement positive.
        #
        # Note that these are completely arbitrary criteria for choosing
        # displacement direction. We are just trying to be as consistent
        # as possible for numerical stability and reproducibility reasons.

        vdotg = v.ravel() @ self.g0
        vdotx = v.ravel() @ self.x0
        sign = 1.
        if abs(vdotg) > 1e-4:
            sign = 2. * (vdotg < 0) - 1.
        elif abs(vdotx) > 1e-4:
            sign = 2. * (vdotx < 0) - 1.
        else:
            for vi in v.ravel():
                if vi > 1e-4:
                    sign = 1.
                    break
                elif vi < -1e-4:
                    sign = -1.
                    break

        vnorm = np.linalg.norm(v) * sign
        _, gplus = self.func(self.x0 + self.eta * v.ravel() / vnorm)
        if self.threepoint:
            fminus, gminus = self.func(self.x0 - self.eta * v.ravel() / vnorm)
            Av = vnorm * (gplus - gminus) / (2 * self.eta)
        else:
            Av = vnorm * (gplus - self.g0) / self.eta

        self.Vs = np.hstack((self.Vs, v.reshape((self.ntrue, -1))))
        self.AVs = np.hstack((self.AVs, Av.reshape((self.ntrue, -1))))

        if self.Uproj is not None:
            Av = self.Uproj.T @ Av

        return Av

    def __add__(self, other):
        return MatrixSum(self, other)

    def _transpose(self):
        return self


class MatrixSum(LinearOperator):
    def __init__(self, *matrices):
        # This makes sure that if matrices of different dtypes are
        # provided, we use the most general type for the sum.

        # For example, if two matrices are provided with the detypes
        # np.int64 and np.float64, then this MatrixSum object will be
        # np.float64.
        self.dtype = sorted([mat.dtype for mat in matrices], reverse=True)[0]
        self.shape = matrices[0].shape

        mnum = None
        self.matrices = []
        for matrix in matrices:
            assert matrix.dtype <= self.dtype
            assert matrix.shape == self.shape, (matrix.shape, self.shape)
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

    def _transpose(self):
        return MatrixSum(*[mat.T for mat in self.matrices])

    def __add__(self, other):
        return MatrixSum(*self.matrices, other)


class ApproximateHessian(LinearOperator):
    def __init__(
        self,
        dim: int,
        ncart: int,
        B0: np.ndarray = None,
        update_method: str = 'TS-BFGS',
        symm: int = 2,
    ) -> None:
        """A wrapper object for the approximate Hessian matrix."""
        self.dim = dim
        self.ncart = ncart
        self.shape = (self.dim, self.dim)
        self.dtype = np.float64
        self.update_method = update_method
        self.symm = symm
        self.initialized = False

        self.set_B(B0)

    def set_B(self, target):
        if target is None:
            self.B = None
            self.evals = None
            self.evecs = None
            return
        elif np.isscalar(target):
            target = target * np.eye(self.dim)
        assert target.shape == self.shape
        self.B = target
        self.evals, self.evecs = eigh(self.B)

    def update(self, dx, dg):
        """Perform a quasi-Newton update on B"""
        if self.B is None:
            B = np.zeros(self.shape, dtype=self.dtype)
        else:
            B = self.B.copy()
        if not self.initialized:
            self.initialized = True
            dx_cart = dx[:self.ncart]
            dg_cart = dg[:self.ncart]
            B[:self.ncart, :self.ncart] = update_H(
                None, dx_cart, dg_cart, method=self.update_method, symm=self.symm,
                lams=None, vecs=None
            )
            self.set_B(B)
            return

        self.set_B(update_H(B, dx, dg, method=self.update_method,
                            symm=self.symm, lams=self.evals, vecs=self.evecs))

    def project(self, U):
        """Project B into the subspace defined by U."""
        m, n = U.shape
        assert m == self.dim

        if self.B is None:
            Bproj = None
        else:
            Bproj = U.T @ self.B @ U

        return ApproximateHessian(n, 0, Bproj, self.update_method,
                                  self.symm)

    def asarray(self):
        if self.B is not None:
            return self.B
        return np.eye(self.dim)

    def _matvec(self, v):
        if self.B is None:
            return v
        return self.B @ v

    def _rmatvec(self, v):
        return self.matvec(v)

    def _matmat(self, X):
        if self.B is None:
            return X
        return self.B @ X

    def _rmatmat(self, X):
        return self.matmat(X)

    def __add__(self, other):
        if isinstance(other, ApproximateHessian):
            other = other.B
        if not self.initialized or other is None:
        #if self.B is None or other is None:
            tot = None
        else:
            tot = self.B + other
        return ApproximateHessian(
            self.dim, self.ncart, tot, self.update_method, self.symm
        )
