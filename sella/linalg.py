#!/usr/bin/env python

import numpy as np

from scipy.sparse.linalg import LinearOperator


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

    def __init__(self, func, x0, g0, eta, threepoint):
        self.func = func
        self.x0 = x0.copy()
        self.g0 = g0.copy()
        self.eta = eta
        self.threepoint = threepoint
        self.calls = 0

        n = len(self.x0)
        self.shape = (n, n)

    def _matvec(self, v):
        self.calls += 1
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
            return vnorm * (gplus - gminus) / (2 * self.eta)
        return vnorm * ((gplus - self.g0) / self.eta).reshape(v.shape)

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
    def __init__(self, A, Tm):
        self.A = A
        self.dtype = A.dtype
        self.Tm = Tm.copy()

        self.dtrue, self.dproj = Tm.shape
        self.shape = (self.dproj, self.dproj)
        self.Vs = np.empty((self.dtrue, 0), dtype=A.dtype)
        self.AVs = np.empty((self.dtrue, 0), dtype=A.dtype)

    def dot(self, v_m):
        v = self.Tm @ v_m
        self.Vs = np.hstack((self.Vs, v.reshape((self.dtrue, -1))))
        w = self.A.dot(v)
        self.AVs = np.hstack((self.AVs, w.reshape((self.dtrue, -1))))
        return self.Tm.T @ w


class MatrixSum(LinearOperator):
    def __init__(self, *args):
        matrices = []
        for arg in args:
            if isinstance(arg, MatrixSum):
                matrices += arg.matrices
            else:
                matrices.append(arg)

        self.dtype = sorted([mat.dtype for mat in matrices], reverse=True)[0]
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
