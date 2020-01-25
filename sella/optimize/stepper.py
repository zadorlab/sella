import numpy as np
from scipy.linalg import eigh


# Classes for optimization algorithms (e.g. MMF, Newton, RFO)
class BaseStepper:
    alpha0 = None
    alphamin = None
    alphamax = None
    # Whether the step size increases or decreases with increasing alpha
    slope = None
    synonyms = None

    def __init__(self, g, H, order=0, d1=None):
        self.g = g
        self.H = H
        self.order = order
        self.d1 = d1
        self._stepper_init()

    @classmethod
    def match(cls, name):
        return name in cls.synonyms

    def _stepper_init(self):
        raise NotImplementedError  # pragma: no cover

    def get_s(self, alpha):
        raise NotImplementedError  # pragma: no cover


class NaiveStepper(BaseStepper):
    synonyms = []  # No synonyms, we don't want someone using this accidentally
    alpha0 = 0.5
    alphamin = 0.
    alphamax = 1.
    slope = 1.

    def __init__(self, dx):
        self.dx = dx

    def get_s(self, alpha):
        return alpha * self.dx, self.dx


class QuasiNewton(BaseStepper):
    alpha0 = 0.
    alphamin = 0.
    alphamax = np.infty
    slope = -1
    synonyms = ['qn', 'quasi-newton', 'quasi newton', 'quasi-newton',
                'newton', 'mmf', 'minimum mode following',
                'minimum-mode following', 'dimer']

    def _stepper_init(self):
        self.L = np.abs(self.H.evals)
        self.L[:self.order] *= -1

        self.V = self.H.evecs
        self.Vg = self.V.T @ self.g

        self.ones = np.ones_like(self.L)
        self.ones[:self.order] = -1

    def get_s(self, alpha):
        denom = self.L + alpha * self.ones
        sproj = self.Vg / denom
        s = -self.V @ sproj
        dsda = self.V @ (sproj / denom)
        return s, dsda


class QuasiNewtonIRC(QuasiNewton):
    synonyms = []

    def _stepper_init(self):
        QuasiNewton._stepper_init(self)
        self.Vd1 = self.V.T @ self.d1

    def get_s(self, alpha):
        denom = np.abs(self.L) + alpha
        sproj = -(self.Vg + alpha * self.Vd1) / denom
        s = self.V @ sproj
        dsda = -self.V @ ((sproj + self.Vd1) / denom)
        return s, dsda


class RationalFunctionOptimization(BaseStepper):
    alpha0 = 1.
    alphamin = 0.
    alphamax = 1.
    slope = 1.
    synonyms = ['rfo', 'rational function optimization']

    def _stepper_init(self):
        self.A = np.block([[self.H.asarray(), self.g[:, np.newaxis]],
                           [self.g, 0]])

    def get_s(self, alpha):
        A = self.A * alpha
        A[:-1, :-1] *= alpha
        L, V = eigh(A)
        s = V[:-1, self.order] * alpha / V[-1, self.order]

        dAda = self.A.copy()
        dAda[:-1, :-1] *= 2 * alpha

        V1 = np.delete(V, self.order, 1)
        L1 = np.delete(L, self.order)
        dVda = V1 @ ((V1.T @ dAda @ V[:, self.order])
                     / (L1 - L[self.order]))

        dsda = (V[:-1, self.order] / V[-1, self.order]
                + (alpha / V[-1, self.order]) * dVda[:-1]
                - (V[:-1, self.order] * alpha
                   / V[-1, self.order]**2) * dVda[-1])
        return s, dsda


class PartitionedRationalFunctionOptimization(RationalFunctionOptimization):
    synonyms = ['prfo', 'p-rfo', 'partitioned rational function optimization']

    def _stepper_init(self):
        self.Vmax = self.H.evecs[:, :self.order]
        self.Vmin = self.H.evecs[:, self.order:]

        self.max = RationalFunctionOptimization(self.Vmax.T @ self.g,
                                                self.H.project(self.Vmax),
                                                order=self.Vmax.shape[1])

        self.min = RationalFunctionOptimization(self.Vmin.T @ self.g,
                                                self.H.project(self.Vmin),
                                                order=0)

    def get_s(self, alpha):
        smax, dsmaxda = self.max.get_s(alpha)
        smin, dsminda = self.min.get_s(alpha)

        s = self.Vmax @ smax + self.Vmin @ smin
        dsda = self.Vmax @ dsmaxda + self.Vmin @ dsminda
        return s, dsda


_all_steppers = [QuasiNewton, RationalFunctionOptimization,
                 PartitionedRationalFunctionOptimization]


def get_stepper(name):
    for stepper in _all_steppers:
        if stepper.match(name):
            return stepper
    raise ValueError("Unknown stepper name: {}".format(name))
