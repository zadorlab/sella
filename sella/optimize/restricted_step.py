from typing import Optional, Union, List

import numpy as np
import inspect

from sella.peswrapper import PES, InternalPES
from .stepper import get_stepper, BaseStepper, NaiveStepper

from sella._constants import _LSTSQ_RCOND


class _CoordinateSelectionTranspose:
    def __init__(self, selection):
        self.selection = selection
        self.shape = (selection.shape[1], selection.shape[0])

    def __matmul__(self, values):
        values = np.asarray(values)
        out = np.zeros((self.shape[0],) + values.shape[1:], dtype=values.dtype)
        out[self.selection.indices] = values
        return out


class _CoordinateSelection:
    """Matrix-like selector used by the exact dummy-coordinate fast path."""
    def __init__(self, indices, dim):
        self.indices = np.asarray(indices, dtype=int)
        self.shape = (len(self.indices), dim)
        self.T = _CoordinateSelectionTranspose(self)

    def __matmul__(self, values):
        return np.asarray(values)[self.indices]


# Classes for restricted step (e.g. trust radius, max atom displacement, etc)
class BaseRestrictedStep:
    synonyms: List[str] = []

    def __init__(
        self,
        pes: Union[PES, InternalPES],
        order: int,
        delta: float,
        method: str = 'qn',
        tol: float = None,
        maxiter: int = 1000,
        d1: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ):
        self.pes = pes
        self.delta = delta
        self.d1 = d1
        self.projection_basis = None
        self.projected_eigenvalues = None
        g0 = self.pes.get_g()

        # W defaults to the identity, in which case Ufree.T @ W == Ufree.T.
        # Skip allocating an n_dof x n_dof eye for the (very common)
        # default case to avoid quadratic zero-fill on large systems.
        self._W_is_identity = (W is None)

        self.scons = self.pes.get_scons()
        # TODO: Should this be HL instead of H?
        g = g0 + self.pes.get_H() @ self.scons

        if inspect.isclass(method) and issubclass(method, BaseStepper):
            stepper = method
        else:
            stepper = get_stepper(method.lower())

        if self.cons(self.scons) - self.delta > 1e-8:
            self.P = self.pes.get_Unred().T
            dx = self.P @ self.scons
            self.stepper = NaiveStepper(dx)
            self.scons[:] *= 0
        else:
            fast_data = None
            dummies = getattr(self.pes, 'dummies', None)
            if (self.d1 is None and dummies is not None
                    and len(dummies) > 0):
                fast_data = self.pes.get_fast_restricted_step_data(
                    g, order, stepper, self._W_is_identity
                )
            if fast_data is not None:
                projection, g_step, H_step = fast_data
                if isinstance(projection, np.ndarray):
                    self.P = _CoordinateSelection(projection, self.pes.dim)
                else:
                    self.P = projection
                self.stepper = stepper(g_step, H_step, order, d1=None)
            else:
                Ufree = self.pes.get_Ufree()
                if self._W_is_identity:
                    self.P = Ufree.T
                    self.projection_basis = Ufree
                else:
                    self.P = Ufree.T @ W
                d1 = self.d1
                if d1 is not None:
                    d1 = np.linalg.lstsq(
                        self.P.T, d1, rcond=_LSTSQ_RCOND
                    )[0]
                self.stepper = stepper(
                    self.P @ g,
                    self.pes.get_HL_projected(self.P.T),
                    order,
                    d1=d1,
                )

        if self.projection_basis is not None:
            self.projected_eigenvalues = getattr(
                self.stepper, 'projected_eigenvalues', None
            )

        if tol is None:
            tol = 1e-10 if self.stepper.newton_safe else 1e-15
        self.tol = tol
        self.maxiter = maxiter

    def cons(self, s, dsda=None):
        raise NotImplementedError

    def eval(self, alpha):
        s, dsda = self.stepper.get_s(alpha)
        stot = self.P.T @ s + self.scons
        val, dval = self.cons(stot, self.P.T @ dsda)
        return stot, val, dval

    def get_s(self):
        alpha = self.stepper.alpha0

        s, val, dval = self.eval(alpha)
        if val < self.delta:
            assert val >= 0.
            return s, val
        err = val - self.delta

        lower = self.stepper.alphamin
        upper = self.stepper.alphamax

        for niter in range(self.maxiter):
            if abs(err) <= self.tol:
                break

            if np.nextafter(lower, upper) >= upper:
                break

            if err * self.stepper.slope > 0:
                upper = alpha
            else:
                lower = alpha

            with np.errstate(divide='ignore', invalid='ignore'):
                a1 = alpha - err / dval
            # Newton-safe paths still contract the sign-changing bracket
            # periodically. This keeps convergence independent of global
            # monotonicity, active-coordinate switches, and inaccurate local
            # derivatives while leaving ordinary fast convergence untouched.
            force_bisection = (
                (niter > 4 and not self.stepper.newton_safe)
                or (self.stepper.newton_safe and niter % 12 == 11)
            )
            if (not np.isfinite(a1) or a1 <= lower or a1 >= upper
                    or force_bisection):
                a2 = (lower + upper) / 2.
                if np.isinf(a2):
                    alpha = alpha + max(1, 0.5 * alpha) * np.sign(a2)
                else:
                    alpha = a2
            else:
                alpha = a1

            s, val, dval = self.eval(alpha)
            err = val - self.delta
        else:
            raise RuntimeError("Restricted step failed to converge!")

        assert val > 0
        return s, self.delta

    @classmethod
    def match(cls, name):
        return name in cls.synonyms


class TrustRegion(BaseRestrictedStep):
    synonyms = [
        'tr',
        'trust region',
        'trust-region',
        'trust radius',
        'trust-radius',
    ]

    def cons(self, s, dsda=None):
        val = np.linalg.norm(s)
        if dsda is None:
            return val

        dval = dsda @ s / max(val, 1e-12)
        return val, dval


class IRCTrustRegion(TrustRegion):
    synonyms = []

    def __init__(self, *args, sqrtm=None, **kwargs):
        assert sqrtm is not None
        self.sqrtm = sqrtm
        TrustRegion.__init__(self, *args, **kwargs)
        assert self.d1 is not None

    def cons(self, s, dsda=None):
        s = (s + self.d1) * self.sqrtm
        if dsda is not None:
            dsda = dsda * self.sqrtm
        return TrustRegion.cons(self, s, dsda)


class RestrictedAtomicStep(BaseRestrictedStep):
    synonyms = ['ras', 'restricted atomic step']

    def __init__(self, pes, *args, wc=1., **kwargs):
        if pes.int is not None:
            raise ValueError(
                "Internal coordinates are not compatible with "
                f"the {self.__class__.__name__} trust region method."
            )
        self.wc = wc  # weight for cell DOF (only used when optimizing the cell)
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        # With cell optimization the step vector is [atomic (3*natoms) | cell
        # (n_cell_dof)]. Only the atomic block is a stack of 3-vectors; the
        # cell DOF are scalars (and n_cell_dof need not be a multiple of 3, so
        # reshaping the whole vector into (-1, 3) would crash). Split the two
        # blocks and take the max over atomic 3-vector norms and weighted cell
        # DOF. When n_cell_dof == 0 this reduces to the original behavior.
        n_cell = getattr(self.pes, 'n_cell_dof', 0)
        if n_cell:
            s_atomic = s[:-n_cell]
            s_cell = s[-n_cell:]
        else:
            s_atomic = s
            s_cell = None

        s_mat = s_atomic.reshape((-1, 3))
        s_norms = np.linalg.norm(s_mat, axis=1)
        atomic_index = int(np.argmax(s_norms))
        val = s_norms[atomic_index]
        cell_index = -1
        if s_cell is not None and n_cell:
            cell_mags = np.abs(s_cell) * self.wc
            c_idx = int(np.argmax(cell_mags))
            if cell_mags[c_idx] > val:
                val = cell_mags[c_idx]
                cell_index = c_idx

        if dsda is None:
            return val

        if cell_index >= 0:
            # The active DOF is a cell parameter: its magnitude is
            # wc * |s_cell[c]|, so d(val)/da = wc * sign(s_cell[c]) * dsda_cell.
            flat = s_cell[cell_index]
            dsda_cell = dsda[-n_cell:][cell_index]
            dval = self.wc * np.sign(flat) * dsda_cell
            return val, dval

        dsda_mat = dsda[:len(s_atomic)].reshape((-1, 3))
        dval = dsda_mat[atomic_index] @ s_mat[atomic_index] / max(val, 1e-12)
        return val, dval


class MaxInternalStep(BaseRestrictedStep):
    synonyms = ['mis', 'max internal step']

    def __init__(self, pes, *args, wc=1., **kwargs):
        if pes.int is None:
            raise ValueError(
                f"Internal coordinates are required for the "
                f"{self.__class__.__name__} trust region method"
            )
        # The per-family weights (translations/bonds/angles/dihedrals/other/
        # rotations) were all fixed at 1.0 and never set by any caller, so only
        # the cell-DOF weight remains configurable.
        self.wc = wc  # Weight for cell DOF
        self._weights_cache = None
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        w = self._get_weights()
        assert len(w) == len(s)

        sw = np.abs(s * w)
        idx = np.argmax(np.abs(sw))
        val = sw[idx]

        if dsda is None:
            return val
        return val, np.sign(s[idx]) * dsda[idx] * w[idx]

    def _get_weights(self):
        """Build the per-DOF weight vector. Cached against
        (counts, weights, n_cell_dof) so the np.array construction
        only runs once per restricted-step instance."""
        cached = self._weights_cache
        n_cell_dof = self.pes.n_cell_dof
        key = (
            self.pes.int.ntrans, self.pes.int.nbonds,
            self.pes.int.nangles, self.pes.int.ndihedrals,
            self.pes.int.nother, self.pes.int.nrotations,
            n_cell_dof,
        )
        if cached is not None and cached[0] == key:
            return cached[1]
        n_coord = (
            self.pes.int.ntrans + self.pes.int.nbonds
            + self.pes.int.nangles + self.pes.int.ndihedrals
            + self.pes.int.nother + self.pes.int.nrotations
        )
        w = np.ones(n_coord)
        if n_cell_dof > 0:
            w = np.concatenate([w, [self.wc] * n_cell_dof])
        self._weights_cache = (key, w)
        return w


_all_restricted_step = [TrustRegion, RestrictedAtomicStep, MaxInternalStep]


def get_restricted_step(name):
    for rs in _all_restricted_step:
        if rs.match(name):
            return rs
    raise ValueError("Unknown restricted step name: {}".format(name))
