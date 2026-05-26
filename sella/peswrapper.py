import logging
from typing import Union, Callable

import numpy as np
from scipy.linalg import eigh, expm, expm_frechet, logm, polar, qr, solve_triangular
from scipy.integrate import LSODA
from ase import Atoms
from ase.build import niggli_reduce
from ase.utils import basestring
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from sella.utilities.math import modified_gram_schmidt
from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.eigensolvers import rayleigh_ritz
from sella.internal import Internals, Constraints, DuplicateInternalError
from sella._gpu import gpu_qr as _gpu_qr, gpu_project

logger = logging.getLogger(__name__)


class _LRU2:
    """2-entry LRU cache keyed by state hash (bytes).

    Two entries match the optimization step cycle, which alternates between
    pre-ODE (post-cell-change) and post-ODE positions.
    """

    __slots__ = ('_entries', '_next')

    def __init__(self):
        self._entries = [None, None]
        self._next = 0

    def get(self, key):
        for entry in self._entries:
            if entry is not None and entry[0] == key:
                return entry[1]
        return None

    def put(self, key, value):
        for entry in self._entries:
            if entry is not None and entry[0] == key:
                return
        self._entries[self._next] = (key, value)
        self._next = 1 - self._next


def _split_cons_subspace(drdxnred, tol_factor=1e-6):
    """Split (n_int) into Ucons (rowspace of drdxnred) and Ufree (its complement).

    Replaces ``np.linalg.svd(drdxnred)`` (which materializes a full
    n_int×n_int V matrix) with rank-revealing QR on ``drdxnred.T``.
    For an (m, n) drdxnred with m << n, this is roughly half the cost of
    the SVD path and returns the same orthonormal subspaces (column order
    differs but the spans match — every downstream consumer is column-
    permutation-invariant).

    Returns ``(Ucons, Ufree)`` of shapes (n, ncons) and (n, n - ncons).
    """
    Q, R, _ = qr(drdxnred.T, mode='full', pivoting=True, check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.size and diag[0] > 0:
        ncons = int(np.sum(diag > tol_factor * diag[0]))
    else:
        ncons = 0
    return Q[:, :ncons], Q[:, ncons:]


def _logm_3x3(F):
    """Closed-form 3x3 matrix logarithm via eigendecomposition.

    Replaces ``scipy.linalg.logm`` (which uses Padé + inverse-squaring
    + onenormest, ~0.9ms per call on 3x3) with a direct
    eigendecomposition: ``log(F) = V diag(log(lam)) V^{-1}``. ~50x
    faster on 3x3, machine-precision agreement on cell-deformation
    inputs (real, near-identity, well-conditioned).

    Falls back to scipy.linalg.logm when ``np.linalg.eig`` produces a
    near-singular eigenvector matrix (defective F). Returns the real
    part directly since cell deformation gradients are real and have
    no negative real eigenvalues for any reasonable cell.
    """
    lam, V = np.linalg.eig(F)
    if np.linalg.cond(V) > 1e10:
        return logm(F)
    return (V @ np.diag(np.log(lam)) @ np.linalg.inv(V)).real


def _expm_frechet_3x3_contracted(U, dEdF):
    """Compute g[mu,nu] = sum_{ab} d expm(U)[E_munu] * dEdF[a,b] for all 9 (mu,nu).

    Replaces the inner ``9x scipy.linalg.expm_frechet`` loop in the
    cell-stress-to-gradient conversion. For diagonalizable 3x3 ``U``
    (which is the typical case for a real positive-definite cell
    deformation logm), the directional derivative of expm has the
    Daleckii–Krein closed form

        d expm(U)[E] = V (f(lam) ⊙ (V^{-1} E V)) V^{-1},

    where ``f(a, b) = (e^a - e^b) / (a - b)`` (or ``e^a`` when ``a==b``).
    Contracting over E_munu and dEdF gives the single matmul chain
    ``g = real(Vinv.T (f ⊙ (V.T dEdF Vinv.T)) V.T)``. ~25x faster than
    the scipy loop on 3x3 inputs (0.67 → 0.03 ms/call).

    Falls back to scipy when ``U`` is too close to zero (eigenvectors
    of a noisy zero matrix are catastrophically ill-conditioned) or
    when ``np.linalg.eig`` produces a near-singular ``V``.
    """
    # When ||U|| ~ 0 the derivative reduces to the identity map E -> E,
    # so the contracted output is dEdF itself. Avoid eig on a noisy zero.
    Unorm = np.linalg.norm(U)
    if Unorm < 1e-10:
        return dEdF.copy()
    lam, V = np.linalg.eig(U)
    if np.linalg.cond(V) > 1e10:
        # Fall back when the eigenvector basis is too ill-conditioned
        # for the closed form to be numerically reliable.
        g = np.zeros((3, 3))
        for mu in range(3):
            for nu in range(3):
                E = np.zeros((3, 3)); E[mu, nu] = 1.0
                ed = expm_frechet(U, E, compute_expm=False)
                g[mu, nu] = np.sum(ed * dEdF)
        return g
    Vinv = np.linalg.inv(V)
    expl = np.exp(lam)
    diff = lam[:, None] - lam[None, :]
    mask = np.abs(diff) > 1e-12 * max(np.abs(lam).max(), 1.0)
    safe = np.where(mask, diff, 1.0)
    fij = np.where(mask, (expl[:, None] - expl[None, :]) / safe, expl[:, None])
    M = V.T @ dEdF @ Vinv.T
    return (Vinv.T @ (fij * M) @ V.T).real


def _niggli_hessian_transform(atoms, orig_cell, exp_cell_factor, cell_mask):
    """Compute the Hessian transformation matrix for Niggli reduction.

    The cell DOF are parameterized as elements of L = logm(F) * factor where
    F = cell @ inv(orig_cell). Niggli reduction changes the lattice basis,
    so the Hessian must be transformed from the old L-parameterization to the
    new one. This computes T such that H_new = T^T @ H_old @ T.

    The transformation is derived from the chain rule through the cell-element
    space: J_old maps old L-perturbations to cell perturbations (via Frechet
    derivative of expm), J_new maps new L-perturbations (at L=0, since
    orig_cell is reset). Then T = J_old^{-1} @ J_new.

    Parameters
    ----------
    atoms : Atoms
        The atoms object. Niggli reduction is applied in-place.
    orig_cell : ndarray, shape (3, 3)
        The old reference cell (before reduction).
    exp_cell_factor : float
        Scaling factor for the log-deformation parameterization.
    cell_mask : ndarray, shape (3, 3), dtype bool
        Mask selecting which cell DOF are free.

    Returns
    -------
    T_masked : ndarray, shape (n_cell_dof, n_cell_dof)
        Transformation matrix for the masked cell DOF.
    """
    # Compute old Jacobian: J_old[ab, ij] = d(cell_ab)/d(L_ij)
    # at the current (pre-reduction) L value
    F_old = atoms.get_cell().array @ np.linalg.inv(orig_cell)
    X_old = _logm_3x3(F_old) / exp_cell_factor  # unscaled log-deformation

    J_old = np.zeros((9, 9))
    for idx in range(9):
        i, j = divmod(idx, 3)
        E = np.zeros((3, 3))
        E[i, j] = 1.0 / exp_cell_factor
        dF = expm_frechet(X_old, E, compute_expm=False)
        dC = dF @ orig_cell  # d(cell)/d(L_ij)
        J_old[:, idx] = dC.ravel()

    # Apply Niggli reduction
    niggli_reduce(atoms)
    orig_cell_new = atoms.get_cell().array.copy()

    # New Jacobian at L=0: d(cell_ab)/d(L_ij) = (1/factor) * delta_ai * O_jb
    # In matrix form: J_new = (1/factor) * kron(I_3, orig_cell_new.T)
    J_new = np.kron(np.eye(3), orig_cell_new.T) / exp_cell_factor

    # T maps new L-perturbations to old L-perturbations (same physical cell change)
    # δL_old = T @ δL_new, so H_new = T^T @ H_old @ T
    T_full = np.linalg.solve(J_old, J_new)

    # Project to masked DOF: T_masked = M @ T_full @ M^T
    mask_flat = cell_mask.ravel()
    mask_indices = np.where(mask_flat)[0]
    T_masked = T_full[np.ix_(mask_indices, mask_indices)]

    return T_masked


class PES:
    n_cell_dof = 0

    def __init__(
        self,
        atoms: Atoms,
        H0: np.ndarray = None,
        constraints: Constraints = None,
        eigensolver: str = 'jd0',
        trajectory: Union[str, Trajectory] = None,
        eta: float = 1e-4,
        v0: np.ndarray = None,
        proj_trans: bool = None,
        proj_rot: bool = None,
        hessian_function: Callable[[Atoms], np.ndarray] = None,
    ) -> None:
        self.atoms = atoms
        if constraints is None:
            constraints = Constraints(self.atoms)
        if proj_trans is None:
            if constraints.internals['translations']:
                proj_trans = False
            else:
                proj_trans = True
        if proj_trans:
            try:
                constraints.fix_translation()
            except DuplicateInternalError:
                pass

        if proj_rot is None:
            if np.any(atoms.pbc):
                proj_rot = False
            else:
                proj_rot = True
        if proj_rot:
            try:
                constraints.fix_rotation()
            except DuplicateInternalError:
                pass
        self.cons = constraints
        self.eigensolver = eigensolver

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                self.traj = Trajectory(trajectory, 'w', self.atoms)
            else:
                self.traj = trajectory
        else:
            self.traj = None

        self.eta = eta
        self.v0 = v0

        self.neval = 0
        self.curr = dict(
            x=None,
            f=None,
            g=None,
        )
        self.last = self.curr.copy()

        # Internal coordinate specific things
        self.int = None
        self.dummies = None

        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        if H0 is None:
            self.set_H(None, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        self.savepoint = dict(apos=None, dpos=None)
        self.first_diag = True

        self.hessian_function = hessian_function

        self._basis_cache = _LRU2()

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def _state_hash(self) -> bytes:
        """Hash of all state that affects cached computations."""
        h = self.atoms.positions.tobytes()
        cell = self.atoms.cell
        if cell is not None and cell.any():
            h += cell.array.tobytes()
        return h

    def save(self):
        self.savepoint = dict(apos=self.apos, dpos=self.dpos)

    def restore(self):
        apos = self.savepoint['apos']
        dpos = self.savepoint['dpos']
        assert apos is not None
        self.atoms.positions = apos
        if dpos is not None:
            self.dummies.positions = dpos

    def close(self):
        """Close any open file handles (e.g., trajectory file)."""
        if self.traj is not None:
            self.traj.close()
            self.traj = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures trajectory is closed."""
        self.close()
        return False

    # Position getter/setter
    def set_x(self, target):
        diff = target - self.get_x()
        self.atoms.positions = target.reshape((-1, 3))
        return diff, diff, self.curr.get('g', np.zeros_like(diff))

    def get_x(self):
        return self.apos.ravel().copy()

    # Hessian getter/setter
    def get_H(self):
        return self.H

    def set_H(self, target, *args, **kwargs):
        self.H = ApproximateHessian(
            self.dim, self.ncart, target, *args, **kwargs
        )

    # Hessian of the constraints
    def get_Hc(self):
        if self.curr['L'] is None:
            raise RuntimeError(
                "PES.get_Hc() called with L=None. "
                f"curr_g_is_none={self.curr.get('g') is None}, "
                f"curr_f_is_none={self.curr.get('f') is None}."
            )
        return self.cons.hessian().ldot(self.curr['L'])

    # Hessian of the Lagrangian
    def get_HL(self):
        return self.get_H() - self.get_Hc()

    def get_HL_projected(self, U):
        """Projected Hessian of the Lagrangian: ApproximateHessian(U.T @ HL @ U).

        Equivalent to ``self.get_HL().project(U)`` but skips constructing the
        full (dim, dim) HL matrix and the intermediate ApproximateHessian.
        """
        H = self.get_H()
        H_B = H.B
        if H_B is None:
            Bproj = None
        else:
            UtHU = gpu_project(H_B, U, H_gpu=H._get_B_gpu())
            # Skip the constraint projection entirely when there are no
            # constraints — Hc is allocated as a (dim, dim) zero block in
            # CellInternalPES, so the matmul would just churn through ~N^3
            # zeros (and force a 44 MB GPU upload at 400 atoms).
            L = self.curr.get('L')
            if L is not None and L.size > 0:
                Hc = self.get_Hc()
                Bproj = UtHU - gpu_project(Hc, U)
            else:
                Bproj = UtHU
        n = U.shape[1]
        return ApproximateHessian(n, 0, Bproj, self.H.update_method, self.H.symm)

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.residual()

    def get_drdx(self):
        return self.cons.jacobian()

    def _calc_basis(self):
        state_hash = self._state_hash()
        cached = self._basis_cache.get(state_hash)
        if cached is not None:
            return cached

        drdx = self.get_drdx()
        Ucons, Ufree = _split_cons_subspace(drdx)
        Unred = np.eye(self.dim)
        result = (drdx, Ucons, Unred, Ufree)

        self._basis_cache.put(state_hash, result)
        return result

    def write_traj(self):
        if self.traj is not None:
            self.traj.write()

    def eval(self):
        self.neval += 1
        f = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        self.write_traj()
        return f, g

    def _calc_eg(self, x):
        self.save()
        self.set_x(x)

        f, g = self.eval()

        self.restore()
        return f, g

    def get_scons(self):
        """Returns displacement vector for linear constraint correction."""
        Ucons = self.get_Ucons()

        scons = -Ucons @ np.linalg.lstsq(
            self.get_drdx() @ Ucons,
            self.get_res(),
            rcond=None,
        )[0]
        return scons

    def _update(self, feval=True):
        state = self._state_hash()
        new_point = True
        if self.curr['x'] is not None and state == self.curr.get('state_hash'):
            if feval and self.curr['f'] is None:
                new_point = False
            else:
                return False
        x = self.get_x()
        basis = self._calc_basis()

        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr['x'] = x
        self.curr['state_hash'] = state
        self.curr['f'] = f
        self.curr['g'] = g
        self._update_basis(basis)
        return True

    def _update_basis(self, basis=None):
        if basis is None:
            basis = self._calc_basis()
        drdx, Ucons, Unred, Ufree = basis
        self.curr['drdx'] = drdx
        self.curr['Ucons'] = Ucons
        self.curr['Unred'] = Unred
        self.curr['Ufree'] = Ufree

        if self.curr['g'] is None:
            L = None
        else:
            L = np.linalg.lstsq(drdx.T, self.curr['g'], rcond=None)[0]

        self.curr['L'] = L

    def _update_H(self, dx, dg):
        if self.last['x'] is None or self.last['g'] is None:
            return
        self.H.update(dx, dg)

    def get_f(self):
        self._update()
        return self.curr['f']

    def get_g(self) -> np.ndarray:
        self._update()
        return self.curr['g'].copy()

    def get_Unred(self):
        self._update(False)
        return self.curr['Unred']

    def get_Ufree(self):
        self._update(False)
        return self.curr['Ufree']

    def get_Ucons(self):
        self._update(False)
        return self.curr['Ucons']

    def diag(self, gamma=0.1, threepoint=False, maxiter=None):
        if self.curr['f'] is None:
            self._update(feval=True)

        Ufree = self.get_Ufree()
        nfree = Ufree.shape[1]

        # If there are no free DOF, there's nothing to diagonalize
        if nfree == 0:
            return

        P = self.get_HL_projected(Ufree)
        P_is_none = P.B is None

        # Determine initial guess vector
        if P_is_none or self.first_diag:
            v0 = self.v0 if self.v0 is not None else self.get_g() @ Ufree
            # If v0 is near-zero, let rayleigh_ritz choose its own initial guess
            if v0 is not None and np.linalg.norm(v0) < 1e-12:
                v0 = None
        else:
            v0 = None

        # Convert P to array
        P = np.eye(nfree) if P_is_none else P.asarray()

        Hproj = NumericalHessian(self._calc_eg, self.get_x(), self.get_g(),
                                 self.eta, threepoint, Ufree)
        Hc = self.get_Hc()
        rayleigh_ritz(Hproj - Ufree.T @ Hc @ Ufree, gamma, P, v0=v0,
                      method=self.eigensolver,
                      maxiter=maxiter)

        # Extract eigensolver iterates
        Vs = Hproj.Vs
        AVs = Hproj.AVs

        # Re-calculate Ritz vectors
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2) - Vs.T @ Hc @ Vs
        _, X = eigh(Atilde)

        # Rotate Vs and AVs into X
        Vs = Vs @ X
        AVs = AVs @ X

        # Update the approximate Hessian
        self.H.update(Vs, AVs)

        self.first_diag = False

    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        return -(Ufree @ (Ufree.T @ g)).reshape((-1, 3))

    def converged(self, fmax, cmax=1e-5):
        fmax1 = np.linalg.norm(self.get_projected_forces(), axis=1).max()
        cmax1 = np.linalg.norm(self.get_res())
        conv = (fmax1 < fmax) and (cmax1 < cmax)
        return conv, fmax1, cmax1

    def wrap_dx(self, dx):
        return dx

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        return g.T @ dx + (dx.T @ H @ dx) / 2.

    def kick(self, dx, diag=False, **diag_kwargs):
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H.asarray()

        dx_initial, dx_final, g_par = self.set_x(x0 + dx)

        df_pred = self.get_df_pred(dx_initial, g0, B0)
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        if df_pred is None or abs(df_pred) < 1e-14:
            ratio = None
        else:
            ratio = df_actual / df_pred

        self._update_H(dx_final, dg_actual)

        if diag:
            if self.hessian_function is not None:
                self.calculate_hessian()
            else:
                self.diag(**diag_kwargs)

        return ratio

    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(self.hessian_function(self.atoms))


class InternalPES(PES):
    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        H0: np.ndarray = None,
        iterative_stepper: int = 0,
        auto_find_internals: bool = True,
        **kwargs
    ):
        self.int_orig = internals
        new_int = internals.copy()
        if auto_find_internals:
            new_int.find_all_bonds()
            new_int.find_all_angles()
            new_int.find_all_dihedrals()
        new_int.validate_basis()

        PES.__init__(
            self,
            atoms,
            *args,
            constraints=new_int.cons,
            H0=None,
            proj_trans=False,
            proj_rot=False,
            **kwargs
        )

        self.int = new_int
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof
        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.jacobian()
            Q, _ = qr(B, mode='economic')
            P = Q @ Q.T
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper

        self._pinv_cache = _LRU2()
        self._qr_cache = _LRU2()
        self._Hc_cache = _LRU2()

    dpos = property(lambda self: self.dummies.positions.copy())

    def _state_hash(self) -> bytes:
        h = super()._state_hash()
        h += self.dummies.positions.tobytes()
        return h

    # =========================================================================
    # Cache optimization: Store and reuse Jacobian QR and pseudo-inverse
    # =========================================================================

    def _get_jacobian_qr(self):
        """Get cached economy QR of internal Jacobian.

        Returns (Q, R) from np.linalg.qr(B, mode='reduced').
        Q (m, n) is the orthonormal basis for range(B) (= Unred).
        R (n, n) upper triangular, with R^{-1} replacing V S^{-1} from SVD.

        Shared between _get_Binv and _calc_basis so the Jacobian is
        decomposed at most once per geometry.  ~2x faster than SVD.
        Falls back to SVD if the Jacobian is rank-deficient.
        """
        state_hash = self._state_hash()
        cached = self._qr_cache.get(state_hash)
        if cached is not None:
            return cached

        B = self.int.jacobian()
        Q, R = _gpu_qr(B)

        # Check for rank deficiency via R diagonal
        rdiag = np.abs(np.diag(R))
        if len(rdiag) > 0 and rdiag.min() < 1e-6 * rdiag.max():
            # Rank-deficient: fall back to SVD for safe truncation
            Ui, Si, VTi = np.linalg.svd(B, full_matrices=False)
            nnred = np.sum(Si > 1e-6)
            Q = Ui[:, :nnred]
            R = np.diag(Si[:nnred]) @ VTi[:nnred]

            # Pre-compute Binv from SVD factors and cache it, since the
            # non-square R can't be used with solve_triangular in _get_Binv
            Siinv = np.diag(1.0 / Si[:nnred])
            Binv = VTi[:nnred].T @ Siinv @ Ui[:, :nnred].T
            self._pinv_cache.put(state_hash, Binv)

        self._qr_cache.put(state_hash, (Q, R))
        return Q, R

    def _get_Binv(self):
        """Get cached pseudo-inverse of internal Jacobian.

        Computes Binv = R^{-1} Q^T from the shared QR cache,
        using a triangular solve instead of a full SVD.
        """
        state_hash = self._state_hash()
        cached = self._pinv_cache.get(state_hash)
        if cached is not None:
            return cached

        Q, R = self._get_jacobian_qr()
        if R.size == 0:
            ncart = 3 * len(self.atoms) + (3 * len(self.dummies) if self.dummies else 0)
            Binv = np.empty((ncart, 0))
        elif R.shape[0] == R.shape[1]:
            Binv = solve_triangular(R, Q.T, check_finite=False)
        else:
            # Non-square R from rank-deficient SVD fallback — Binv should
            # already have been cached by _get_jacobian_qr, but recompute
            # as a safety net (e.g., if the 2-entry cache evicted it).
            B = self.int.jacobian()
            Binv = np.linalg.pinv(B)

        self._pinv_cache.put(state_hash, Binv)
        return Binv

    # =========================================================================
    # Iterative stepper with improved convergence checking
    # =========================================================================
    # Uses Newton-Raphson iteration with robust convergence detection:
    # - Strict absolute tolerance (1e-8) for convergence
    # - Divergence detection (2x initial error)
    # - Stagnation detection (3 consecutive iterations without progress)
    # - Final verification pass before accepting solution
    # Falls back to ODE integrator on failure.
    # =========================================================================

    def _set_x_iterative(self, target, max_iter=20):
        """Fast iterative stepper for internal coordinate updates.

        Uses Newton-Raphson iteration to update Cartesian positions to match
        target internal coordinates. Returns None if convergence fails.
        """
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        x0 = self.get_x()
        dx_initial = target - x0

        # Get initial gradient in Cartesian space
        g0 = self._get_Binv() @ self.curr.get('g', np.zeros_like(dx_initial))

        rms_prev = np.inf
        initial_rms = None
        pos_first = None
        dpos_first = None
        stagnation_count = 0

        for iteration in range(max_iter):
            residual = self.wrap_dx(target - self.get_x())
            rms = np.linalg.norm(residual) / np.sqrt(len(residual))

            if initial_rms is None:
                initial_rms = rms

            # Converged
            if rms < 1e-8:
                break

            # Check for divergence (getting significantly worse)
            if rms > initial_rms * 2.0:
                # Diverging, restore and fall back
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                return None

            # Check for stagnation (after first few iterations)
            if iteration > 3:
                if rms > rms_prev * 0.95:
                    stagnation_count += 1
                    if stagnation_count >= 3:
                        # Stagnating, give up if we haven't made progress
                        if rms > initial_rms * 0.5:
                            self.atoms.positions = pos0
                            self.dummies.positions = dpos0
                            return None
                        break  # Accept partial convergence
                else:
                    stagnation_count = 0

            rms_prev = rms

            # Newton step
            dx = np.linalg.lstsq(
                self.int.jacobian(),
                residual,
                rcond=None,
            )[0].reshape((-1, 3))

            # Update positions
            self.atoms.positions += dx[:len(self.atoms)]
            self.dummies.positions += dx[len(self.atoms):]

            # Save first iteration result as fallback
            if pos_first is None:
                pos_first = self.atoms.positions.copy()
                dpos_first = self.dummies.positions.copy()

            # Check for bad internals during iteration
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                # Restore and return None to trigger ODE fallback
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                self.bad_int = None
                return None

        # After loop, verify we actually converged well enough
        final_residual = self.wrap_dx(target - self.get_x())
        final_rms = np.linalg.norm(final_residual) / np.sqrt(len(dx_initial))
        if final_rms > 1e-6:
            # Didn't converge well enough, fall back to ODE
            self.atoms.positions = pos0
            self.dummies.positions = dpos0
            return None

        dx_final = self.get_x() - x0
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    def _set_x_ode(self, target):
        """ODE-based stepper for internal coordinate updates.

        Uses LSODA to integrate the geodesic equation for reliable convergence
        on large or ill-conditioned steps.
        """
        dx = target - self.get_x()
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ self.curr.get('g', np.zeros_like(dx))))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
                view(self.atoms + self.dummies)
                raise RuntimeError("Geometry update ODE is taking too long "
                                   "to converge!")

        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")

        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    # Position getter/setter
    def set_x(self, target):
        """Update internal coordinates to target values.

        Uses fast iterative stepper by default, with ODE fallback for robustness.
        """
        if self.iterative_stepper:
            res = self._set_x_iterative(target)
            if res is not None:
                q_after_ode = self.int.calc().copy()
                proj_moved = self._project_to_constraints()
                dx_initial, dx_final_ode, g_final = res
                dx_final = self._add_proj_delta(dx_final_ode, q_after_ode,
                                                proj_moved)
                return dx_initial, dx_final, g_final
        # Fall back to ODE solver
        res = self._set_x_ode(target)
        q_after_ode = self.int.calc().copy()
        proj_moved = self._project_to_constraints()
        dx_initial, dx_final_ode, g_final = res
        dx_final = self._add_proj_delta(dx_final_ode, q_after_ode, proj_moved)
        return dx_initial, dx_final, g_final

    def _add_proj_delta(self, dx_int_final, q_after_ode, proj_moved):
        """Combine ODE-tangent dx with the projection's IC delta.

        ``dx_int_final`` is the tangent-integrated displacement returned
        by the ODE/iterative stepper — what BFGS expects as the secant.
        If the projection then nudged atoms, that extra motion is
        captured as ``delta_proj = int.calc() - q_after_ode`` (raw IC
        difference, with dihedrals wrapped to (-π, π] for safety).
        BFGS sees the sum so its `s = dx` matches `dg = g_after - g_before`.
        """
        if not proj_moved:
            return dx_int_final
        delta_proj = self.int.calc() - q_after_ode
        dih_start = (self.int.ntrans + self.int.nbonds
                     + self.int.nangles)
        dih_end = dih_start + self.int.ndihedrals
        if dih_end > dih_start:
            delta_proj[dih_start:dih_end] = (
                (delta_proj[dih_start:dih_end] + np.pi)
                % (2 * np.pi) - np.pi
            )
        return dx_int_final + delta_proj

    def _project_to_constraints(self, target_tol=1e-7, max_iter=8,
                                safety_limit=0.05):
        """Newton projection onto the constraint manifold (IC null-space).

        Drives ``cons.residual()`` to zero with corrections that, to
        first order, do not change any *free* internal coordinate.
        This avoids the failure mode of a Cartesian min-norm projection,
        which would tilt the dummy atom in directions that couple back
        into the free improper-dihedral bending coordinate.

        Algorithm (one Newton iteration, repeated):

            r       = cons.residual()                          # (ncons,)
            drdx    = d(cons) / d(int_coords)                   # (ncons, n_int)
            Ucons   = IC-space basis spanned by constraints     # (n_int, ncons')
            s       = lstsq(drdx @ Ucons, -r)                   # min-norm in Ucons
            dq_int  = Ucons @ s                                 # IC-space step
            dx_cart = Binv @ dq_int                             # back to Cartesian

        Because ``dq_int`` lives entirely in ``Ucons`` (orthogonal to
        ``Ufree`` in the IC inner product), every free internal — including
        the improper dihedral that parametrizes a linear-bend — is
        unchanged to first order. The Cartesian step is the
        minimum-norm representative of that IC-space step (``Binv`` is
        the pseudoinverse), so real atoms only move when the
        constraint *requires* it (e.g. ``FixBondLengths``).

        ``safety_limit`` caps ``|dx_cart|_inf`` per iteration. If the
        Newton step would exceed it, we bail and accept the partial
        residual — the alternative (damped re-iteration) was tested
        and found to *increase* opt step counts (~+30%) on the tier1+2
        benchmarks because the partial corrections accumulate as
        Hessian noise. Bailing leaves the projection as a strict
        improvement: it can only help, never hurt.
        """
        if self.cons.residual().size == 0:
            return False

        n_real = 3 * len(self.atoms)
        n_dummy = 3 * len(self.dummies)
        moved = False

        for _ in range(max_iter):
            r = self.cons.residual()
            if np.linalg.norm(r, ord=np.inf) < target_tol:
                return moved

            # _compute_basis_int returns (drdx, Ucons, Unred, Ufree) for the
            # internal-only block (no cell DOF). drdx is ncons × n_int in
            # IC space; Ucons is n_int × ncons'.
            drdx, Ucons, _, _ = self._compute_basis_int()
            if Ucons.shape[1] == 0:
                return moved  # no constraint subspace — nothing to project

            s, *_ = np.linalg.lstsq(drdx @ Ucons, -r, rcond=None)
            dq_int = Ucons @ s                       # IC-space step (n_int,)
            dx = self._get_Binv() @ dq_int            # Cartesian (n_cart,)

            if np.linalg.norm(dx, ord=np.inf) > safety_limit:
                return moved  # would override optimizer's step — bail

            self.atoms.positions += dx[:n_real].reshape(-1, 3)
            if n_dummy > 0:
                self.dummies.positions += dx[n_real:n_real + n_dummy].reshape(-1, 3)
            moved = True

        return moved

    def get_x(self):
        x = self.int.calc()
        if self.curr['x'] is not None:
            dih_start = (self.int.ntrans + self.int.nbonds
                         + self.int.nangles)
            dih_end = dih_start + self.int.ndihedrals
            if dih_end > dih_start:
                dx = x[dih_start:dih_end] - self.curr['x'][dih_start:dih_end]
                x[dih_start:dih_end] = (
                    self.curr['x'][dih_start:dih_end]
                    + (dx + np.pi) % (2 * np.pi) - np.pi
                )
        return x

    # Hessian of the constraints
    def _compute_Hc_int(self):
        """Compute the internal-coords-only constraint Hessian (uncached)."""
        if self.curr['L'] is None:
            raise RuntimeError(
                "InternalPES.get_Hc() called with L=None. "
                f"curr_g_is_none={self.curr.get('g') is None}, "
                f"curr_f_is_none={self.curr.get('f') is None}."
            )

        # No constraints → L is empty → Hc is identically zero. Skip the
        # expensive ldot/matmul chain (~95ms at 400 atoms).
        Binv_int = self._get_Binv()
        n_dof = Binv_int.shape[1]
        if self.curr['L'].size == 0:
            return np.zeros((n_dof, n_dof))

        D_cons = self.cons.hessian().ldot(self.curr['L'])
        B_cons = self.cons.jacobian()
        L_int = self.curr['L'] @ B_cons @ Binv_int
        D_int = self.int.hessian().ldot(L_int)
        return Binv_int.T @ (D_cons - D_int) @ Binv_int

    def get_Hc(self):
        # Subclasses (CellInternalPES) cache the cell-extended form themselves
        # and call _compute_Hc_int directly, so we only cache here when this
        # *is* the runtime class.
        state_hash = self._state_hash()
        cached = self._Hc_cache.get(state_hash)
        if cached is not None:
            return cached

        Hc = self._compute_Hc_int()
        self._Hc_cache.put(state_hash, Hc)
        return Hc

    def get_drdx(self):
        # dr/dq = dr/dx dx/dq
        return PES.get_drdx(self) @ self._get_Binv()

    def _compute_basis_int(self):
        """Compute the internal-coords-only basis (uncached, fast path).

        Uses the cached jacobian QR factors. Subclasses (CellInternalPES) call
        this to obtain the internal block, then add their own cell extension
        and cache the combined result themselves.
        """
        cons = self.cons
        Q, R = self._get_jacobian_qr()
        Unred = Q

        n_int = Q.shape[0]
        cons_jac = cons.jacobian()
        if cons_jac.shape[0] == 0:
            # No constraints: all non-redundant DOF are free
            drdx = np.zeros((0, n_int))
            Ucons = np.zeros((n_int, 0))
            Ufree = Unred
        else:
            if R.shape[0] == R.shape[1]:
                # Full rank: cons_jac @ R^{-1} via triangular solve
                drdxnred = solve_triangular(
                    R.T, cons_jac.T, lower=True, check_finite=False
                ).T
            else:
                # Rank-deficient (SVD fallback in _get_jacobian_qr)
                Binv = self._get_Binv()
                drdxnred = cons_jac @ (Binv @ Q)
            drdx = drdxnred @ Q.T
            Vcons, Vfree = _split_cons_subspace(drdxnred)
            Ucons = Unred @ Vcons
            Ufree = Unred @ Vfree
        return drdx, Ucons, Unred, Ufree

    def _calc_basis(self, internal=None, cons=None):
        # If custom internal/cons provided, bypass cache (used by refine paths)
        if internal is not None or cons is not None:
            if internal is None:
                internal = self.int
            if cons is None:
                cons = self.cons
            B = internal.jacobian()
            Ui, Si, VTi = np.linalg.svd(B, full_matrices=False)
            nnred = np.sum(Si > 1e-6)
            Unred = Ui[:, :nnred]
            Vnred = VTi[:nnred].T
            Siinv = np.diag(1 / Si[:nnred])
            cons_jac = cons.jacobian()
            n_int = B.shape[0]
            if cons_jac.shape[0] == 0:
                # No constraints: all non-redundant DOF are free
                drdx = np.zeros((0, n_int))
                Ucons = np.zeros((n_int, 0))
                Ufree = Unred
            else:
                drdxnred = cons_jac @ Vnred @ Siinv
                drdx = drdxnred @ Unred.T
                Vcons, Vfree = _split_cons_subspace(drdxnred)
                Ucons = Unred @ Vcons
                Ufree = Unred @ Vfree
            return drdx, Ucons, Unred, Ufree

        # Subclasses (CellInternalPES) cache the cell-extended form themselves
        # and call _compute_basis_int directly, so we only cache here when
        # this *is* the runtime class.
        state_hash = self._state_hash()
        cached = self._basis_cache.get(state_hash)
        if cached is not None:
            return cached

        result = self._compute_basis_int()
        self._basis_cache.put(state_hash, result)
        return result

    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = self._get_Binv()
        return f, g_cart @ Binv[:len(g_cart)]

    def update_internals(self, dx):
        self._update(True)

        nold = 3 * (len(self.atoms) + len(self.dummies))

        # Find new internals, constraints, and dummies
        new_int = self.int_orig.copy()
        new_int.find_all_bonds()
        new_int.find_all_angles()
        new_int.find_all_dihedrals()
        new_int.validate_basis()
        new_cons = new_int.cons

        # Calculate B matrix and its inverse for new and old internals
        Blast = self.int.jacobian()
        B = new_int.jacobian()
        Binv = np.linalg.pinv(B)
        Dlast = self.int.hessian()
        D = new_int.hessian()

        # # Projection matrices
        # P2 = B[:, nold:] @ Binv[nold:, :]

        # Update the info in self.curr
        x = new_int.calc()
        g = -self.atoms.get_forces().ravel() @ Binv[:3*len(self.atoms)]
        drdx, Ucons, Unred, Ufree = self._calc_basis(
            internal=new_int,
            cons=new_cons,
        )
        L = np.linalg.lstsq(drdx.T, g, rcond=None)[0]

        # Update H using old data where possible. For new (dummy) atoms,
        # use the guess hessian info.
        H = self.get_H().asarray()
        Hcart = Blast.T @ H @ Blast
        Hcart += Dlast.ldot(self.curr['g'])
        Hnew = Binv.T[:, :nold] @ (Hcart - D.ldot(g)) @ Binv
        self.dim = len(x)
        self.set_H(Hnew)

        self.int = new_int
        self.cons = new_cons

        self.curr.update(x=x, g=g, drdx=drdx, Ufree=Ufree,
                         Unred=Unred, Ucons=Ucons, L=L, B=B, Binv=Binv)

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.

    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        # Use cached jacobian from curr if available
        if 'B' in self.curr and self.curr['B'] is not None:
            B = self.curr['B']
        else:
            B = self.int.jacobian()
        return -(Ufree @ (Ufree.T @ g) @ B).reshape((-1, 3))

    def wrap_dx(self, dx):
        return self.int.wrap(dx)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        # Use direct HVP computation instead of forming full Hessians.
        # Batch the two D_rdot @ vector products into one (D_rdot @ matrix)
        # matmul, then one Binv @ matrix matmul, halving the matmul count.
        D_rdot = self.int.hessian_rdot(dxdt)
        Binv = self._ode_Binv
        rhs = np.column_stack((dxdt, g))     # (ndof, 2)
        out = -Binv @ (D_rdot @ rhs)          # (ndof, 2)
        dydt[1] = out[:, 0]
        dydt[2] = out[:, 1]

        return dydt.ravel()

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        return ratio

    def write_traj(self):
        if self.traj is not None:
            energy = self.atoms.calc.results['energy']
            forces = np.zeros((len(self.atoms) + len(self.dummies), 3))
            forces[:len(self.atoms)] = self.atoms.calc.results['forces']
            atoms_tmp = self.atoms + self.dummies
            atoms_tmp.calc = SinglePointCalculator(atoms_tmp, energy=energy,
                                                   forces=forces)
            self.traj.write(atoms_tmp)

    def _update(self, feval=True):
        if not PES._update(self, feval=feval):
            return

        B = self.int.jacobian()
        Binv = self._get_Binv()  # Use cached version instead of recomputing
        self.curr.update(B=B, Binv=Binv)
        return True

    def _convert_cartesian_hessian_to_internal(
        self,
        Hcart: np.ndarray,
    ) -> np.ndarray:
        ncart = 3 * len(self.atoms)
        # Get Jacobian and calculate redundant and non-redundant spaces
        B = self.int.jacobian()[:, :ncart]
        Ui, Si, VTi = np.linalg.svd(B, full_matrices=True)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Ured = Ui[:, nnred:]

        # Calculate inverse Jacobian in non-redundant space
        Bnred_inv = VTi[:nnred].T @ np.diag(1 / Si[:nnred])

        # Convert Cartesian Hessian to non-redundant internal Hessian
        Hcart_coupled = self.int.hessian().ldot(self.get_g())[:ncart, :ncart]
        Hcart_corr = Hcart - Hcart_coupled
        Hnred = Bnred_inv.T @ Hcart_corr @ Bnred_inv

        # Find eigenvalues of non-redundant internal Hessian
        lnred, _ = np.linalg.eigh(Hnred)

        # The redundant part of the Hessian will be initialized to the
        # geometric mean of the non-redundant eigenvalues
        lnred_mean = np.exp(np.log(np.abs(lnred)).mean())

        # finish reconstructing redundant internal Hessian
        return Unred @ Hnred @ Unred.T + lnred_mean * Ured @ Ured.T

    def _convert_internal_hessian_to_cartesian(
        self,
        Hint: np.ndarray,
    ) -> np.ndarray:
        B = self.int.jacobian()
        return B.T @ Hint @ B + self.int.hessian().ldot(self.get_g())

    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(self._convert_cartesian_hessian_to_internal(
            self.hessian_function(self.atoms)
        ))


# =============================================================================
# Utility functions for cell optimization
# =============================================================================

def voigt_6_to_full_3x3_stress(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert 6-component Voigt stress to full 3x3 stress tensor.

    ASE uses the convention: [xx, yy, zz, yz, xz, xy]
    """
    xx, yy, zz, yz, xz, xy = stress_voigt
    return np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz]
    ])


def full_3x3_to_voigt_6_stress(stress_3x3: np.ndarray) -> np.ndarray:
    """Convert 3x3 stress tensor to 6-component Voigt notation."""
    return np.array([
        stress_3x3[0, 0],  # xx
        stress_3x3[1, 1],  # yy
        stress_3x3[2, 2],  # zz
        stress_3x3[1, 2],  # yz
        stress_3x3[0, 2],  # xz
        stress_3x3[0, 1],  # xy
    ])


