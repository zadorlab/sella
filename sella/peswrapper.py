import logging
from typing import Union, Callable

import numpy as np
from scipy.linalg import eigh, expm, expm_frechet, logm, polar, qr, solve_triangular
from scipy.integrate import RK45 as _GeodesicSolver, BDF as _GeodesicStiffSolver
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

        Integrates the geodesic equation for reliable convergence on large or
        ill-conditioned steps; see :meth:`_run_geodesic` for the solver (RK45
        with a BDF stiff fallback) and why LSODA is avoided.
        """
        dx = target - self.get_x()
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ self.curr.get('g', np.zeros_like(dx))))
        y, t0 = self._run_geodesic(y0, t0, debug_view=True)

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

    def _run_geodesic(self, y0, t0, debug_view=False):
        """Integrate the geodesic ODE from ``(t0, y0)`` to ``t_bound=1``.

        Returns ``(y, t)`` at the final integration point.

        Solver choice and the memory leak it avoids: scipy's ``LSODA`` wraps
        the Fortran lsoda integrator, which leaks its native ``rwork``/``iwork``
        workspace on *every* instantiation (and reset) — memory the Python GC
        cannot reclaim. Because this stepper builds a fresh integrator on every
        optimizer step, that leak grows to many GB and OOMs over a few hundred
        geometry optimizations. The geodesic RHS is smooth and, in practice,
        non-stiff, so we integrate with the pure-Python explicit ``RK45``
        (``_GeodesicSolver``): same ``OdeSolver`` API, no native workspace, and
        empirically faster than LSODA on real (incl. saddle-point) searches.

        As insurance for a genuinely stiff / ill-conditioned step, if RK45
        stalls (``nfev`` exceeds the budget) or fails, we retry from the same
        initial state with ``BDF`` (``_GeodesicStiffSolver``) — a pure-Python
        *implicit* solver that handles stiffness without LSODA's native leak
        (its only retained state is a reference cycle, which cyclic GC
        reclaims). Both restart cleanly from ``y0`` because ``_q_ode`` resets
        the atom positions from ``y`` on its first call.

        Sets ``self.bad_int`` and returns the partial ``(y, t)`` if
        ``check_for_bad_internals`` trips mid-integration — a legitimate early
        stop, not a failure.
        """
        for Solver, is_stiff_fallback in (
            (_GeodesicSolver, False), (_GeodesicStiffSolver, True),
        ):
            ode = Solver(self._q_ode, t0, y0, t_bound=1., atol=1e-6)
            stalled = False
            while ode.status == 'running':
                ode.step()
                self.bad_int = self.int.check_for_bad_internals()
                if self.bad_int is not None:
                    return ode.y, ode.t
                if ode.nfev > 1000:
                    stalled = True
                    break
            if not stalled and ode.status != 'failed':
                return ode.y, ode.t
            if is_stiff_fallback:
                if debug_view:
                    view(self.atoms + self.dummies)
                raise RuntimeError(
                    "Geometry update ODE failed to converge with both RK45 "
                    f"and BDF (status={ode.status}, nfev={ode.nfev})."
                )
            # else: RK45 stalled/failed -> fall through to the BDF retry

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


class CellInternalPES(InternalPES):
    """Internal coordinate PES with unit cell optimization.

    This class extends InternalPES to simultaneously optimize both internal
    coordinates (bonds, angles, dihedrals) and the unit cell parameters.

    The cell is parameterized using the log of the deformation gradient:
        F = cell @ inv(orig_cell)
        cell_params = _logm_3x3(F) * exp_cell_factor

    This parameterization ensures that:
    1. The identity corresponds to zero cell parameters
    2. Small deformations are approximately linear in the parameters
    3. Large deformations are handled smoothly

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with periodic boundary conditions.
    internals : Internals
        Internal coordinate system definition.
    exp_cell_factor : float, optional
        Scaling factor for cell parameterization. Default is number of atoms.
    cell_mask : ndarray, optional
        Boolean mask of shape (3, 3) indicating which cell DOF are free.
        Default is all True (full cell optimization).
    scalar_pressure : float, optional
        External pressure in eV/Å³. Default is 0.
    rigid_fragments : bool, optional
        If True, cell changes translate fragment centers of mass to maintain
        fractional CoM positions while preserving intramolecular geometry.
        Auto-detected: defaults to True when internals have translations
        (allow_fragments=True), else False.
    refine_initial_hessian : bool, optional
        If True, compute cell-coordinate coupling and cell-cell Hessian blocks
        via finite differences. This requires additional force evaluations
        (2 * n_cell_dof) but can improve convergence for coupled systems.
        Default is False.
    hessian_delta : float, optional
        Finite difference step size for Hessian refinement. Default is 1e-5.
    """

    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        exp_cell_factor: float = None,
        cell_mask: np.ndarray = None,
        scalar_pressure: float = 0.0,
        rigid_fragments: bool = None,
        refine_initial_hessian: Union[bool, int] = False,
        hessian_delta: float = 1e-5,
        save_hessian: str = None,
        H0: np.ndarray = None,
        **kwargs
    ):
        """Initialize CellInternalPES.

        Parameters
        ----------
        rigid_fragments : bool, optional
            If True, cell changes translate fragment centers of mass to maintain
            fractional CoM positions while preserving intramolecular geometry.
            This zeroes out cell-intramolecular Hessian coupling while keeping
            physical cell-TRIC coupling. Auto-detected: defaults to True when
            internals have translations (allow_fragments=True), else False.
        refine_initial_hessian : bool or int
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - True or 1: Refine cell-related blocks only (2 * n_cell_dof evals)
            - 2: Also refine translation/rotation blocks (adds 2 * n_tric evals)
            - 3: Refine full internal Hessian (2 * n_internal evals, expensive!)
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        # Store original cell as reference before any optimization
        self.orig_cell = atoms.get_cell().array.copy()

        # Cell parameterization scaling (like ASE's FrechetCellFilter)
        if exp_cell_factor is None:
            exp_cell_factor = float(len(atoms))
        self.exp_cell_factor = exp_cell_factor

        # Cell mask: which of the 9 cell matrix elements are free
        if cell_mask is None:
            cell_mask = np.ones((3, 3), dtype=bool)
        self.cell_mask = np.asarray(cell_mask, dtype=bool).reshape((3, 3))
        self.n_cell_dof = int(self.cell_mask.sum())

        # External pressure
        self.scalar_pressure = scalar_pressure

        # Store rigid_fragments request; auto-detection deferred until after
        # parent init populates translations via find_all_bonds()
        self._rigid_fragments_request = rigid_fragments

        # Flag to control get_x behavior during parent initialization
        # When True, get_x returns only internal coords (for parent __init__)
        self._initializing = True
        self.n_internal = None  # Will be set by parent

        # Initialize parent class - this will set up internal coords
        # (including finding bonds/angles/translations if auto_find_internals)
        InternalPES.__init__(self, atoms, internals, *args, H0=H0, **kwargs)

        # Now parent is initialized. Store internal-only dimension.
        self.n_internal = self.dim  # Parent set dim to internal coords count

        # Rigid fragment mode: auto-detect from translations in internals
        # (must be after parent init, which calls find_all_bonds and adds translations)
        if self._rigid_fragments_request is None:
            self.rigid_fragments = bool(self.int.internals.get('translations', []))
        else:
            self.rigid_fragments = self._rigid_fragments_request

        if self.rigid_fragments:
            # Extract fragment atom groups from Translation coordinates
            self.fragment_groups, self.fragment_dummy_groups = \
                self._extract_fragment_groups(self.int)

        # Update dimension to include cell DOF
        self.dim = self.n_internal + self.n_cell_dof

        # Cache for the cell-extended constraint Hessian (separate from the
        # parent's internal-only _Hc_cache).
        self._Hc_cell_cache = _LRU2()

        # Cache for the cell-extended basis (parent's _basis_cache only covers
        # the internal-coords-only basis; CellInternalPES._calc_basis adds the
        # cell-DOF zero-padding which we cache here).
        self._cell_basis_cache = _LRU2()

        # Done initializing - now get_x returns full vector
        self._initializing = False

        # Create proper Hessian with correct dimensions
        # Use block-diagonal structure: internal Hessian + cell Hessian
        H_old = self.H.B if self.H is not None and self.H.B is not None else None

        # Pad internal Hessian and add cell block
        H0_full = np.zeros((self.dim, self.dim))
        if H_old is not None:
            H0_full[:self.n_internal, :self.n_internal] = H_old
        else:
            B = self.int.jacobian()
            Q, _ = qr(B, mode='economic')
            P = Q @ Q.T
            H_internal = P @ self.int.guess_hessian() @ P
            H0_full[:self.n_internal, :self.n_internal] = H_internal

        # Convert bool to int for refinement level
        if refine_initial_hessian is True:
            refine_level = 1
        elif refine_initial_hessian is False:
            refine_level = 0
        else:
            refine_level = int(refine_initial_hessian)

        if refine_level >= 1:
            # Level 1: Refine cell-related blocks
            H_cell_cols = self._compute_cell_hessian_columns(hessian_delta)
            # Set internal-cell coupling (and its transpose for symmetry)
            H0_full[:self.n_internal, self.n_internal:] = H_cell_cols[:self.n_internal, :]
            H0_full[self.n_internal:, :self.n_internal] = H_cell_cols[:self.n_internal, :].T
            # Set cell-cell block with explicit symmetrization
            H_cell_cell = H_cell_cols[self.n_internal:, :]
            H0_full[self.n_internal:, self.n_internal:] = (H_cell_cell + H_cell_cell.T) / 2

        if refine_level >= 2:
            # Level 2: Also refine translation and rotation blocks
            H_tric_cols = self._compute_tric_hessian_columns(hessian_delta)
            tric_indices = self._get_tric_indices()
            for i, idx in enumerate(tric_indices):
                H0_full[:, idx] = H_tric_cols[:, i]
                H0_full[idx, :] = H_tric_cols[:, i]

        if refine_level >= 3:
            # Level 3: Refine full internal Hessian (expensive!)
            H_int_cols = self._compute_internal_hessian_columns(hessian_delta)
            # Symmetrize and set the internal-internal block
            H0_full[:self.n_internal, :self.n_internal] = (H_int_cols + H_int_cols.T) / 2

        if refine_level == 0:
            # No refinement: use diagonal guess for cell block
            h0_cell = 1.0
            H0_full[self.n_internal:, self.n_internal:] = h0_cell * np.eye(self.n_cell_dof)

        # Save Hessian if requested
        if save_hessian is not None:
            np.save(save_hessian, H0_full)
            logger.info("Initial Hessian saved to %s", save_hessian)

        # With FD-refined Hessian (refine_level >= 1), use initialized=False
        # to preserve the refined cell block on the first BFGS update — the
        # uninitialized path only updates B[:ncart, :ncart] (internal block),
        # which is appropriate since the FD-refined cell block is better than
        # what one BFGS update would produce.
        # Without refinement, use initialized=True so the first BFGS update
        # covers all DOF including cell.
        self.set_H(H0_full, initialized=(refine_level == 0))

    def maybe_niggli_reduce(self, angle_threshold=30.0):
        """Apply Niggli reduction if cell angles deviate too far from 90 deg.

        When the unit cell becomes highly skewed during optimization, this
        remaps to the most compact (Niggli-reduced) cell and resets the
        log-deformation reference. The cell block of the Hessian is
        transformed to the new parameterization basis via the Jacobian of
        the log-deformation map.

        Parameters
        ----------
        angle_threshold : float
            Maximum deviation from 90 deg before triggering reduction.
            Default 30 means reduction triggers when any angle < 60 or > 120.

        Returns
        -------
        bool
            True if reduction was applied.
        """
        angles = self.atoms.get_cell().angles()
        max_deviation = max(abs(a - 90.0) for a in angles)
        if max_deviation <= angle_threshold:
            return False

        H = self.H.B.copy()
        n = self.n_internal
        T_masked = _niggli_hessian_transform(
            self.atoms, self.orig_cell, self.exp_cell_factor, self.cell_mask
        )

        # Transform cell-cell block: H_new = T^T @ H_old @ T
        H_cell_new = T_masked.T @ H[n:, n:] @ T_masked
        H[n:, n:] = H_cell_new

        # Transform coupling blocks
        H[:n, n:] = H[:n, n:] @ T_masked
        H[n:, :n] = T_masked.T @ H[n:, :n]

        self.orig_cell = self.atoms.get_cell().array.copy()
        self.set_H(H, initialized=True)

        # Reset cached state so next evaluation recomputes everything
        self.curr = dict(x=None, f=None, g=None)
        self.last = self.curr.copy()

        return True

    def save(self):
        """Save current state including cell."""
        InternalPES.save(self)
        self.savepoint['cell'] = self.atoms.get_cell().array.copy()

    def restore(self):
        """Restore saved state including cell."""
        InternalPES.restore(self)
        if 'cell' in self.savepoint:
            self.atoms.set_cell(self.savepoint['cell'], scale_atoms=False)

    def refine_hessian(self, refine_level: int = 1, delta: float = 1e-5):
        """Re-refine Hessian blocks via finite differences during optimization.

        This can help recover from accumulated bad curvature in the Hessian
        that develops during BFGS updates.

        Parameters
        ----------
        refine_level : int
            Level of refinement (1=cell, 2=cell+TRIC, 3=full internal).
        delta : float
            Finite difference step size.
        """
        if refine_level < 1:
            return

        # Get current Hessian
        H = self.H.asarray()

        if refine_level >= 1:
            # Level 1: Refine cell-related blocks
            H_cell_cols = self._compute_cell_hessian_columns(delta)
            # Set internal-cell coupling (and its transpose for symmetry)
            H[:self.n_internal, self.n_internal:] = H_cell_cols[:self.n_internal, :]
            H[self.n_internal:, :self.n_internal] = H_cell_cols[:self.n_internal, :].T
            # Set cell-cell block with explicit symmetrization
            H_cell_cell = H_cell_cols[self.n_internal:, :]
            H[self.n_internal:, self.n_internal:] = (H_cell_cell + H_cell_cell.T) / 2

        if refine_level >= 2:
            # Level 2: Also refine translation and rotation blocks
            H_tric_cols = self._compute_tric_hessian_columns(delta)
            tric_indices = self._get_tric_indices()
            for i, idx in enumerate(tric_indices):
                H[:, idx] = H_tric_cols[:, i]
                H[idx, :] = H_tric_cols[:, i]

        if refine_level >= 3:
            # Level 3: Refine full internal Hessian (expensive!)
            H_int_cols = self._compute_internal_hessian_columns(delta)
            # Symmetrize and set the internal-internal block
            H[:self.n_internal, :self.n_internal] = (H_int_cols + H_int_cols.T) / 2

        # Update the Hessian (preserves eigenvalue tracking, etc.)
        self.set_H(H, initialized=True)
        logger.info("Hessian re-refined at level %d", refine_level)

    def _compute_cell_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for cell DOF via finite differences.

        This computes d(gradient)/d(cell_param) for all cell parameters,
        giving us both the internal-cell coupling block and the cell-cell block.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_cols : ndarray
            Array of shape (dim, n_cell_dof) containing Hessian columns.
        """
        H_cols = np.zeros((self.dim, self.n_cell_dof))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * self.n_cell_dof
        logger.info("Refining initial Hessian: 0/%d force calls", n_evals)

        for i in range(self.n_cell_dof):
            # Restore state before each FD probe to ensure path-independence
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)

            # Displace cell parameter +delta
            x_plus = x0.copy()
            x_plus[self.n_internal + i] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()
            logger.info("Refining initial Hessian: %d/%d force calls", 2*i + 1, n_evals)

            # Restore before -delta
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)

            # Displace cell parameter -delta
            x_minus = x0.copy()
            x_minus[self.n_internal + i] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()
            logger.info("Refining initial Hessian: %d/%d force calls", 2*i + 2, n_evals)

            # Central difference
            H_cols[:, i] = (g_plus - g_minus) / (2 * delta)


        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_cols

    def _get_tric_indices(self) -> np.ndarray:
        """Get indices of translation and rotation coordinates in internal space."""
        n_trans = len(self.int.internals['translations'])
        n_bonds = len(self.int.internals['bonds'])
        n_angles = len(self.int.internals['angles'])
        n_dihedrals = len(self.int.internals['dihedrals'])
        n_rot = len(self.int.internals['rotations'])

        # Internal coord order: translations, bonds, angles, dihedrals, other, rotations
        trans_indices = list(range(n_trans))
        rot_start = n_trans + n_bonds + n_angles + n_dihedrals + len(self.int.internals['other'])
        rot_indices = list(range(rot_start, rot_start + n_rot))

        return np.array(trans_indices + rot_indices)

    def _compute_tric_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for translation/rotation DOF via finite differences.

        This refines the coupling between TRICs and all other coordinates,
        which is important for molecular crystals where fragment motions are coupled.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_cols : ndarray
            Array of shape (dim, n_tric) containing Hessian columns.
        """
        tric_indices = self._get_tric_indices()
        n_tric = len(tric_indices)
        H_cols = np.zeros((self.dim, n_tric))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * n_tric
        logger.info("Refining TRIC Hessian: 0/%d force calls", n_evals)

        for i, idx in enumerate(tric_indices):
            # Displace TRIC parameter +delta
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)
            x_plus = x0.copy()
            x_plus[idx] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()
            logger.info("Refining TRIC Hessian: %d/%d force calls", 2*i + 1, n_evals)

            # Displace TRIC parameter -delta
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)
            x_minus = x0.copy()
            x_minus[idx] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()
            logger.info("Refining TRIC Hessian: %d/%d force calls", 2*i + 2, n_evals)

            # Central difference
            H_cols[:, i] = (g_plus - g_minus) / (2 * delta)


        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_cols

    def _compute_internal_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute full internal-internal Hessian block via finite differences.

        This is expensive: requires 2 * n_internal force evaluations.
        Only use when a highly accurate initial Hessian is needed.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_int : ndarray
            Array of shape (n_internal, n_internal) containing the internal Hessian.
        """
        H_int = np.zeros((self.n_internal, self.n_internal))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * self.n_internal
        logger.info("Refining internal Hessian: 0/%d force calls", n_evals)

        for i in range(self.n_internal):
            # Displace internal coordinate +delta
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)
            x_plus = x0.copy()
            x_plus[i] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()

            # Displace internal coordinate -delta
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)
            x_minus = x0.copy()
            x_minus[i] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()

            # Central difference - only internal part
            H_int[:, i] = (g_plus[:self.n_internal] - g_minus[:self.n_internal]) / (2 * delta)

            # Progress update every 10 columns or at the end
            if (i + 1) % 10 == 0 or i == self.n_internal - 1:
                logger.info("Refining internal Hessian: %d/%d force calls", 2*(i+1), n_evals)


        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_int

    def get_x(self) -> np.ndarray:
        """Return combined internal coordinates + cell parameters.

        During initialization (_initializing=True), returns only internal coords
        to be compatible with parent class initialization.
        """
        q = self.int.calc()  # Internal coordinates

        # During parent initialization, return only internal coords
        if self._initializing:
            return q

        cell_params = self._masked_cell_params()  # Cell DOF
        x = np.concatenate([q, cell_params])

        # Unwrap dihedrals to prevent ±π branch cut jumps
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

    def _get_deformation_gradient(self) -> np.ndarray:
        """Get current deformation gradient F = cell @ inv(orig_cell)."""
        return self.atoms.get_cell().array @ np.linalg.inv(self.orig_cell)

    def _get_log_deform(self) -> np.ndarray:
        """Get log of deformation gradient, scaled by exp_cell_factor."""
        F = self._get_deformation_gradient()
        return _logm_3x3(F) * self.exp_cell_factor

    def _set_cell_from_log_deform(self, log_deform_scaled: np.ndarray) -> None:
        """Set cell from scaled log-deformation gradient.

        When rigid_fragments is False, scales atomic positions with cell
        (fixed fractional coords), matching the virial stress definition.

        When rigid_fragments is True, keeps Cartesian positions fixed here.
        The rigid fragment CoM translation in set_x() then moves each fragment
        to maintain its fractional CoM position while preserving intramolecular
        geometry.
        """
        log_deform = log_deform_scaled / self.exp_cell_factor
        F = expm(log_deform.real)
        new_cell = F @ self.orig_cell
        self.atoms.set_cell(new_cell, scale_atoms=not self.rigid_fragments)

    def _masked_cell_params(self) -> np.ndarray:
        """Get cell parameters as flat array (only free DOF)."""
        log_deform = self._get_log_deform()
        return log_deform[self.cell_mask]

    def _set_masked_cell_params(self, params: np.ndarray) -> None:
        """Set cell from flat array of free DOF."""
        log_deform = self._get_log_deform()
        log_deform[self.cell_mask] = params
        self._set_cell_from_log_deform(log_deform)

    @staticmethod
    def _extract_fragment_groups(internals):
        """Extract fragment atom groups for rigid-body operations.

        Uses the original fragment groups stored during find_all_bonds,
        which contain all real atoms. Translation coordinates get
        corrupted by add_dummy_to_internals (drops the last real atom),
        so we avoid using those. Dummy indices are found via dinds.

        Returns
        -------
        list of ndarray
            Each element is an array of real atom indices for one fragment.
        list of ndarray
            Each element is an array of dummy atom indices for the same fragment.
        """
        if internals.fragment_atom_groups is not None:
            groups = internals.fragment_atom_groups
        else:
            natoms = internals.natoms
            groups = []
            for trans in internals.internals.get('translations', []):
                if trans.kwargs['dim'] == 0:
                    indices = np.array(trans.indices)
                    groups.append(indices[indices < natoms])

        dummy_groups = []
        for group in groups:
            dummies = []
            for atom_idx in group:
                didx = internals.dinds[atom_idx]
                if didx >= 0:
                    dummies.append(didx)
            dummy_groups.append(np.array(dummies, dtype=np.int32))

        return groups, dummy_groups

    def _compute_delta_r(self):
        """Compute positions relative to fragment center of mass.

        Returns Δr = r - r_CoM_expanded, where each atom's position is
        relative to its fragment's geometric center. Uses geometric center
        (unweighted mean) for consistency with how TRICs define translations.

        Returns
        -------
        delta_r : ndarray, shape (n_atoms, 3)
        """
        positions = self.atoms.get_positions()
        delta_r = positions.copy()
        for group in self.fragment_groups:
            if len(group) > 0:
                com = positions[group].mean(axis=0)
                delta_r[group] -= com
        return delta_r

    def set_x(self, target: np.ndarray):
        """Set internal coordinates and cell parameters.

        This is more complex than InternalPES.set_x because:
        1. We first update the cell (which changes internal coord values)
        2. Then apply only the internal coordinate *step* (dq) on top

        The cell change moves atoms according to the mode:
        - rigid_fragments=True: scale_atoms=False + CoM translation
          (preserves intramolecular geometry)
        - rigid_fragments=False: scale_atoms=True
          (atoms scale with cell, changing bonds/angles)

        In both cases, the internal coord solver only applies the dq
        displacement requested by the optimizer, NOT the full q_target.
        This ensures the gradient is consistent with the actual displacement.

        Returns
        -------
        dx_initial, dx_final, g_par : tuple of np.ndarray
            Displacement information for Hessian update.
        """
        x0 = self.get_x()
        dx_initial = target - x0

        # Split target into internal and cell parts
        # dq is the internal coordinate step the optimizer requested
        q0 = x0[:self.n_internal]
        dq = target[:self.n_internal] - q0
        cell_target = target[self.n_internal:]

        # Get initial cell params
        cell_params0 = self._masked_cell_params()

        # Save state before cell change for rigid fragment mode
        if self.rigid_fragments:
            pos_before = self.atoms.get_positions().copy()
            cell_before = self.atoms.get_cell().array.copy()

        # Update cell (scales atoms if rigid_fragments=False)
        self._set_masked_cell_params(cell_target)

        # Rigid fragment mode: translate fragment CoMs to maintain
        # fractional positions, and rotate fragments by R from polar
        # decomposition of the incremental deformation gradient.
        if self.rigid_fragments:
            cell_after = self.atoms.get_cell().array
            cell_before_inv = np.linalg.inv(cell_before)
            F_inc = cell_after @ cell_before_inv
            R_inc, _ = polar(F_inc)
            for group, dgroup in zip(self.fragment_groups,
                                     self.fragment_dummy_groups):
                com_old = pos_before[group].mean(axis=0)
                # Convert old CoM to fractional, then to new Cartesian
                com_frac = com_old @ cell_before_inv
                com_new = com_frac @ cell_after
                # Rotate relative positions by R (row-vector: r_new = r @ R^T)
                delta_r = pos_before[group] - com_old
                self.atoms.positions[group] = com_new + delta_r @ R_inc.T
                # Move dummy atoms with the same transformation
                if len(dgroup) > 0:
                    didx = dgroup - self.int.natoms  # Convert to dummies index
                    delta_d = self.dummies.positions[didx] - com_old
                    self.dummies.positions[didx] = com_new + delta_d @ R_inc.T

        # Read back internal coords AFTER the cell change moved atoms.
        # The solver targets q_after_cell + dq, not the raw q_target.
        # This ensures we don't undo the atom motion from the cell change.
        q_after_cell = self.int.calc()
        q_target = q_after_cell + dq

        # If there are no internal coordinates, we're done
        if self.n_internal == 0:
            # Cell-only case: dx_final equals the cell displacement
            dx_cell = cell_target - cell_params0
            dx_final = dx_cell.copy()
            # Return actual cell gradient at starting position for proper Hessian update
            # (dg_actual = get_g() - g_par needs g_par to be the old gradient)
            g_old = self.curr.get('g', None)
            if g_old is not None:
                g_final = g_old[-self.n_cell_dof:].copy()
            else:
                g_final = np.zeros(self.n_cell_dof)
            return dx_initial, dx_final, g_final

        # Now update atomic positions to match internal coordinate target
        res = self._set_x_ode_internal(q_target)

        # Project onto constraint manifold (decoupled from trust radius).
        # Only when ODE succeeded — the InternalPES.set_x fallback below
        # runs its own projection internally.
        proj_moved = False
        if res is not None:
            q_after_ode = self.int.calc().copy()
            proj_moved = self._project_to_constraints()

        # Get old cell gradient for parallel transport (needed for correct
        # BFGS secant condition on the cell block of the Hessian)
        g_old = self.curr.get('g', None)
        if g_old is not None:
            g_old_cell = g_old[self.n_internal:].copy()
        else:
            g_old_cell = np.zeros(self.n_cell_dof)

        if res is None:
            # Fallback: just do parent set_x ignoring cell
            dx_int, _, g_int = InternalPES.set_x(self, q_target)
            dx_final = np.concatenate([dx_int, cell_target - cell_params0])
            g_final = np.concatenate([g_int, g_old_cell])
        else:
            dx_int_initial, dx_int_final, g_int = res
            # Combine the ODE-tangent step with the projection's IC delta
            # so BFGS sees a coherent secant. When the projection didn't
            # fire this is a pure passthrough of dx_int_final.
            dx_int_realized = self._add_proj_delta(dx_int_final, q_after_ode,
                                                    proj_moved)
            dx_final = np.concatenate([dx_int_realized, cell_target - cell_params0])
            g_final = np.concatenate([g_int, g_old_cell])

        return dx_initial, dx_final, g_final

    def _set_x_ode_internal(self, q_target: np.ndarray, old_g_cart=None):
        """ODE-based stepper for internal coords only (cell already updated)."""
        x0 = self.int.calc()
        dx = q_target - x0
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv

        if 'g' in self.curr and self.curr['g'] is not None:
            g_cart_for_ode = Binv @ self.curr['g'][:self.n_internal]
        else:
            g_cart_for_ode = np.zeros(3 * (len(self.atoms) + len(self.dummies)))

        y0 = np.hstack((
            self.apos.ravel(),
            self.dpos.ravel(),
            Binv @ dx,
            g_cart_for_ode,
        ))
        # See _run_geodesic: RK45 (leak-free) with a BDF fallback for stiff
        # steps, replacing the Fortran LSODA that leaks native workspace.
        y, t0 = self._run_geodesic(y0, t0)

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

    def eval(self) -> tuple:
        """Evaluate energy and combined gradient (internal + cell)."""
        self.neval += 1
        f = self.atoms.get_potential_energy()

        # Add pressure contribution: H = E + P*V
        if self.scalar_pressure != 0.0:
            f += self.scalar_pressure * self.atoms.get_volume()

        # Atomic forces -> internal coordinate gradient
        forces = self.atoms.get_forces()
        g_cart = -forces.ravel()
        Binv = self._get_Binv()
        g_internal = g_cart @ Binv[:len(g_cart)]

        # Stress tensor -> cell gradient
        stress = self.atoms.get_stress()  # 6-component Voigt, eV/Å³
        g_cell = self._stress_to_cell_gradient(stress, forces=forces)

        self.write_traj()
        return f, np.concatenate([g_internal, g_cell])

    def _stress_to_cell_gradient(self, stress_voigt: np.ndarray, forces: np.ndarray = None) -> np.ndarray:
        """Convert stress tensor to gradient w.r.t. log-deformation cell parameters.

        Uses the Frechet derivative of the matrix exponential to correctly
        transform dE/dF into dE/dU = dE/d(log F).

        The virial stress V*σ relates to dE/dC via (ASE row-vector convention):
            V*σ = dE/dC^T @ C - f^T @ r

        For different atom-motion modes (where Δr = r - r_CoM_expanded):
            default:            dE/dC = C^{-T} @ V*σ                    @ C₀^T
            rigid_fragments:    dE/dC = C^{-T} @ (V*σ + Δr^T @ f)       @ C₀^T

        Parameters
        ----------
        stress_voigt : ndarray
            6-component Voigt stress tensor.
        forces : ndarray, optional
            Atomic forces, shape (n_atoms, 3). Required when rigid_fragments=True.
        """
        volume = self.atoms.get_volume()
        stress_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)

        # Add external pressure contribution
        if self.scalar_pressure != 0.0:
            stress_3x3 += self.scalar_pressure * np.eye(3)

        # The virial V*σ is the base term
        virial = volume * stress_3x3

        if self.rigid_fragments and forces is not None:
            # Rigid fragment mode: use Δr^T @ f correction
            # Δr = positions relative to fragment CoM
            delta_r = self._compute_delta_r()
            virial_corrected = virial + delta_r.T @ forces
        else:
            virial_corrected = virial

        # dE/dC = C^{-T} @ virial_corrected, then dE/dF = dE/dC @ C₀^T
        C = self.atoms.get_cell().array
        C_inv_T = np.linalg.inv(C.T)
        dEdF = C_inv_T @ virial_corrected @ self.orig_cell.T

        if self.rigid_fragments and forces is not None:
            # Rotation correction: fragments rotate by R from polar decomposition
            # of F, so dE/dF gets an additional term from ∂R/∂F.
            # rot_correction_mn = -Σ_{kl} (∂R_kl/∂F_mn) * [f^T @ Δr⁰]_kl
            # where Δr⁰ = Δr @ R (back-rotated to reference frame)
            F = self._get_deformation_gradient()
            R_polar, _ = polar(F)
            delta_r_ref = delta_r @ R_polar
            M = forces.T @ delta_r_ref

            eps = 1e-7
            rot_correction = np.zeros((3, 3))
            for m in range(3):
                for n in range(3):
                    F_pert = F.copy()
                    F_pert[m, n] += eps
                    R_pert, _ = polar(F_pert)
                    dR = (R_pert - R_polar) / eps
                    rot_correction[m, n] = -np.sum(dR * M)
            dEdF += rot_correction

        # Convert dE/dF to dE/dU via Frechet derivative of expm
        F = self._get_deformation_gradient()
        U = _logm_3x3(F)
        g_cell_3x3 = _expm_frechet_3x3_contracted(U, dEdF)

        # Apply cell mask and scale
        g_cell_3x3 = g_cell_3x3 * self.cell_mask
        g_cell_3x3 = g_cell_3x3 / self.exp_cell_factor

        return g_cell_3x3[self.cell_mask]

    def _calc_basis(self, internal=None, cons=None):
        """Calculate basis including cell DOF.

        The cell DOF are treated as unconstrained additional coordinates.
        """
        # Refine paths pass custom internal/cons; fall back to the parent's
        # full path (bypasses parent's cache too) and return without caching
        # on this side.
        if internal is not None or cons is not None:
            result = InternalPES._calc_basis(self, internal=internal, cons=cons)
            return self._extend_basis_with_cell(result)

        state_hash = self._state_hash()
        cached = self._cell_basis_cache.get(state_hash)
        if cached is not None:
            return cached

        # Compute the internal-only basis directly (bypass parent's cache —
        # we cache the cell-extended form here instead, so the parent cache
        # would just hold a redundant unpadded copy).
        result = self._compute_basis_int()
        out = self._extend_basis_with_cell(result)
        self._cell_basis_cache.put(state_hash, out)
        return out

    def _extend_basis_with_cell(self, basis_int):
        """Pad an internal-only basis with cell DOF (identity in Unred/Ufree)."""
        drdx_int, Ucons_int, Unred_int, Ufree_int = basis_int
        n_int = drdx_int.shape[1]
        n_total = n_int + self.n_cell_dof

        # Cell DOF are not constrained, so they're all in Ufree
        drdx = np.zeros((drdx_int.shape[0], n_total))
        drdx[:, :n_int] = drdx_int

        Ucons = np.zeros((n_total, Ucons_int.shape[1]))
        Ucons[:n_int, :] = Ucons_int

        Unred = self._pad_with_cell_identity(Unred_int)

        # When there are no constraints, _compute_basis_int returns
        # ``Ufree is Unred`` (same object). Share the extended array too —
        # avoids a second ~3 MB slice copy on every call.
        if Ufree_int is Unred_int:
            Ufree = Unred
        else:
            Ufree = self._pad_with_cell_identity(Ufree_int)

        return drdx, Ucons, Unred, Ufree

    def _pad_with_cell_identity(self, M_int):
        """Build (n_int + n_cell, M_int.shape[1] + n_cell) with M_int and an
        identity block in the cell-DOF corner. Uses np.empty and explicit
        zero strips to skip the bulk np.zeros memset for the n_int block
        that's about to be overwritten anyway."""
        n_int, n_cols = M_int.shape
        n_cell = self.n_cell_dof
        out = np.empty((n_int + n_cell, n_cols + n_cell))
        out[:n_int, :n_cols] = M_int
        out[:n_int, n_cols:] = 0
        out[n_int:, :n_cols] = 0
        out[n_int:, n_cols:] = np.eye(n_cell)
        return out

    def converged(self, fmax: float, smax: float = None, cmax: float = 1e-5):
        """Check convergence of forces and stress.

        Parameters
        ----------
        fmax : float
            Maximum force tolerance (eV/Å).
        smax : float, optional
            Maximum stress tolerance. If None, uses fmax.
        cmax : float, optional
            Constraint residual tolerance.

        Returns
        -------
        conv : bool
            True if converged.
        fmax_actual : float
            Maximum force.
        cmax_actual : float
            Constraint residual norm.
        smax_actual : float
            Maximum stress gradient.
        """
        if smax is None:
            smax = fmax

        # Force convergence (project out constraints).
        # Associate as U @ (U.T @ g) — two cheap matvecs — instead of
        # (U @ U.T) @ g which materializes a (n_int, n_int) intermediate.
        g = self.get_g()
        g_internal = g[:self.n_internal]
        Ufree_int = self.curr['Ufree'][:self.n_internal, :self.curr['Ufree'].shape[1] - self.n_cell_dof]
        g_proj = Ufree_int @ (Ufree_int.T @ g_internal)

        # Convert to Cartesian for force norm
        B = self.int.jacobian()
        g_cart = (g_proj @ B).reshape((-1, 3))
        fmax_actual = np.linalg.norm(g_cart, axis=1).max()

        # Stress convergence
        g_cell = g[self.n_internal:]
        smax_actual = np.abs(g_cell).max() if len(g_cell) > 0 else 0.0

        # Constraint residual
        cmax_actual = np.linalg.norm(self.get_res())

        conv = (fmax_actual < fmax) and (smax_actual < smax) and (cmax_actual < cmax)
        return conv, fmax_actual, cmax_actual, smax_actual

    def get_projected_forces(self) -> np.ndarray:
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_internal = g[:self.n_internal]
        Ufree = self.get_Ufree()
        Ufree_int = Ufree[:self.n_internal, :]
        B = self.int.jacobian()
        return -(Ufree_int @ (Ufree_int.T @ g_internal) @ B).reshape((-1, 3))

    def get_drdx(self):
        """Get constraint Jacobian extended for cell DOF.

        The constraint Jacobian from the parent class only has columns for
        internal coordinates. We extend it with zero columns for cell DOF
        since there are no constraints on the cell.
        """
        # Get internal constraint Jacobian from parent
        drdx_int = InternalPES.get_drdx(self)

        # Extend with zeros for cell DOF
        n_cons = drdx_int.shape[0]
        drdx = np.zeros((n_cons, self.dim))
        drdx[:, :self.n_internal] = drdx_int

        return drdx

    def get_Hc(self):
        """Get constraint Hessian extended for cell DOF.

        The constraint Hessian from InternalPES has shape (n_internal, n_internal).
        We extend it with zeros to (dim, dim) since there are no constraints
        on cell DOF.
        """
        state_hash = self._state_hash()
        cached = self._Hc_cell_cache.get(state_hash)
        if cached is not None:
            return cached

        # Compute the internal-only Hc directly (bypass parent's _Hc_cache —
        # we cache the cell-extended form here instead, so the parent cache
        # would just hold a redundant unpadded copy).
        Hc_int = self._compute_Hc_int()

        # Extend to full dimension
        Hc = np.zeros((self.dim, self.dim))
        n_int = self.n_internal
        if Hc_int.size > 0:
            Hc[:n_int, :n_int] = Hc_int

        self._Hc_cell_cache.put(state_hash, Hc)
        return Hc


class CellCartesianPES(PES):
    """Cartesian PES with unit cell optimization.

    This class extends PES to simultaneously optimize both atomic Cartesian
    positions and the unit cell parameters.

    The cell is parameterized using the log of the deformation gradient:
        F = cell @ inv(orig_cell)
        cell_params = _logm_3x3(F) * exp_cell_factor

    This parameterization ensures that:
    1. The identity corresponds to zero cell parameters
    2. Small deformations are approximately linear in the parameters
    3. Large deformations are handled smoothly

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with periodic boundary conditions.
    exp_cell_factor : float, optional
        Scaling factor for cell parameterization. Default is number of atoms.
    cell_mask : ndarray, optional
        Boolean mask of shape (3, 3) indicating which cell DOF are free.
        Default is all True (full cell optimization).
    scalar_pressure : float, optional
        External pressure in eV/Å³. Default is 0.
    refine_initial_hessian : bool or int, optional
        Level of Hessian refinement via finite differences:
        - False or 0: No refinement (default)
        - True or 1: Refine cell-related blocks only (2 * n_cell_dof force calls)
        Note: Level 2 (TRICs) is not applicable for Cartesian coordinates.
    hessian_delta : float, optional
        Finite difference step size for Hessian refinement. Default is 1e-5.
    save_hessian : str, optional
        Path to save the initial Hessian as .npy file for analysis.
    """

    def __init__(
        self,
        atoms: Atoms,
        *args,
        exp_cell_factor: float = None,
        cell_mask: np.ndarray = None,
        scalar_pressure: float = 0.0,
        refine_initial_hessian: Union[bool, int] = False,
        hessian_delta: float = 1e-5,
        save_hessian: str = None,
        H0: np.ndarray = None,
        **kwargs
    ):
        """Initialize CellCartesianPES.

        Parameters
        ----------
        refine_initial_hessian : bool or int
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - True or 1: Refine cell-related blocks only
            Note: Level 2 (TRICs) is not applicable for Cartesian coordinates.
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        # Store original cell as reference before any optimization
        self.orig_cell = atoms.get_cell().array.copy()

        # Cell parameterization scaling (like ASE's FrechetCellFilter)
        if exp_cell_factor is None:
            exp_cell_factor = float(len(atoms))
        self.exp_cell_factor = exp_cell_factor

        # Cell mask: which of the 9 cell matrix elements are free
        if cell_mask is None:
            cell_mask = np.ones((3, 3), dtype=bool)
        self.cell_mask = np.asarray(cell_mask, dtype=bool).reshape((3, 3))
        self.n_cell_dof = int(self.cell_mask.sum())

        # External pressure
        self.scalar_pressure = scalar_pressure

        # Flag to control get_x behavior during parent initialization
        self._initializing = True

        # Initialize parent class - PES uses 3*natoms as dimension
        PES.__init__(self, atoms, *args, H0=H0, **kwargs)

        # Store Cartesian dimension (set by parent)
        self.n_cart = self.dim  # 3 * natoms

        # Update dimension to include cell DOF
        self.dim = self.n_cart + self.n_cell_dof

        # Done initializing - now get_x returns full vector
        self._initializing = False

        # Create proper Hessian with correct dimensions
        # Use block-diagonal structure: Cartesian Hessian + cell Hessian
        H_old = self.H.B if self.H is not None and self.H.B is not None else None

        H0_full = np.zeros((self.dim, self.dim))
        if H_old is not None:
            H0_full[:self.n_cart, :self.n_cart] = H_old
        else:
            # Default: 70 eV/Å² is reasonable for stiff materials
            H0_full[:self.n_cart, :self.n_cart] = 70.0 * np.eye(self.n_cart)

        # Convert bool to int for refinement level
        if refine_initial_hessian is True:
            refine_level = 1
        elif refine_initial_hessian is False:
            refine_level = 0
        else:
            refine_level = int(refine_initial_hessian)

        if refine_level >= 1:
            # Level 1: Refine cell-related blocks
            H_cell_cols = self._compute_cell_hessian_columns(hessian_delta)
            # Set Cartesian-cell coupling (and its transpose for symmetry)
            H0_full[:self.n_cart, self.n_cart:] = H_cell_cols[:self.n_cart, :]
            H0_full[self.n_cart:, :self.n_cart] = H_cell_cols[:self.n_cart, :].T
            # Set cell-cell block with explicit symmetrization
            H_cell_cell = H_cell_cols[self.n_cart:, :]
            H0_full[self.n_cart:, self.n_cart:] = (H_cell_cell + H_cell_cell.T) / 2

        if refine_level == 0:
            # No refinement: use diagonal guess for cell block
            h0_cell = 1.0
            H0_full[self.n_cart:, self.n_cart:] = h0_cell * np.eye(self.n_cell_dof)

        # Save Hessian if requested
        if save_hessian is not None:
            np.save(save_hessian, H0_full)
            logger.info("Initial Hessian saved to %s", save_hessian)

        self.set_H(H0_full, initialized=(refine_level == 0))

    def maybe_niggli_reduce(self, angle_threshold=30.0):
        """Apply Niggli reduction if cell angles deviate too far from 90 deg.

        When the unit cell becomes highly skewed during optimization, this
        remaps to the most compact (Niggli-reduced) cell and resets the
        log-deformation reference. The cell block of the Hessian is
        transformed to the new parameterization basis via the Jacobian of
        the log-deformation map.

        Parameters
        ----------
        angle_threshold : float
            Maximum deviation from 90 deg before triggering reduction.
            Default 30 means reduction triggers when any angle < 60 or > 120.

        Returns
        -------
        bool
            True if reduction was applied.
        """
        angles = self.atoms.get_cell().angles()
        max_deviation = max(abs(a - 90.0) for a in angles)
        if max_deviation <= angle_threshold:
            return False

        H = self.H.B.copy()
        n = self.n_cart
        T_masked = _niggli_hessian_transform(
            self.atoms, self.orig_cell, self.exp_cell_factor, self.cell_mask
        )

        # Transform cell-cell block: H_new = T^T @ H_old @ T
        H_cell_new = T_masked.T @ H[n:, n:] @ T_masked
        H[n:, n:] = H_cell_new

        # Transform coupling blocks
        H[:n, n:] = H[:n, n:] @ T_masked
        H[n:, :n] = T_masked.T @ H[n:, :n]

        self.orig_cell = self.atoms.get_cell().array.copy()
        self.set_H(H, initialized=True)

        # Reset cached state so next evaluation recomputes everything
        self.curr = dict(x=None, f=None, g=None)
        self.last = self.curr.copy()

        return True

    def save(self):
        """Save current state including cell."""
        PES.save(self)
        self.savepoint['cell'] = self.atoms.get_cell().array.copy()

    def restore(self):
        """Restore saved state including cell."""
        PES.restore(self)
        if 'cell' in self.savepoint:
            self.atoms.set_cell(self.savepoint['cell'], scale_atoms=False)

    def refine_hessian(self, refine_level: int = 1, delta: float = 1e-5):
        """Re-refine Hessian blocks via finite differences during optimization.

        This can help recover from accumulated bad curvature in the Hessian
        that develops during BFGS updates.

        Parameters
        ----------
        refine_level : int
            Level of refinement (only level 1 supported for Cartesian).
        delta : float
            Finite difference step size.
        """
        if refine_level < 1:
            return

        # Get current Hessian
        H = self.H.asarray()

        # Level 1: Refine cell-related blocks
        H_cell_cols = self._compute_cell_hessian_columns(delta)
        # Set Cartesian-cell coupling (and its transpose for symmetry)
        H[:self.n_cart, self.n_cart:] = H_cell_cols[:self.n_cart, :]
        H[self.n_cart:, :self.n_cart] = H_cell_cols[:self.n_cart, :].T
        # Set cell-cell block with explicit symmetrization
        H_cell_cell = H_cell_cols[self.n_cart:, :]
        H[self.n_cart:, self.n_cart:] = (H_cell_cell + H_cell_cell.T) / 2

        # Update the Hessian (preserves eigenvalue tracking, etc.)
        self.set_H(H, initialized=True)
        logger.info("Hessian re-refined at level %d", refine_level)

    def _compute_cell_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for cell DOF via finite differences.

        This computes d(gradient)/d(cell_param) for all cell parameters,
        giving us both the Cartesian-cell coupling block and the cell-cell block.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_cols : ndarray
            Array of shape (dim, n_cell_dof) containing Hessian columns.
        """
        H_cols = np.zeros((self.dim, self.n_cell_dof))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * self.n_cell_dof
        logger.info("Refining initial Hessian: 0/%d force calls", n_evals)

        for i in range(self.n_cell_dof):
            # Restore state before each FD probe
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)

            # Displace cell parameter +delta
            x_plus = x0.copy()
            x_plus[self.n_cart + i] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()
            logger.info("Refining initial Hessian: %d/%d force calls", 2*i + 1, n_evals)

            # Restore before -delta
            self.atoms.positions = pos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)

            # Displace cell parameter -delta
            x_minus = x0.copy()
            x_minus[self.n_cart + i] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()
            logger.info("Refining initial Hessian: %d/%d force calls", 2*i + 2, n_evals)

            # Central difference
            H_cols[:, i] = (g_plus - g_minus) / (2 * delta)


        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_cols

    def get_x(self) -> np.ndarray:
        """Return Cartesian positions + cell parameters.

        During initialization (_initializing=True), returns only Cartesian coords
        to be compatible with parent class initialization.
        """
        x_cart = self.apos.ravel().copy()

        # During parent initialization, return only Cartesian coords
        if self._initializing:
            return x_cart

        cell_params = self._masked_cell_params()
        return np.concatenate([x_cart, cell_params])

    def _get_deformation_gradient(self) -> np.ndarray:
        """Get current deformation gradient F = cell @ inv(orig_cell)."""
        return self.atoms.get_cell().array @ np.linalg.inv(self.orig_cell)

    def _get_log_deform(self) -> np.ndarray:
        """Get log of deformation gradient, scaled by exp_cell_factor."""
        F = self._get_deformation_gradient()
        return _logm_3x3(F) * self.exp_cell_factor

    def _set_cell_from_log_deform(self, log_deform_scaled: np.ndarray) -> None:
        """Set cell from scaled log-deformation gradient.

        Does not scale atoms — in CellCartesianPES, positions are set
        explicitly by set_x() after the cell change. The gradient formula
        already accounts for fixed-Cartesian positions via the r^T @ f term.
        """
        log_deform = log_deform_scaled / self.exp_cell_factor
        F = expm(log_deform.real)
        new_cell = F @ self.orig_cell
        self.atoms.set_cell(new_cell, scale_atoms=False)

    def _masked_cell_params(self) -> np.ndarray:
        """Get cell parameters as flat array (only free DOF)."""
        log_deform = self._get_log_deform()
        return log_deform[self.cell_mask]

    def _set_masked_cell_params(self, params: np.ndarray) -> None:
        """Set cell from flat array of free DOF."""
        log_deform = self._get_log_deform()
        log_deform[self.cell_mask] = params
        self._set_cell_from_log_deform(log_deform)

    def set_x(self, target: np.ndarray):
        """Set Cartesian positions and cell parameters.

        Much simpler than CellInternalPES since Cartesian positions can be
        set directly without iterative or ODE-based solvers.

        Returns
        -------
        dx_initial, dx_final, g_par : tuple of np.ndarray
            Displacement information for Hessian update.
        """
        x0 = self.get_x()
        dx_initial = target - x0

        # Split target into Cartesian and cell parts
        x_cart_target = target[:self.n_cart]
        cell_target = target[self.n_cart:]

        # Get initial cell params
        cell_params0 = self._masked_cell_params()

        # Update cell first
        self._set_masked_cell_params(cell_target)

        # Update positions directly (simple for Cartesian!)
        x_cart0 = self.apos.ravel()
        diff = x_cart_target - x_cart0
        self.atoms.positions = x_cart_target.reshape((-1, 3))

        dx_final = np.concatenate([diff, cell_target - cell_params0])

        # Return parallel gradient for Hessian update
        g_old = self.curr.get('g', None)
        if g_old is not None:
            g_par = g_old.copy()
        else:
            g_par = np.zeros(self.dim)

        return dx_initial, dx_final, g_par

    def eval(self) -> tuple:
        """Evaluate energy and combined gradient (Cartesian + cell)."""
        self.neval += 1
        f = self.atoms.get_potential_energy()

        # Add pressure contribution: H = E + P*V
        if self.scalar_pressure != 0.0:
            f += self.scalar_pressure * self.atoms.get_volume()

        # Cartesian gradient: Sella works in actual Cartesian positions
        # (not the undeformed frame), so no F transformation needed
        forces = self.atoms.get_forces()
        g_cart = -forces.ravel()

        # Stress tensor -> cell gradient
        stress = self.atoms.get_stress()  # 6-component Voigt, eV/Å³
        g_cell = self._stress_to_cell_gradient(stress, forces)

        self.write_traj()
        return f, np.concatenate([g_cart, g_cell])

    def _stress_to_cell_gradient(self, stress_voigt: np.ndarray,
                                 forces: np.ndarray) -> np.ndarray:
        """Convert stress tensor to gradient w.r.t. log-deformation cell parameters.

        Uses the Frechet derivative of the matrix exponential to correctly
        transform dE/dF into dE/dU = dE/d(log F).

        The virial stress V*σ relates to dE/dC via (ASE row-vector convention):
            V*σ = dE/dC^T @ C - f^T @ r

        So dE/dC = C^{-T} @ (V*σ + r^T @ f)  at fixed Cartesian positions, or
           dE/dC = C^{-T} @ V*σ              at fixed fractional positions.

        Then dE/dF = dE/dC @ C₀^T via chain rule through cell = F @ C₀.
        """
        volume = self.atoms.get_volume()
        stress_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)

        # Add external pressure contribution
        if self.scalar_pressure != 0.0:
            stress_3x3 += self.scalar_pressure * np.eye(3)

        C = self.atoms.get_cell().array
        C_inv_T = np.linalg.inv(C.T)

        # V*σ from the virial stress
        virial = volume * stress_3x3

        # In CellCartesianPES, positions are always set independently after
        # cell changes (set_x overrides any position scaling), so we always
        # need the fixed-Cartesian gradient: dE/dC = C^{-T} @ (V*σ + r^T @ f)
        positions = self.atoms.get_positions()
        dEdC = C_inv_T @ (virial + positions.T @ forces)

        # Chain rule: cell = F @ C₀, so dE/dF = dE/dC @ C₀^T
        dEdF = dEdC @ self.orig_cell.T

        # Convert dE/dF to dE/dU via Frechet derivative of expm
        F = self._get_deformation_gradient()
        U = _logm_3x3(F)
        g_cell_3x3 = _expm_frechet_3x3_contracted(U, dEdF)

        # Apply cell mask and scale
        g_cell_3x3 = g_cell_3x3 * self.cell_mask
        g_cell_3x3 = g_cell_3x3 / self.exp_cell_factor

        return g_cell_3x3[self.cell_mask]

    def _calc_basis(self):
        """Calculate basis including cell DOF.

        The cell DOF are treated as unconstrained additional coordinates.
        """
        # Compute Cartesian basis directly (not via parent, since parent uses self.dim)
        # This mirrors PES._calc_basis but uses n_cart instead of self.dim
        state_hash = self._state_hash()
        cached = self._basis_cache.get(state_hash)
        if cached is not None:
            return cached

        drdx_cart = self.cons.jacobian()  # Constraint Jacobian for Cartesian coords
        U, S, VT = np.linalg.svd(drdx_cart)
        ncons = np.sum(S > 1e-6)
        Ucons_cart = VT[:ncons].T
        Ufree_cart = VT[ncons:].T
        Unred_cart = np.eye(self.n_cart)

        # Extend to include cell DOF
        n_total = self.n_cart + self.n_cell_dof

        # drdx extended with zeros for cell columns
        drdx = np.zeros((drdx_cart.shape[0], n_total))
        drdx[:, :self.n_cart] = drdx_cart

        # Ucons stays the same (no cell constraints)
        Ucons = np.zeros((n_total, Ucons_cart.shape[1]))
        Ucons[:self.n_cart, :] = Ucons_cart

        # Unred extended with identity for cell DOF
        Unred = np.zeros((n_total, Unred_cart.shape[1] + self.n_cell_dof))
        Unred[:self.n_cart, :Unred_cart.shape[1]] = Unred_cart
        Unred[self.n_cart:, Unred_cart.shape[1]:] = np.eye(self.n_cell_dof)

        # Ufree extended with identity for cell DOF
        Ufree = np.zeros((n_total, Ufree_cart.shape[1] + self.n_cell_dof))
        Ufree[:self.n_cart, :Ufree_cart.shape[1]] = Ufree_cart
        Ufree[self.n_cart:, Ufree_cart.shape[1]:] = np.eye(self.n_cell_dof)

        result = drdx, Ucons, Unred, Ufree

        # Cache the result
        self._basis_cache.put(state_hash, result)
        return result

    def converged(self, fmax: float, smax: float = None, cmax: float = 1e-5):
        """Check convergence of forces and stress.

        Parameters
        ----------
        fmax : float
            Maximum force tolerance (eV/Å).
        smax : float, optional
            Maximum stress tolerance. If None, uses fmax.
        cmax : float, optional
            Constraint residual tolerance.

        Returns
        -------
        conv : bool
            True if converged.
        fmax_actual : float
            Maximum force.
        cmax_actual : float
            Constraint residual norm.
        smax_actual : float
            Maximum stress gradient.
        """
        if smax is None:
            smax = fmax

        # Force convergence (project out constraints)
        g = self.get_g()
        g_cart = g[:self.n_cart]
        Ufree = self.get_Ufree()
        Ufree_cart = Ufree[:self.n_cart, :Ufree.shape[1] - self.n_cell_dof]
        g_proj = (Ufree_cart @ (Ufree_cart.T @ g_cart)).reshape((-1, 3))

        fmax_actual = np.linalg.norm(g_proj, axis=1).max()

        # Stress convergence
        g_cell = g[self.n_cart:]
        smax_actual = np.abs(g_cell).max() if len(g_cell) > 0 else 0.0

        # Constraint residual
        cmax_actual = np.linalg.norm(self.get_res())

        conv = (fmax_actual < fmax) and (smax_actual < smax) and (cmax_actual < cmax)
        return conv, fmax_actual, cmax_actual, smax_actual

    def get_projected_forces(self) -> np.ndarray:
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_cart = g[:self.n_cart]
        Ufree = self.get_Ufree()
        Ufree_cart = Ufree[:self.n_cart, :]
        return -(Ufree_cart @ (Ufree_cart.T @ g_cart)).reshape((-1, 3))

    def get_drdx(self):
        """Get constraint Jacobian extended for cell DOF."""
        drdx_cart = PES.get_drdx(self)
        n_cons = drdx_cart.shape[0]
        drdx = np.zeros((n_cons, self.dim))
        drdx[:, :self.n_cart] = drdx_cart
        return drdx

    def get_Hc(self):
        """Get constraint Hessian extended for cell DOF."""
        Hc_cart = PES.get_Hc(self)
        Hc = np.zeros((self.dim, self.dim))
        Hc[:self.n_cart, :self.n_cart] = Hc_cart
        return Hc
