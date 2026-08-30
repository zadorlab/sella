import logging
from typing import Union, Callable

import numpy as np
from scipy.linalg import eigh, expm, expm_frechet, logm, polar, qr, solve_triangular, svd as _scipy_svd
from scipy.integrate import LSODA
from ase import Atoms
from ase.build import niggli_reduce
from ase.utils import basestring
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.eigensolvers import rayleigh_ritz
from sella.internal import Internals, Constraints, DuplicateInternalError
from sella._gpu import (gpu_qr as _gpu_qr, gpu_project,
                        gpu_svd as _gpu_svd, gpu_pinv as _gpu_pinv,
                        gpu_solve_triangular as _gpu_solve_triangular,
                        gpu_qr_with_pinv as _gpu_qr_with_pinv,
                        gpu_left_matmul as _gpu_left_matmul,
                        gpu_left_factor as _gpu_left_factor)

logger = logging.getLogger(__name__)

from sella._constants import _LSTSQ_RCOND


_LINEAR_DUMMY_PROJECTION_TOL = 1e-7


def _hessian_matvec(H, v):
    if hasattr(H, 'matvec'):
        return H.matvec(v)
    return H @ v


def _robust_svd(M, full_matrices=True):
    """SVD that falls back to the QR-based LAPACK driver on non-convergence.

    numpy's default ``gesdd`` (divide-and-conquer) SVD is fast but fragile on
    near-singular / clustered-singular-value matrices, where it can raise
    ``LinAlgError: SVD did not converge``. The internal-coordinate Jacobian is
    routinely near-singular (redundant internals + rigid-body null space, plus
    transient near-degenerate coordinates during the geodesic ODE), so this
    happens in practice -- especially with ``exact_geodesic=True``, which
    recomputes the pseudoinverse on every ODE RHS evaluation. The older
    ``gesvd`` (QR iteration) driver is slower but robust and succeeds on the
    exact matrices where ``gesdd`` fails.

    When a CUDA GPU is present this first routes through cuSOLVER
    (:func:`sella._gpu.gpu_svd`), which is both faster on the ~1e3-square
    Jacobians hit under ``exact_geodesic=True`` and uses a different driver
    that sidesteps the gesdd convergence bug entirely; the CPU gesdd->gesvd
    path below remains as the fallback.
    """
    r = _gpu_svd(M, full_matrices=full_matrices)
    if r is not None:
        return r
    try:
        return np.linalg.svd(M, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        return _scipy_svd(M, full_matrices=full_matrices, lapack_driver='gesvd')


def _robust_pinv(B, rcond=1e-15):
    """Moore-Penrose pseudoinverse with the same gesdd->gesvd fallback as
    :func:`_robust_svd` (numpy's ``pinv`` uses ``gesdd`` internally). Routes
    through the GPU (:func:`sella._gpu.gpu_pinv`) first when available."""
    r = _gpu_pinv(B, rcond=rcond)
    if r is not None:
        return r
    try:
        return np.linalg.pinv(B, rcond=rcond)
    except np.linalg.LinAlgError:
        U, s, Vt = _scipy_svd(B, full_matrices=False, lapack_driver='gesvd')
        cutoff = rcond * (s[0] if s.size else 0.0)
        sinv = np.where(s > cutoff, 1.0 / s, 0.0)
        return (Vt.T * sinv) @ U.T


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


def _constraint_normal_subspace(drdxnred):
    """Economy-size orthonormal basis for independent constraint normals."""
    Q, R, _ = qr(
        drdxnred.T, mode='economic', pivoting=True, check_finite=False
    )
    ncons = drdxnred.shape[0]
    if (Q.shape[1] != ncons
            or not np.all(np.isfinite(R))
            or np.any(np.diag(R) == 0.0)):
        return None
    return Q


class _LowRankTangentProjectionTranspose:
    def __init__(self, projection):
        self.projection = projection
        self.shape = (projection.shape[1], projection.shape[0])

    def __matmul__(self, values):
        return self.projection.basis @ self.projection.project(values)


class _LowRankTangentProjection:
    """Matrix-like ``T @ U.T`` without materializing the tangent projector."""
    def __init__(self, basis, normals):
        self.basis = basis
        self.normals = normals
        self.shape = (basis.shape[1], basis.shape[0])
        self.T = _LowRankTangentProjectionTranspose(self)

    def project(self, values):
        values = np.asarray(values)
        return values - self.normals @ (self.normals.T @ values)

    def __matmul__(self, values):
        return self.project(self.basis.T @ values)


def _svd_rank(s, rtol=1e-6):
    """Numerical rank from a singular-value spectrum, relative to the largest.

    Counts singular values above ``rtol * s[0]``; returns 0 for an empty
    spectrum. Using a relative (rather than absolute) cutoff keeps the rank
    scale-invariant, which matters for poorly-scaled Jacobians.
    """
    if len(s) == 0:
        return 0
    return int(np.sum(s > rtol * s[0]))


def _range_space_projector(B):
    """Orthogonal projector onto range(B) with rank truncation via pivoting QR."""
    if B.size == 0:
        # scipy's pivoted QR rejects a zero-sized matrix (LAPACK geqp3), but the
        # range of an empty B is {0}, so the projector is the (m, m) zero matrix.
        # This matches the rank-0 result the QR path would produce for a nonzero
        # but rank-deficient B.
        m = B.shape[0]
        return np.zeros((m, m), dtype=B.dtype)
    Q, R, _ = qr(B, mode='full', pivoting=True, check_finite=False)
    rdiag = np.abs(np.diag(R))
    rcond = max(B.shape) * np.finfo(B.dtype).eps
    if rdiag.size and rdiag[0] > 0:
        nkeep = int(np.sum(rdiag > rcond * rdiag[0]))
    else:
        nkeep = 0
    Q_r = Q[:, :nkeep]
    return Q_r @ Q_r.T


def _rigid_motion_nullspace(positions, tol=1e-10):
    """Orthonormal rigid translation/rotation modes for Cartesian positions."""
    n = len(positions)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    modes = []
    for dim in range(3):
        mode = np.zeros((n, 3), dtype=np.float64)
        mode[:, dim] = 1.0
        modes.append(mode.ravel())

    rel = np.asarray(positions, dtype=np.float64) - np.mean(positions, axis=0)
    for axis in np.eye(3):
        modes.append(np.cross(axis, rel).ravel())

    Z0 = np.column_stack(modes)
    Qz, Rz = np.linalg.qr(Z0, mode='reduced')
    diag = np.abs(np.diag(Rz))
    if diag.size == 0 or diag[0] == 0:
        return np.empty((3 * n, 0), dtype=np.float64)
    rank = int(np.sum(diag > tol * diag[0]))
    if rank == len(diag):
        return Qz

    # Collinear and otherwise rank-deficient geometries can place a dependent
    # rotation before an independent one. Pivot only that uncommon case so the
    # returned leading columns still span every valid rigid motion.
    Qz, Rz, _ = qr(
        Z0, mode='economic', pivoting=True, check_finite=False
    )
    diag = np.abs(np.diag(Rz))
    if diag.size == 0 or diag[0] == 0:
        return np.empty((3 * n, 0), dtype=np.float64)
    rank = int(np.sum(diag > tol * diag[0]))
    return Qz[:, :rank]


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


def _expm_frechet_3x3_contracted_taylor(U, dEdF):
    """Fourth-order small-``U`` series for ``_expm_frechet_3x3_contracted``.

    This evaluates the adjoint of
    ``L_exp(U, E) = sum_n 1/n! sum_k U^k E U^(n-1-k)`` contracted with
    ``dEdF``.  For ``||U|| < 1e-4`` the omitted terms are far below double
    precision for the 3x3 cell updates this path serves, while avoiding the
    much slower Padé-based SciPy Frechet loop.
    """
    UT = U.T
    U2T = UT @ UT
    U3T = U2T @ UT
    return (
        dEdF
        + 0.5 * (dEdF @ UT + UT @ dEdF)
        + (dEdF @ U2T + UT @ dEdF @ UT + U2T @ dEdF) / 6.0
        + (
            dEdF @ U3T + UT @ dEdF @ U2T
            + U2T @ dEdF @ UT + U3T @ dEdF
        ) / 24.0
    )


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

    Uses a Taylor contraction when ``U`` is small (the eigendecomposition is
    ill-conditioned near zero), and falls back to scipy when ``np.linalg.eig``
    produces a near-singular ``V`` or nearly-degenerate finite eigenvalues.
    """
    # When ||U|| ~ 0 the derivative reduces to the identity map E -> E,
    # so the contracted output is dEdF itself. Avoid eig on a noisy zero.
    Unorm = np.linalg.norm(U)
    if Unorm < 1e-10:
        return dEdF.copy()

    def _scipy_contract():
        g = np.zeros((3, 3))
        for mu in range(3):
            for nu in range(3):
                E = np.zeros((3, 3)); E[mu, nu] = 1.0
                ed = expm_frechet(U, E, compute_expm=False)
                g[mu, nu] = np.sum(ed * dEdF)
        return g

    # For small ||U|| the identity-map shortcut is only first-order accurate
    # (error ~ O(||U||)), and eig on a small nonsymmetric matrix yields
    # ill-conditioned eigenvectors whose closed-form divided difference loses
    # precision well before the cond(V) test below trips.  A direct Frechet
    # Taylor series is both accurate and much cheaper than the SciPy loop here.
    if Unorm < 1e-4:
        return _expm_frechet_3x3_contracted_taylor(U, dEdF)

    lam, V = np.linalg.eig(U)
    if np.linalg.cond(V) > 1e10:
        # Fall back when the eigenvector basis is too ill-conditioned
        # for the closed form to be numerically reliable.
        return _scipy_contract()
    # Near-degenerate eigenvalues make the divided-difference
    # (e^a - e^b)/(a - b) lose precision even when V is well-conditioned
    # (e.g. symmetric U). Detect small relative gaps directly and use the
    # accurate scipy path. Fill the diagonal with +inf via fill_diagonal to
    # avoid the 0*inf = nan trap of ``np.eye(3) * np.inf``.
    lam_scale = max(np.abs(lam).max(), 1.0)
    gaps = np.abs(lam[:, None] - lam[None, :])
    np.fill_diagonal(gaps, np.inf)
    if gaps.min() < 1e-6 * lam_scale:
        return _scipy_contract()
    Vinv = np.linalg.inv(V)
    expl = np.exp(lam)
    diff = lam[:, None] - lam[None, :]
    mask = np.abs(diff) > 1e-12 * max(np.abs(lam).max(), 1.0)
    safe = np.where(mask, diff, 1.0)
    fij = np.where(mask, (expl[:, None] - expl[None, :]) / safe, expl[:, None])
    M = V.T @ dEdF @ Vinv.T
    return (Vinv.T @ (fij * M) @ V.T).real


def _cell_deformation_jacobian(F, orig_cell, exp_cell_factor):
    """Jacobian ``d(cell_ab)/d(L_ij)`` of the cell parameterization.

    The cell is parameterized as ``cell(L) = expm(L / factor) @ orig_cell``
    with ``L = logm(F) * factor``. The Frechet derivative of expm is therefore
    evaluated at the *unscaled* log-deformation ``logm(F)`` (the base point),
    with perturbation direction ``E_ij / factor`` (the chain-rule dL->dX term).
    """
    X = _logm_3x3(F)
    J = np.zeros((9, 9))
    for idx in range(9):
        i, j = divmod(idx, 3)
        E = np.zeros((3, 3))
        E[i, j] = 1.0 / exp_cell_factor
        dF = expm_frechet(X, E, compute_expm=False)
        dC = dF @ orig_cell  # d(cell)/d(L_ij)
        J[:, idx] = dC.ravel()
    return J


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
    # Old Jacobian J_old[ab, ij] = d(cell_ab)/d(L_ij) at the current
    # (pre-reduction) log-deformation.
    F_old = atoms.get_cell().array @ np.linalg.inv(orig_cell)
    J_old = _cell_deformation_jacobian(F_old, orig_cell, exp_cell_factor)

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
        refine_initial_hessian: Union[bool, int] = False,
        hessian_delta: float = 1e-4,
        save_hessian: str = None,
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

        refine_level = self._coerce_refine_level(refine_initial_hessian)
        self.initial_hessian_refinement_level = refine_level
        if refine_level >= 3:
            self.set_H(
                self._fd_cartesian_hessian(hessian_delta),
                initialized=False,
            )

        if save_hessian is not None:
            np.save(save_hessian, self.H.asarray())
            logger.info("Initial Hessian saved to %s", save_hessian)

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def _state_hash(self) -> bytes:
        """Hash of all state that affects cached computations."""
        h = self.atoms.positions.tobytes()
        cell = self.atoms.cell
        if cell is not None and cell.any():
            h += cell.array.tobytes()
        # Inequality toggles (disable_satisfied_inequalities / validate_
        # inequalities) flip the constraint active mask without moving atoms,
        # which changes drdx / residual and thus the constraint basis. Fold the
        # mask into the hash so geometry-keyed caches invalidate on a toggle.
        h += np.asarray(self.cons._active_mask, dtype=bool).tobytes()
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

    @staticmethod
    def _coerce_refine_level(refine_initial_hessian) -> int:
        if refine_initial_hessian is True:
            return 1
        if refine_initial_hessian is False:
            return 0
        return int(refine_initial_hessian)

    def _fd_hessian_columns(self, delta, indices, n_rows, label="",
                            forward=False):
        """Finite-difference Hessian columns for coordinate indices."""
        n_cols = len(indices)
        H_cols = np.zeros((n_rows, n_cols))
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()
        dpos0 = None if self.dummies is None else self.dummies.positions.copy()

        suffix = " (forward)" if forward else ""
        if forward:
            _, g0 = self.eval()
            n_evals = n_cols + 1
            logger.info("Refining %s Hessian (forward): 1/%d force calls",
                        label, n_evals)
        else:
            n_evals = 2 * n_cols
            logger.info("Refining %s Hessian: 0/%d force calls",
                        label, n_evals)

        def _probe(idx, sign):
            # Restore before every probe because curvilinear and cell updates
            # are path-dependent.
            self.atoms.positions = pos0.copy()
            if dpos0 is not None:
                self.dummies.positions = dpos0.copy()
            self.atoms.set_cell(cell0, scale_atoms=False)
            x_probe = x0.copy()
            x_probe[idx] += sign * delta
            self.set_x(x_probe)
            return self.eval()[1]

        for col, idx in enumerate(indices):
            g_plus = _probe(idx, +1)
            if forward:
                H_cols[:, col] = (g_plus[:n_rows] - g0[:n_rows]) / delta
            else:
                g_minus = _probe(idx, -1)
                H_cols[:, col] = (
                    (g_plus[:n_rows] - g_minus[:n_rows]) / (2 * delta)
                )

            if (col + 1) % max(n_cols // 10, 1) == 0 or col == n_cols - 1:
                count = (col + 2) if forward else 2 * (col + 1)
                logger.info("Refining %s Hessian%s: %d/%d force calls",
                            label, suffix, count, n_evals)

        self.atoms.positions = pos0
        if dpos0 is not None:
            self.dummies.positions = dpos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None
        return H_cols

    def _fd_cartesian_hessian(self, delta: float) -> np.ndarray:
        """Build the Cartesian Hessian by central force differences."""
        positions = self.atoms.positions.copy()
        ncart = positions.size
        H_cols = np.empty((ncart, ncart))
        logger.info("Refining Cartesian Hessian: 0/%d force calls", 2 * ncart)

        for column in range(ncart):
            displaced = positions.copy().reshape(-1)
            displaced[column] += delta
            self.atoms.positions = displaced.reshape((-1, 3))
            _, g_plus = PES.eval(self)

            displaced[column] -= 2 * delta
            self.atoms.positions = displaced.reshape((-1, 3))
            _, g_minus = PES.eval(self)
            H_cols[:, column] = (g_plus - g_minus) / (2 * delta)

            if ((column + 1) % max(ncart // 10, 1) == 0
                    or column == ncart - 1):
                logger.info(
                    "Refining Cartesian Hessian: %d/%d force calls",
                    2 * (column + 1), 2 * ncart,
                )

        self.atoms.positions = positions
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None
        return (H_cols + H_cols.T) / 2

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
        H_gpu = H._get_B_gpu()
        H_B = H.B if getattr(H, '_cpu_current', True) else None
        if H_B is None and H_gpu is None:
            Bproj = None
        else:
            UtHU = gpu_project(H_B, U, H_gpu=H_gpu)
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

    def get_fast_restricted_step_data(self, g, order, stepper, W_is_identity):
        return None

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.residual()

    def get_convergence_res(self):
        return self.get_res()

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

    def _get_potential_energy_raw(self):
        return self.atoms.get_potential_energy(apply_constraint=False)

    def _get_forces_raw(self):
        return self.atoms.get_forces(apply_constraint=False)

    def _get_stress_raw(self):
        return self.atoms.get_stress(apply_constraint=False)

    def eval(self):
        self.neval += 1
        f = self._get_potential_energy_raw()
        g = -self._get_forces_raw().ravel()
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
        if Ucons.shape[1] == 0:
            return np.zeros(self.dim)

        scons = -Ucons @ np.linalg.lstsq(
            self.get_drdx() @ Ucons,
            self.get_res(),
            rcond=_LSTSQ_RCOND,
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
        self.curr.pop('Ufree_materialized', None)
        self.curr.pop('fast_dummy_selection_indices', None)
        self.curr.pop('fast_dummy_redundant_projection', None)
        self.curr['drdx'] = drdx
        self.curr['Ucons'] = Ucons
        self.curr['Unred'] = Unred
        self.curr['Ufree'] = Ufree

        if self.curr['g'] is None:
            L = None
        elif drdx.shape[0] == 0:
            L = np.zeros(0)
        else:
            L = np.linalg.lstsq(drdx.T, self.curr['g'], rcond=_LSTSQ_RCOND)[0]

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

        # A dense escape restart can sample two directions and continue from
        # their linear combination. Rank-reveal the accumulated numerical
        # secants before the multi-secant update so an exact dependence does
        # not make S.T @ S singular. Keep original sample pairs rather than
        # rotating finite-difference data through an SVD.
        if Hproj.requires_secant_rank_cleanup and Vs.shape[1] > 1:
            _, R_samples, piv = qr(
                Vs, mode='economic', pivoting=True, check_finite=False
            )
            diagonal = np.abs(np.diag(R_samples))
            cutoff = (
                max(Vs.shape) * np.finfo(Vs.dtype).eps * diagonal[0]
                if len(diagonal) and diagonal[0] > 0 else 0.0
            )
            rank = int(np.sum(diagonal > cutoff))
            if rank < Vs.shape[1]:
                keep = np.asarray(piv[:rank], dtype=int)
                Vs = Vs[:, keep]
                AVs = AVs[:, keep]

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
        cmax1 = np.linalg.norm(self.get_convergence_res())
        conv = (fmax1 < fmax) and (cmax1 < cmax)
        return conv, fmax1, cmax1

    def wrap_dx(self, dx, origin=None):
        return dx

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Hdx = _hessian_matvec(H, dx)
        return g.T @ dx + (dx.T @ Hdx) / 2.

    def kick(self, dx, diag=False, **diag_kwargs):
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H

        dx_initial, dx_final, g_par = self.set_x(x0 + dx)

        df_pred = self.get_df_pred(dx_initial, g0, B0)
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        df_threshold = 1e-14 * max(abs(f0), 1.0)
        if df_pred is not None and abs(df_pred) >= df_threshold:
            ratio = df_actual / df_pred
        else:
            if df_pred is not None:
                logger.debug(
                    "Trust ratio skipped: |df_pred|=%.2e < threshold=%.2e",
                    abs(df_pred), df_threshold,
                )
            ratio = None

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
        exact_geodesic: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        hessian_delta: float = 1e-4,
        save_hessian: str = None,
        **kwargs
    ):
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
        self.n_internal = self.dim
        self.ncart = self.int.ndof
        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.jacobian()
            P = _range_space_projector(B)
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper
        self.exact_geodesic = exact_geodesic

        self._pinv_cache = _LRU2()
        self._qr_cache = _LRU2()
        self._Hc_cache = _LRU2()

        refine_level = self._coerce_refine_level(refine_initial_hessian)
        self.initial_hessian_refinement_level = refine_level
        refined = False
        H_refined = self.H.asarray().copy()
        if refine_level >= 3:
            H_cart = self._fd_cartesian_hessian(hessian_delta)
            H_refined = self._convert_cartesian_hessian_to_internal(H_cart)
            refined = True
        elif refine_level >= 2:
            tric_indices = self._get_tric_indices()
            if len(tric_indices):
                H_tric_cols = self._compute_tric_hessian_columns(
                    hessian_delta
                )
                for col, idx in enumerate(tric_indices):
                    H_refined[:, idx] = H_tric_cols[:, col]
                    H_refined[idx, :] = H_tric_cols[:, col]
                refined = True

        if refined:
            self.set_H(H_refined, initialized=False)
        if save_hessian is not None:
            np.save(save_hessian, self.H.asarray())
            logger.info("Initial Hessian saved to %s", save_hessian)

    dpos = property(lambda self: self.dummies.positions.copy())

    def _state_hash(self) -> bytes:
        h = super()._state_hash()
        h += self.dummies.positions.tobytes()
        return h

    # =========================================================================
    # Cache optimization: Store and reuse Jacobian QR and pseudo-inverse
    # =========================================================================

    def _rigid_nullspace_qr(self, B, rank_rtol=1e-6):
        """QR/Binv for Jacobians whose nullspace is rigid-body motion.

        For free internal-coordinate systems, the only Cartesian null modes are
        often global translations and rotations.  If those explicit modes span
        ``null(B)``, drop a well-conditioned set of gauge columns, QR the
        reduced full-rank Jacobian, then project the particular inverse back to
        the minimum-norm Cartesian inverse.  This avoids an SVD of the dense
        QR factor while preserving the same range basis and Moore-Penrose
        pseudoinverse.
        """
        ncart = B.shape[1]
        if ncart == 0 or ncart % 3 != 0:
            return None

        if len(self.dummies):
            positions = np.vstack((self.atoms.positions,
                                   self.dummies.positions))
        else:
            positions = self.atoms.positions
        Z = _rigid_motion_nullspace(positions)
        nullity = Z.shape[1]
        rank = ncart - nullity
        if nullity == 0 or rank < 0 or rank > B.shape[0]:
            return None

        norm_B = np.linalg.norm(B)
        if norm_B == 0:
            return None
        if np.linalg.norm(B @ Z) > 1e-8 * norm_B:
            return None

        try:
            _, _, piv = qr(
                Z.T, mode='economic', pivoting=True, check_finite=False
            )
            drop = np.asarray(piv[:nullity], dtype=int)
            Z_drop = Z[drop, :]
            if np.linalg.cond(Z_drop) > 1e8:
                return None
            keep_mask = np.ones(ncart, dtype=bool)
            keep_mask[drop] = False
            keep = np.flatnonzero(keep_mask)

            B_keep = B[:, keep]
            qr_pinv = _gpu_qr_with_pinv(B_keep)
            if qr_pinv is None:
                Q, R_keep = _gpu_qr(B_keep)
                Binv_keep = None
            else:
                Q, R_keep, Binv_keep = qr_pinv

            if R_keep.shape != (rank, rank):
                return None
            rdiag = np.abs(np.diag(R_keep))
            if (rdiag.size == 0 or rdiag.max() == 0
                    or rdiag.min() < rank_rtol * rdiag.max()):
                return None

            if Binv_keep is None:
                Binv_keep = _gpu_solve_triangular(
                    R_keep, Q.T, upper=True
                )
                if Binv_keep is None:
                    Binv_keep = solve_triangular(
                        R_keep, Q.T, check_finite=False
                    )

            X = np.zeros((ncart, B.shape[0]), dtype=B.dtype)
            X[keep, :] = Binv_keep
            Binv = X - Z @ (Z.T @ X)

            # Rebuild a non-square R satisfying B = Q @ R.  Downstream
            # constraint handling uses the non-square shape to choose the
            # cached-Binv path, while Q remains the exact range(B) basis.
            Z_keep = Z[keep, :]
            T = np.linalg.solve(Z_drop.T, Z_keep.T).T
            R = np.empty((rank, ncart), dtype=B.dtype)
            R[:, keep] = R_keep
            R[:, drop] = -(R_keep @ T)
        except (RuntimeError, MemoryError, np.linalg.LinAlgError, ValueError):
            return None

        return Q, R, Binv

    def _get_jacobian_qr(self):
        """Get cached economy QR of internal Jacobian.

        Returns (Q, R) from np.linalg.qr(B, mode='reduced').
        Q (m, n) is the orthonormal basis for range(B) (= Unred).
        R (n, n) upper triangular, with R^{-1} replacing V S^{-1} from SVD.

        Shared between _get_Binv and _calc_basis so the Jacobian is
        decomposed at most once per geometry.  ~2x faster than SVD.
        On rank deficiency, SVD-truncates the small R factor (reusing this
        QR) rather than re-factorizing B.
        """
        state_hash = self._state_hash()
        cached = self._qr_cache.get(state_hash)
        if cached is not None:
            return cached

        B = self.int.jacobian()
        # Free molecular internals normally have exactly the six rigid-body
        # Cartesian null modes.  The rigid-nullspace path fully factorizes B
        # after dropping a well-conditioned gauge, so try it before doing a
        # full QR that would only detect the expected rank deficiency and then
        # factorize the reduced Jacobian a second time.  Its consistency and
        # conditioning checks retain the general QR/SVD path as a fallback.
        # Periodic systems and explicit fragment frames do not have the six
        # global Cartesian rigid motions as their complete Jacobian nullspace.
        # Skip the failed applicability work there and use the general path.
        has_fragment_frames = bool(
            getattr(self.int, 'fragment_atom_groups', None)
        )
        rigid_qr = None
        if not np.any(self.atoms.pbc) and not has_fragment_frames:
            rigid_qr = self._rigid_nullspace_qr(B)
        if rigid_qr is not None:
            Q, R, Binv = rigid_qr
            self._pinv_cache.put(state_hash, Binv)
            self._qr_cache.put(state_hash, (Q, R))
            return Q, R

        qr_pinv = _gpu_qr_with_pinv(B)
        if qr_pinv is None:
            Q, R = _gpu_qr(B)
            Binv = None
        else:
            Q, R, Binv = qr_pinv
            if Binv is not None:
                self._pinv_cache.put(state_hash, Binv)

        # Check for rank deficiency.  Overcomplete free molecules usually give
        # square R with a small diagonal; sparse/minimal internals can give a
        # non-square R with explicit Cartesian null directions.
        rdiag = np.abs(np.diag(R))
        rank_deficient = (
            (R.size != 0 and R.shape[0] < R.shape[1])
            or (len(rdiag) > 0 and rdiag.min() < 1e-6 * rdiag.max())
        )
        if rank_deficient:
            # Rank-deficient. Rather than re-factorizing the big (m x n) B
            # with a fresh SVD, reuse the economy QR already computed above
            # (on the GPU when available): with B = Q R and Q orthonormal,
            # range(B) = Q @ range(R) and pinv(B) = pinv(R) @ Q.T, so the
            # rank-revealing work shrinks to the small (k x n) R factor.
            # This is ~1.5x faster than SVD-of-B on large systems and matches
            # it to machine precision. Free molecules with only rigid-body
            # null modes returned above; this handles other rank deficiencies.
            Ur, Sr, VTr = _robust_svd(R, full_matrices=False)
            nnred = _svd_rank(Sr)

            # Binv = pinv(R) @ Q.T, computed before Q/R are reassigned below.
            Binv = (VTr[:nnred].T / Sr[:nnred]) @ (Ur[:, :nnred].T @ Q.T)
            self._pinv_cache.put(state_hash, Binv)

            Q = Q @ Ur[:, :nnred]  # orthonormal basis for range(B) = Unred
            # (nnred, n) non-square -> signals deficiency
            R = np.diag(Sr[:nnred]) @ VTr[:nnred]

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
        # For a rank-deficient Jacobian, _get_jacobian_qr computes and caches
        # Binv itself. Re-check here (our first check ran before that populate)
        # so we don't recompute it, and so the else below is a true
        # eviction-only path.
        cached = self._pinv_cache.get(state_hash)
        if cached is not None:
            return cached

        if R.size == 0:
            ncart = 3 * len(self.atoms) + (3 * len(self.dummies) if self.dummies else 0)
            Binv = np.empty((ncart, 0))
        elif R.shape[0] == R.shape[1]:
            # Full rank: R is the upper-triangular QR factor, so R^{-1} Q^T via
            # a triangular solve is cheaper than any SVD-based pseudoinverse.
            Binv = _gpu_solve_triangular(R, Q.T, upper=True)
            if Binv is None:
                Binv = solve_triangular(R, Q.T, check_finite=False)
        else:
            # Non-square R => rank-deficient. We only get here on eviction:
            # _qr_cache still holds the truncated (Q, R) but the 2-entry
            # _pinv_cache dropped Binv. Rebuild pinv(B) = pinv(R) @ Q.T from
            # those cached factors (Q orthonormal, R the rank-truncated
            # factor) -- cheaper than re-SVDing the full Jacobian, and
            # consistent with the QR path's 1e-6 rank truncation. Route pinv(R)
            # through _robust_pinv for the gesdd->gesvd fallback, since R stays
            # near-singular in the cases that trip gesdd.
            Binv = _robust_pinv(R) @ Q.T

        self._pinv_cache.put(state_hash, Binv)
        return Binv

    def _get_tric_indices(self) -> np.ndarray:
        """Indices of translation and rotation internal coordinates."""
        n_trans = len(self.int.internals['translations'])
        n_bonds = len(self.int.internals['bonds'])
        n_angles = len(self.int.internals['angles'])
        n_dihedrals = len(self.int.internals['dihedrals'])
        n_other = len(self.int.internals['other'])
        n_rot = len(self.int.internals['rotations'])

        trans_indices = list(range(n_trans))
        rot_start = n_trans + n_bonds + n_angles + n_dihedrals + n_other
        rot_indices = list(range(rot_start, rot_start + n_rot))
        return np.array(trans_indices + rot_indices, dtype=int)

    def _compute_tric_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for translation/rotation coordinates."""
        tric_indices = self._get_tric_indices()
        return self._fd_hessian_columns(
            delta, tric_indices, self.dim, "TRIC"
        )

    def _compute_internal_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute the complete internal-coordinate Hessian by FD."""
        indices = list(range(self.n_internal))
        return self._fd_hessian_columns(
            delta, indices, self.n_internal, "internal"
        )

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

    def _restore_internal_positions(self, pos0, dpos0):
        self.atoms.positions = pos0
        self.dummies.positions = dpos0

    def _iterative_residual_rms(self, target):
        current = self.get_x()
        residual = self.wrap_dx(target - current, origin=current)
        rms = np.linalg.norm(residual) / np.sqrt(len(residual))
        return residual, rms

    @staticmethod
    def _iterative_step_status(iteration, rms, initial_rms, rms_prev,
                               stagnation_count):
        """Classify one Newton iteration without changing convergence thresholds."""
        if rms < 1e-8:
            return 'converged', stagnation_count
        if rms > initial_rms * 2.0:
            return 'failed', stagnation_count

        # Check for stagnation after the first few iterations. Three stagnant
        # iterations are accepted only if the residual has at least halved.
        if iteration <= 3:
            return 'running', stagnation_count
        if rms <= rms_prev * 0.95:
            return 'running', 0

        stagnation_count += 1
        if stagnation_count < 3:
            return 'running', stagnation_count
        if rms > initial_rms * 0.5:
            return 'failed', stagnation_count
        return 'converged', stagnation_count

    def _apply_internal_newton_step(self, residual):
        """Apply one Cartesian Newton correction for the current IC residual."""
        dx = np.linalg.lstsq(
            self.int.jacobian(),
            residual,
            rcond=_LSTSQ_RCOND,
        )[0].reshape((-1, 3))
        self.atoms.positions += dx[:len(self.atoms)]
        self.dummies.positions += dx[len(self.atoms):]

    def _bad_internals_after_iterative_step(self):
        self.bad_int = self.int.check_for_bad_internals()
        if self.bad_int is None:
            return False
        self.bad_int = None
        return True

    def _iterative_final_result(self, target, x0, dx_initial, g0):
        """Return the realized step only if the final residual avoids ODE fallback."""
        current = self.get_x()
        final_residual = self.wrap_dx(target - current, origin=current)
        final_rms = np.linalg.norm(final_residual) / np.sqrt(len(dx_initial))
        if final_rms > 1e-6:
            return None
        dx_final = self.wrap_dx(current - x0, origin=x0)
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    def _set_x_iterative(self, target, max_iter=20):
        """Fast iterative stepper for internal coordinate updates.

        Uses Newton-Raphson iteration to update Cartesian positions to match
        target internal coordinates. Returns None if convergence fails.
        """
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        x0 = self.get_x()
        dx_initial = self.wrap_dx(target - x0, origin=x0)

        # Get initial gradient in Cartesian space.
        g0 = self._get_Binv() @ self.curr.get('g', np.zeros_like(dx_initial))

        rms_prev = np.inf
        initial_rms = None
        stagnation_count = 0

        for iteration in range(max_iter):
            residual, rms = self._iterative_residual_rms(target)
            if initial_rms is None:
                initial_rms = rms

            status, stagnation_count = self._iterative_step_status(
                iteration, rms, initial_rms, rms_prev, stagnation_count
            )
            if status == 'converged':
                break
            if status == 'failed':
                self._restore_internal_positions(pos0, dpos0)
                return None

            rms_prev = rms
            self._apply_internal_newton_step(residual)
            if self._bad_internals_after_iterative_step():
                self._restore_internal_positions(pos0, dpos0)
                return None

        result = self._iterative_final_result(target, x0, dx_initial, g0)
        if result is None:
            self._restore_internal_positions(pos0, dpos0)
        return result

    def _set_x_ode(self, target):
        """ODE-based stepper for internal coordinate updates.

        Uses LSODA to integrate the geodesic equation for reliable convergence
        on large or ill-conditioned steps.
        """
        x0 = self.get_x()
        dx = self.wrap_dx(target - x0, origin=x0)
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv
        self._ode_Binv_gpu = (
            _gpu_left_factor(Binv) if not self.exact_geodesic else None
        )
        g_current = self.curr.get('g')
        if g_current is None:
            g_current = np.zeros_like(dx)
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ g_current))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
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

        Uses ODE integrator by default. If iterative_stepper is enabled,
        tries Newton-Raphson first and falls back to ODE on failure.
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
        delta_proj = self.wrap_dx(
            self.int.calc() - q_after_ode, origin=q_after_ode
        )
        filter_rows = getattr(self, '_last_projection_filter_rows', ())
        if filter_rows:
            delta_proj[np.asarray(filter_rows, dtype=int)] = 0.0
            self._last_projection_filter_rows = ()
        return dx_int_final + delta_proj

    def _project_to_constraints(
        self, target_tol=_LINEAR_DUMMY_PROJECTION_TOL, max_iter=8,
        safety_limit=0.05,
    ):
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

        For the general constraint path, ``safety_limit`` caps
        ``|dx_cart|_inf`` per iteration. If the Newton step would exceed it,
        we bail and accept the partial
        residual — the alternative (damped re-iteration) was tested
        and found to *increase* opt step counts (~+30%) on the tier1+2
        benchmarks because the partial corrections accumulate as
        Hessian noise. Bailing leaves the projection as a strict
        improvement: it can only help, never hurt.
        """
        self._last_projection_filter_rows = ()
        if self.cons.residual().size == 0:
            return False

        r = self.cons.residual()
        if np.linalg.norm(r, ord=np.inf) < target_tol:
            return False

        self._last_linear_dummy_projection_mode = None
        moved = self._project_linear_bend_dummies(
            target_tol=target_tol,
        )
        if moved is not None:
            return moved

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

            s, *_ = np.linalg.lstsq(drdx @ Ucons, -r, rcond=_LSTSQ_RCOND)
            dq_int = Ucons @ s                       # IC-space step (n_int,)
            dx = self._get_Binv() @ dq_int            # Cartesian (n_cart,)

            if np.linalg.norm(dx, ord=np.inf) > safety_limit:
                return moved  # would override optimizer's step — bail

            self.atoms.positions += dx[:n_real].reshape(-1, 3)
            if n_dummy > 0:
                self.dummies.positions += dx[n_real:n_real + n_dummy].reshape(-1, 3)
            moved = True

        return moved

    def _dummy_dihedral_values(self):
        natoms = self.int.natoms
        vals = []
        atoms = self.int.all_atoms
        for coord, active in zip(self.int.internals['dihedrals'],
                                 self.int._active['dihedrals']):
            if active and np.any(np.asarray(coord.indices) >= natoms):
                vals.append(float(coord.calc(atoms)))
        return np.asarray(vals)

    def _dummy_containing_coord_rows(self):
        natoms = self.int.natoms
        rows = []
        row = 0
        for name in self.int._names:
            for coord, active in zip(self.int.internals[name],
                                     self.int._active[name]):
                if not active:
                    continue
                if np.any(np.asarray(coord.indices) >= natoms):
                    rows.append(row)
                row += 1
        return tuple(sorted(set(rows)))

    def _dummy_projection_gauge_filter_rows(self):
        natoms = self.int.natoms
        rows = []
        row = 0
        for name in self.int._names:
            for coord, active in zip(self.int.internals[name],
                                     self.int._active[name]):
                if not active:
                    continue
                if (name in ('translations', 'rotations')
                        and np.any(np.asarray(coord.indices) >= natoms)):
                    rows.append(row)
                row += 1
        return tuple(sorted(set(rows)))

    def _linear_dummy_auxiliary_optimizer_rows(self):
        records = self._linear_bend_dummy_projection_records()
        if records is None:
            return ()
        auxiliaries = {
            name: tuple(
                record[index]
                for record in records
            )
            for name, index in (('bonds', 2), ('angles', 4))
        }
        rows = []
        row = 0
        for name in self.int._names:
            for coord, active in zip(self.int.internals[name],
                                     self.int._active[name]):
                if not active:
                    continue
                if (name in auxiliaries
                        and any(coord == auxiliary
                                for auxiliary in auxiliaries[name])):
                    rows.append(row)
                row += 1
        return tuple(rows)

    @staticmethod
    def _wrapped_angle_delta(after, before):
        return (after - before + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _project_one_linear_dummy_on_cone(positions, cell, dummy, parent,
                                          length, angle_coord, theta,
                                          dummy_is_first):
        ids = np.asarray(angle_coord.indices, dtype=int)
        tvecs = angle_coord.kwargs['ncvecs'] @ cell

        if dummy_is_first:
            neighbor_vec = positions[ids[2]] - positions[parent] + tvecs[1]
            dummy_shift = tvecs[0]
            current = positions[dummy] - positions[parent] - dummy_shift
        else:
            neighbor_vec = positions[ids[0]] - positions[parent] - tvecs[0]
            dummy_shift = -tvecs[1]
            current = positions[dummy] - positions[parent] - dummy_shift

        neighbor_norm = np.linalg.norm(neighbor_vec)
        if neighbor_norm <= 1e-12:
            return False
        axis = neighbor_vec / neighbor_norm
        perp = current - (current @ axis) * axis
        perp_norm = np.linalg.norm(perp)
        if perp_norm <= 1e-12:
            fallback = np.zeros(3)
            fallback[np.argmin(np.abs(axis))] = 1.0
            perp = fallback - (fallback @ axis) * axis
            perp_norm = np.linalg.norm(perp)
        azimuth = perp / perp_norm

        new_vec = length * (
            np.cos(theta) * axis + np.sin(theta) * azimuth
        )
        positions[dummy] = positions[parent] + dummy_shift + new_vec
        return True

    def _linear_bend_dummy_projection_records(self):
        """Return dummy projection records, or None for the general path."""
        if self.cons.nint == 0 or len(self.dummies) == 0:
            return None

        natoms = self.int.natoms
        parent_of = self._linear_bend_dummy_parent_map()
        if parent_of is None:
            return None

        bond_rec = {}
        angle_rec = {}
        if not self._linear_bend_dummy_constraints_are_projectable():
            return None

        for coord, active, target in zip(self.cons.internals['bonds'],
                                         self.cons._active['bonds'],
                                         self.cons._targets['bonds']):
            if not active:
                continue
            ids, reals, dummies = self._split_real_dummy_ids(
                coord.indices, natoms
            )
            if len(dummies) != 1 or len(reals) != 1:
                return None
            dummy = int(dummies[0])
            parent = int(reals[0])
            if parent_of.get(dummy) != parent or dummy in bond_rec:
                return None
            bond_rec[dummy] = (parent, coord, float(target))

        for coord, active, target in zip(self.cons.internals['angles'],
                                         self.cons._active['angles'],
                                         self.cons._targets['angles']):
            if not active:
                continue
            ids, _, dummies = self._split_real_dummy_ids(coord.indices, natoms)
            if len(dummies) != 1 or ids[1] >= natoms:
                return None
            dummy = int(dummies[0])
            parent = int(ids[1])
            if parent_of.get(dummy) != parent or dummy in angle_rec:
                return None
            other = int(ids[2] if ids[0] == dummy else ids[0])
            if other >= natoms:
                return None
            angle_rec[dummy] = (coord, float(target), ids[0] == dummy)

        if (set(bond_rec) != set(angle_rec)
                or set(bond_rec) != set(parent_of)):
            return None

        return [
            (dummy, bond_rec[dummy][0], bond_rec[dummy][1],
             bond_rec[dummy][2],
             angle_rec[dummy][0], angle_rec[dummy][1], angle_rec[dummy][2])
            for dummy in sorted(bond_rec)
        ]

    def _linear_bend_dummy_parent_map(self):
        parent_of = {
            int(dummy): int(parent)
            for parent, dummy in enumerate(self.int.dinds)
            if dummy >= 0
        }
        if len(parent_of) != len(self.dummies):
            return None
        return parent_of

    def _linear_bend_dummy_constraints_are_projectable(self):
        for name in self.cons._names:
            for active, kind in zip(self.cons._active[name],
                                    self.cons._kind[name]):
                if active and (kind != 'eq' or name not in ('bonds', 'angles')):
                    return False
        return True

    @staticmethod
    def _split_real_dummy_ids(indices, natoms):
        ids = np.asarray(indices, dtype=int)
        return ids, ids[ids < natoms], ids[ids >= natoms]

    def _project_linear_bend_dummies(
        self, target_tol=_LINEAR_DUMMY_PROJECTION_TOL,
    ):
        """Project linear-bend dummy constraints without a full IC basis.

        A linear-bend dummy contributes exactly two active constraints: the
        parent-dummy bond length and one parent-neighbor-dummy angle. Holding
        real atoms fixed, those constraints are satisfied by putting the dummy
        on a fixed cone around the parent-neighbor direction while preserving
        the current azimuth. This avoids a full constrained-basis/QR solve for
        the common dummy-only constraint case and falls back otherwise.
        """
        records = self._linear_bend_dummy_projection_records()
        if records is None:
            return None

        old_dpos = self.dummies.positions.copy()
        dummy_dihedrals0 = self._dummy_dihedral_values()
        all_positions = np.vstack([self.atoms.positions,
                                   self.dummies.positions])
        cell = self.atoms.cell.array

        for (dummy, parent, bond_coord, length, angle_coord, theta,
             dummy_is_first) in records:
            if length <= 0 or theta <= 1e-8 or abs(np.pi - theta) <= 1e-8:
                self.dummies.positions[:] = old_dpos
                return None

            if not self._project_one_linear_dummy_on_cone(
                all_positions, cell, dummy, parent, length, angle_coord,
                theta, dummy_is_first,
            ):
                self.dummies.positions[:] = old_dpos
                return None

        self.dummies.positions[:] = all_positions[self.int.natoms:]
        residual = self.cons.residual()
        if np.linalg.norm(residual, ord=np.inf) >= target_tol:
            self.dummies.positions[:] = old_dpos
            return None
        dummy_dihedrals1 = self._dummy_dihedral_values()
        if dummy_dihedrals0.size:
            ddih = self._wrapped_angle_delta(dummy_dihedrals1,
                                             dummy_dihedrals0)
            if not np.all(np.isfinite(ddih)):
                self.dummies.positions[:] = old_dpos
                return None
            self._last_linear_dummy_projection_ddih = float(
                np.linalg.norm(ddih, ord=np.inf)
            )
        else:
            self._last_linear_dummy_projection_ddih = 0.0
        self._last_linear_dummy_projection_mode = 'cone'
        self._last_projection_filter_rows = (
            self._dummy_projection_gauge_filter_rows()
        )
        return True

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

    def _fast_dummy_selection_indices(self):
        key = 'fast_dummy_selection_indices'
        if key in self.curr:
            return self.curr[key]

        indices = None
        Unred = self.curr.get('Unred')
        Ucons = self.curr.get('Ucons')
        if (self.curr.get('Ufree') is Unred
                and Unred is not None
                and Unred.shape[0] == Unred.shape[1]
                and Ucons is not None
                and Ucons.shape[1] > 0):
            auxiliary = np.argmax(np.abs(Ucons), axis=0)
            indexed_normals = np.eye(self.dim)[:, auxiliary]
            if (len(np.unique(auxiliary)) == Ucons.shape[1]
                    and np.array_equal(Ucons, indexed_normals)):
                mask = np.ones(self.dim, dtype=bool)
                mask[auxiliary] = False
                indices = np.flatnonzero(mask)

        self.curr[key] = indices
        return indices

    def _fast_dummy_redundant_projection(self):
        key = 'fast_dummy_redundant_projection'
        if key in self.curr:
            return self.curr[key]

        projection = None
        Unred = self.curr.get('Unred')
        Ucons = self.curr.get('Ucons')
        if (self.curr.get('Ufree') is Unred
                and Unred is not None
                and Ucons is not None
                and Ucons.shape[1] > 0
                and self._fast_dummy_selection_indices() is None):
            normals = Unred.T @ Ucons
            projection = _LowRankTangentProjection(Unred, normals)
        self.curr[key] = projection
        return projection

    def get_scons(self):
        # Certified linear-dummy constraints are restored by the cone
        # projection after each geometry step. Avoid a least-squares solve for
        # a correction that is already below that projection's target.
        if len(self.dummies):
            self._update(False)
            fast_dummy = (
                self._fast_dummy_selection_indices() is not None
                or self._fast_dummy_redundant_projection() is not None
            )
            if fast_dummy:
                residual = self.get_res()
                if (residual.size == 0
                        or np.linalg.norm(residual, ord=np.inf)
                        < _LINEAR_DUMMY_PROJECTION_TOL):
                    return np.zeros(self.dim)
        return super().get_scons()

    def get_Ufree(self):
        self._update(False)
        indices = self._fast_dummy_selection_indices()
        projection = self._fast_dummy_redundant_projection()
        if indices is None and projection is None:
            return self.curr['Ufree']
        cached = self.curr.get('Ufree_materialized')
        if cached is None:
            if indices is not None:
                cached = np.eye(self.dim)[:, indices]
            else:
                drdxnred = self.curr['drdx'] @ self.curr['Unred']
                _, Vfree = _split_cons_subspace(drdxnred)
                cached = self.curr['Unred'] @ Vfree
            self.curr['Ufree_materialized'] = cached
        return cached

    def _project_free_vector(self, vector):
        self._update(False)
        indices = self._fast_dummy_selection_indices()
        projection = self._fast_dummy_redundant_projection()
        if indices is None and projection is None:
            Ufree = self.curr['Ufree']
            return Ufree @ (Ufree.T @ vector)
        if projection is not None:
            return projection.T @ (projection @ vector)
        projected = np.zeros_like(vector)
        projected[indices] = np.asarray(vector)[indices]
        return projected

    def get_fast_restricted_step_data(self, g, order, stepper, W_is_identity):
        if (not W_is_identity
                or getattr(stepper, '__name__', '') != 'QuasiNewton'
                or len(self.dummies) == 0):
            return None
        self._update(False)
        indices = self._fast_dummy_selection_indices()
        projection = self._fast_dummy_redundant_projection()

        if indices is not None:
            if not getattr(self.H, '_cpu_current', True):
                return None
            # Each certified constraint is literally q_aux - target, so its
            # Hessian with respect to q is zero. Computing get_Hc() would only
            # form and cancel Cartesian chain-rule terms.
            H = self.H.asarray()[np.ix_(indices, indices)].copy()
            P = indices
            g_step = np.asarray(g)[indices]
        elif projection is not None and order == 0:
            Unred = projection.basis
            H_gpu = self.H._get_B_gpu()
            H_B = self.H.B if getattr(self.H, '_cpu_current', True) else None
            if H_B is None and H_gpu is None:
                Hred = np.eye(Unred.shape[1])
            else:
                Hred = gpu_project(H_B, Unred, H_gpu=H_gpu)
            C = projection.normals
            HC = Hred @ C
            CtHC = C.T @ HC
            H = (Hred - HC @ C.T - C @ HC.T
                 + C @ CtHC @ C.T + C @ C.T)
            H = (H + H.T) / 2.0
            P = projection
            g_step = projection @ g
        else:
            return None

        H_step = ApproximateHessian(
            H.shape[0], 0, H, self.H.update_method, self.H.symm
        )
        return P, g_step, H_step

    def _compute_basis_int(self):
        """Compute the internal-coords-only basis (uncached, fast path).

        Uses the cached jacobian QR factors. Subclasses (CellInternalPES) call
        this to obtain the internal block, then add their own cell extension
        and cache the combined result themselves.
        """
        Q, R = self._get_jacobian_qr()
        n_int = Q.shape[0]
        cons_jac = self.cons.jacobian()
        if cons_jac.shape[0] == 0:
            # No constraints: every non-redundant DOF is free.
            return np.zeros((0, n_int)), np.zeros((n_int, 0)), Q, Q
        if R.shape[0] == R.shape[1]:
            # Full rank: cons_jac @ R^{-1} via triangular solve.
            drdxnred = solve_triangular(
                R.T, cons_jac.T, lower=True, check_finite=False
            ).T
        else:
            # Rank-deficient (SVD-of-R fallback in _get_jacobian_qr).
            drdxnred = cons_jac @ (self._get_Binv() @ Q)
        drdx = drdxnred @ Q.T
        records = self._linear_bend_dummy_projection_records()
        if (Q.shape[0] == Q.shape[1]
                and R.shape[0] == R.shape[1]
                and records is not None):
            auxiliary = self._linear_dummy_auxiliary_optimizer_rows()
            # The records check proves that every active constraint is the
            # corresponding bond or angle in ``self.int``.  With a square,
            # full-rank internal Jacobian those coordinates are independent,
            # so their exact tangent complement is obtained by simply omitting
            # their coordinate axes.  No geometry-dependent cutoff is needed.
            if (len(auxiliary) == drdx.shape[0]
                    and len(set(auxiliary)) == len(auxiliary)
                    and all(0 <= row < n_int for row in auxiliary)):
                identity = np.eye(n_int)
                Ucons = identity[:, np.asarray(auxiliary, dtype=int)]
                return drdx, Ucons, identity, identity
        elif records is not None:
            Vcons = _constraint_normal_subspace(drdxnred)
            if Vcons is not None:
                Ucons = Q @ Vcons
                return drdx, Ucons, Q, Q
        Vcons, Vfree = _split_cons_subspace(drdxnred)
        return drdx, Q @ Vcons, Q, Q @ Vfree

    def _calc_basis(self):
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

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        dx_proj = Unred @ dx_r
        Hdx = _hessian_matvec(H, dx_proj)
        return g.T @ dx_proj + (dx_proj.T @ Hdx) / 2.

    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_projected = self._project_free_vector(g)
        # Use cached jacobian from curr if available
        if 'B' in self.curr and self.curr['B'] is not None:
            B = self.curr['B']
        else:
            B = self.int.jacobian()
        return -(g_projected @ B).reshape((-1, 3))

    def wrap_dx(self, dx, origin=None):
        return self.int.wrap(dx, origin=origin)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        # Use direct HVP contractions instead of forming full Hessians or the
        # sparse D_rdot matrix.  The ODE only needs D_rdot @ [dxdt, g].
        if self.exact_geodesic:
            Binv = self._get_Binv()
            Binv_gpu = None
        else:
            Binv = self._ode_Binv
            Binv_gpu = getattr(self, '_ode_Binv_gpu', None)
        rhs = np.column_stack((dxdt, g))     # (ndof, 2)
        hrdot = self.int.hessian_rdot_mat(dxdt, rhs)
        out = None
        if Binv_gpu is not None:
            out = _gpu_left_matmul(None, hrdot, A_gpu=Binv_gpu)
        if out is None:
            out = Binv @ hrdot
        out = -out  # (ndof, 2)
        dydt[1] = out[:, 0]
        dydt[2] = out[:, 1]

        return dydt.ravel()

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
        Ui, Si, VTi = _robust_svd(B, full_matrices=True)
        # Relative rank threshold (was an absolute 1e-6 here, inconsistent with
        # the other SVD sites; matters only for poorly-scaled Jacobians).
        nnred = _svd_rank(Si)
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


class _CellPESMixin:
    """Shared cell-parameterization logic for CellInternalPES and CellCartesianPES.

    Provides the log-deformation cell parameterization, Niggli reduction,
    finite-difference Hessian refinement, and save/restore of cell state.

    Subclasses must define:
    - ``n_coords`` property: number of non-cell DOF (n_internal or n_cart)
    - ``_scale_atoms_on_cell_change`` property: whether set_cell uses
      scale_atoms=True (fractional coords fixed) or False
    """

    @property
    def n_coords(self) -> int:
        raise NotImplementedError

    @property
    def _scale_atoms_on_cell_change(self) -> bool:
        return False

    def _init_cell_params(
        self, atoms, exp_cell_factor, cell_mask, scalar_pressure,
    ):
        self.orig_cell = atoms.get_cell().array.copy()
        self._exp_cell_factor_request = exp_cell_factor
        if exp_cell_factor is None:
            exp_cell_factor = float(len(atoms))
        self.exp_cell_factor = exp_cell_factor
        if cell_mask is None:
            cell_mask = np.ones((3, 3), dtype=bool)
        self.cell_mask = np.asarray(cell_mask, dtype=bool).reshape((3, 3))
        self.n_cell_dof = int(self.cell_mask.sum())
        self.scalar_pressure = scalar_pressure

    def _state_hash(self) -> bytes:
        h = super()._state_hash()
        h += np.float64(self.scalar_pressure).tobytes()
        return h

    def _get_deformation_gradient(self) -> np.ndarray:
        """Get current deformation gradient F = cell @ inv(orig_cell)."""
        return self.atoms.get_cell().array @ np.linalg.inv(self.orig_cell)

    def _get_log_deform(self) -> np.ndarray:
        """Get log of deformation gradient, scaled by exp_cell_factor."""
        F = self._get_deformation_gradient()
        return _logm_3x3(F) * self.exp_cell_factor

    def _set_cell_from_log_deform(self, log_deform_scaled: np.ndarray) -> None:
        """Set cell from scaled log-deformation gradient."""
        log_deform = log_deform_scaled / self.exp_cell_factor
        F = expm(log_deform.real)
        new_cell = F @ self.orig_cell
        self.atoms.set_cell(new_cell,
                            scale_atoms=self._scale_atoms_on_cell_change)

    def _masked_cell_params(self) -> np.ndarray:
        """Get cell parameters as flat array (only free DOF)."""
        log_deform = self._get_log_deform()
        return log_deform[self.cell_mask]

    def _set_masked_cell_params(self, params: np.ndarray) -> None:
        """Set cell from flat array of free DOF."""
        log_deform = self._get_log_deform()
        log_deform[self.cell_mask] = params
        self._set_cell_from_log_deform(log_deform)

    def save(self):
        """Save current state including cell."""
        super().save()
        self.savepoint['cell'] = self.atoms.get_cell().array.copy()

    def restore(self):
        """Restore saved state including cell."""
        super().restore()
        if 'cell' in self.savepoint:
            self.atoms.set_cell(self.savepoint['cell'], scale_atoms=False)

    def _compute_cell_hessian_columns(self, delta: float,
                                       forward: bool = False) -> np.ndarray:
        """Compute Hessian columns for cell DOF via finite differences.

        ``delta`` is interpreted in physical log(F) space and scaled by
        ``exp_cell_factor`` so the perturbation magnitude is independent
        of the cell parameterization scaling.

        Parameters
        ----------
        delta : float
            Step size in physical log(F) space.
        forward : bool
            If True, use forward differences (n_cell_dof evals) instead
            of central differences (2 * n_cell_dof evals).

        Returns array of shape (dim, n_cell_dof).
        """
        n = self.n_coords
        indices = list(range(n, n + self.n_cell_dof))
        delta_u = delta * self.exp_cell_factor
        return self._fd_hessian_columns(
            delta_u, indices, self.dim, "cell", forward=forward
        )

    def maybe_niggli_reduce(self, angle_threshold=30.0):
        """Apply Niggli reduction if cell angles deviate too far from 90 deg.

        Returns True if reduction was applied.
        """
        angles = self.atoms.get_cell().angles()
        max_deviation = max(abs(a - 90.0) for a in angles)
        if max_deviation <= angle_threshold:
            return False

        H = self.H.asarray().copy()
        n = self.n_coords
        T_masked = _niggli_hessian_transform(
            self.atoms, self.orig_cell, self.exp_cell_factor, self.cell_mask
        )
        H_cell_new = T_masked.T @ H[n:, n:] @ T_masked
        H[n:, n:] = H_cell_new
        H[:n, n:] = H[:n, n:] @ T_masked
        H[n:, :n] = T_masked.T @ H[n:, :n]

        self.orig_cell = self.atoms.get_cell().array.copy()
        self.set_H(H, initialized=True)
        self.curr = dict(x=None, f=None, g=None)
        self.last = self.curr.copy()
        return True

    def refine_hessian(self, refine_level: int = 1, delta: float = 1e-4):
        """Re-refine Hessian cell blocks via finite differences."""
        if refine_level < 1:
            return
        H = self.H.asarray()
        n = self.n_coords
        H_cell_cols = self._compute_cell_hessian_columns(delta)
        H[:n, n:] = H_cell_cols[:n, :]
        H[n:, :n] = H_cell_cols[:n, :].T
        H_cell_cell = H_cell_cols[n:, :]
        H[n:, n:] = (H_cell_cell + H_cell_cell.T) / 2
        self.set_H(H, initialized=True)
        logger.info("Hessian re-refined at level %d", refine_level)

    def _init_cell_hessian(self, H_old, refine_initial_hessian, hessian_delta,
                           save_hessian, H_coords_default):
        """Build the block-diagonal initial Hessian (shared init logic).

        H_coords_default is used when H_old is None (caller-specific guess).
        """
        H0_full = np.zeros((self.dim, self.dim))
        n = self.n_coords
        if H_old is not None:
            H0_full[:n, :n] = H_old
        else:
            H0_full[:n, :n] = H_coords_default

        refine_level = self._coerce_refine_level(refine_initial_hessian)

        if refine_level == 0:
            H0_full[n:, n:] = np.eye(self.n_cell_dof)
        else:
            H_cell_cols = self._compute_cell_hessian_columns(hessian_delta, forward=refine_level < 0)
            H0_full[:n, n:] = H_cell_cols[:n, :]
            H0_full[n:, :n] = H_cell_cols[:n, :].T
            H_cell_cell = H_cell_cols[n:, :]
            H0_full[n:, n:] = (H_cell_cell + H_cell_cell.T) / 2

        if save_hessian is not None:
            np.save(save_hessian, H0_full)
            logger.info("Initial Hessian saved to %s", save_hessian)

        return H0_full, refine_level

    def _install_cell_hessian(self, H_coords_or_full):
        H_coords_or_full = np.asarray(H_coords_or_full, dtype=np.float64)
        n = self.n_coords
        if H_coords_or_full.shape == (self.dim, self.dim):
            H_full = H_coords_or_full
        elif H_coords_or_full.shape == (n, n):
            H_full = self.H.asarray().copy()
            H_full[:n, :n] = H_coords_or_full
        else:
            raise ValueError(
                'hessian_function returned shape {}, expected {} or {}'
                .format(
                    H_coords_or_full.shape,
                    (n, n),
                    (self.dim, self.dim),
                )
            )
        self.set_H(H_full, initialized=True)

    def get_drdx(self):
        """Get constraint Jacobian extended with zeros for cell DOF."""
        drdx_coords = super().get_drdx()
        n_cons = drdx_coords.shape[0]
        drdx = np.zeros((n_cons, self.dim))
        drdx[:, :self.n_coords] = drdx_coords
        return drdx

    def _virial_to_dEdF(self, virial, forces):
        """Convert the virial V*sigma to dE/dF. Subclass-specific.

        CellInternalPES adds the rigid-fragment corrections; CellCartesianPES
        uses the fixed-Cartesian form. Must be overridden.
        """
        raise NotImplementedError

    def _stress_to_cell_gradient(self, stress_voigt, forces=None):
        """Convert a stress tensor to the gradient w.r.t. the log-deformation
        cell parameters.

        Shared prologue/epilogue for both cell PES flavors: build the virial
        (with external pressure), delegate the virial->dE/dF step to the
        subclass ``_virial_to_dEdF`` hook, then map dE/dF -> dE/dU = dE/d(logF)
        via the Frechet derivative of expm and apply the cell mask / scaling.
        """
        volume = self.atoms.get_volume()
        stress_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)

        # Add external pressure contribution
        if self.scalar_pressure != 0.0:
            stress_3x3 += self.scalar_pressure * np.eye(3)

        # The virial V*sigma is the base term
        virial = volume * stress_3x3

        dEdF = self._virial_to_dEdF(virial, forces)

        # Convert dE/dF to dE/dU via Frechet derivative of expm
        F = self._get_deformation_gradient()
        U = _logm_3x3(F)
        g_cell_3x3 = _expm_frechet_3x3_contracted(U, dEdF)

        # Apply cell mask and scale
        g_cell_3x3 = g_cell_3x3 * self.cell_mask
        g_cell_3x3 = g_cell_3x3 / self.exp_cell_factor

        return g_cell_3x3[self.cell_mask]

    def _effective_stress_convergence_max(self, g_cell):
        """Maximum effective stress for user-specified ``smax``.

        ``g_cell`` is Sella's corrected gradient with respect to scaled
        log-cell coordinates.  Converting it back to eV/A^3 keeps explicit
        ``smax`` on the same cell-motion path that Sella optimizes, including
        rigid-fragment and constraint-projection corrections.
        """
        if self.n_cell_dof == 0 or len(g_cell) == 0:
            return 0.0
        effective_stress = (
            np.asarray(g_cell) * self.exp_cell_factor
            / self.atoms.get_volume()
        )
        return np.abs(effective_stress).max()

    def _cell_convergence_max(self, g_cell, smax):
        if smax is None:
            return np.abs(g_cell).max() if len(g_cell) > 0 else 0.0
        return self._effective_stress_convergence_max(g_cell)

    def _extend_basis_with_cell(self, basis_int):
        """Pad a coordinate-only basis with cell DOF (identity in Unred/Ufree)."""
        drdx_int, Ucons_int, Unred_int, Ufree_int = basis_int
        n_int = drdx_int.shape[1]
        n_total = n_int + self.n_cell_dof

        # Cell DOF are not constrained, so they're all in Ufree
        drdx = np.zeros((drdx_int.shape[0], n_total))
        drdx[:, :n_int] = drdx_int

        Ucons = np.zeros((n_total, Ucons_int.shape[1]))
        Ucons[:n_int, :] = Ucons_int

        Unred = self._pad_with_cell_identity(Unred_int)

        # When there are no constraints, the internal basis returns
        # ``Ufree is Unred`` (same object). Share the extended array too —
        # avoids a second slice copy on every call.
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


class CellInternalPES(_CellPESMixin, InternalPES):
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
        Scaling factor for cell parameterization. Default is sqrt(n_atoms)
        when rigid_fragments is active, else n_atoms.
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
    refine_initial_hessian : bool or int, optional
        Level of Hessian refinement via finite differences:
        - False or 0: No refinement (default)
        - True or 1: Central differences (2 * n_cell_dof force evals)
        - -1: Forward differences (n_cell_dof force evals, half the cost)
    hessian_delta : float, optional
        Finite difference step size for Hessian refinement. Default is 1e-4.
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
        hessian_delta: float = 1e-4,
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
            - -1: Forward differences for cell blocks (n_cell_dof evals)
            - True or 1: Central differences for cell blocks (2 * n_cell_dof evals)
            - 2: Also refine translation/rotation blocks (adds 2 * n_tric evals)
            - 3: Refine full internal Hessian (2 * n_internal evals, expensive!)
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        self._init_cell_params(atoms, exp_cell_factor, cell_mask,
                               scalar_pressure)

        # Store rigid_fragments request; auto-detection deferred until after
        # parent init populates translations via find_all_bonds()
        self._rigid_fragments_request = rigid_fragments

        self._initializing = True
        self.n_internal = None

        InternalPES.__init__(self, atoms, internals, *args, H0=H0, **kwargs)

        self.n_internal = self.dim

        if self._rigid_fragments_request is None:
            # Only genuine fragments (from find_all_bonds with
            # allow_fragments=True) set fragment_atom_groups. Constraint-derived
            # translations (e.g. FixAtoms/FixCartesian copied into internals)
            # also produce Translation coordinates, but those are not rigid
            # fragments and must not trigger the rigid-fragment cell path.
            self.rigid_fragments = bool(
                getattr(self.int, 'fragment_atom_groups', None)
            )
        else:
            self.rigid_fragments = self._rigid_fragments_request

        # When rigid_fragments is active and the user didn't specify
        # exp_cell_factor, use sqrt(n_atoms).  Cell strain doesn't deform
        # intramolecular bonds, so the effective stiffness scales more
        # weakly with system size than the n_atoms default.
        if self.rigid_fragments and self._exp_cell_factor_request is None:
            self.exp_cell_factor = float(np.sqrt(len(atoms)))

        if self.rigid_fragments:
            self.fragment_groups, self.fragment_dummy_groups = \
                self._extract_fragment_groups(self.int)
            self._validate_fragment_groups()

        self.dim = self.n_internal + self.n_cell_dof

        self._Hc_cell_cache = _LRU2()
        self._cell_basis_cache = _LRU2()
        self._initializing = False

        H_old = self.H.B if self.H is not None and self.H.B is not None else None
        if H_old is None:
            B = self.int.jacobian()
            P = _range_space_projector(B)
            H_coords_default = P @ self.int.guess_hessian() @ P
        else:
            H_coords_default = None

        H0_full, refine_level = self._init_cell_hessian(
            H_old, refine_initial_hessian, hessian_delta,
            save_hessian, H_coords_default,
        )
        self.initial_hessian_refinement_level = refine_level

        if refine_level >= 2:
            H_tric_cols = self._compute_tric_hessian_columns(hessian_delta)
            tric_indices = self._get_tric_indices()
            for i, idx in enumerate(tric_indices):
                H0_full[:, idx] = H_tric_cols[:, i]
                H0_full[idx, :] = H_tric_cols[:, i]

        if refine_level >= 3:
            H_int_cols = self._compute_internal_hessian_columns(hessian_delta)
            H0_full[:self.n_internal, :self.n_internal] = (H_int_cols + H_int_cols.T) / 2

        self.set_H(H0_full, initialized=(refine_level == 0))

    @property
    def n_coords(self) -> int:
        return self.n_internal

    @property
    def _scale_atoms_on_cell_change(self) -> bool:
        return not self.rigid_fragments

    def maybe_niggli_reduce(self, angle_threshold=30.0):
        """Niggli reduction is unsafe with internal coordinates; skip it.

        Niggli reduction recombines the lattice vectors and wraps atoms
        between periodic images. That invalidates internal-coordinate state
        that Cartesian PES does not carry:

        - Stored bond/angle/dihedral ``ncvecs`` are integer image offsets
          interpreted against the cell (``ncvecs @ cell``). Recombining the
          lattice changes which physical image each offset points to, so the
          coordinates silently jump (e.g. a 0.5 A periodic bond becomes
          5.5 A) unless the offsets are remapped by the integer lattice
          transform -- which ASE's ``niggli_reduce`` does not expose.
        - Dummy atoms (added for linear angles) are not part of the ASE atoms
          object, so they are neither wrapped nor basis-transformed and are
          left at stale Cartesian positions.

        Reconstructing this correctly amounts to re-detecting the internals,
        so we instead decline the reduction. Cell optimization still proceeds
        via the log-deformation parameterization; it just does not get the
        compactness benefit of Niggli. ``CellCartesianPES`` keeps the
        reduction, where atom positions are the coordinates and Niggli is
        self-consistent.
        """
        return False

    def refine_hessian(self, refine_level: int = 1, delta: float = 1e-4):
        """Re-refine Hessian blocks (extends base with levels 2-3)."""
        _CellPESMixin.refine_hessian(self, min(refine_level, 1), delta)
        if refine_level < 2:
            return
        H = self.H.asarray()
        if refine_level >= 2:
            H_tric_cols = self._compute_tric_hessian_columns(delta)
            tric_indices = self._get_tric_indices()
            for i, idx in enumerate(tric_indices):
                H[:, idx] = H_tric_cols[:, i]
                H[idx, :] = H_tric_cols[:, i]
        if refine_level >= 3:
            H_int_cols = self._compute_internal_hessian_columns(delta)
            H[:self.n_internal, :self.n_internal] = (H_int_cols + H_int_cols.T) / 2
        self.set_H(H, initialized=True)
        logger.info("Hessian re-refined at level %d", refine_level)

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

    def _validate_fragment_groups(self):
        """Ensure rigid_fragments has usable fragment groups.

        The rigid cell move iterates over ``fragment_groups`` to translate and
        rotate each fragment. If that list is empty (e.g. rigid_fragments=True
        requested explicitly on an Internals built without allow_fragments, so
        no fragment TRICs exist), the loop is a no-op: the cell changes but the
        atoms stay fixed in Cartesian space -- a silent corruption. The groups
        must also partition every real atom exactly once, or some atoms would
        not move with the cell. Fail loudly instead.
        """
        natoms = self.int.natoms
        covered = []
        for group in self.fragment_groups:
            covered.extend(int(a) for a in group)
        if (not self.fragment_groups
                or sorted(covered) != list(range(natoms))):
            raise ValueError(
                "rigid_fragments=True requires fragment groups that partition "
                "all {} atoms exactly once, but got {}. Fragment groups come "
                "from find_all_bonds(allow_fragments=True); pass "
                "allow_fragments=True (or rigid_fragments=False for a "
                "non-rigid cell).".format(
                    natoms, [list(g) for g in self.fragment_groups]
                )
            )

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

    def _cell_target_parts(self, target, x0):
        """Split target into optimizer-requested internal and cell steps."""
        q0 = x0[:self.n_internal]
        dq = self.wrap_dx(
            target[:self.n_internal] - q0, origin=q0
        )
        cell_target = target[self.n_internal:]
        cell_params0 = self._masked_cell_params()
        return dq, cell_target, cell_params0

    def _move_rigid_fragments_after_cell_change(self, cell_before, pos_before):
        """Move rigid fragments after a cell update without changing shape.

        Rigid fragment mode translates fragment CoMs to maintain fractional
        positions, and rotates each fragment rigidly.

        For ASE row-vector cells a global rotation is
        ``cell_after = cell_before @ Q.T``, i.e. the incremental deformation
        acting on row vectors is the RIGHT factor
        ``M = inv(cell_before) @ cell_after`` (so a pure rotation gives M = Q.T
        and r_new = r @ Q.T rotates correctly). Using the left factor
        ``cell_after @ inv(cell_before)`` with ``@ R.T`` is only correct for
        orthogonal cells and breaks objectivity on skewed cells (a pure rotation
        would change the energy).
        """
        cell_after = self.atoms.get_cell().array
        cell_before_inv = np.linalg.inv(cell_before)
        M_inc = cell_before_inv @ cell_after
        R_inc, _ = polar(M_inc)
        for group, dgroup in zip(
            self.fragment_groups, self.fragment_dummy_groups
        ):
            com_old = pos_before[group].mean(axis=0)
            # Convert old CoM to fractional, then to new Cartesian.
            com_frac = com_old @ cell_before_inv
            com_new = com_frac @ cell_after
            # Rotate relative positions by R (row-vector: r_new = r @ R).
            delta_r = pos_before[group] - com_old
            self.atoms.positions[group] = com_new + delta_r @ R_inc
            # Move dummy atoms with the same transformation.
            if len(dgroup) > 0:
                didx = dgroup - self.int.natoms  # Convert to dummies index.
                delta_d = self.dummies.positions[didx] - com_old
                self.dummies.positions[didx] = com_new + delta_d @ R_inc

    def _move_nonrigid_dummies_after_cell_change(self, cell_before):
        """Apply ASE's fractional-preserving real-atom cell move to dummies."""
        if len(self.dummies) == 0:
            return

        # Non-rigid mode: ASE's set_cell(scale_atoms=True) scaled the real atoms
        # by the deformation gradient F = inv(cell_before) @ cell_after
        # (fractional coords preserved), but it does not know about dummy atoms.
        # Apply the same transform so dummies track the cell shear/scale instead
        # of being left behind.
        F = np.linalg.inv(cell_before) @ self.atoms.get_cell().array
        self.dummies.positions = self.dummies.positions @ F

    def _set_cell_and_move_attached_positions(self, cell_target):
        """Update the cell and move atoms/dummies according to fragment mode."""
        # Save cell before the change so we can transform atoms/dummies that
        # ASE's set_cell does not touch.
        cell_before = self.atoms.get_cell().array.copy()
        pos_before = (
            self.atoms.get_positions().copy() if self.rigid_fragments else None
        )

        # Update cell (scales atoms if rigid_fragments=False).
        self._set_masked_cell_params(cell_target)

        if self.rigid_fragments:
            self._move_rigid_fragments_after_cell_change(cell_before, pos_before)
        else:
            self._move_nonrigid_dummies_after_cell_change(cell_before)

    def _old_cell_gradient(self):
        """Cell block of the old gradient for Hessian parallel transport."""
        g_old = self.curr.get('g', None)
        if g_old is not None:
            return g_old[self.n_internal:].copy()
        return np.zeros(self.n_cell_dof)

    def _cell_only_step_result(self, dx_initial, cell_delta):
        """Return the Hessian-update tuple for a pure cell step."""
        # Cell-only case: dx_final equals the cell displacement.
        # Return actual cell gradient at starting position for proper Hessian
        # update (dg_actual = get_g() - g_par needs g_par to be the old gradient).
        return dx_initial, cell_delta.copy(), self._old_cell_gradient()

    def _solve_internal_after_cell(self, q_target):
        """Run internal-coordinate ODE, then project constraints if it succeeds."""
        # Now update atomic positions to match internal coordinate target.
        res = self._set_x_ode_internal(q_target)
        if res is None:
            return None, None, False

        # Project onto constraint manifold (decoupled from trust radius).
        # Only when ODE succeeded -- the InternalPES.set_x fallback below
        # runs its own projection internally.
        q_after_ode = self.int.calc().copy()
        proj_moved = self._project_to_constraints()
        return res, q_after_ode, proj_moved

    def _fallback_internal_step_after_cell(self, q_target, cell_delta,
                                           g_old_cell):
        """Fallback to parent internal set_x when the cell-aware ODE fails."""
        dx_int, _, g_int = InternalPES.set_x(self, q_target)
        dx_final = np.concatenate([dx_int, cell_delta])
        g_final = np.concatenate([g_int, g_old_cell])
        return dx_final, g_final

    def _ode_internal_step_after_cell(self, res, q_after_ode, proj_moved,
                                      cell_delta, g_old_cell):
        """Combine ODE internal step, projection delta, and cell secant data."""
        _, dx_int_final, g_int = res
        # Combine the ODE-tangent step with the projection's IC delta so BFGS
        # sees a coherent secant. When the projection didn't fire this is a pure
        # passthrough of dx_int_final.
        dx_int_realized = self._add_proj_delta(
            dx_int_final, q_after_ode, proj_moved
        )
        dx_final = np.concatenate([dx_int_realized, cell_delta])
        g_final = np.concatenate([g_int, g_old_cell])
        return dx_final, g_final

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
        dq, cell_target, cell_params0 = self._cell_target_parts(target, x0)
        cell_delta = cell_target - cell_params0

        self._set_cell_and_move_attached_positions(cell_target)

        # Read back internal coords AFTER the cell change moved atoms.
        # The solver targets q_after_cell + dq, not the raw q_target.
        # This ensures we don't undo the atom motion from the cell change.
        q_target = self.int.calc() + dq

        if self.n_internal == 0:
            return self._cell_only_step_result(dx_initial, cell_delta)

        res, q_after_ode, proj_moved = self._solve_internal_after_cell(q_target)
        g_old_cell = self._old_cell_gradient()
        if res is None:
            dx_final, g_final = self._fallback_internal_step_after_cell(
                q_target, cell_delta, g_old_cell
            )
        else:
            dx_final, g_final = self._ode_internal_step_after_cell(
                res, q_after_ode, proj_moved, cell_delta, g_old_cell
            )

        return dx_initial, dx_final, g_final

    def _set_x_ode_internal(self, q_target: np.ndarray):
        """ODE-based stepper for internal coords only (cell already updated)."""
        x0 = self.int.calc()
        dx = self.wrap_dx(q_target - x0, origin=x0)
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv
        self._ode_Binv_gpu = (
            _gpu_left_factor(Binv) if not self.exact_geodesic else None
        )

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
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
                raise RuntimeError("Geometry update ODE is taking too long!")

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

    def eval(self) -> tuple:
        """Evaluate energy and combined gradient (internal + cell)."""
        self.neval += 1
        f = self._get_potential_energy_raw()

        # Add pressure contribution: H = E + P*V
        if self.scalar_pressure != 0.0:
            f += self.scalar_pressure * self.atoms.get_volume()

        # Atomic forces -> internal coordinate gradient
        forces = self._get_forces_raw()
        g_cart = -forces.ravel()
        Binv = self._get_Binv()
        g_internal = g_cart @ Binv[:len(g_cart)]

        # Stress tensor -> cell gradient
        stress = self._get_stress_raw()  # 6-component Voigt, eV/Å³
        g_cell = self._stress_to_cell_gradient(stress, forces=forces)

        # Add the constraint-projection work term (nonzero only when internal
        # constraints are present in the non-rigid path -- see the method).
        g_cell = g_cell + self._constraint_projection_cell_gradient()

        self.write_traj()
        return f, np.concatenate([g_internal, g_cell])

    def _constraint_projection_cell_gradient(self):
        """Cell-gradient contribution from constraint re-projection.

        With an *internal* constraint (fixed bond/angle/dihedral) and cell
        optimization, ``set_x`` changes the cell (moving atoms fractionally),
        which perturbs the constrained coordinate, then re-projects the atoms
        back onto the constraint manifold. That projection does virtual work
        against the forces, contributing to dE/d(cell) -- but the stress/virial
        term computes the gradient at fixed (projected-out) coordinates and
        misses it.

        The smooth linearized projection is
        ``dq_proj/du = -Uc @ pinv(Jc @ Uc) @ Rc_u`` (Jc = constraint Jacobian
        in the non-redundant internal basis, Uc = constraint subspace, Rc_u =
        d(residual)/d(cell)), so the correction is
        ``g_proj = g_int^T @ dq_proj/du = -Rc_u^T @ y`` with
        ``(Jc @ Uc)^T y = Uc^T g_int``.

        Returns a zero vector unless there are active constraints. In
        rigid-fragment mode a constraint that lies entirely within one rigid
        fragment is invariant under the rigid cell move (zero contribution),
        but a constraint that *spans* fragments does acquire a residual under
        cell strain -- the interfragment separation follows the full cell
        deformation -- so it gets the same projection correction, differing
        only in how the unprojected cell motion (``X_C``) moves the atoms.
        """
        n_cell = self.n_cell_dof
        if n_cell == 0 or self.cons.nint == 0:
            return np.zeros(n_cell)
        if self._constraints_are_intrafragment_shape_constraints():
            return np.zeros(n_cell)
        drdx, Ucons, _, _ = self._compute_basis_int()
        if Ucons.shape[1] == 0:
            return np.zeros(n_cell)  # no active constraints

        # Raw internal gradient: use forces WITHOUT the ASE constraint applied.
        # ASE's constrained forces have already removed the exact component that
        # does the projection work, which would zero out this term.
        forces_raw = self.atoms.get_forces(apply_constraint=False)
        g_int = (-forces_raw.ravel()) @ self._get_Binv()[:forces_raw.size]

        A = drdx @ Ucons
        y, *_ = np.linalg.lstsq(A.T, Ucons.T @ g_int, rcond=_LSTSQ_RCOND)

        # Residual sensitivity to the raw cell C, via the derivative of the
        # unprojected cell motion: Rc_C = Jc_cart @ dr/dC + dRc/dC.
        X_C = self._unprojected_cell_motion_derivative()
        Rc_C = self.cons.jacobian() @ X_C + self.cons.cell_jacobian()
        G_C_proj = -(Rc_C.T @ y).reshape(3, 3)

        # Map raw-cell gradient to the scaled log-cell params, same as stress.
        dEdF_proj = G_C_proj @ self.orig_cell.T
        U = _logm_3x3(self._get_deformation_gradient())
        g_u = _expm_frechet_3x3_contracted(U, dEdF_proj) / self.exp_cell_factor
        return g_u[self.cell_mask]

    def _constraints_are_intrafragment_shape_constraints(self):
        """True when rigid-cell motion cannot perturb active constraints."""
        if not self.rigid_fragments:
            return False

        shape_constraint_names = {'bonds', 'angles', 'dihedrals'}
        frag_id = np.full(self.int.natoms + self.int.ndummies, -1, dtype=int)
        for fid, group in enumerate(self.fragment_groups):
            frag_id[np.asarray(group, dtype=int)] = fid
        for fid, dgroup in enumerate(self.fragment_dummy_groups):
            frag_id[np.asarray(dgroup, dtype=int)] = fid

        saw_active = False
        for name in self.cons._names:
            for coord, active in zip(self.cons.internals[name],
                                     self.cons._active[name]):
                if not active:
                    continue
                saw_active = True
                if name not in shape_constraint_names:
                    return False
                if np.any(coord.kwargs['ncvecs'] != 0):
                    return False
                ids = frag_id[np.asarray(coord.indices, dtype=int)]
                if np.any(ids < 0) or np.unique(ids).size != 1:
                    return False
        return saw_active

    def _unprojected_cell_motion_derivative(self):
        """d(atom+dummy positions)/d(raw cell C) for the unprojected cell move.

        Shape (3*ntot, 9), column ``3a+b`` = d r / d C[a,b] flattened.

        Non-rigid: ASE's set_cell(scale_atoms=True) is fractional-preserving,
        so ``r_i = frac_i @ C`` and ``dr_i/dC[a,b] = frac_i[a] * e_b``.

        Rigid fragments: each fragment translates with its CoM (fractional-
        preserving) and rotates rigidly, ``r_i = frac_com_g @ C + delta_i @ R``
        with ``R = polar(inv(C) @ C_new)``. To first order at C_new = C,
        ``dR/dC[a,b] = skew(inv(C) @ E_ab)``, giving
        ``dr_i/dC[a,b] = frac_com_g[a] * e_b + delta_i @ skew(inv(C) @ E_ab)``.
        Intrafragment vectors then change only by a rotation (zero residual);
        interfragment separations follow the full strain.
        """
        C = self.atoms.get_cell().array
        Cinv = np.linalg.inv(C)
        if len(self.dummies) > 0:
            positions = np.vstack([self.atoms.positions,
                                   self.dummies.positions])
        else:
            positions = self.atoms.positions
        ntot = positions.shape[0]
        X_C = np.zeros((3 * ntot, 9))

        if not self.rigid_fragments:
            frac = positions @ Cinv
            for i, s in enumerate(frac):
                for a in range(3):
                    for b in range(3):
                        X_C[3 * i + b, 3 * a + b] = s[a]
            return X_C

        # Rigid: reference each atom (and dummy) to its fragment CoM.
        natoms = self.int.natoms
        frag_of = {}
        for fid, group in enumerate(self.fragment_groups):
            for a in group:
                frag_of[int(a)] = fid
        for d in range(len(self.dummies)):
            parent = int(np.where(self.int.dinds == natoms + d)[0][0])
            frag_of[natoms + d] = frag_of[parent]
        com = np.zeros((ntot, 3))
        for fid, group in enumerate(self.fragment_groups):
            c = self.atoms.positions[list(group)].mean(axis=0)
            for i in range(ntot):
                if frag_of.get(i) == fid:
                    com[i] = c
        frac_com = com @ Cinv
        delta = positions - com
        eye3 = np.eye(3)
        for a in range(3):
            for b in range(3):
                E = np.zeros((3, 3)); E[a, b] = 1.0
                Amat = Cinv @ E
                skew = 0.5 * (Amat - Amat.T)
                # frac_com @ E scatters frac_com[:,a] into cartesian component b
                # (E has its 1 at [a,b], so r @ E = r[a] * e_b); delta @ skew is
                # the rigid rotation of the intrafragment offset.
                dr = np.outer(frac_com[:, a], eye3[b]) + delta @ skew
                X_C[:, 3 * a + b] = dr.ravel()
        return X_C

    def _virial_to_dEdF(self, virial, forces):
        """Rigid-fragment-aware virial -> dE/dF for internal-coordinate cells.

        dE/dC = C^{-T} @ (V*sigma [+ sym(dr^T @ f)]), then dE/dF = dE/dC @ C0^T.
        ``dr`` is positions relative to fragment CoM.

        For rigid fragments the relative vectors rotate with R = polar(inv(
        C_before) @ C_after) rather than following the full affine (fractional)
        motion the ASE virial assumes. The difference between affine motion
        (delta_i @ A) and rigid rotation (delta_i @ skew(A)), with A = inv(C) @
        dC, collapses to the *symmetric* part of P = dr^T @ f:

            relative correction = P:A - skew(P):A = sym(P):A

        so the correction is simply ``virial + sym(P)``. This is objective: a
        pure global rotation lies in the skew(A) direction, which sym(P)
        annihilates, so it contributes no spurious cell gradient. (The previous
        ``virial + P`` plus a finite-difference dR/dF term was derived for the
        non-objective left-multiplied motion and left a spurious rotational
        derivative on skewed cells.)
        """
        if self.rigid_fragments and forces is not None:
            delta_r = self._compute_delta_r()
            P = delta_r.T @ forces
            virial_corrected = virial + 0.5 * (P + P.T)
        else:
            virial_corrected = virial

        # dE/dC = C^{-T} @ virial_corrected, then dE/dF = dE/dC @ C0^T
        C = self.atoms.get_cell().array
        C_inv_T = np.linalg.inv(C.T)
        return C_inv_T @ virial_corrected @ self.orig_cell.T

    def _calc_basis(self):
        """Calculate basis including cell DOF.

        The cell DOF are treated as unconstrained additional coordinates.
        """
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

    def converged(self, fmax: float, smax: float = None, cmax: float = 1e-5):
        """Check convergence of forces and cell criterion.

        Parameters
        ----------
        fmax : float
            Maximum force tolerance (eV/Å).
        smax : float, optional
            Maximum effective stress tolerance. If None, compare the scaled
            log-cell gradient to fmax for ASE-like behavior.
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
            Maximum effective stress when smax is provided, otherwise maximum
            scaled log-cell gradient.
        """
        smax_threshold = fmax if smax is None else smax

        # Force convergence (project out constraints).
        # Associate as U @ (U.T @ g) — two cheap matvecs — instead of
        # (U @ U.T) @ g which materializes a (n_int, n_int) intermediate.
        g = self.get_g()
        g_internal = g[:self.n_internal]
        g_atomic = np.zeros_like(g)
        g_atomic[:self.n_internal] = g_internal
        g_proj = self._project_free_vector(g_atomic)[:self.n_internal]

        # Convert to Cartesian for force norm
        B = self.int.jacobian()
        g_cart = (g_proj @ B).reshape((-1, 3))
        fmax_actual = np.linalg.norm(g_cart, axis=1).max()

        # Stress convergence
        g_cell = g[self.n_internal:]
        smax_actual = self._cell_convergence_max(g_cell, smax)

        # Constraint residual
        cmax_actual = np.linalg.norm(self.get_convergence_res())

        conv = ((fmax_actual < fmax)
                and (smax_actual < smax_threshold)
                and (cmax_actual < cmax))
        return conv, fmax_actual, cmax_actual, smax_actual

    def get_projected_forces(self) -> np.ndarray:
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_internal = g[:self.n_internal]
        g_atomic = np.zeros_like(g)
        g_atomic[:self.n_internal] = g_internal
        g_projected = self._project_free_vector(g_atomic)[:self.n_internal]
        B = self.int.jacobian()
        return -(g_projected @ B).reshape((-1, 3))

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

    def calculate_hessian(self):
        assert self.hessian_function is not None
        H_raw = np.asarray(self.hessian_function(self.atoms), dtype=np.float64)
        ncart = 3 * len(self.atoms)
        if H_raw.shape == (self.dim, self.dim):
            self._install_cell_hessian(H_raw)
        elif H_raw.shape == (ncart, ncart):
            self._install_cell_hessian(
                self._convert_cartesian_hessian_to_internal(H_raw)
            )
        elif H_raw.shape == (self.n_internal, self.n_internal):
            self._install_cell_hessian(H_raw)
        else:
            raise ValueError(
                'hessian_function returned shape {}, expected {}, {}, or {}'
                .format(
                    H_raw.shape,
                    (ncart, ncart),
                    (self.n_internal, self.n_internal),
                    (self.dim, self.dim),
                )
            )


class CellCartesianPES(_CellPESMixin, PES):
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
        Scaling factor for cell parameterization. Default is n_atoms.
    cell_mask : ndarray, optional
        Boolean mask of shape (3, 3) indicating which cell DOF are free.
        Default is all True (full cell optimization).
    scalar_pressure : float, optional
        External pressure in eV/Å³. Default is 0.
    refine_initial_hessian : bool or int, optional
        Level of Hessian refinement via finite differences:
        - False or 0: No refinement (default)
        - -1: Forward differences for cell blocks (n_cell_dof force calls)
        - True or 1: Central differences for cell blocks (2 * n_cell_dof force calls)
    hessian_delta : float, optional
        Finite difference step size for Hessian refinement. Default is 1e-4.
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
        hessian_delta: float = 1e-4,
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
        self._init_cell_params(atoms, exp_cell_factor, cell_mask,
                               scalar_pressure)

        self._initializing = True
        PES.__init__(self, atoms, *args, H0=H0, **kwargs)
        self.n_cart = self.dim
        self.dim = self.n_cart + self.n_cell_dof
        self._initializing = False

        H_old = self.H.B if self.H is not None and self.H.B is not None else None
        H_coords_default = 70.0 * np.eye(self.n_cart) if H_old is None else None

        H0_full, refine_level = self._init_cell_hessian(
            H_old, refine_initial_hessian, hessian_delta,
            save_hessian, H_coords_default,
        )
        self.initial_hessian_refinement_level = refine_level
        self.set_H(H0_full, initialized=(refine_level == 0))

    @property
    def n_coords(self) -> int:
        return self.n_cart

    def get_x(self) -> np.ndarray:
        """Return Cartesian positions + cell parameters."""
        x_cart = self.apos.ravel().copy()
        if self._initializing:
            return x_cart
        cell_params = self._masked_cell_params()
        return np.concatenate([x_cart, cell_params])

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
        f = self._get_potential_energy_raw()

        # Add pressure contribution: H = E + P*V
        if self.scalar_pressure != 0.0:
            f += self.scalar_pressure * self.atoms.get_volume()

        # Cartesian gradient: Sella works in actual Cartesian positions
        # (not the undeformed frame), so no F transformation needed
        forces = self._get_forces_raw()
        g_cart = -forces.ravel()

        # Stress tensor -> cell gradient
        stress = self._get_stress_raw()  # 6-component Voigt, eV/Å³
        g_cell = self._stress_to_cell_gradient(stress, forces)

        self.write_traj()
        return f, np.concatenate([g_cart, g_cell])

    def calculate_hessian(self):
        assert self.hessian_function is not None
        self._install_cell_hessian(self.hessian_function(self.atoms))

    def _virial_to_dEdF(self, virial, forces):
        """Fixed-Cartesian virial -> dE/dF for Cartesian cells.

        In CellCartesianPES positions are always set independently after cell
        changes (set_x overrides any position scaling), so the fixed-Cartesian
        gradient applies: dE/dC = C^{-T} @ (V*sigma + r^T @ f), then chain-rule
        through cell = F @ C0 gives dE/dF = dE/dC @ C0^T.
        """
        C = self.atoms.get_cell().array
        C_inv_T = np.linalg.inv(C.T)
        positions = self.atoms.get_positions()
        dEdC = C_inv_T @ (virial + positions.T @ forces)
        return dEdC @ self.orig_cell.T

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
        U, S, VT = _robust_svd(drdx_cart)
        ncons = _svd_rank(S)
        Ucons_cart = VT[:ncons].T
        Ufree_cart = VT[ncons:].T
        Unred_cart = np.eye(self.n_cart)

        # Extend to include cell DOF (cell DOF are unconstrained -> identity
        # block in Unred/Ufree). Shared with CellInternalPES via the mixin.
        result = self._extend_basis_with_cell(
            (drdx_cart, Ucons_cart, Unred_cart, Ufree_cart)
        )

        # Cache the result
        self._basis_cache.put(state_hash, result)
        return result

    def converged(self, fmax: float, smax: float = None, cmax: float = 1e-5):
        """Check convergence of forces and cell criterion.

        Parameters
        ----------
        fmax : float
            Maximum force tolerance (eV/Å).
        smax : float, optional
            Maximum effective stress tolerance. If None, compare the scaled
            log-cell gradient to fmax for ASE-like behavior.
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
            Maximum effective stress when smax is provided, otherwise maximum
            scaled log-cell gradient.
        """
        smax_threshold = fmax if smax is None else smax

        # Force convergence (project out constraints)
        g = self.get_g()
        g_cart = g[:self.n_cart]
        Ufree = self.get_Ufree()
        Ufree_cart = Ufree[:self.n_cart, :Ufree.shape[1] - self.n_cell_dof]
        g_proj = (Ufree_cart @ (Ufree_cart.T @ g_cart)).reshape((-1, 3))

        fmax_actual = np.linalg.norm(g_proj, axis=1).max()

        # Stress convergence
        g_cell = g[self.n_cart:]
        smax_actual = self._cell_convergence_max(g_cell, smax)

        # Constraint residual
        cmax_actual = np.linalg.norm(self.get_convergence_res())

        conv = ((fmax_actual < fmax)
                and (smax_actual < smax_threshold)
                and (cmax_actual < cmax))
        return conv, fmax_actual, cmax_actual, smax_actual

    def get_projected_forces(self) -> np.ndarray:
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_cart = g[:self.n_cart]
        Ufree = self.get_Ufree()
        # Drop the trailing cell-DOF columns (as converged() does) so the
        # projection is over atomic DOF only. These columns are zero in the
        # Cartesian rows, so this does not change the result; it just keeps the
        # two methods consistent.
        Ufree_cart = Ufree[:self.n_cart, :Ufree.shape[1] - self.n_cell_dof]
        return -(Ufree_cart @ (Ufree_cart.T @ g_cart)).reshape((-1, 3))

    def get_Hc(self):
        """Get constraint Hessian extended for cell DOF."""
        Hc_cart = PES.get_Hc(self)
        Hc = np.zeros((self.dim, self.dim))
        Hc[:self.n_cart, :self.n_cart] = Hc_cart
        return Hc
