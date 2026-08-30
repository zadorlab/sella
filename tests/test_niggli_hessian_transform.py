import numpy as np
from scipy.linalg import expm, expm_frechet

from sella.peswrapper import _cell_deformation_jacobian, \
    _expm_frechet_3x3_contracted


def test_cell_deformation_jacobian_matches_fd():
    """d(cell)/d(L) must match finite differences of the parameterization
    cell(L) = expm(L / factor) @ orig_cell at a *deformed* reference (F != I).

    Regression for the Niggli Hessian transform: the Frechet base point was
    wrongly scaled by exp_cell_factor (logm(F)/factor instead of logm(F)),
    which is only harmless when F is near the identity. With a real cell
    deformation the analytic Jacobian diverged from finite differences.
    """
    orig_cell = np.array([[4.0, 0.0, 0.0],
                          [0.1, 4.2, 0.0],
                          [0.2, 0.15, 4.1]])
    factor = 3.0  # a realistic exp_cell_factor (defaults to ~natoms, never 1)

    # A nonzero (deformed) log-deformation parameter matrix.
    L = np.array([[0.10, 0.05, 0.00],
                  [0.05, -0.08, 0.02],
                  [0.00, 0.02, 0.12]])
    F = expm(L / factor)

    J = _cell_deformation_jacobian(F, orig_cell, factor)

    delta = 1e-6
    J_num = np.zeros((9, 9))
    for idx in range(9):
        i, j = divmod(idx, 3)
        Lp = L.copy(); Lp[i, j] += delta
        Lm = L.copy(); Lm[i, j] -= delta
        cp = expm(Lp / factor) @ orig_cell
        cm = expm(Lm / factor) @ orig_cell
        J_num[:, idx] = ((cp - cm) / (2 * delta)).ravel()

    np.testing.assert_allclose(J, J_num, atol=1e-6, rtol=1e-5)


def _scipy_frechet_contract(U, dEdF):
    """Reference contraction via scipy's expm_frechet, one E_munu at a time."""
    g = np.zeros((3, 3))
    for mu in range(3):
        for nu in range(3):
            E = np.zeros((3, 3)); E[mu, nu] = 1.0
            ed = expm_frechet(U, E, compute_expm=False)
            g[mu, nu] = np.sum(ed * dEdF)
    return g


def test_expm_frechet_contracted_stable_for_tiny_logs():
    """The contracted expm Frechet must stay accurate as ||U|| -> 0.

    The closed-form Daleckii-Krein path uses eig(U), which is
    ill-conditioned for tiny nonsymmetric U; the exact-zero shortcut only
    covered ||U|| < 1e-10, leaving a gap around 1e-8 where the result drifted
    from scipy by ~1e-4. The small-norm Taylor path closes that gap without
    paying the scipy Frechet-loop cost.
    """
    rng = np.random.RandomState(0)
    for scale in [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        worst_rel = 0.0
        for _ in range(100):
            U = rng.randn(3, 3) * scale
            dEdF = rng.randn(3, 3)
            g = _expm_frechet_3x3_contracted(U, dEdF)
            g_ref = _scipy_frechet_contract(U, dEdF)
            denom = max(np.abs(g_ref).max(), 1e-30)
            worst_rel = max(worst_rel, np.abs(g - g_ref).max() / denom)
        assert worst_rel < 1e-6, f"scale {scale:.0e}: rel err {worst_rel:.2e}"


def test_expm_frechet_contracted_near_degenerate_eigenvalues():
    """Accurate even when U has a moderate norm but near-degenerate eigenvalues."""
    rng = np.random.RandomState(1)
    for gap in [1e-4, 1e-7, 1e-10]:
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        U = Q @ np.diag([0.3, 0.3 + gap, -0.2]) @ Q.T
        for _ in range(50):
            dEdF = rng.randn(3, 3)
            g = _expm_frechet_3x3_contracted(U, dEdF)
            g_ref = _scipy_frechet_contract(U, dEdF)
            np.testing.assert_allclose(g, g_ref, atol=1e-9)
