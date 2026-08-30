import pytest

import numpy as np

from ase.build import molecule
from ase.calculators.emt import EMT

from sella.peswrapper import PES, InternalPES, CellInternalPES, \
    _range_space_projector, _rigid_motion_nullspace
from sella.internal import Internals

@pytest.mark.parametrize("name,traj,cons",
                         [("CH4", "CH4.traj", None),
                          ("CH4", None, dict(bonds=((0, 1)))),
                          ("C6H6", None, None),
                          ])
def test_PES(name, traj, cons):
    tol = dict(atol=1e-6, rtol=1e-6)

    atoms = molecule(name)

    # EMT is *not* appropriate for molecules like this, but this is one
    # of the few calculators that is guaranteed to be available to all
    # users, and we don't need to use a physical PES to test this.
    atoms.calc = EMT()
    for MyPES in [PES, InternalPES]:
        with PES(atoms, trajectory=traj) as pes:

            pes.kick(0., diag=True, gamma=0.1)

            for i in range(2):
                pes.kick(-pes.get_g() * 0.01)

            assert pes.H is not None
            assert not pes.converged(0.)[0]
            assert pes.converged(1e100)
            A = pes.get_Ufree().T @ pes.get_Ucons()
            np.testing.assert_allclose(A, 0, **tol)

            pes.kick(-pes.get_g() * 0.001, diag=True, gamma=0.1)


# NH3/CH4/C2H6/C6H6 are rank-deficient (redundant internals + rigid-body
# null space at machine epsilon); H2O is full rank and exercises the
# non-deficient branch.
@pytest.mark.parametrize("name", ['H2O', 'NH3', 'CH4', 'C2H6', 'C6H6'])
def test_range_space_projector_matches_svd_truth(name):
    """The guess-Hessian projector must equal the SVD projector onto range(B).

    The internal-coordinate Jacobian of a free (non-periodic) molecule is
    typically rank-deficient: redundant internals plus the molecule's
    rigid-body modes produce singular values at machine epsilon. A
    non-pivoting economic QR keeps those near-null directions, leaking that
    subspace into the projected Hessian; _range_space_projector uses pivoting
    QR with rank truncation and must match the SVD-truth projector to machine
    precision.
    """
    atoms = molecule(name)
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    B = internals.jacobian()

    S = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(S > 1e-6 * S[0]))
    full_rank = min(B.shape)

    # SVD-truth projector onto range(B).
    U = np.linalg.svd(B, full_matrices=False)[0]
    P_true = U[:, :rank] @ U[:, :rank].T

    P = _range_space_projector(B)
    np.testing.assert_allclose(P, P_true, atol=1e-10)

    if rank < full_rank:
        # The old non-pivoting economic-QR projector retains the null space
        # and is materially wrong; this guards against a regression to it.
        Q, _ = np.linalg.qr(B, mode='reduced')
        P_old = Q @ Q.T
        assert np.linalg.norm(P_old - P_true) > 1e-3


def test_range_space_projector_exercises_rank_deficient_input():
    """At least one parametrized molecule must be genuinely rank-deficient,
    otherwise the regression guard above never runs."""
    atoms = molecule('C2H6')
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    B = internals.jacobian()
    S = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(S > 1e-6 * S[0]))
    assert rank < min(B.shape)


@pytest.mark.parametrize("axis", range(3))
def test_linear_rigid_motion_nullspace_spans_true_modes(axis):
    """A zero axial rotation must not displace a later transverse mode."""
    positions = np.zeros((3, 3))
    positions[:, axis] = [-1.0, 0.0, 2.0]

    modes = []
    for dim in range(3):
        mode = np.zeros_like(positions)
        mode[:, dim] = 1.0
        modes.append(mode.ravel())
    rel = positions - positions.mean(axis=0)
    for rotation_axis in np.eye(3):
        modes.append(np.cross(rotation_axis, rel).ravel())
    modes = np.column_stack(modes)

    expected = modes @ np.linalg.pinv(modes)
    actual_modes = _rigid_motion_nullspace(positions)
    actual = actual_modes @ actual_modes.T

    assert actual_modes.shape == (9, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_internal_df_pred_matches_projected_hessian_formula():
    """Trust prediction must match the explicit nonredundant projection.

    The optimized implementation avoids materializing ``U.T @ H @ U`` and
    instead evaluates the same quadratic form with the projected step vector.
    """
    rng = np.random.default_rng(7)
    dim = 18
    nred = 11
    U, _ = np.linalg.qr(rng.normal(size=(dim, nred)), mode='reduced')
    dx = rng.normal(size=dim)
    g = rng.normal(size=dim)
    A = rng.normal(size=(dim, dim))
    H = (A + A.T) * 0.5

    class DummyPES:
        def get_Unred(self):
            return U

    dx_r = dx @ U
    g_r = g @ U
    H_r = U.T @ H @ U
    expected = g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.
    actual = InternalPES.get_df_pred(DummyPES(), dx, g, H)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_linear_dummy_projection_accounts_extra_dummy_dihedral():
    """Projection changes to non-auxiliary dummy dihedrals must reach BFGS."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    atoms = Atoms(
        'OCOH',
        positions=[
            [0.0, 0.0, 0.0],
            [1.16, 0.0, 0.0],
            [2.32, 0.0, 0.0],
            [4.0, 1.7, 0.9],
        ],
    )
    atoms.calc = LennardJones()

    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()

    natoms = internals.natoms
    extra_dihedral = (natoms, 0, 3, 1)
    internals.add_dihedral(extra_dihedral)

    pes = InternalPES(atoms, internals, auto_find_internals=False)
    pes.get_g()

    row = 0
    extra_row = None
    for name in pes.int._names:
        for coord, active in zip(pes.int.internals[name],
                                 pes.int._active[name]):
            if not active:
                continue
            if name == 'dihedrals' and tuple(coord.indices) == extra_dihedral:
                extra_row = row
            row += 1
    assert extra_row is not None

    pes.dummies.positions += np.array([[8e-4, -1.7e-3, 1.1e-3]])
    target = pes.get_x().copy()

    dx_initial, dx_final, _ = pes.set_x(target)
    actual_delta = pes._wrapped_angle_delta(
        np.array([pes.get_x()[extra_row]]),
        np.array([target[extra_row]]),
    )[0]

    np.testing.assert_allclose(dx_initial, 0.0, atol=1e-14)
    np.testing.assert_allclose(dx_final[extra_row], actual_delta, atol=1e-12)


def test_independent_rectangular_dummy_jacobian_keeps_tangent_directions():
    """Dense dummy normals need low-rank projection even with square Q."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella.optimize.restricted_step import MaxInternalStep

    def make_pes():
        atoms = Atoms(
            'OCOH',
            positions=[
                [0.0, 0.0, 0.0],
                [1.16, 0.0, 0.0],
                [2.32, 0.0, 0.0],
                [3.10, 0.8, 0.2],
            ],
        )
        atoms.calc = LennardJones()
        return InternalPES(
            atoms, Internals(atoms), auto_find_internals=True
        )

    pes = make_pes()
    dense = make_pes()
    dense._linear_bend_dummy_projection_records = lambda: None

    Q, R = pes._get_jacobian_qr()
    assert Q.shape[0] == Q.shape[1]
    assert R.shape[0] != R.shape[1]

    rng = np.random.default_rng(29)
    A = rng.normal(size=(pes.dim, pes.dim))
    H = A.T @ A / pes.dim + np.eye(pes.dim)
    pes.set_H(H, initialized=True)
    dense.set_H(H, initialized=True)
    pes.get_g()
    dense.get_g()

    Ufree = pes.get_Ufree()
    drdx = pes.get_drdx()

    assert Ufree.shape == (pes.dim, pes.dim - pes.cons.nint)
    np.testing.assert_allclose(Ufree.T @ Ufree, np.eye(Ufree.shape[1]),
                               atol=1e-12)
    np.testing.assert_allclose(drdx @ Ufree, 0.0, atol=1e-12)
    assert pes._fast_dummy_selection_indices() is None
    assert pes._fast_dummy_redundant_projection() is not None

    fast_step, _ = MaxInternalStep(pes, 0, 0.1).get_s()
    dense_step, _ = MaxInternalStep(dense, 0, 0.1).get_s()
    np.testing.assert_allclose(fast_step, dense_step, atol=1e-11,
                               rtol=1e-11)
    assert np.linalg.norm(fast_step) > 1e-8


def test_range_space_projector_periodic_rank_deficient():
    """Periodic systems are also column-rank deficient when redundant internals
    (bonds/angles/dihedrals) are used: the intramolecular coordinates fail to
    span the inter-fragment rigid-body motions, and the cell does not rescue
    that. (Overcompleteness alone does not -- a fully bonded network like a
    metal is overcomplete but full column rank.) The projector must still match
    SVD truth here, exactly as in the non-periodic case.
    """
    from ase.calculators.lj import LennardJones

    atoms = None
    for pos in [(1, 1, 1), (4, 1, 1), (1, 4, 1), (1, 1, 4)]:
        w = molecule('H2O')
        w.positions += pos
        atoms = w if atoms is None else atoms + w
    atoms.set_cell([8, 8, 8])
    atoms.pbc = True
    atoms.calc = LennardJones()

    # allow_fragments=False forces redundant bond/angle/dihedral internals
    # (the minimal TRIC fragment set would instead be square and full rank).
    internals = Internals(atoms, allow_fragments=False)
    pes = CellInternalPES(atoms, internals, auto_find_internals=True)
    B = pes.int.jacobian()

    S = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(S > 1e-6 * S[0]))
    full_rank = min(B.shape)
    assert rank < full_rank, "expected a column-rank-deficient periodic Jacobian"

    U = np.linalg.svd(B, full_matrices=False)[0]
    P_true = U[:, :rank] @ U[:, :rank].T

    P = _range_space_projector(B)
    np.testing.assert_allclose(P, P_true, atol=1e-10)

    # Guard against regression to the non-pivoting economic-QR projector.
    Q, _ = np.linalg.qr(B, mode='reduced')
    P_old = Q @ Q.T
    assert np.linalg.norm(P_old - P_true) > 1e-3


def test_fixatoms_does_not_enable_rigid_fragments():
    """A FixAtoms constraint must not trigger rigid-fragment cell mode.

    FixAtoms becomes Translation coordinates, but those are constraints, not
    rigid fragments. rigid_fragments auto-detection must key on genuine
    fragment_atom_groups (set only by find_all_bonds with allow_fragments),
    not on the mere presence of any translation coordinate.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
                  cell=np.eye(3) * 8.0, pbc=True)
    atoms.calc = LennardJones()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.warns(UserWarning, match="do not span the full coordinate space"):
        pes = Sella(atoms, order=0, internal=True, optimize_cell=True).pes
    assert pes.rigid_fragments is False


def test_genuine_fragments_enable_rigid_fragments():
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms('OH2OH2',
                  positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0],
                             [6, 0, 0], [6.96, 0, 0], [6, 0.96, 0]],
                  cell=np.eye(3) * 12.0, pbc=True)
    atoms.calc = LennardJones()
    pes = Sella(atoms, order=0, internal=True, optimize_cell=True,
                allow_fragments=True).pes
    assert pes.rigid_fragments is True


@pytest.mark.parametrize("optimize_cell", [False, True])
def test_allow_fragments_unwraps_pbc_split_molecule(optimize_cell):
    """TRIC fragments must be contiguous before translations/rotations are used."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms(
        'OH2',
        positions=[
            [9.70, 5.00, 5.00],
            [0.66, 5.00, 5.00],  # bonded to O through +x periodic image
            [9.70, 5.90, 5.00],
        ],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atoms.calc = LennardJones()
    initial_extent = np.ptp(atoms.positions, axis=0).max()
    assert initial_extent > 9.0

    dyn = Sella(
        atoms, order=0, internal=True, optimize_cell=optimize_cell,
        allow_fragments=True, logfile=None,
    )

    final_extent = np.ptp(atoms.positions, axis=0).max()
    assert final_extent < 1.5
    assert dyn.pes.int.fragment_atom_groups is not None
    assert [len(g) for g in dyn.pes.int.fragment_atom_groups] == [3]
    assert len(dyn.pes.int.internals['translations']) == 3
    assert len(dyn.pes.int.internals['rotations']) == 3


@pytest.mark.parametrize(
    "symbols,positions,expected_nullity",
    [
        ('H2', [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], 5),
        (
            'CH4CH4',
            [
                [0.00, 0.00, 0.00],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
                [5.00, 0.00, 0.00],
                [5.63, 0.63, 0.63],
                [4.37, -0.63, 0.63],
                [4.37, 0.63, -0.63],
                [5.63, -0.63, -0.63],
            ],
            6,
        ),
    ],
)
def test_rank_deficient_jacobian_uses_rigid_nullspace(
    monkeypatch, symbols, positions, expected_nullity,
):
    """Global rigid-body null modes should not require an SVD fallback."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms(symbols, positions=positions)
    atoms.calc = LennardJones()
    dyn = Sella(
        atoms, order=0, internal=True, allow_fragments=False,
        logfile=None,
    )

    def fail_svd(*args, **kwargs):
        raise AssertionError("rigid-nullspace path should avoid SVD")

    from sella import peswrapper
    qr_shapes = []
    original_qr_with_pinv = peswrapper._gpu_qr_with_pinv

    def record_qr_shape(matrix):
        qr_shapes.append(matrix.shape)
        return original_qr_with_pinv(matrix)

    monkeypatch.setattr("sella.peswrapper._robust_svd", fail_svd)
    monkeypatch.setattr(
        "sella.peswrapper._gpu_qr_with_pinv", record_qr_shape
    )
    B = dyn.pes.int.jacobian()
    Q, R = dyn.pes._get_jacobian_qr()
    Binv = dyn.pes._get_Binv()

    assert B.shape not in qr_shapes
    assert (B.shape[0], B.shape[1] - expected_nullity) in qr_shapes
    assert Q.shape[1] == B.shape[1] - expected_nullity
    assert R.shape == (Q.shape[1], B.shape[1])
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
    np.testing.assert_allclose(Q @ R, B, atol=1e-10)
    np.testing.assert_allclose(B @ Binv @ B, B, atol=1e-10)
    np.testing.assert_allclose(Binv @ B @ Binv, Binv, atol=1e-10)


def test_periodic_jacobian_skips_rigid_nullspace(monkeypatch):
    """Periodic cell optimizations should go directly to the general QR."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms(
        'OH2',
        positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    atoms.calc = LennardJones()
    dyn = Sella(
        atoms, order=0, internal=True, optimize_cell=True,
        allow_fragments=True, logfile=None,
    )

    def fail_rigid_qr(*args, **kwargs):
        raise AssertionError("periodic systems should skip the rigid fast path")

    monkeypatch.setattr(
        "sella.peswrapper.InternalPES._rigid_nullspace_qr", fail_rigid_qr
    )
    Q, R = dyn.pes._get_jacobian_qr()
    np.testing.assert_allclose(Q @ R, dyn.pes.int.jacobian(), atol=1e-10)


def test_rigid_nullspace_failure_uses_general_qr(monkeypatch):
    """An eligible but rejected rigid fast path must retain the fallback."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella, peswrapper

    atoms = Atoms(
        'CH4',
        positions=[
            [0.00, 0.00, 0.00],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
    )
    atoms.calc = LennardJones()
    dyn = Sella(
        atoms, order=0, internal=True, allow_fragments=False, logfile=None,
    )
    B = dyn.pes.int.jacobian()
    qr_shapes = []
    original_qr_with_pinv = peswrapper._gpu_qr_with_pinv

    def record_qr_shape(matrix):
        qr_shapes.append(matrix.shape)
        return original_qr_with_pinv(matrix)

    monkeypatch.setattr(
        "sella.peswrapper.InternalPES._rigid_nullspace_qr",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sella.peswrapper._gpu_qr_with_pinv", record_qr_shape
    )
    dyn.pes._get_jacobian_qr()
    assert B.shape in qr_shapes


def test_copy_preserves_fragment_atom_groups():
    from ase import Atoms
    atoms = Atoms('OH2OH2',
                  positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0],
                             [6, 0, 0], [6.96, 0, 0], [6, 0.96, 0]],
                  cell=np.eye(3) * 12.0, pbc=True)
    ic = Internals(atoms, allow_fragments=True)
    ic.find_all_bonds()
    cp = ic.copy()
    assert cp.fragment_atom_groups is not None
    assert len(cp.fragment_atom_groups) == len(ic.fragment_atom_groups)


def test_inequality_toggle_invalidates_basis_cache():
    """Disabling a satisfied inequality must refresh the cached basis.

    _calc_basis / QR / Binv are cached by _state_hash, which keyed on geometry
    only. disable_satisfied_inequalities flips the constraint active mask
    without moving atoms, so drdx shrinks but Ucons stayed stale until the
    geometry changed. The mask is now folded into _state_hash.
    """
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella
    from sella.internal import Constraints

    atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    atoms.calc = LennardJones()
    cons = Constraints(atoms)
    # bond(0,1) currently 1.0, constrained < 2.0 -> satisfied inequality.
    cons.fix_bond((0, 1), target=2.0, comparator='lt')
    pes = Sella(atoms, order=0, constraints=cons, internal=False).pes

    ncols_before = pes.get_Ucons().shape[1]
    nrows_before = pes.get_drdx().shape[0]

    cons.disable_satisfied_inequalities()
    nrows_after = pes.get_drdx().shape[0]
    ncols_after = pes.get_Ucons().shape[1]

    # One inequality row dropped; the cached constraint basis must follow.
    assert nrows_after == nrows_before - 1
    assert ncols_after == ncols_before - 1
