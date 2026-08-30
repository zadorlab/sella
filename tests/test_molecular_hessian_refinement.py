import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.calculators.lj import LennardJones

from sella import Sella


def _assert_refined_optimizer_can_step(
    optimizer, expected_initialization_evals=None
):
    H = optimizer.pes.H.asarray()
    assert H.shape == (optimizer.pes.dim, optimizer.pes.dim)
    assert np.isfinite(H).all()
    np.testing.assert_allclose(H, H.T, atol=1e-10)

    neval_before_init = optimizer.pes.neval
    optimizer._ensure_initialized()
    if expected_initialization_evals is not None:
        # A complete refined Hessian must suppress the otherwise adaptive
        # initial HVP diagonalization. Cartesian refinement still needs one
        # reference gradient; internal conversion has already cached it.
        assert (
            optimizer.pes.neval
            == neval_before_init + expected_initialization_evals
        )
    optimizer.step()
    assert np.isfinite(optimizer.atoms.positions).all()


@pytest.mark.parametrize("order", [0, 1])
def test_level3_refines_nonperiodic_internal_hessian(order):
    atoms = molecule("H2O")
    positions = atoms.positions.copy()
    atoms.calc = LennardJones()

    optimizer = Sella(
        atoms,
        order=order,
        internal=True,
        refine_initial_hessian=3,
        hessian_delta=1e-3,
        logfile=None,
    )

    assert optimizer.pes.neval == 6 * len(atoms) + 1
    np.testing.assert_allclose(atoms.positions, positions, atol=1e-12)
    _assert_refined_optimizer_can_step(
        optimizer, expected_initialization_evals=0
    )


@pytest.mark.parametrize("order", [0, 1])
def test_level3_refines_nonperiodic_cartesian_hessian(order):
    atoms = Atoms(
        "Ar2",
        positions=[[0, 0, 0], [1.5, 0, 0]],
        calculator=LennardJones(),
    )
    positions = atoms.positions.copy()

    optimizer = Sella(
        atoms,
        order=order,
        internal=False,
        refine_initial_hessian=3,
        hessian_delta=1e-3,
        logfile=None,
    )

    assert optimizer.pes.neval == 2 * optimizer.pes.dim
    np.testing.assert_allclose(atoms.positions, positions, atol=1e-12)
    _assert_refined_optimizer_can_step(
        optimizer, expected_initialization_evals=1
    )


@pytest.mark.parametrize("order", [0, 1])
def test_level2_refines_nonperiodic_tric_hessian(order):
    atoms = molecule("H2O") + molecule("H2O")
    atoms.positions[3:] += [5, 0, 0]
    positions = atoms.positions.copy()
    atoms.calc = LennardJones()

    optimizer = Sella(
        atoms,
        order=order,
        internal=True,
        allow_fragments=True,
        refine_initial_hessian=2,
        hessian_delta=1e-3,
        logfile=None,
    )
    tric_indices = optimizer.pes._get_tric_indices()

    assert len(tric_indices) == 12
    assert optimizer.pes.neval == 2 * len(tric_indices)
    np.testing.assert_allclose(atoms.positions, positions, atol=1e-12)
    _assert_refined_optimizer_can_step(optimizer)


def test_level2_without_trics_is_noop():
    atoms = molecule("H2O")
    atoms.calc = LennardJones()

    optimizer = Sella(
        atoms,
        order=0,
        internal=True,
        refine_initial_hessian=2,
        hessian_delta=1e-3,
        logfile=None,
    )

    assert len(optimizer.pes._get_tric_indices()) == 0
    assert optimizer.pes.neval == 0


def test_level3_saves_nonperiodic_hessian(tmp_path):
    atoms = molecule("H2O")
    atoms.calc = LennardJones()
    hessian_path = tmp_path / "initial_hessian.npy"

    optimizer = Sella(
        atoms,
        order=1,
        internal=True,
        refine_initial_hessian=3,
        hessian_delta=1e-3,
        save_hessian=hessian_path,
        logfile=None,
    )

    np.testing.assert_allclose(
        np.load(hessian_path), optimizer.pes.H.asarray()
    )


def test_level3_is_not_repeated_on_internal_pes_rebuild():
    atoms = molecule("H2O")
    atoms.calc = LennardJones()
    optimizer = Sella(
        atoms,
        order=1,
        internal=True,
        hessian_function=None,
        refine_initial_hessian=3,
        hessian_delta=1e-3,
        logfile=None,
    )

    assert optimizer.pes.initial_hessian_refinement_level == 3
    assert optimizer.pes.neval == 6 * len(atoms) + 1

    optimizer._rebuild_after_bad_internals()

    assert optimizer.pes.initial_hessian_refinement_level == 0
    assert optimizer.pes.neval == 0
