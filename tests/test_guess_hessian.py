import numpy as np
from ase import Atoms

from sella.internal import Internals


def _two_fragment_internals():
    """Two water molecules far apart -> per-fragment translations + rotations."""
    atoms = Atoms(
        symbols=['O', 'H', 'H', 'O', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.0, 0.96, 0.0],
            [10.0, 0.0, 0.0],
            [10.96, 0.0, 0.0],
            [10.0, 0.96, 0.0],
        ],
    )
    ints = Internals(atoms, allow_fragments=True)
    ints.find_all_bonds()
    ints.find_all_angles()
    ints.find_all_dihedrals()
    return ints


def test_guess_hessian_includes_other_and_rotations():
    """guess_hessian must assign curvature to every active coordinate.

    Regression: guess_hessian walked translations/bonds/angles/dihedrals/
    rotations but skipped the 'other' family, so its diagonal ordering
    diverged from jacobian()'s. With an 'other' coordinate present ahead of
    rotations, the trailing rotation slot was left at zero (rotation
    curvature landed in the 'other' slot instead).
    """
    ints = _two_fragment_internals()
    # Precondition: this system exercises the buggy ordering (rotations after
    # an 'other' coordinate).
    assert ints.nrotations > 0
    ints.add_other(ints.internals['bonds'][0])
    assert ints.nother > 0

    H = ints.guess_hessian()
    d = np.diag(H)

    assert H.shape == (ints.nint, ints.nint)
    # Every active coordinate (including 'other' and rotations) gets a
    # positive diagonal curvature; no slot is left at zero.
    assert np.all(d > 0), d
