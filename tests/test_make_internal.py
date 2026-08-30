import numpy as np
import jax.numpy as jnp
from ase import Atoms

from sella.internal import make_internal


def _scaled_distance(pos, scale):
    return scale * jnp.linalg.norm(pos[1] - pos[0])


def test_make_internal_preserves_kwargs():
    """Fixed eval kwargs passed to make_internal must survive instantiation.

    Regression: the generated class stored kwargs at class level, but
    Coordinate.__init__ reset self.kwargs to {}, so they were silently dropped
    and never reached the eval functions.
    """
    Scaled = make_internal("ScaledBond", _scaled_distance, 2, use_jit=False,
                           scale=3.0)
    coord = Scaled([0, 1])

    assert coord.kwargs == {"scale": 3.0}

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # calc passes **self.kwargs into _eval0 -> 3.0 * |r1 - r0| = 3.0.
    assert np.isclose(coord.calc(atoms), 3.0)


def test_make_internal_kwargs_not_shared_between_instances():
    """Each instance gets its own kwargs dict (no shared class-level state)."""
    Scaled = make_internal("ScaledBond2", _scaled_distance, 2, use_jit=False,
                           scale=3.0)
    a = Scaled([0, 1])
    b = Scaled([0, 1])
    a.kwargs["scale"] = 99.0
    assert b.kwargs["scale"] == 3.0
