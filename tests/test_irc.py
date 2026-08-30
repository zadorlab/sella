import numpy as np

from ase import Atoms
from ase.build import molecule
from ase.calculators.lj import LennardJones

from sella import IRC


# A first-order saddle of the 6-atom Lennard-Jones cluster (one imaginary mode).
_LJ6_TS = [[-0.819098, -0.456198, -0.221436], [0.233188, -0.375765, -0.567537],
           [-0.259168, 0.501054, -0.063452], [-0.504031, -0.163298, 0.804587],
           [0.800850, 0.577070, -0.410659], [0.548259, -0.082862, 0.458498]]


def _ts():
    atoms = Atoms('Ar6', positions=_LJ6_TS)
    atoms.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)
    return atoms


def test_irc_peskwargs_preserved():
    """A non-None peskwargs must be stored.

    Regression: self.peskwargs used to be assigned only in the
    ``if peskwargs is None`` branch, so passing an explicit peskwargs left
    the attribute unset and irun() crashed with AttributeError.
    """
    atoms = molecule('H2O')
    atoms.calc = LennardJones()
    opt = IRC(atoms, peskwargs={'gamma': 0.2}, logfile=None)
    assert opt.peskwargs == {'gamma': 0.2}


def test_irc_default_peskwargs():
    """The default path still derives peskwargs from gamma."""
    atoms = molecule('H2O')
    atoms.calc = LennardJones()
    opt = IRC(atoms, gamma=0.3, logfile=None)
    assert opt.peskwargs == {'gamma': 0.3}


def test_irc_takes_first_step_from_converged_ts():
    """IRC must apply the initial displacement even when the input TS already
    has |F| < fmax.

    Regression for the ASE >= 3.28 optimizer loop, which checks
    ``gradient_converged()`` rather than ``converged()``. IRC only overrode
    ``converged()`` (where the ``first``-step guard lives), so an IRC started
    from a converged TS "converged" at 0 steps and returned the TS unchanged.
    """
    x0 = _ts().get_positions()
    ends = {}
    for direction in ('forward', 'reverse'):
        atoms = _ts()
        irc = IRC(atoms, dx=0.1, keep_going=True, logfile=None)
        irc.run(fmax=0.05, steps=100, direction=direction)
        assert irc.nsteps > 0, f"{direction} IRC took no steps"
        assert np.abs(atoms.get_positions() - x0).max() > 1e-3, \
            f"{direction} IRC did not leave the TS"
        ends[direction] = atoms.get_positions()

    # forward and reverse must descend to opposite sides of the TS
    assert np.abs(ends['forward'] - ends['reverse']).max() > 1e-2
