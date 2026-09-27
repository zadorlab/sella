import numpy as np
import pytest

from ase import Atoms
from ase.build import add_adsorbate, fcc100, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms

from sella import Constraints, IRC, Sella


# A first-order saddle of the 6-atom Lennard-Jones cluster (one imaginary mode).
_LJ6_TS = [[-0.819098, -0.456198, -0.221436], [0.233188, -0.375765, -0.567537],
           [-0.259168, 0.501054, -0.063452], [-0.504031, -0.163298, 0.804587],
           [0.800850, 0.577070, -0.410659], [0.548259, -0.082862, 0.458498]]


class _SaddleWithFrozenForce(Calculator):
    """Quadratic saddle with a nonzero force on the frozen atom."""

    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions.ravel()
        curvature = np.ones_like(x)
        curvature[3] = -1.0
        gradient = curvature * x
        gradient[0] += 1.0
        self.results = {
            'energy': 0.5 * np.dot(curvature * x, x) + x[0],
            'forces': -gradient.reshape((-1, 3)),
        }


class _NoisyEMT(Calculator):
    """EMT with seeded Gaussian noise of 1e-10 eV/Ang on the forces.

    Stands in for a calculator whose forces are not bitwise reproducible.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, seed):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        ref = self.atoms.copy()
        ref.calc = EMT()
        # get_forces() applies FixAtoms, so fixed atoms carry only noise
        noise = 1e-10 * self.rng.standard_normal((len(ref), 3))
        self.results = {
            'energy': ref.get_potential_energy(),
            'forces': ref.get_forces() + noise,
        }


def _ts():
    atoms = Atoms('Ar6', positions=_LJ6_TS)
    atoms.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)
    return atoms


@pytest.fixture(scope='module')
def slab_ts():
    """Cu adatom hopping over a bridge of Cu(100); bottom layer fixed."""
    atoms = fcc100('Cu', size=(3, 3, 2), vacuum=6.0)
    add_adsorbate(atoms, 'Cu', 1.9, 'bridge')
    atoms.set_constraint(FixAtoms([atom.index for atom in atoms
                                   if atom.tag == 2]))
    atoms.positions[-1, 1] += 0.05
    atoms.calc = EMT()
    opt = Sella(atoms, order=1, internal=False, logfile=None)
    assert opt.run(fmax=1e-4, steps=300)
    atoms.calc = None
    return atoms


def _initial_v0ts(atoms, calc):
    """Build a new IRC object and return its initial displacement."""
    atoms = atoms.copy()
    atoms.calc = calc
    irc = IRC(atoms, dx=0.05, eta=1e-4, gamma=0.4, logfile=None)
    irc.run(fmax=1e-3, steps=0)
    return irc.v0ts.copy()


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
        converged = irc.run(fmax=0.05, steps=100, direction=direction)
        assert converged
        assert irc.converged()
        assert irc.nsteps > 0, f"{direction} IRC took no steps"
        assert np.abs(atoms.get_positions() - x0).max() > 1e-3, \
            f"{direction} IRC did not leave the TS"
        ends[direction] = atoms.get_positions()

    # forward and reverse must descend to opposite sides of the TS
    assert np.abs(ends['forward'] - ends['reverse']).max() > 1e-2


@pytest.mark.parametrize('constraint_source', ['ase', 'sella'])
def test_irc_inner_loop_ignores_forces_on_fixed_atoms(constraint_source):
    atoms = Atoms('HH', positions=np.zeros((2, 3)))
    atoms.calc = _SaddleWithFrozenForce()
    kwargs = {}
    if constraint_source == 'ase':
        atoms.set_constraint(FixAtoms([0]))
    else:
        constraints = Constraints(atoms)
        constraints.fix_translation(0)
        kwargs['constraints'] = constraints

    irc = IRC(
        atoms, dx=0.1, ninner_iter=3, eta=1e-4, logfile=None, **kwargs
    )
    irc.run(direction='forward', fmax=1e-4, fmax_inner=1e-4, steps=1)

    assert irc.nsteps == 1


def test_irc_log_uses_projected_forces(tmp_path):
    atoms = Atoms('HH', positions=np.zeros((2, 3)))
    atoms.calc = _SaddleWithFrozenForce()
    constraints = Constraints(atoms)
    constraints.fix_translation(0)
    logfile = tmp_path / 'irc.log'

    irc = IRC(atoms, constraints=constraints, logfile=str(logfile))
    irc.run(fmax=1e-4, steps=0)
    irc.close()

    row = next(
        line for line in logfile.read_text().splitlines()
        if line.startswith('IRC:')
    )
    assert float(row.split()[-1]) == pytest.approx(0.0)


@pytest.mark.parametrize('seed', range(8))
def test_irc_initial_sign_ignores_fixed_atom_residue(slab_ts, seed):
    """Two IRC objects on the same saddle must orient v0ts the same way.

    Regression: the sign of v0ts was taken from its first non-zero
    component. On this slab that component belongs to a fixed atom and is
    numerical residue (~1e-9 Ang), so force noise of 1e-10 eV/Ang flipped
    the orientation between objects, and a forward run in one object and a
    reverse run in another could follow the same side of the barrier.
    """
    v1 = _initial_v0ts(slab_ts, _NoisyEMT(2 * seed))
    v2 = _initial_v0ts(slab_ts, _NoisyEMT(2 * seed + 1))
    cos = v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)
    assert cos == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('seed', range(8))
def test_irc_initial_sign_decided_by_free_atom(slab_ts, seed):
    """The component that fixes the sign of v0ts is not fixed-atom residue."""
    fixed = slab_ts.constraints[0].get_indices()
    v0ts = _initial_v0ts(slab_ts, _NoisyEMT(seed))
    v0abs = np.abs(v0ts)
    assert v0abs.reshape((-1, 3))[fixed].max() < 1e-5 * v0abs.max()

    first = np.flatnonzero(v0abs > 1e-3 * v0abs.max())[0]
    assert first // 3 not in fixed
    assert v0ts[first] > 0


def test_irc_initial_sign_unchanged_without_constraints():
    """For this unconstrained LJ6 saddle the first non-zero component of
    v0ts is not negligible, so v0ts keeps the orientation of the
    first-non-zero rule.
    """
    irc = IRC(_ts(), dx=0.1, logfile=None)
    irc.run(fmax=0.05, steps=0)
    v0ts = irc.v0ts
    first_nonzero = np.nonzero(v0ts)[0][0]
    assert abs(v0ts[first_nonzero]) > 1e-3 * np.abs(v0ts).max()
    assert v0ts[first_nonzero] > 0
