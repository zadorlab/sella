import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io import read

from sella import Sella


def _read_sella_log(path):
    lines = path.read_text().splitlines()
    header = next(line for line in lines if "Step" in line)
    rows = [line for line in lines if line.startswith("Sella")]
    return header, rows


@pytest.mark.parametrize("internal", [False, True])
def test_logged_trajectory_ids_are_accepted_geometries(tmp_path, internal):
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.calc = LennardJones()
    trajectory = tmp_path / "optimization.traj"
    logfile = tmp_path / "optimization.log"
    accepted_positions = []

    opt = Sella(
        atoms,
        order=0,
        internal=internal,
        trajectory=str(trajectory),
        logfile=str(logfile),
        diag_every_n=1,
        nsteps_per_diag=0,
    )
    opt.attach(lambda: accepted_positions.append(atoms.positions.copy()))
    opt.run(fmax=1e-8, steps=3)
    opt.close()

    frames = read(trajectory, index=":")
    header, rows = _read_sella_log(logfile)
    trajectory_ids = [int(row.split()[-1]) for row in rows]

    assert header.split()[-1] == "trjid"
    assert len(trajectory_ids) == len(accepted_positions)
    assert len(frames) > len(accepted_positions)
    for trajectory_id, positions in zip(trajectory_ids, accepted_positions):
        np.testing.assert_allclose(frames[trajectory_id].positions, positions)


def test_log_omits_trajectory_id_without_trajectory(tmp_path):
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.calc = LennardJones()
    logfile = tmp_path / "optimization.log"

    opt = Sella(atoms, order=0, internal=False, logfile=str(logfile))
    opt.run(fmax=0.0, steps=0)
    opt.close()

    header, rows = _read_sella_log(logfile)
    assert "trjid" not in header.split()
    assert header.split()[-1] == "rho"
    assert len(rows[0].split()) == len(header.split()) + 1


def test_cell_log_includes_trajectory_id(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    trajectory = tmp_path / "cell.traj"
    logfile = tmp_path / "cell.log"

    opt = Sella(
        atoms,
        order=0,
        internal=False,
        optimize_cell=True,
        trajectory=str(trajectory),
        logfile=str(logfile),
    )
    opt.run(fmax=0.0, steps=0)
    opt.close()

    header, rows = _read_sella_log(logfile)
    assert header.split()[-1] == "trjid"
    assert int(rows[0].split()[-1]) == 0
    np.testing.assert_allclose(read(trajectory).cell, atoms.cell)
