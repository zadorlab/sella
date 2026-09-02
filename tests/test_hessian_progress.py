from ase import Atoms
from ase.calculators.lj import LennardJones

from sella import Sella


def test_sella_logs_hessian_force_evaluation_progress(tmp_path):
    atoms = Atoms('Ar2', positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.calc = LennardJones()
    logfile = tmp_path / 'sella.log'

    opt = Sella(
        atoms,
        order=1,
        internal=False,
        logfile=str(logfile),
        hessian_progress=True,
    )
    opt._ensure_initialized()
    opt.close()

    progress = [
        line for line in logfile.read_text().splitlines()
        if line.startswith('# Sella ')
    ]
    assert progress[0].endswith(': starting Hessian diagonalization')
    assert 'Hessian force evaluation 1 completed' in progress[1]
    assert 'Hessian diagonalization completed after' in progress[-1]

    evaluations = [
        int(line.split()[-2])
        for line in progress
        if line.endswith('completed')
    ]
    assert evaluations == list(range(1, len(evaluations) + 1))
    suffix = 'evaluation' if len(evaluations) == 1 else 'evaluations'
    assert progress[-1].endswith(
        f'{len(evaluations)} force {suffix}'
    )


def test_hessian_progress_is_disabled_by_default(tmp_path):
    atoms = Atoms('Ar2', positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.calc = LennardJones()
    logfile = tmp_path / 'sella.log'

    opt = Sella(atoms, order=1, internal=False, logfile=str(logfile))
    opt._ensure_initialized()
    opt.close()

    # ASE >= 3.28 wraps the logfile in a Log object that opens the file
    # lazily on the first write, so with progress logging disabled the file
    # is never created at all. Older ASE opens it eagerly. Both mean the
    # same thing here: nothing was logged.
    contents = logfile.read_text() if logfile.exists() else ''
    assert '# Sella ' not in contents
