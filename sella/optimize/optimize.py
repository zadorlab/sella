#!/usr/bin/env python3

import logging
import warnings
from time import localtime, strftime
from typing import Union, Callable, Optional

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils import basestring
from ase.io.trajectory import Trajectory

from .restricted_step import get_restricted_step, MaxInternalStep
from sella.peswrapper import PES, InternalPES, CellInternalPES, CellCartesianPES
from sella.internal import Internals, Constraints

logger = logging.getLogger(__name__)

_default_kwargs = dict(
    minimum=dict(
        delta0=1e-1,
        sigma_inc=1.15,
        sigma_dec=0.90,
        rho_inc=1.035,
        rho_dec=100,
        method='qn',
        eig=False
    ),
    saddle=dict(
        delta0=0.1,
        sigma_inc=1.15,
        sigma_dec=0.65,
        rho_inc=1.035,
        rho_dec=5.0,
        method='prfo',
        eig=True
    )
)


class Sella(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        restart: bool = None,
        logfile: str = '-',
        trajectory: Union[str, Trajectory] = None,
        master: bool = None,
        delta0: float = None,
        sigma_inc: float = None,
        sigma_dec: float = None,
        rho_dec: float = None,
        rho_inc: float = None,
        order: int = 1,
        eig: bool = None,
        eta: float = 1e-4,
        method: str = None,
        gamma: float = 0.1,
        threepoint: bool = False,
        constraints: Constraints = None,
        constraints_tol: float = 1e-5,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        append_trajectory: bool = False,
        rs: str = None,
        nsteps_per_diag: int = 3,
        diag_every_n: Optional[int] = None,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        optimize_cell: bool = False,
        cell_mask: np.ndarray = None,
        exp_cell_factor: float = None,
        scalar_pressure: float = 0.0,
        smax: float = None,
        allow_fragments: bool = False,
        niggli: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        save_hessian: str = None,
        **kwargs
    ):
        """Initialize Sella optimizer.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to optimize.
        optimize_cell : bool, optional
            If True, optimize unit cell parameters along with atomic positions.
            Requires order=0. Default is False.
        cell_mask : ndarray, optional
            Boolean mask of shape (3, 3) indicating which cell DOF are free.
            Default is all True (full cell optimization).
        exp_cell_factor : float, optional
            Scaling factor for cell parameterization. Default is number of atoms.
        scalar_pressure : float, optional
            External pressure in eV/Å³ for cell optimization. Default is 0.
        smax : float, optional
            Maximum stress tolerance for convergence when optimize_cell=True.
            If None, uses fmax.
        allow_fragments : bool, optional
            If True, allow disconnected molecular fragments when using internal
            coordinates. Adds translation and rotation coordinates (TRICs) for
            each fragment. Useful for molecular crystals. Default is False.
        niggli : bool, optional
            If True, apply Niggli reduction during cell optimization when cell
            angles deviate more than 30 deg from 90 deg. This remaps to the
            most compact unit cell and resets the Hessian cell block.
            Default is False.
        refine_initial_hessian : bool or int, optional
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - True or 1: Refine cell-related blocks only (2 * n_cell_dof force calls)
            - 2: Also refine translation/rotation blocks for molecular crystals
              (adds 2 * n_tric force calls, where n_tric = n_fragments * 6)
            - 3: Refine full internal Hessian (2 * n_internal force calls, expensive!)
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        if order == 0:
            default = _default_kwargs['minimum']
        else:
            default = _default_kwargs['saddle']

        # Validate cell optimization parameters
        self.optimize_cell = optimize_cell
        self.allow_fragments = allow_fragments
        self.niggli = niggli
        self.smax = smax
        if optimize_cell:
            if order != 0:
                raise ValueError(
                    "Cell optimization is only supported for minima (order=0), "
                    f"got order={order}."
                )
            if not np.any(atoms.pbc):
                raise ValueError(
                    "Cell optimization requires periodic boundary conditions. "
                    "Set atoms.pbc = True for periodic systems."
                )

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)
            # Register trajectory for cleanup when close() is called
            self.closelater(trajectory)

        asetraj = None
        self.peskwargs = kwargs.copy()
        self.user_internal = internal
        self.initialize_pes(
            atoms,
            trajectory,
            order,
            eta,
            constraints,
            v0,
            internal,
            hessian_function,
            optimize_cell=optimize_cell,
            cell_mask=cell_mask,
            exp_cell_factor=exp_cell_factor,
            scalar_pressure=scalar_pressure,
            allow_fragments=allow_fragments,
            refine_initial_hessian=refine_initial_hessian,
            save_hessian=save_hessian,
            **kwargs
        )

        if rs is None:
            rs = 'mis' if internal else 'ras'
        self.rs = get_restricted_step(rs)
        Optimizer.__init__(self, atoms, restart=restart,
                           logfile=logfile, trajectory=asetraj,
                           master=master)

        if delta0 is None:
            delta0 = default['delta0']
        if rs in ['mis', 'ras']:
            self.delta = delta0
        else:
            self.delta = delta0 * self.pes.get_Ufree().shape[1]
        self.delta_cell = delta0

        self.sigma_inc = sigma_inc if sigma_inc is not None else default['sigma_inc']
        self.sigma_dec = sigma_dec if sigma_dec is not None else default['sigma_dec']
        self.rho_inc = rho_inc if rho_inc is not None else default['rho_inc']
        self.rho_dec = rho_dec if rho_dec is not None else default['rho_dec']
        self.method = method if method is not None else default['method']
        self.eig = eig if eig is not None else default['eig']

        self.ord = order
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.diagkwargs = dict(gamma=gamma, threepoint=threepoint)
        self.rho = 1.

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.
        self.nsteps_per_diag = nsteps_per_diag

        # Set by run() / first converged() call.
        self.fmax = None
        self._last_converged = None
        self.nsteps_since_diag = 0
        self.diag_every_n = np.inf if diag_every_n is None else diag_every_n

    def initialize_pes(
        self,
        atoms: Atoms,
        trajectory: str = None,
        order: int = 1,
        eta: float = 1e-4,
        constraints: Constraints = None,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        optimize_cell: bool = False,
        cell_mask: np.ndarray = None,
        exp_cell_factor: float = None,
        scalar_pressure: float = 0.0,
        allow_fragments: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        save_hessian: str = None,
        **kwargs
    ):
        if internal:
            if isinstance(internal, Internals):
                auto_find_internals = False
                if constraints is not None:
                    raise ValueError(
                        "Internals object and Constraint object cannot both "
                        "be provided to Sella. Instead, you must pass the "
                        "Constraints object to the constructor of the "
                        "Internals object."
                    )
            else:
                auto_find_internals = True
                internal = Internals(
                    atoms, cons=constraints, allow_fragments=allow_fragments,
                )
            self.internal = internal.copy()
            self.constraints = None

            if optimize_cell:
                # Use CellInternalPES for combined internal + cell optimization
                self.pes = CellInternalPES(
                    atoms,
                    internals=internal,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    auto_find_internals=auto_find_internals,
                    hessian_function=hessian_function,
                    exp_cell_factor=exp_cell_factor,
                    cell_mask=cell_mask,
                    scalar_pressure=scalar_pressure,
                    refine_initial_hessian=refine_initial_hessian,
                    save_hessian=save_hessian,
                    **kwargs
                )
            else:
                self.pes = InternalPES(
                    atoms,
                    internals=internal,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    auto_find_internals=auto_find_internals,
                    hessian_function=hessian_function,
                    **kwargs
                )
        else:
            self.internal = None
            if constraints is None:
                constraints = Constraints(atoms)
            self.constraints = constraints
            if optimize_cell:
                # Use CellCartesianPES for Cartesian + cell optimization
                self.pes = CellCartesianPES(
                    atoms,
                    constraints=constraints,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    hessian_function=hessian_function,
                    exp_cell_factor=exp_cell_factor,
                    cell_mask=cell_mask,
                    scalar_pressure=scalar_pressure,
                    refine_initial_hessian=refine_initial_hessian,
                    save_hessian=save_hessian,
                    **kwargs
                )
            else:
                self.pes = PES(
                atoms,
                constraints=constraints,
                trajectory=trajectory,
                eta=eta,
                v0=v0,
                hessian_function=hessian_function,
                **kwargs
            )
        self.trajectory = self.pes.traj

    def _predict_step(self):
        if not self.initialized:
            self.pes.get_g()
            if self.eig:
                if self.pes.hessian_function is not None:
                    self.pes.calculate_hessian()
                else:
                    self.pes.diag(**self.diagkwargs)
                self.nsteps_since_diag = -1
            self.initialized = True

        self.pes.cons.disable_satisfied_inequalities()
        self.pes._update_basis()
        self.pes.save()
        x0 = self.pes.get_x()

        rs_kwargs = {}
        if self.optimize_cell and isinstance(self.rs, type) and issubclass(
            self.rs, MaxInternalStep
        ):
            rs_kwargs['wc'] = self.delta / self.delta_cell

        if self.pes.cons.has_inequalities():
            all_valid = False
            while not all_valid:
                s, smag = self.rs(
                    self.pes, self.ord, self.delta, method=self.method,
                    **rs_kwargs
                ).get_s()
                self.pes.set_x(x0 + s)
                all_valid = self.pes.cons.validate_inequalities()
                self.pes._update_basis()
                self.pes.restore()
            self.pes._update_basis()
        else:
            s, smag = self.rs(
                self.pes, self.ord, self.delta, method=self.method,
                **rs_kwargs
            ).get_s()

        return s, smag

    def step(self):
        s, smag = self._predict_step()

        # Determine if we need to call the eigensolver, then step
        if self.nsteps_since_diag >= self.diag_every_n:
            ev = True
        elif self.eig and self.nsteps_since_diag >= self.nsteps_per_diag:
            if self.pes.H.evals is None:
                ev = True
            else:
                Unred = self.pes.get_Unred()
                ev = (self.pes.get_HL_projected(Unred)
                                       .evals[:self.ord] > 0).any()
        else:
            ev = False

        if ev:
            self.nsteps_since_diag = 0
        else:
            self.nsteps_since_diag += 1

        rho = self.pes.kick(s, ev, **self.diagkwargs)

        # Check for bad internals, and if found, reset PES object.
        # This skips the trust radius update.
        if self.internal and self.pes.int.check_for_bad_internals():
            if isinstance(self.pes, CellInternalPES):
                cell_mask = self.pes.cell_mask
                exp_cell_factor = self.pes.exp_cell_factor
                scalar_pressure = self.pes.scalar_pressure
            else:
                cell_mask = None
                exp_cell_factor = None
                scalar_pressure = 0.0
            self.initialize_pes(
                atoms=self.pes.atoms,
                trajectory=self.pes.traj,
                order=self.ord,
                eta=self.pes.eta,
                constraints=self.constraints,
                v0=None,  # TODO: use leftmost eigenvector from old H
                internal=self.user_internal,
                hessian_function=self.pes.hessian_function,
                optimize_cell=self.optimize_cell,
                cell_mask=cell_mask,
                exp_cell_factor=exp_cell_factor,
                scalar_pressure=scalar_pressure,
                allow_fragments=self.allow_fragments,
            )
            self.initialized = False
            self.rho = 1
            return

        # Update trust radius
        if rho is not None:
            if self.optimize_cell and isinstance(self.pes, CellInternalPES):
                n_int = self.pes.n_internal
                smag_int = np.max(np.abs(s[:n_int])) if n_int > 0 else 0
                smag_cell = np.max(np.abs(s[n_int:])) if len(s) > n_int else 0
            else:
                smag_int = smag
                smag_cell = 0

            if rho < 1./self.rho_dec or rho > self.rho_dec:
                self.delta = max(smag_int * self.sigma_dec, self.delta_min)
                if smag_cell > 0:
                    self.delta_cell = max(self.delta_cell * self.sigma_dec,
                                          self.delta_min)
            elif 1./self.rho_inc < rho < self.rho_inc:
                self.delta = max(self.sigma_inc * smag_int, self.delta)
                if smag_cell > 0:
                    self.delta_cell = max(self.sigma_inc * smag_cell,
                                          self.delta_cell)
            self.rho = rho
        else:
            self.rho = 1.

        # Apply Niggli reduction if cell becomes too skewed
        if self.optimize_cell and self.niggli and self.pes.maybe_niggli_reduce():
            logger.info("Applied Niggli reduction to reduce cell skewness")
            self.initialized = False
            self.rho = 1.

    def gradient_converged(self, gradient=None):
        return self.converged()

    def converged(self, forces=None):
        # fmax may still be None if converged() is called before run()
        fmax = self.fmax if self.fmax is not None else 0.05  # Default threshold
        if self.optimize_cell:
            smax = self.smax if self.smax is not None else fmax
            result = self.pes.converged(fmax, smax=smax)
            self._last_converged = result
            return result[0]
        result = self.pes.converged(fmax)
        self._last_converged = result
        return result[0]

    def log(self, forces=None):
        if self.logfile is None:
            return
        if self.optimize_cell:
            smax = self.smax if self.smax is not None else self.fmax
            result = self._last_converged
            if result is None or len(result) != 4:
                result = self.pes.converged(self.fmax, smax=smax)
            _, fmax, cmax, smax_actual = result
            e = self.pes.get_f()
            T = strftime("%H:%M:%S", localtime())
            name = self.__class__.__name__
            buf = " " * len(name)
            if self.nsteps == 0:
                self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                                   "{:>12s} {:>12s} {:>12s} {:>12s}\n"
                                   .format("Step", "Time", "Energy", "fmax",
                                           "smax", "cmax", "rtrust",
                                           "strust", "rho"))
            self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                               "{:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}\n"
                               .format(name, self.nsteps, T, e, fmax, smax_actual,
                                       cmax, self.delta, self.delta_cell,
                                       self.rho))
        else:
            result = self._last_converged
            if result is None or len(result) != 3:
                result = self.pes.converged(self.fmax)
            _, fmax, cmax = result
            e = self.pes.get_f()
            T = strftime("%H:%M:%S", localtime())
            name = self.__class__.__name__
            buf = " " * len(name)
            if self.nsteps == 0:
                self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                                   "{:>12s} {:>12s}\n"
                                   .format("Step", "Time", "Energy", "fmax",
                                           "cmax", "rtrust", "rho"))
            self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                               "{:>12.4f} {:>12.4f}\n"
                               .format(name, self.nsteps, T, e, fmax, cmax,
                                       self.delta, self.rho))
        try:
            self.logfile.flush()
        except (AttributeError, TypeError):
            pass
