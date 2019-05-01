# Proof of concept MinMode implementation for Atoms which explicitly
# removes constraints from the degrees of freedom exposed to the
# optimizer

import warnings

import numpy as np
from scipy.linalg import eigh, lstsq

from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

from .eigensolvers import davidson
from .linalg import NumericalHessian, ProjectedMatrix
from .hessian_update import update_H, symmetrize_Y
from .constraints import initialize_constraints, calc_constr_basis


class MinModeAtoms(object):
    def __init__(self, atoms, calc, eigensolver=davidson,
                 project_translations=True, project_rotations=None,
                 constraints=None, trajectory=None, shift=1000,
                 v0=None, maxres=1e-5):
        self.atoms = atoms.copy()
        self.H = None

        if self.atoms.constraints:
            warnings.warn('ASE Atoms object has attached constraints, '
                          'but these will be ignored! Please provide '
                          'constraints to the MinMode object '
                          'initializer instead!')
            self.atoms.set_constraint()

        # We don't want to pass the dummy atoms onto the calculator,
        # which might not know what to do with them, so we have a
        # second Atoms object with dummy atoms removed, and attach
        # the calculator to *that*
        self._atoms_nodummy = self.atoms
        self.dummy_indices = [x.index for x in self.atoms if x.symbol == 'X']
        if self.dummy_indices:
            self._atoms_nodummy = self.atoms.copy()
        self.dummy_pos = self.atoms.positions[self.dummy_indices]
        del self._atoms_nodummy[self.dummy_indices]

        self._atoms_nodummy.set_calculator(calc)

        self.eigensolver = eigensolver
        self.shift = shift
        self.v0 = v0
        self.maxres = maxres

        # Default to projecting out rotations for aperiodic systems, but
        # not for periodic systems.
        if project_rotations is None:
            project_rotations = not np.any(self.atoms.pbc)
        # Extract the initial positions for use in fixed atom constraints
        self.pos0 = self.atoms.get_positions().copy()
        self.constraints = dict()
        self.set_constraints(constraints,
                             project_translations,
                             project_rotations)

        # The true dimensionality of the problem
        self.d = 3 * len(self.atoms)

        if trajectory is not None:
            self.trajectory = Trajectory(trajectory, 'w', self.atoms)
        else:
            self.trajectory = None

        # The position, function value, its gradient, and some other
        # information from the *previous* function call.
        # This dictionary should never be *updated*; instead, it
        # should be *replaced* by an entirely new dictionary with
        # new values. This will avoid the entries becoming out of
        # sync from one another.
        self.last = dict(x=None,
                         f=None,
                         g=None,
                         h=None,
                         x_m=None,
                         g_m=None)
        self.calls = 0
        self._basis_xlast = None
        self.ratio = None

        self._basis_update()

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, target):
        if target is None:
            self._H = None
            self.Hred = None
            self.lams = None
            self.vecs = None
            return
        self._H = target
        Tproj = self.Tm.T @ self.Tfree
        self.Hred = Tproj @ target @ Tproj.T
        lams, vecs = eigh(self.Hred)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
        self.lams = lams[indices]
        self.vecs = vecs[:, indices]

    @property
    def x(self):
        # The current atomic positions in a flattened array.
        # Full dimensional.
        xout = np.zeros(self.d).reshape((-1, 3))
        x_real = self._atoms_nodummy.positions
        x_dummy = self.dummy_pos
        nreal = 0
        ndummy = 0
        for i, row in enumerate(xout):
            if i in self.dummy_indices:
                row[:] = x_dummy[ndummy]
                ndummy += 1
            else:
                row[:] = x_real[nreal]
                nreal += 1
        return xout.ravel()

    @x.setter
    def x(self, target):
        xin = target.reshape((-1, 3))
        xreal = self.atoms.positions.copy()
        nreal = 0
        ndummy = 0
        for i, row in enumerate(xin):
            if i in self.dummy_indices:
                self.dummy_pos[ndummy] = row
                ndummy += 1
            else:
                xreal[nreal] = row
                nreal += 1
        #print('POSITIONS:', xin)
        self.atoms.set_positions(xin)
        self._atoms_nodummy.set_positions(xreal)
        self._basis_update()

    @property
    def x_m(self):
        # The current atomic positions in the subspace orthogonal to
        # any constraints.
        return self.Tm.T @ self.x

    def _basis_update(self):
        if self._basis_xlast is None or np.any(self.x != self._basis_xlast):
            out = calc_constr_basis(self.x, self.constraints, self.nconstraints,
                                    self.rot_center, self.rot_axes)
            self.res, self.drdx, self.Tm, self.Tfree, self.Tc = out
            self._basis_xlast = self.x.copy()

    def kick(self, dx_m, minmode=False, **kwargs):
        if np.linalg.norm(dx_m) == 0 and self.last['f'] is not None:
            return self.last['f'], self.Tm.T @ self.last['g'], dx_m

        res_orig = self.res.copy()

        dx_c, _, _, _ = lstsq(-self.drdx.T, self.res)

        dx = self.Tm @ dx_m + dx_c
        f1, g1 = self.f_update(self.x + dx)

        if minmode:
            self.f_minmode(**kwargs)

        return f1, self.Tm.T @ self.last['g'], dx_m

    def set_constraints(self, constraints, p_t, p_r):
        if self.H is not None:
            assert self.Tfree is not None
            Hfull = self.Tfree @ self.H @ self.Tfree.T
        else:
            Hfull = None

        con, ncon, rc, ra = initialize_constraints(self.atoms, self.pos0,
                                                   constraints, p_t, p_r)
        self.constraints = con
        self.nconstraints = ncon
        self.rot_center = rc
        self.rot_axes = ra

        if Hfull is not None:
            # Project into new basis
            self._basis_update()
            self.H = self.Tfree.T @ Hfull @ self.Tfree

    def calc_eg(self, x=None):
        if x is not None:
            self.x = x

        e = self._atoms_nodummy.get_potential_energy()
        gin = self._atoms_nodummy.get_forces()
        gout = np.zeros(self.d).reshape((-1, 3))
        nreal = 0
        for i, row in enumerate(gout):
            if i not in self.dummy_indices:
                row[:] = gin[nreal]
                nreal += 1
        g = -gout.ravel()
        self.calls += 1

        if self.trajectory is not None:
            if self.dummy_indices:
                calc = SinglePointCalculator(self.atoms, energy=e, forces=gout)
                self.atoms.set_calculator(calc)
            self.trajectory.write(self.atoms)
        return e, g

    def f_update(self, x):
        if self.last['f'] is not None and np.all(x == self.x):
            return self.last['f'], self.last['g']

        f, g = self.calc_eg(x)
        h = g - self.drdx @ self.Tc.T @ g

        if self.last['f'] is not None:
            self.df = f - self.last['f']
            dx = self.x - self.last['x']
            dx_free = self.Tfree.T @ dx
            dx_m = (self.Tfree.T @ (self.Tm @ self.Tm.T)) @ dx
            dx_c = (self.Tfree.T @ (self.Tc @ self.Tc.T)) @ dx

        if self.last['f'] is not None and self.H is not None:
            # Calculate predicted vs actual change in energy
            self.df_pred = (self.last['g'].T @ dx - (dx_m @ self.H @ dx_c)
                            + (dx_m @ self.H @ dx_m) / 2.
                            + (dx_c @ self.H @ dx_c) / 2.)

            self.ratio = self.df_pred / self.df

        if self.last['h'] is not None:
            # Update Hessian matrix
            dh_free = self.Tfree.T @ (h - self.last['h'])
            self.H = update_H(self.H, dx_free, dh_free)

        g_m = self.Tm.T @ g
        if self.H is not None:
            Tproj = self.Tm.T @ self.Tfree
            g_m -= Tproj @ self.H @ (self.Tfree.T @ (self.Tc @ self.res))

        self.last = dict(x=self.x.copy(),
                         f=f,
                         g=g.copy(),
                         h=h.copy(),
                         x_m=self.x_m.copy(),
                         g_m=g_m.copy())

        return f, g

    def f_minmode(self, dxL, maxres, threepoint=False, **kwargs):
        if self.last['g'] is None:
            self.f_update(self.x)

        x = self.last['x']
        g = self.last['g']

        # If we don't have an approximate Hessian yet, then
        H = self.H
        if H is None:
            v = self.v0
            if v is None:
                v = self.last['g']
            v = self.Tfree.T @ v
            H = np.eye(len(v)) - 2 * np.outer(v, v) / (v @ v)

        # Htrue is a representation of the *true* Hessian matrix, which
        # can be probed only through Hessian-vector products that are
        # evaluated using finite difference of the gradient
        Htrue = NumericalHessian(self.calc_eg, x, g, dxL, threepoint)

        # We project the true Hessian into the space of free coordinates
        Hproj = ProjectedMatrix(Htrue, self.Tm)

        Pproj = (self.Tm.T @ self.Tfree) @ H @ (self.Tfree.T @ self.Tm)

        x_orig = self.x.copy()
        lams, Vs, AVs = self.eigensolver(Hproj, maxres, Pproj)
        self.x = x_orig

        Vs = Hproj.Vs
        AVs = Hproj.AVs
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2)
        lams, vecs = eigh(Atilde)

        Vs = Vs @ vecs
        AVs = AVs @ vecs
        AVstilde = AVs - self.drdx @ self.Tc.T @ AVs
        self.H = update_H(self.H, self.Tfree.T @ Vs, self.Tfree.T @ AVstilde)

    def converged(self, ftol):
        return ((np.linalg.norm(self.Tm.T @ self.last['g']) < ftol)
                and np.all(np.abs(self.res) < self.maxres))
