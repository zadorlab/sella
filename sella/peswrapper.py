# Proof of concept MinMode implementation for Atoms which explicitly
# removes constraints from the degrees of freedom exposed to the
# optimizer

import numpy as np
from scipy.linalg import eigh, lstsq

from ase.io import Trajectory

from .eigensolvers import rayleigh_ritz
from .linalg import NumericalHessian, ProjectedMatrix
from .hessian_update import update_H, symmetrize_Y
from .constraints import initialize_constraints, calc_constr_basis


class PESWrapper(object):
    def __init__(self, atoms, calc, eigensolver=rayleigh_ritz,
                 project_translations=True, project_rotations=None,
                 constraints=None, trajectory=None, shift=1000,
                 v0=None, maxres=1e-5):
        self.atoms = atoms
        self.H = None
        self.HPSB = None
        self.HBFGS = None
        self.HSR1 = None
        self.HDFP = None

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
            out = calc_constr_basis(self.atoms, self.constraints,
                                    self.nconstraints, self.rot_center,
                                    self.rot_axes)
            self.res, self.drdx, self.Tm, self.Tfree, self.Tc = out
            self._basis_xlast = self.x.copy()

    def kick(self, dx_m, diag=False, **kwargs):
        if np.linalg.norm(dx_m) == 0 and self.last['f'] is not None:
            return self.last['f'], self.Tm.T @ self.last['g'], dx_m

        dx_c, _, _, _ = lstsq(-self.drdx.T, self.res)

        dx = self.Tm @ dx_m + dx_c
        f1, g1 = self.evaluate(self.x + dx)

        if diag:
            self.diag(**kwargs)

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
            self.trajectory.write(self.atoms, energy=e, forces=gout)
        return e, g

    def evaluate(self, x):
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
            self.err = np.abs(self.df_pred - self.df) / np.abs(self.df_pred)

        if self.last['h'] is not None:
            # Update Hessian matrix
            dh_free = self.Tfree.T @ (h - self.last['h'])
            self.H = update_H(self.H, dx_free, dh_free)

            self.HPSB = update_H(self.HPSB, dx_free, dh_free, method='PSB')
            self.HBFGS = update_H(self.HBFGS, dx_free, dh_free, method='BFGS')
            self.HSR1 = update_H(self.HSR1, dx_free, dh_free, method='SR1')
            self.HDFP = update_H(self.HDFP, dx_free, dh_free, method='DFP')

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

    def diag(self, eta, gamma=0.5, threepoint=False, shift=False,
             shiftfactr=1000, maxiter=None, **kwargs):
        if self.last['g'] is None:
            self.evaluate(self.x)
            if maxiter is not None:
                maxiter -= 1

        x = self.last['x']
        g = self.last['g']

        # If we don't have an approximate Hessian yet, then
        H = self.H
        v0 = None
        if H is None:
            v0 = self.v0
            if v0 is None:
                v0 = self.last['g']
            v0 = self.Tfree.T @ v0
            H = np.eye(self.Tfree.shape[1])

        # Htrue is a representation of the *true* Hessian matrix, which
        # can be probed only through Hessian-vector products that are
        # evaluated using finite difference of the gradient
        Htrue = NumericalHessian(self.calc_eg, x, g, eta, threepoint)

        # We project the true Hessian into the space of free coordinates
        if shift:
            Hproj = Htrue + shiftfactr * self.Tc @ self.Tc.T
            Pproj = H + shiftfactr * self.Tc @ self.Tc.T
        else:
            if v0 is not None:
                v0 = self.Tm.T @ self.Tfree @ v0
            Hproj = ProjectedMatrix(Htrue, self.Tm)
            Pproj = (self.Tm.T @ self.Tfree) @ H @ (self.Tfree.T @ self.Tm)

        x_orig = self.x.copy()
        vref = kwargs.pop('vref', None)
        vreftol = kwargs.pop('vreftol', 0.99)
        if vref is not None and not shift:
            vref = self.Tm.T @ vref
        method = kwargs.pop('method', 'jd0')
        lams, Vs, AVs = self.eigensolver(Hproj, gamma, Pproj, v0=v0,
                                         vref=vref, vreftol=vreftol,
                                         method=method, maxiter=maxiter)
        self.x = x_orig

        if not shift:
            Vs = Hproj.Vs
            AVs = Hproj.AVs
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2)
        lams, vecs = eigh(Atilde, Vs.T @ Vs)

        Vs = Vs @ vecs
        AVs = AVs @ vecs
        AVstilde = AVs - self.drdx @ self.Tc.T @ AVs

        Vs_free = self.Tfree.T @ Vs
        AVstilde_free = self.Tfree.T @ AVstilde

        self.H = update_H(self.H, Vs_free, AVstilde_free)

        self.HPSB = update_H(self.HPSB, Vs_free, AVstilde_free, method='PSB')
        self.HBFGS = update_H(self.HBFGS, Vs_free, AVstilde_free,
                              method='BFGS')
        self.HSR1 = update_H(self.HSR1, Vs_free, AVstilde_free, method='SR1')
        self.HDFP = update_H(self.HDFP, Vs_free, AVstilde_free, method='DFP')

    def converged(self, ftol):
        return ((np.linalg.norm(self.Tm.T @ self.last['g']) < ftol)
                and np.all(np.abs(self.res) < self.maxres))
