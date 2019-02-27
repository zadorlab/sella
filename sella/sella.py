#!/usr/bin/env python

import numpy as np
from scipy.linalg import eigh, null_space
from ase.io import Trajectory
from ase.constraints import FixAtoms
from .eigensolvers import NumericalHessian, ProjectedMatrix, atoms_tr_projection, davidson, project_translation, project_rotation
from .hessian_update import update_H
from .internal import Internal

class MinMode(object):
    def __init__(self, f, d, minmode=davidson, H0=None, v0=None,
                 trshift=1000, trshift_factor=4.):
        self.f = f
        self.d = d
        self.minmode = minmode
        self.H = None
        self.flast = None
        self.xlast2 = None
        self.xlast = None
        self.glast = None
        self.shift = trshift
        self.shift_factor = trshift_factor
        self.calls = 0
        self.df = None
        self.df_pred = None
        self.ratio = None
        self.v0 = v0

        self.H = H0
        if self.H is not None:
            self.lams, self.vecs = eigh(self.H)
        else:
            self.lams, self.vecs = None, None

    def dx(self, x):
        return self.xlast - x

    def kick(self, dx, minmode=False, **kwargs):
        # Kick geometry forwards and evaluate
        x1 = self.xlast + dx
        if minmode:
            f1, g1 = self.f_minmode(x1, **kwargs)
        else:
            f1, g1 = self.f_update(x1)
        return f1, g1, dx

    def f_update(self, x):
        if self.xlast is not None and np.all(x == self.xlast):
            return self.flast, self.glast

        self.calls += 1
        f, g = self.f(x)

        if self.flast is not None:
            self.df = f - self.flast
            dx = x - self.xlast
            self.df_pred = self.glast.T @ dx + (dx.T @ self.H @ dx) / 2.
            self.ratio = self.df_pred / self.df

        if self.xlast is not None and self.glast is not None:
            self.H = update_H(self.H, x - self.xlast, g - self.glast)
            self.lams, self.vecs = eigh(self.H)

        self.flast = f
        self.xlast2 = self.xlast
        self.xlast = x.copy()
        self.glast = g.copy()
        return f, g

    def f_minmode(self, x0, dxL=1e-5, maxres=5e-3, threepoint=False, **kwargs):
        d = len(x0)

        f0, g0 = self.f_update(x0)

        Htrue = NumericalHessian(self.f, x0, g0, dxL, threepoint)

        I = np.eye(self.d)
        T = np.empty((self.d, 0))

        P = self.H
        if P is None:
            u = g0 / np.linalg.norm(g0)
            P = I - 2 * np.outer(u, u)
        H = Htrue

        lams, Vs, AVs = self.minmode(H, maxres, P, T, shift=self.shift, **kwargs)
        self.calls += H.calls

        if self.H is None:
            self.H = np.average(np.abs(lams)) * I

        self.H = update_H(self.H, Vs, AVs)

        lams_all, vecs_all = eigh(self.H)
        self.shift = self.shift_factor * lams_all[-1]

        self.H = self.H
        self.lams, self.vecs = eigh(self.H)

        return f0, g0

    def f_dimer(self, x0, *args, **kwargs):
        f, g = self.f_minmode(self, x0, *args, **kwargs)
        gpar = self.vecs[:, 0] * (self.vecs[:, 0] @ g)
        if self.lams[0] > 0:
            return f, -gpar
        return f, g - 2 * gpar

    def xpolate(self, alpha):
        return self.xlast2 + alpha * (self.xlast - self.xlast2)

class MinModeAtoms(MinMode):
    def __init__(self, atoms, calculator, minmode=davidson, H0=None, v0=None, trshift=1000,
                 trshift_factor=4., project_translations=True, project_rotations=False, trajectory=None):
        d = 3 * len(atoms)

        MinMode.__init__(self, self.eval_eg, d, minmode, H0, v0, trshift,
                         trshift_factor)
        self.atoms = atoms.copy()
        self.x0 = self.atoms.positions.ravel().copy()
        self.atoms.set_calculator(calculator)
        self.d = 3 * len(self.atoms)
        if self.atoms.constraints:
            self.d -= 3 * len(self.atoms.constraints[0].index)

        self.project_translations = project_translations
        self.project_rotations = project_rotations

        if trajectory is not None:
            self.trajectory = Trajectory(trajectory, 'w', self.atoms)
        else:
            self.trajectory = None

    def eval_eg(self, x):
        xin = x.reshape((-1, 3))
        if self.atoms.constraints:
            pos = np.zeros_like(self.atoms.positions)
            # Currently assume there is only one constraint, and it is FixAtoms
            assert len(self.atoms.constraints) == 1
            constr = self.atoms.constraints[0]
            assert isinstance(constr, FixAtoms)
            n = 0
            for i, xi in enumerate(pos):
                if i in constr.index:
                    pos[i] = self.x0[i]
                else:
                    pos[i] = xin[n]
                    n += 1
        else:
            pos = xin

        self.atoms.set_positions(pos)
        e = self.atoms.get_potential_energy()
        f = -self.atoms.get_forces()
        if self.atoms.constraints:
            g = np.zeros_like(xin)
            n = 0
            for i, xi in enumerate(pos):
                if i not in constr.index:
                    g[n] = f[i]
                    n += 1
        else:
            g = f

        if self.trajectory is not None:
            self.trajectory.write(self.atoms)
        return e, g.ravel()

    def f_update(self, x):
        if self.xlast is not None and np.all(x == self.xlast):
            return self.flast, self.glast

        self.calls += 1
        f, g = self.eval_eg(x)

        if self.flast is not None:
            self.df = f - self.flast
            dx = x - self.xlast
            self.df_pred = self.glast.T @ dx + (dx.T @ self.H @ dx) / 2.
            self.ratio = self.df_pred / self.df

        if self.xlast is not None and self.glast is not None:
            self.H = update_H(self.H, x - self.xlast, g - self.glast)
            self.lams, self.vecs = eigh(self.H)

        self.flast = f
        self.xlast2 = self.xlast
        self.xlast = x.copy()
        self.glast = g.copy()
        return f, g

    def f_minmode(self, x0, dxL=1e-5, maxres=5e-3, threepoint=False, **kwargs):
        d = len(x0)

        f0, g0 = self.f_update(x0)

        Htrue = NumericalHessian(self.f, x0, g0, dxL, threepoint)

        I = np.eye(self.d)
        T = np.empty((self.d, 0))
        if self.project_translations:
            T = np.hstack((T, project_translation(x0)))
        if self.project_rotations:
            T = np.hstack((T, project_rotation(x0)))
        _, ntr = T.shape

        if ntr > 0:
            Tnull = null_space(T.T)
            Proj = Tnull @ Tnull.T
        else:
            Proj = np.eye(self.d)

        P = self.H
        if P is None:
            if self.v0 is not None:
                u = self.v0.copy()
            else:
                u = g0.copy()
            u /= np.linalg.norm(u)
            P = I - 2 * np.outer(u, u)
        H = Htrue

        #if self.v is None or np.abs(self.lams[0]) < 1e-8:
        #    if self.H is not None:
        #        self.v = self.vecs[:, 0]
        #    else:
        #        self.v = np.random.normal(size=d)
        #        self.v /= np.linalg.norm(self.v)

        #v0 = self.v

        lams, Vs, AVs = self.minmode(H, maxres, P, T, shift=self.shift, **kwargs)
        self.calls += H.calls

        #Proj = I - T @ T.T
        if ntr > 0:
            Tnull = null_space(T.T)
            Proj = Tnull @ Tnull.T
        else:
            Proj = np.eye(self.d)

        if self.H is None:
            if ntr == 0:
                lam0 = np.average(np.abs(lams))
            else:
                lam0 = np.average(np.abs(lams[:-ntr]))
            self.H = lam0 * Proj + self.shift * T @ T.T

        self.H = update_H(self.H, Vs, AVs)

        lams_all, vecs_all = eigh(self.H)
        self.shift = self.shift_factor * lams_all[-(ntr + 1)]

        self.H = Proj @ (self.H) @ Proj + self.shift * (T @ T.T)

        self.lams, self.vecs = eigh(self.H)

        return f0, g0

class MinModeInternal(MinModeAtoms):
    def __init__(self, atoms, calculator, minmode=davidson, H0=None, v0=None, trshift=1000,
                 trshift_factor=4., use_angles=True, use_dihedrals=True, trajectory=None,
                 extra_bonds=None):
        MinModeAtoms.__init__(self, atoms, None, minmode, H0, v0, trshift, trshift_factor,
                              True, False, None)
        self.use_angles = use_angles
        self.use_dihedrals = use_dihedrals
        self.internal = Internal(self.atoms, angles=self.use_angles, dihedrals=self.use_dihedrals, extra_bonds=extra_bonds)
        if self.internal.ninternal < 3 * len(self.atoms) - 6:
            raise RuntimeError('Not enough internal coordinates found! '
                               'Consider using angles or dihedrals.')
        self.d = 3 * self.internal.natoms
        self.ndof = 3 * len(self.internal.atoms) - 6
        self.atoms = self.internal.atoms
        self.atoms.set_calculator(calculator)

        if trajectory is not None:
            self.trajectory = Trajectory(trajectory, 'w', self.atoms)
        else:
            self.trajectory = None

    def xpolate(self, alpha):
        self.internal.xpolate(alpha)
        return self.internal.atoms.get_positions().ravel().copy()

    def kick(self, dx, minmode=False, **kwargs):
        # Project out translational/rotational motion
        lvecs, lams, rvecs = np.linalg.svd(self.internal.B, full_matrices=False)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
        dx = rvecs[indices, :].T @ (rvecs[indices, :] @ dx)
        x = self.xlast + dx
        if minmode:
            f, g = self.f_minmode(x, **kwargs)
        else:
            f, g = self.f_update(x)
        return f, g, self.internal.v1

    def f_update(self, x):
        if self.xlast is not None:
            dx = x - self.xlast
            if np.linalg.norm(dx) == 0.:
                return self.flast, self.glast

            # Predicted change in energy in Cartesian coordinates
            df_pred = dx @ self.glast + (dx @ self.H @ dx) / 2.

            ## Predicted change in energy in internal coordinates
            #dq = self.internal.B @ dx
            #Binv = rvecs[indices, :].T @ (lvecs[:, indices].T / lams[indices, np.newaxis])
            #Proj = rvecs[indices, :].T @ rvecs[indices, :]
            #gq = Binv.T @ self.glast
            #Hq = Binv.T @ (Proj @ self.H @ Proj - self.internal.D.ldot(gq)) @ Binv
            #df_pred = dq @ gq + (dq @ Hq @ dq) / 2.

            self.internal.p = self.internal.p + self.internal.B @ dx
            x = self.internal.atoms.get_positions().ravel().copy()
        else:
            df_pred = 0

        f, g = self.f(x)
        self.calls += 1

        lvecs, lams, rvecs = np.linalg.svd(self.internal.B, full_matrices=False)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
        Proj = rvecs[indices, :].T @ rvecs[indices, :]
        g = Proj @ g

        if self.xlast is not None and self.glast is not None:
            self.H = update_H(self.H, x - self.xlast, g - self.glast)
            Hlams, Hvecs = eigh(self.H)
            indices = [i for i, lam in enumerate(Hlams) if abs(lam) > 1e-12]
            self.lams = Hlams[indices]
            self.vecs = Hvecs[:, indices]

        if self.flast is not None:
            self.ratio = df_pred / (f - self.flast)

        self.flast = f
        self.xlast2 = self.xlast
        self.xlast = x.copy()
        self.glast = g.copy()

        return f, g

    def f_minmode(self, x, dxL=1e-5, maxres=5e-3, threepoint=False, **kwargs):
        f, g = self.f_update(x)
        x = self.xlast

        Htrue = NumericalHessian(self.f, x, g, dxL, threepoint)

        I = np.eye(self.d)

        lvecs, lams, rvecs = np.linalg.svd(self.internal.B, full_matrices=False)
        indices = [i for i, lam in enumerate(lams) if abs(lam) < 1e-12]

        T = rvecs[indices, :].T
        _, ntr = T.shape

        Proj = I - T @ T.T

        P = self.H
        if P is None:
            u = g / np.linalg.norm(g)
            P = I - 2 * np.outer(u, u)
        else:
            P = Proj @ P @ Proj + self.shift * T @ T.T
        H = Htrue

        lams, Vs, AVs = self.minmode(H, maxres, P, T, shift=self.shift, **kwargs)
        self.calls += H.calls

        # This sets up an initial H so that update_H has something to
        # work with.
        if self.H is None:
            if ntr == 0:
                lam0 = np.average(np.abs(lams))
            else:
                lam0 = np.average(np.abs(lams[:-ntr]))
            self.H = lam0 * Proj + self.shift * T @ T.T

        self.H = Proj @ update_H(self.H, Vs, AVs) @ Proj

        lams_all, vecs_all = eigh(self.H)
        self.shift = self.shift_factor * lams_all[-(ntr + 1)]

        lams, vecs = eigh(self.H)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
        self.lams = lams[indices]
        self.vecs = vecs[:, indices]

        return f, g
