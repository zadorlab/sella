import numpy as np

from ase.io import Trajectory
from ase.utils import basestring

from scipy.linalg import eigh, null_space
from scipy.integrate import LSODA

from sella.utilities.math import modified_gram_schmidt
from sella.constraints import merge_user_constraints, get_constraints
from sella.internal.get_internal import get_internal
#from sella.cython_routines import modified_gram_schmidt
from sella.hessian_update import update_H, symmetrize_Y
from sella.linalg import NumericalHessian, ProjectedMatrix
from sella.eigensolvers import rayleigh_ritz


class DummyTrajectory:
    def write(self):
        pass


class BasePES:
    def __init__(self, atoms, eigensolver='jd0', trajectory=None, eta=1e-4,
                 v0=None):
        self.atoms = atoms
        self.last = dict(x=None, f=None, g=None)
        self.lastlast = dict(x=None, f=None, g=None)
        self.neval = 0
        if trajectory is not None:
            if isinstance(trajectory, basestring):
                self.traj = Trajectory(trajectory, 'w', self.atoms)
            else:
                self.traj = trajectory
        else:
            self.traj = DummyTrajectory()
        #self._test_traj = Trajectory('test.traj', 'w', self.atoms)
        self.H = None
        self.eigensolver = eigensolver
        self.eta = eta
        self.v0 = v0

    calls = property(lambda self: self.neval)

    def _project_H(self):
        return self.H

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
        self.Hred = self.Ufree.T @ self._H @ self.Ufree
        self.lams, self.vecs = eigh(self.Hred)

    def _update(self):
        x = self.atoms.positions.ravel()
        if self.last['x'] is not None and np.all(x == self.last['x']):
            return False
        g = -self.atoms.get_forces().ravel()
        f = self.atoms.get_potential_energy()
        self.lastlast = self.last
        self.last = dict(x=x.copy(),
                         f=f,
                         g=g.copy())

        self.neval += 1
        self.traj.write()
        #self._test_traj.write(self.atoms + self.int.dummies)
        return True

    def converged(self, fmax, maxres=1e-5):
        return ((self.forces**2).sum(1).max() < fmax**2
                and (np.linalg.norm(self.res) < maxres))

    @property
    def B(self):
        return np.eye(len(self.x))

    @property
    def Binv(self):
        return np.eye(len(self.x))

    @property
    def x(self):
        raise NotImplementedError

    @property
    def f(self):
        self._update()
        return self.last['f']

    @property
    def g(self):
        raise NotImplementedError

    def kick(self, dxfree, diag=False, **diag_kwargs):
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        f0 = self.f
        #g0 = self.g.copy()

        # Update "free" coordinates, which in turn will also update
        # the "constrained" coordinates to reduce self.res
        self.xfree = self.xfree + dxfree
        df_actual = self.f - f0
        g0 = self.glast.copy()

        dx = self.dx(pos0, dpos0)
        if self.H is not None:
            H = self._project_H()
            df_pred = g0.T @ dx + (dx.T @ H @ dx) / 2.

        if self.H is not None:
            ratio = df_pred / df_actual
        else:
            ratio = None

        self.update_H()

        if diag:
            self.diag(**diag_kwargs)

        return self.f, self.gfree, ratio

    def dx(self, pos0):
        x1 = self.x.copy()
        pos1 = self.atoms.positions.copy()
        self.atoms.positions = pos0
        x0 = self.x.copy()
        self.atoms.positions = pos1
        return x1 - x0

    def _calc_eg(self, x):
        pos0 = self.atoms.positions.copy()
        self.x = x
        g = self.g
        f = self.f
        self.atoms.positions = pos0
        return f, g

    def diag(self, gamma=0.5, threepoint=False, maxiter=None):
        lastlast = self.lastlast.copy()
        last = self.last.copy()

        x0 = self.x.copy()
        P = self.Hred
        v0 = None
        if P is None:
            P = np.eye(len(self.xfree))
            v0 = self.gfree.copy()
        Htrue = NumericalHessian(self._calc_eg, x0, self.g.copy(), self.eta,
                                 threepoint)
        Hproj = ProjectedMatrix(Htrue, self.Ufree)
        lams, Vs, AVs = rayleigh_ritz(Hproj, gamma, P, v0=v0,
                                      method=self.eigensolver,
                                      maxiter=maxiter)
        Vs = Hproj.Vs
        AVs = Hproj.AVs
        self.x = x0
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2)
        theta, X = eigh(Atilde)
        Vs = Vs @ X
        AVs = AVs @ X
        AVstilde = AVs - self.drdx.T @ self.Ucons.T @ AVs
        self.H = update_H(self.H, Vs, AVstilde)
        self.lastlast = lastlast
        self.last = last


class CartPES(BasePES):
    def __init__(self, atoms, eigensolver='jd0', constraints=None,
                 trajectory=None, eta=1e-4, v0=None):
        BasePES.__init__(self, atoms, eigensolver, trajectory, eta, v0)
        self.last.update(gfree=None, h=None)
        self.con_user, self.target_user = merge_user_constraints(atoms,
                                                                 constraints)
        self.cons = get_constraints(atoms, self.con_user, self.target_user)

    res = property(lambda self: self.cons.get_res(self.atoms.positions))
    drdx = property(lambda self: self.cons.get_drdx(self.atoms.positions))
    Ufree = property(lambda self: self.cons.get_Ufree(self.atoms.positions))
    Ucons = property(lambda self: self.cons.get_Ucons(self.atoms.positions))

    def _update(self):
        if not BasePES._update(self):
            return False
        g = self.last['g']

        gfree = self.Ufree.T @ g
        h = g - (self.drdx.T @ self.Ucons.T) @ g

        self.last.update(gfree=gfree, h=h)
        return True

    @property
    def x(self):
        return self.atoms.positions.ravel()

    @x.setter
    def x(self, target):
        self.atoms.positions = target.reshape((-1, 3))

    @property
    def g(self):
        self._update()
        return self.last['g']

    @property
    def glast(self):
        self._update()
        return self.lastlast['g']

    @property
    def xfree(self):
        Ufree = self.cons.get_Ufree(self.atoms.positions)
        return Ufree.T @ self.x

    @xfree.setter
    def xfree(self, target):
        dx_cons = -np.linalg.pinv(self.drdx) @ self.res
        dx_free = self.Ufree @ (target - self.xfree)
        self.x = self.x + dx_free + dx_cons

    @property
    def gfree(self):
        self._update()
        return self.last['gfree']

    @property
    def forces(self):
        self._update()
        return -((self.Ufree @ self.Ufree.T) @ self.g).reshape((-1, 3))

    @property
    def h(self):
        self._update()
        return self.last['h']

    @property
    def Winv(self):
        return np.eye(self.Ufree.shape[1])

    def update_H(self):
        if self.lastlast['x'] is None:
            return
        dx = self.x - self.lastlast['x']
        dg = self.g - self.lastlast['g']
        self.H = update_H(self.H, dx, dg)


class IntPES(BasePES):
    def __init__(self, atoms, eigensolver='jd0', constraints=None,
                 trajectory=None, eta=1e-4, v0=None, angles=True,
                 dihedrals=True, extra_bonds=None):
        BasePES.__init__(self, atoms, eigensolver, trajectory, eta, v0)
        self.last.update(xint=None, gint=None, hint=None, xdummy=None)
        self.con_user, self.target_user = merge_user_constraints(atoms,
                                                                 constraints)
        self.int, self.cons, self.dummies = get_internal(atoms, self.con_user,
                                                         self.target_user)

        self._H0 = self.int.guess_hessian(atoms, self.dummies)
        self.Binvlast = self.Binv.copy()
        self._natoms = len(atoms)
        self.int_last = None

    apos = property(lambda self: self.atoms.positions)
    dpos = property(lambda self: self.dummies.positions)

    res = property(lambda self: self.cons.get_res(self.apos, self.dpos))

    @property
    def drdx(self):
        return self.cons.get_drdx(self.apos, self.dpos) @ self.Binv

    @property
    def drdxinv(self):
        return self.B @ self.cons.get_Binv(self.apos, self.dpos)

    @property
    def Ufree(self):
        # This is a bit convoluted.
        # There might be a better way to accomplish this.
        Ufree = self.cons.get_Ufree(self.apos, self.dpos)
        B = self.B @ (Ufree @ Ufree.T)
        G = B @ B.T
        lams, vecs = eigh(G)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-8]
        return vecs[:, indices]

    @property
    def Ucons(self):
        Ucons = modified_gram_schmidt(self.drdxinv.T).T
        return Ucons

    def _update(self):
        if not BasePES._update(self):
            # The geometry has not changed, so nothing needs to be done
            return
        g = self.last['g']
        xint = self.x
        gint = self.Binv[:3*len(self.atoms)].T @ g
        #h = g - self.int.B(self.atoms.positions).T @ self.drdx @ self.Ucons.T @ gint
        #hint = self.int.Binv(self.atoms.positions).T @ h

        #self.last.update(xint=xint, gint=gint, h=h, hint=hint)
        self.last.update(xint=xint, gint=gint,
                         xdummy=self.dummies.positions.ravel().copy())
        #if (self.lastlast['xint'] is not None
        #        and self.H is not None
        #        and len(xint) != len(self.lastlast['xint'])):
        #    P = self.int.B(self.lastlast['x']) @ self.Binvlast
        #    self.H = P @ self.H @ P.T
        #self.Binvlast = self.Binv.copy()

    @property
    def x(self):
        return self.int.get_q(self.apos, self.dpos)

    @x.setter
    def x(self, target):
        dq = self.int.dq_wrap(target - self.x)
        pos0 = self.atoms.positions.ravel().copy()
        dpos0 = self.dummies.positions.ravel().copy()
        B0 = self.B.copy()

        t0 = 0.
        running = True
        while running:
            pos = self.atoms.positions.ravel().copy()
            dpos = self.dummies.positions.ravel().copy()
            nx = len(pos)
            ndx = len(dpos)
            y0 = np.zeros(2 * (nx + ndx))
            y0[:nx] = pos
            y0[nx:nx+ndx] = dpos
            y0[nx+ndx:] = self.Binv @ dq
            ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-9)
            while ode.status == 'running':
                ode.step()
                t0 = ode.t
                # FIXME: This needs to be updated for the new internal stuff
                if self.int.check_for_bad_internal(self.x):
                    print("Bad internals found")
                    if self.int_last is None:
                        self.int_last = self.int
                    out = get_internal(self.atoms, self.con_user,
                                       self.target_user,
                                       intlast=self.int_last)
                    self.int, self.cons, self.dummies = out
                    dq = self.B[:, :nx+ndx] @ ode.y[nx+ndx:]
                    break
                if ode.nfev > 200:
                    raise RuntimeError("Geometry update ODE is taking "
                                       "too long to converge!")
            else:
                running = False
        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")
        self.atoms.positions = ode.y[:nx].reshape((-1, 3))
        self.dummies.positions = ode.y[nx:nx+ndx].reshape((-1, 3))

    def _q_ode(self, t, y):
        nx = 3 * len(self.atoms)
        ndx = 3 * len(self.dummies)
        x = y[:nx+ndx]
        dxdt = y[nx+ndx:]

        dydt = np.zeros_like(y)
        dydt[:nx+ndx] = dxdt

        self.atoms.positions = x[:nx].reshape((-1, 3)).copy()
        self.dummies.positions = x[nx:].reshape((-1, 3)).copy()
        D = self.int.get_D(self.apos, self.dpos)
        dydt[nx+ndx:] = -self.Binv @ D.ddot(dxdt, dxdt)

        return dydt

    @property
    def f(self):
        self._update()
        return self.last['f']

    @property
    def g(self):
        self._update()
        return self.last['gint']

    @property
    def glast(self):
        self._update()
        x = self.lastlast['x'].reshape((-1, 3))
        xd = self.lastlast['xdummy'].reshape((-1, 3))
        Binvlast = self.int.get_Binv(x, xd)
        return Binvlast[:3*len(self.atoms)].T @ self.lastlast['g']

    @property
    def h(self):
        self._update()
        return self.last['hint']

    @property
    def xfree(self):
        return self.Ufree.T @ self.x

    @xfree.setter
    def xfree(self, target):
        dx_cons = -self.drdxinv @ self.res
        dx_free = self.Ufree @ (target - self.xfree)
        self.x = self.x + self.int.dq_wrap(dx_free + dx_cons)

    @property
    def gfree(self):
        self._update()
        return self.Ufree.T @ self.g

    @property
    def forces(self):
        self._update()
        forces_int = -((self.Ufree @ self.Ufree.T) @ self.g)
        forces_cart = forces_int @ self.B
        return forces_cart.reshape((-1, 3))

    def dx(self, pos0, dpos0):
        x0 = self.int.get_q(pos0, dpos0)
        return self.int.dq_wrap(self.x - x0)

    @property
    def B(self):
        return self.int.get_B(self.apos, self.dpos)

    @property
    def Binv(self):
        return self.int.get_Binv(self.apos, self.dpos)

    @property
    def Winv(self):
        #h0 = []
        #h0 += [1.] * self.int.ncart
        #h0 += [1.] * self.int.nbonds
        #h0 += [0.25] * self.int.nangles
        #h0 += [0.10] * self.int.ndihedrals
        #h0 += [0.25] * self.int.nangle_sums
        #h0 += [0.25] * self.int.nangle_diffs
        #h0 = np.array(h0)
        #h0 = np.ones(len(self.x))
        h0 = np.diag(self.int.guess_hessian(self.atoms, self.dummies))
        Winv = self.Ufree.T @ np.diag(1./np.sqrt(h0)) @ self.Ufree
        return Winv / np.linalg.det(Winv)**(1./len(Winv))

    def _project_H(self):
        if self.int_last is None:
            return self.H
        nd0 = len(self.int_last.dummies)
        nold = 3 * (len(self.atoms) + nd0)
        self.int_last.dummies.positions[:nd0] = self.int.dummies.positions[:nd0]
        Blast = self.int_last.get_B(self.atoms.positions)
        Binvlast = self.int_last.get_Binv(self.atoms.positions)
        B = self.int.get_B(self.atoms.positions)
        Binv = self.int.get_Binv(self.atoms.positions)
        P = B[:, :nold] @ Binvlast

        if self._conin is None:
            con = dict()
        else:
            con = self._conin.copy()
        for key, val in self.int.cons.items():
            con[key] = con.get(key, []) + val
        # FIXME
        self.cons = Constraints(self.atoms + self.dummies, con, p_t=False, p_r=False)
        P2 = B[:, nold:] @ Binv[nold:, :]

        return P @ self.H @ P.T + P2 @ self.int.guess_hessian(self.atoms, self.dummies) @ P2.T

    def update_H(self):
        if self.int_last is not None:
            # TODO: This logic should be moved out of update_H
            #
            # Internal coordinate definitions changed.
            # Update the Hessian using the old internal coordinate definition,
            # then transform to the new internal coordinates.
            nd0 = len(self.int_last.dummies)
            self.int_last.dummies.positions[:nd0] = self.int.dummies.positions[:nd0]
            q1 = self.int_last.get_q(self.atoms.positions)
            dx = self.int_last.dq_wrap(q1 - self.lastlast['xint'])
            nx = 3 * len(self.atoms)

            dg = self.int_last.get_Binv(self.atoms.positions).T @ self.last['g'] - self.lastlast['gint']
            if self.H is not None:
                H = self.int_last.guess_hessian(self.atoms, self.dummies)
            else:
                H = self.H
            Blast = self.int_last.get_B(self.atoms.positions)
            Binvlast = self.int_last.get_Binv(self.atoms.positions)
            P = Blast @ Binvlast
            H = update_H(P @ H @ P.T, dx, dg)
            P2 = self.B[:, :nx + 3*nd0] @ Binvlast
            self.H = P2 @ H @ P2.T
            self.int_last = None

            # FIXME: constraints stuff has been completely changed, this needs
            # to be entirely reworked
            # now create new constraints
            if self._conin is None:
                con = dict()
            else:
                con = self._conin.copy()
            for key, val in self.int.cons.items():
                con[key] = con.get(key, []) + val
            # FIXME
            self.cons = Constraints(self.atoms + self.dummies, con, p_t=False, p_r=False)
            return

        qlast = self.int.get_q(self.lastlast['x'].reshape((-1, 3)),
                               self.lastlast['xdummy'].reshape((-1, 3)))
        dx = self.int.dq_wrap(self.x - qlast)
        #dg = self.g - self.int.Binv(self.lastlast['x']).T @ self.lastlast['g']
        dg = self.g - self.glast
        if self.H is None:
            H = self.int.guess_hessian(self.atoms, self.dummies)
        else:
            H = self.H
        P = self.B @ self.Binv
        self.H = update_H(P @ H @ P.T, dx, dg)
