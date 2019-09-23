import numpy as np

from ase.io import Trajectory

from scipy.linalg import eigh
from scipy.integrate import LSODA, BDF, RK23, RK45, Radau

from sella.constraints import Constraints
from sella.internal import Internal
from sella.cython_routines import simple_ortho, modified_gram_schmidt

class DummyTrajectory:
    def write(self):
        pass


class Geom:
    def __init__(self, atoms, trajectory=None):
        self.atoms = atoms
        self.last = dict(x=None, f=None, g=None)
        self.neval = 0
        if trajectory is not None:
            self.traj = Trajectory(trajectory, 'w', self.atoms)
        else:
            self.traj = DummyTrajectory()

    def _update(self):
        x = self.atoms.positions.ravel()
        if self.last['x'] is not None and np.all(x == self.last['x']):
            return False
        g = -self.atoms.get_forces().ravel()
        f = self.atoms.get_potential_energy()
        self.last = dict(x=x.copy(),
                         f=f,
                         g=g.copy())

        self.neval += 1
        self.traj.write()
        return True

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

    def kick(self, dxfree):
        pos0 = self.atoms.positions.copy()
        self.xfree = self.xfree + dxfree
        self._update()
        x1 = self.x.copy()
        pos1 = self.atoms.positions.copy()
        self.atoms.positions = pos0
        x0 = self.x.copy()
        self.atoms.positions = pos1
        return x1 - x0

    def H_to_cart(self, H):
        return H

    def H_to_int(self, H):
        return H

    @property
    def w(self):
        return np.eye(len(self.x))


class CartGeom(Geom):
    def __init__(self, atoms, constraints=None, trajectory=None):
        Geom.__init__(self, atoms, trajectory)
        self.last.update(gfree=None, h=None)
        self.cons = Constraints(atoms, constraints)

    res = property(lambda self: self.cons.res(self.atoms.positions))
    drdx = property(lambda self: self.cons.drdx(self.atoms.positions))
    Ufree = property(lambda self: self.cons.Ufree(self.atoms.positions))
    Ucons = property(lambda self: self.cons.Ucons(self.atoms.positions))

    def _update(self):
        if not Geom._update(self):
            return False
        g = self.last['g']

        gfree = self.Ufree.T @ g
        h = g - (self.drdx @ self.Ucons.T) @ g

        self.last.update(gfree=gfree, h=h)
        return True

    @property
    def x(self):
        return self.atoms.positions.ravel()

    @x.setter
    def x(self, target):
        self.atoms.positions = target.reshape((-1, 3))

    def dx(self, x0, split=False):
        dx = self.x - x0
        if split:
            dx_free = (self.Ufree @ self.Ufree.T) @ dx
            dx_cons = (self.Ucons @ self.Ucons.T) @ dx
            return dx_free, dx_cons
        else:
            return dx

    @property
    def g(self):
        self._update()
        return self.last['g']

    @property
    def xfree(self):
        Ufree = self.cons.Ufree(self.atoms.positions)
        return Ufree.T @ self.x

    @xfree.setter
    def xfree(self, target):
        dx_cons = -np.linalg.pinv(self.drdx.T) @ self.res
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


class IntGeom(Geom):
    def __init__(self, atoms, constraints=None, trajectory=None, angles=True,
                 dihedrals=True, extra_bonds=None):
        Geom.__init__(self, atoms, trajectory)
        self.int = Internal(self.atoms, angles, dihedrals, extra_bonds)
        self.cons = Constraints(self.atoms, constraints, p_t=False, p_r=False)
        self._H0 = self.int.guess_hessian(atoms)

    res = property(lambda self: self.cons.res(self.atoms.positions))

    @property
    def drdx(self):
        drdx = self.cons.drdx(self.atoms.positions)
        return self.int.Binv(self.atoms.positions).T @ drdx

    @property
    def Ufree(self):
        # This is a bit convoluted. There might be a better way to accomplish this.
        Ufree = self.cons.Ufree(self.atoms.positions)
        B = self.int.B(self.atoms.positions)
        B = B @ (Ufree @ Ufree.T)
        G = B @ B.T
        lams, vecs = eigh(G)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-8]
        return vecs[:, indices]

    @property
    def Ucons(self):
        return modified_gram_schmidt(self.drdx)

    def _update(self):
        if not Geom._update(self):
            # The geometry has not changed, so nothing needs to be done
            return
        xint = self.int.q(self.atoms.positions)
        gint = self.int.Binv(self.atoms.positions).T @ self.last['g']
        h = gint - (self.drdx @ self.Ucons.T) @ gint

        self.last.update(xint=xint, gint=gint, h=h)

    @property
    def x(self):
        return self.int.q(self.atoms.positions)

    @x.setter
    def x(self, target):
        pos0 = self.atoms.positions.ravel().copy()
        nx = len(pos0)
        y0 = np.zeros(2 * nx)
        y0[:nx] = pos0
        y0[nx:] = self.int.Binv(self.atoms.positions) @ self.int.q_wrap(target - self.x)
        ode = LSODA(self._q_ode, 0., y0, t_bound=1., jac=self._q_jac, atol=1e-8)
        #ode = BDF(self._q_ode, 0., y0, t_bound=1., jac=self._q_jac)
        while ode.status == 'running':
            ode.step()
            if ode.nfev > 200:
                raise RuntimeError("Geometry update ODE is taking too long to converge!")
        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")
        self.atoms.positions = ode.y[:nx].reshape((-1, 3))

    def _q_ode(self, t, y):
        nx = len(y) // 2
        x = y[:nx]
        dxdt = y[nx:]

        dydt = np.zeros_like(y)
        dydt[:nx] = dxdt

        self.atoms.positions = x.reshape((-1, 3)).copy()
        D = self.int.D(self.atoms.positions)
        Binv = self.int.Binv(self.atoms.positions)
        dydt[nx:] = -Binv @ D.ddot(dxdt, dxdt)
        #print(dydt)

        return dydt

    def _q_jac(self, t, y):
        nx = len(y) // 2
        dxdt = y[nx:]
        jac = np.zeros((2 * nx, 2 * nx))
        jac[:nx, nx:] = np.eye(nx)
        D = self.int.D(self.atoms.positions)
        Binv = self.int.Binv(self.atoms.positions)
        jac[nx:, nx:] = -2 * Binv @ D.rdot(dxdt)
        return jac

    @property
    def f(self):
        self._update()
        return self.last['f']

    @property
    def g(self):
        self._update()
        return self.last['gint']

    @property
    def h(self):
        self._update()
        return self.last['h']

    @property
    def xfree(self):
        return self.Ufree.T @ self.x

    @xfree.setter
    def xfree(self, target):
        dx_cons = -np.linalg.pinv(self.drdx.T) @ self.res
        dx_free = self.Ufree @ (target - self.xfree)
        self.x = self.x + self.int.q_wrap(dx_free + dx_cons)

    def kick(self, dxfree):
        return self.int.q_wrap(Geom.kick(self, dxfree))

    @property
    def gfree(self):
        self._update()
        return self.Ufree.T @ self.g

    @property
    def forces(self):
        self._update()
        forces_int = -((self.Ufree @ self.Ufree.T) @ self.g)
        forces_cart = forces_int @ self.int.B(self.atoms.positions)
        return forces_cart.reshape((-1, 3))

    def dx(self, x0, split=False):
        dx = self.int.q_wrap(self.x - x0)
        if split:
            dx_free = (self.Ufree @ self.Ufree.T) @ dx
            dx_cons = (self.Ucons @ self.Ucons.T) @ dx
            return dx_free, dx_cons
        else:
            return dx

    @property
    def B(self):
        return self.int.B(self.atoms.positions)

    @property
    def Binv(self):
        return self.int.Binv(self.atoms.positions)

    def H_to_cart(self, Hint):
        if Hint is None:
            return Hint
        D = self.int.D(self.atoms.positions)
        return self.B.T @ Hint @ self.B + D.ldot(self.g)

    def H_to_int(self, Hcart):
        if Hcart is None:
            return Hcart
        D = self.int.D(self.atoms.positions)
        return self.Binv.T @ (Hcart - D.ldot(self.g)) @ self.Binv

    @property
    def Winv(self):
        H0free = self.Ufree.T @ self._H0 @ self.Ufree
        lams, vecs = np.linalg.eigh(H0free)
        lams = np.sqrt(lams/ np.prod(lams)**(1./len(lams)))
        return vecs @ np.diag(1. / lams) @ vecs.T


