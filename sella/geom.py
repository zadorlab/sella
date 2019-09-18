import numpy as np

from ase.io import Trajectory

from sella.constraints import Constraints


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
    def x(self):
        raise NotImplementedError

    @property
    def f(self):
        self._update()
        return self.last['f']

    @property
    def g(self):
        raise NotImplementedError


class CartGeom(Geom):
    def __init__(self, atoms, constraints=None, trajectory=None):
        Geom.__init__(self, atoms, trajectory)
        self.last.update(gfree=None, h=None)
        self.cons = Constraints(atoms, constraints)

    # Is this kosher?
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

    def split_dx(self, dx):
        """Splits a real Cartesian displacement vector into its
        "free" and "cons" components"""
        dx_free = (self.Ufree @ self.Ufree.T) @ dx
        dx_cons = (self.Ucons @ self.Ucons.T) @ dx
        return dx_free, dx_cons


#class IntGeom(Geom):
#    def __init__(self, atoms):
#        Geom.__init__(self, atoms)
#        self.int = Internal(self.atoms)
#        self.last.update(x_int=None, g_int=None)
#
#    def _update(self):
#        if not Geom._update(self):
#            # The geometry has not changed, so nothing needs to be done
#            return
#        x_int = self.int.p.copy()
#        g_int = self.int.Binv @ self.last['g']
#
#        self.last.update(x_int=x_int, g_int=g_int)
#
#    @property
#    def x(self):
#        return self.int.p
#
#    @x.setter
#    def x(self, target):
#        self.int.p = target
#
#    @property
#    def f(self):
#        self._update()
#        return self.last['f']
#
#    @property
#    def g(self):
#        self._update()
#        return self.last['g_int']
