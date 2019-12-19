import numpy as np

from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.constraints import get_constraints, merge_user_constraints
from sella.eigensolvers import rayleigh_ritz


class BasePES:
    def __init__(self,
                 atoms,
                 constraints=None,
                 eigensolver='jd0',
                 trajectory=None,
                 eta=1e-4,
                 v0=None):
        self.atoms = atoms
        self.set_constraints(constraints)
        self.eigensolver = eigensolver

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                self.traj = Trajectory(trajectory, 'w', self.atoms)
            else:
                self.traj = trajectory
        else:
            self.traj = None

        self.eta = eta
        self.v0 = v0

        # Initialize empty constraints
        self.neval = 0
        self.last = dict(x=None,
                         f=None,
                         g=None,
                         gfree=None,
                         hfree=None,
                         )
        self.lastlast = self.last.copy()
        self.dummies = None
        self.dim = None

    def set_constraints(self, c=None):
        if c is None:
            self.con_user = dict()
            self.target_user = dict()
            self.cons = None
            return

        self.con_user, self.target_user = merge_user_constraints(self.atoms, c)
        self.cons = get_constraints(self.atoms, self.con_user, self.target_user)

    def get_H(self, reduced=False):
        if reduced:
            return self.H.project(self.Ufree)
        return self.H

    def set_H(self, target):
        self.H = ApproximateHessian(self.dim, target)

    def set_x(self, target):
        raise NotImplementedError

    def get_x(self):
        raise NotImplementedError

    def get_Dcons(self):
        raise NotImplementedError

    def _calc_basis(self):
        raise NotImplementedError

    def _calc_eg(self, x=None):
        raise NotImplementedError

    def _update(self, feval=True):
        x = self.get_x()
        new_point = True
        if self.last['x'] is not None and np.all(x == self.last['x']):
            if feval and self.last['f'] is None:
                new_point = False
            else:
                return False
        drdx, Ucons, Ufree = self._calc_basis()

        if feval:
            f, g = self._calc_eg()
            L = np.linalg.lstsq(drdx @ Ucons, Ucons.T @ g, rcond=None)[0]
        else:
            f = None
            g = None
            L = None

        if new_point:
            self.lastlast = last
            self.last = dict(x=x, f=f, g=g, drdx=drdx, Ufree=Ufree,
                             Ucons=Ucons, L=L)
        else:
            self.last['f'] = f
            self.last['g'] = g
            self.last['L'] = L
        return True

    def get_f(self):
        self._update()
        return self.last['f']

    def get_g(self, projected=False):
        self._update()
        if projected:
            return self.last['Ufree'].T @ self.last['g']
        return self.last['g'].copy()

    def get_Ufree(self):
        self._update(False)
        return self.last['Ufree']

    def get_Ucons(self):
        self._update(False)
        return self.last['Ufree']

    def diag(self, gamma=0.5, threepoint=False, maxiter=None):
        # Backup current state (this might not be necessary)
        x0 = self.get_x()
        last = self.last.copy()
        lastlast = self.lastlast.copy()

        P = self.get_H(reduced=True)
        v0 = None
        if P is None:
            v0 = self.v0
            if v0 is None:
                v0 = self.get_g(projected=True)
        Hproj = NumericalHessian(self._calc_eg, x0, self.get_g(), self.eta,
                                 threepoint, self.get_Ufree())
        rayleigh_ritz(Hproj, gamma, P, v0=v0, method=self.eigensolver,
                      maxiter=maxiter)

        # Restore original state (this might not be necessary)
        self.set_x(x0)
        self.last = last
        self.lastlast = lastlast

        # Extract eigensolver iterates
        Vs = Hproj.Vs
        AVs = Hproj.AVs

        # Re-calculate Ritz vectors
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2)
        _, X = eigh(Atilde)

        # Rotate Vs and AVs into X
        Vs = Vs @ X
        AVs = AVs @ X

        # Add the Lagrange multiplier gradient contribution
        AVstilde = AVs - self.get_Dcons().ldot(self.last['L']) @ Vs

        # Update the approximate Hessian
        self.H.update(Vs, AVstilde)




class CartPES(BasePES):
    def __init__(self,
                 atoms,
                 H0=None,
                 constraints=None,
                 eigensolver='jd0',
                 trajectory=None,
                 eta=1e-4,
                 v0=None):
        BasePES.__init__(self, atoms, constraints, eigensolver, trajectory,
                         eta, v0)
        self.dim = len(self.get_x())
        self.set_H(H0)

    def set_x(self, target):
        self.atoms.positions = target.reshape((-1, 3))

    def get_x(self):
        return self.atoms.positions.ravel().copy()

    def get_Dcons(self):
        return self.cons.get_D(self.atoms.positions)

    def _calc_basis(self):
        drdx = self.cons.get_drdx(atoms.positions)
        Ucons = modified_gram_schmidt(drdx.T)
        Ufree = modified_gram_schmidt(np.eye(self.dim), Ucons)
        return drdx, Ucons, Ufree

    def _calc_eg(self, x=None):
        if x is not None:
            self.set_x(x)
        g = -atoms.get_forces().ravel()
        f = atoms.get_potential_energy()
        return f, g
