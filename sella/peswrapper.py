import numpy as np
from scipy.linalg import eigh
from scipy.integrate import LSODA
from ase.utils import basestring
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from sella.utilities.math import modified_gram_schmidt
from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.constraints import get_constraints, merge_user_constraints, ConstraintBuilder
from sella.eigensolvers import rayleigh_ritz
from sella.internal.get_internal import InternalBuilder


class PES:
    def __init__(self,
                 atoms,
                 H0=None,
                 constraints=None,
                 eigensolver='jd0',
                 trajectory=None,
                 eta=1e-4,
                 v0=None,
                 proj_trans=True,
                 proj_rot=True):
        self.atoms = atoms
        self.proj_trans = proj_trans
        self.proj_rot = proj_rot
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

        self.neval = 0
        self.curr = dict(x=None,
                         f=None,
                         g=None,
                         gfree=None,
                         hfree=None,
                         )
        self.last = self.curr.copy()

        # Internal coordinate specific things
        self.int = None
        self.dummies = None

        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        self.set_H(H0)

        self.savepoint = dict(apos=None, dpos=None)
        self.first_diag = True

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def save(self):
        self.savepoint = dict(apos=self.apos, dpos=self.dpos)

    def restore(self):
        apos = self.savepoint['apos']
        dpos = self.savepoint['dpos']
        assert apos is not None
        self.atoms.positions = apos
        if dpos is not None:
            self.dummies.positions = dpos

    def set_constraints(self, c):
        self.conbuilder = ConstraintBuilder(self.atoms, c)
        self.cons = self.conbuilder.get_constraints(
            self.atoms,
            proj_trans=self.proj_trans,
            proj_rot=self.proj_rot,
        )

    # Position getter/setter
    def set_x(self, target):
        diff = target - self.get_x()
        self.atoms.positions = target.reshape((-1, 3))
        return diff, diff, self.curr.get('g', np.zeros_like(diff))

    def get_x(self):
        return self.apos.ravel().copy()

    # Hessian getter/setter
    def get_H(self):
        return self.H

    def set_H(self, target):
        self.H = ApproximateHessian(self.dim, self.ncart, target)

    # Hessian of the constraints
    def get_Hc(self):
        return self.cons.get_D(self.apos, self.dpos).ldot(self.curr['L'])

    # Hessian of the Lagrangian
    def get_HL(self):
        return self.get_H() - self.get_Hc()

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.get_res(self.apos, self.dpos).copy()

    def get_drdx(self):
        return self.cons.get_drdx(self.apos, self.dpos).copy()

    def _calc_basis(self):
        drdx = self.get_drdx()
        Ucons = modified_gram_schmidt(drdx.T)
        Unred = np.eye(self.dim)
        Ufree = modified_gram_schmidt(Unred, Ucons)
        return drdx, Ucons, Unred, Ufree

    def write_traj(self):
        if self.traj is not None:
            self.traj.write()

    def eval(self):
        self.neval += 1
        f = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        self.write_traj()
        return f, g

    def _calc_eg(self, x):
        self.save()
        self.set_x(x)

        f, g = self.eval()

        self.restore()
        return f, g

    def get_scons(self):
        """Returns displacement vector for linear constraint correction."""
        Ucons = self.get_Ucons()

        scons = -Ucons @ np.linalg.lstsq(self.get_drdx() @ Ucons,
                                         self.get_res(), rcond=None)[0]
        return scons

    def _update(self, feval=True):
        x = self.get_x()
        new_point = True
        if self.curr['x'] is not None and np.all(x == self.curr['x']):
            if feval and self.curr['f'] is None:
                new_point = False
            else:
                return False
        drdx, Ucons, Unred, Ufree = self._calc_basis()


        if feval:
            f, g = self.eval()
            L = np.linalg.lstsq(drdx.T, g, rcond=None)[0]
        else:
            f = None
            g = None
            L = None

        if new_point:
            self.last = self.curr
            self.curr = dict(x=x, f=f, g=g, drdx=drdx, Ucons=Ucons,
                             Unred=Unred, Ufree=Ufree, L=L)
        else:
            self.curr['f'] = f
            self.curr['g'] = g
            self.curr['L'] = L
        return True

    def _update_H(self, dx, dg):
        if self.last['x'] is None or self.last['g'] is None:
            return
        self.H.update(dx, dg)

    def get_f(self):
        self._update()
        return self.curr['f']

    def get_g(self):
        self._update()
        return self.curr['g'].copy()

    def get_Unred(self):
        self._update(False)
        return self.curr['Unred']

    def get_Ufree(self):
        self._update(False)
        return self.curr['Ufree']

    def get_Ucons(self):
        self._update(False)
        return self.curr['Ucons']

    def diag(self, gamma=0.5, threepoint=False, maxiter=None):
        Unred = self.get_Unred()

        P = self.get_HL().project(Unred)

        if P.B is None or self.first_diag:
            v0 = self.v0
            if v0 is None:
                v0 = self.get_g() @ Unred
        else:
            v0 = None

        if P.B is None:
            P = np.eye(len(self.get_x()))
        else:
            P = P.asarray()

        Hproj = NumericalHessian(self._calc_eg, self.get_x(), self.get_g(),
                                 self.eta, threepoint, Unred)
        Hc = self.get_Hc()
        rayleigh_ritz(Hproj - Unred.T @ Hc @ Unred, gamma, P, v0=v0,
                      method=self.eigensolver,
                      maxiter=maxiter)

        # Extract eigensolver iterates
        Vs = Hproj.Vs
        AVs = Hproj.AVs

        # Re-calculate Ritz vectors
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2) - Vs.T @ Hc @ Vs
        _, X = eigh(Atilde)

        # Rotate Vs and AVs into X
        Vs = Vs @ X
        AVs = AVs @ X

        # Update the approximate Hessian
        self.H.update(Vs, AVs)

        self.first_diag = False

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        return -((Ufree @ Ufree.T) @ g).reshape((-1, 3))

    def converged(self, fmax, cmax=1e-5):
        fmax1 = np.linalg.norm(self.get_projected_forces(), axis=1).max()
        cmax1 = np.linalg.norm(self.get_res())
        conv = (fmax1 < fmax) and (cmax1 < cmax)
        return conv, fmax1, cmax1

    def get_W(self):
        return np.eye(self.dim)

    def wrap_dx(self, dx):
        return dx

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        return g.T @ dx + (dx.T @ H @ dx) / 2.

    def kick(self, dx, diag=False, **diag_kwargs):
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H.asarray()

        dx_initial, dx_final, g_par = self.set_x(x0 + dx)

        #dx_actual = self.wrap_dx(self.get_x() - x0)
        df_pred = self.get_df_pred(dx_initial, g0, B0)
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        if df_pred is None:
            ratio = None
        else:
            ratio = df_actual / df_pred

        self._update_H(dx_final, dg_actual)

        if diag:
            self.diag(**diag_kwargs)

        return ratio


class InternalPES(PES):
    def __init__(self, atoms, *args, H0=None, angles=True, dihedrals=True,
                 extra_bonds=None, atol=15, **kwargs):
        PES.__init__(self, atoms, *args, H0=None, **kwargs)
        self.atol = atol
        intbuilder = InternalBuilder(self.atoms, self.conbuilder)
        intbuilder.find_bonds()
        intbuilder.find_angles(self.atol)
        intbuilder.find_dihedrals()
        self.int = intbuilder.to_internal()
        self.cons = intbuilder.to_constraints()
        self.dummies = intbuilder.dummies
        self.dinds = intbuilder.dinds

        self.dim = len(self.get_x())
        self.ncart = self.int.ncart
        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.get_B(self.apos, self.dpos)
            P = B @ self.int.get_Binv(self.apos, self.dpos)
            H0 = P @ self.int.guess_hessian(self.atoms, self.dummies, h0cart=1) @ P
        self.set_H(H0)
        self.H.initialized = False

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None

    dpos = property(lambda self: self.dummies.positions.copy())

    # Position getter/setter
    def set_x(self, target):
        dx = target - self.get_x()

        t0 = 0.
        Binv = self.int.get_Binv(self.apos, self.dpos)
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ self.curr.get('g', np.zeros_like(dx))))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internal(
                self.apos, self.dpos
            )
            if self.bad_int is not None:
                print('Bad internals found!')
                break
            if ode.nfev > 1000:
                view(self.atoms + self.dummies)
                raise RuntimeError("Geometry update ODE is taking too long "
                                   "to converge!")

        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")


        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.get_B(self.apos, self.dpos)
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    def get_x(self):
        return self.int.get_q(self.apos, self.dpos)

    # Hessian of the constraints
    def get_Hc(self):
        D_cons = self.cons.get_D(self.apos, self.dpos)
        Binv_int = self.int.get_Binv(self.apos, self.dpos)
        B_cons = self.cons.get_B(self.apos, self.dpos)
        D_int = self.int.get_D(self.apos, self.dpos)
        L_int = self.curr['L'] @ B_cons @ Binv_int
        Hc = Binv_int.T @ (D_cons.ldot(self.curr['L'])
                           - D_int.ldot(L_int)) @ Binv_int
        return Hc

    def get_drdx(self):
        # dr/dq = dr/dx dx/dq
        return PES.get_drdx(self) @ self.int.get_Binv(self.apos, self.dpos)

    def _calc_basis(self):
        drdx = self.get_drdx()
        Ucons = modified_gram_schmidt(drdx.T)
        na = 3 * len(self.atoms)
        B = self.int.get_B(self.apos, self.dpos)
        Udummy = modified_gram_schmidt(B[:, na:])
        Binv = self.int.get_Binv(self.apos, self.dpos)
        Unred = modified_gram_schmidt(B @ Binv, Udummy)
        Ufree = modified_gram_schmidt(Unred, Ucons)
        return drdx, Ucons, Unred, Ufree

    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = self.int.get_Binv(self.apos, self.dpos)
        return f, g_cart @ Binv[:len(g_cart)]

    def update_internals(self, dx):
        self._update(True)

        if self.bad_int is None:
            check_bonds = None
            check_angles = None
        else:
            check_bonds = self.bad_int['bonds']
            check_angles = self.bad_int['angles']

        # Find new internals, constraints, and dummies
        intbuilder = InternalBuilder(self.atoms, self.conbuilder, self.dummies,
                                     self.dinds)
        intbuilder.find_bonds(check_bonds)
        intbuilder.find_angles(self.atol, check_angles)
        intbuilder.find_dihedrals()
        new_int = intbuilder.to_internal()
        new_cons = intbuilder.to_constraints()
        new_dummies = intbuilder.dummies
        nold = 3 * (len(self.atoms) + len(self.dummies))

        # Calculate B matrix and its inverse for new and old internals
        Blast = self.int.get_B(self.apos, self.dpos)
        B = new_int.get_B(self.apos, new_dummies.positions)
        Binv = new_int.get_Binv(self.apos, new_dummies.positions)
        Dlast = self.int.get_D(self.apos, self.dpos)
        D = new_int.get_D(self.apos, new_dummies.positions)

        # Projection matrices
        P2 = B[:, nold:] @ Binv[nold:, :]

        # Update the info in self.curr
        x = new_int.get_q(self.apos, new_dummies.positions)
        g = -self.atoms.get_forces().ravel() @ Binv[:3*len(self.atoms)]
        drdx = new_cons.get_drdx(self.apos, new_dummies.positions) @ Binv
        L = np.linalg.lstsq(drdx.T, g, rcond=None)[0]
        Ucons = modified_gram_schmidt(drdx.T)
        Unred = modified_gram_schmidt(B @ B.T)
        Ufree = modified_gram_schmidt(Unred, Ucons)

        # Update H using old data where possible. For new (dummy) atoms,
        # use the guess hessian info.
        H = self.get_H().asarray()
        Hcart = Blast.T @ H @ Blast + Dlast.ldot(self.curr['g'])
        Hnew = Binv.T[:, :nold]@Hcart@Binv[:nold] - Binv.T@D.ldot(g) @ Binv
        self.dim = len(x)
        self.set_H(Hnew)

        self.int = new_int
        self.cons = new_cons
        self.dummies = new_dummies

        self.curr.update(x=x, g=g, drdx=drdx, Ufree=Ufree,
                         Unred=Unred, Ucons=Ucons, L=L, B=B, Binv=Binv)

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        #dx_r = self.wrap_dx(dx) @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        B = self.int.get_B(self.apos, self.dpos)
        return -((Ufree @ Ufree.T) @ g @ B).reshape((-1, 3))

    def wrap_dx(self, dx):
        return self.int.dq_wrap(dx)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        D = self.int.get_D(self.apos, self.dpos)
        Binv = self.int.get_Binv(self.apos, self.dpos)
        dydt[1] = -Binv @ D.ddot(dxdt, dxdt)
        dydt[2] = -Binv @ D.ddot(dxdt, g)

        return dydt.ravel()

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        if self.bad_int is not None:
            self.update_internals(dx)
            self.bad_int = None

        return ratio

    def write_traj(self):
        if self.traj is not None:
            energy = self.atoms.calc.results['energy']
            forces = np.zeros((len(self.atoms) + len(self.dummies), 3))
            forces[:len(self.atoms)] = self.atoms.calc.results['forces']
            atoms_tmp = self.atoms + self.dummies
            atoms_tmp.calc = SinglePointCalculator(atoms_tmp, energy=energy,
                                                   forces=forces)
            self.traj.write(atoms_tmp)

    def _update(self, feval=True):
        if not PES._update(self, feval=feval):
            return

        B = self.int.get_B(self.apos, self.dpos)
        Binv = self.int.get_Binv(self.apos, self.dpos)
        self.curr.update(B=B, Binv=Binv)
        return True
