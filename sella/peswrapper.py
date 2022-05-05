from typing import Union

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import LSODA
from ase import Atoms
from ase.utils import basestring
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from sella.utilities.math import modified_gram_schmidt
from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.eigensolvers import rayleigh_ritz
from sella.internal import Internals, Constraints, DuplicateInternalError


class PES:
    def __init__(
        self,
        atoms: Atoms,
        H0: np.ndarray = None,
        constraints: Constraints = None,
        eigensolver: str = 'jd0',
        trajectory: Union[str, Trajectory] = None,
        eta: float = 1e-4,
        v0: np.ndarray = None,
        proj_trans: bool = None,
        proj_rot: bool = None
    ) -> None:
        self.atoms = atoms
        if constraints is None:
            constraints = Constraints(self.atoms)
        if proj_trans is None:
            if constraints.internals['translations']:
                proj_trans = False
            else:
                proj_trans = True
        if proj_trans:
            try:
                constraints.fix_translation()
            except DuplicateInternalError:
                pass

        if proj_rot is None:
            if np.any(atoms.pbc):
                proj_rot = False
            else:
                proj_rot = True
        if proj_rot:
            try:
                constraints.fix_rotation()
            except DuplicateInternalError:
                pass
        self.cons = constraints
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
        self.curr = dict(
            x=None,
            f=None,
            g=None,
        )
        self.last = self.curr.copy()

        # Internal coordinate specific things
        self.int = None
        self.dummies = None

        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        if H0 is None:
            self.set_H(None, initialized=False)
        else:
            self.set_H(H0, initialized=True)

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

    def set_H(self, target, *args, **kwargs):
        self.H = ApproximateHessian(
            self.dim, self.ncart, target, *args, **kwargs
        )

    # Hessian of the constraints
    def get_Hc(self):
        return self.cons.hessian().ldot(self.curr['L'])

    # Hessian of the Lagrangian
    def get_HL(self):
        return self.get_H() - self.get_Hc()

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.residual()

    def get_drdx(self):
        return self.cons.jacobian()

    def _calc_basis(self):
        drdx = self.get_drdx()
        U, S, VT = np.linalg.svd(drdx)
        ncons = np.sum(S > 1e-6)
        Ucons = VT[:ncons].T
        Ufree = VT[ncons:].T
        Unred = np.eye(self.dim)
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

        scons = -Ucons @ np.linalg.lstsq(
            self.get_drdx() @ Ucons,
            self.get_res(),
            rcond=None,
        )[0]
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
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr['x'] = x
        self.curr['f'] = f
        self.curr['g'] = g
        self._update_basis()
        return True

    def _update_basis(self):
        drdx, Ucons, Unred, Ufree = self._calc_basis()
        self.curr['drdx'] = drdx
        self.curr['Ucons'] = Ucons
        self.curr['Unred'] = Unred
        self.curr['Ufree'] = Ufree

        if self.curr['g'] is None:
            L = None
        else:
            L = np.linalg.lstsq(drdx.T, self.curr['g'], rcond=None)[0]
        self.curr['L'] = L

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

    def diag(self, gamma=0.1, threepoint=False, maxiter=None):
        if self.curr['f'] is None:
            self._update(feval=True)

        Ufree = self.get_Ufree()
        nfree = Ufree.shape[1]

        P = self.get_HL().project(Ufree)

        if P.B is None or self.first_diag:
            v0 = self.v0
            if v0 is None:
                v0 = self.get_g() @ Ufree
        else:
            v0 = None

        if P.B is None:
            P = np.eye(nfree)
        else:
            P = P.asarray()

        Hproj = NumericalHessian(self._calc_eg, self.get_x(), self.get_g(),
                                 self.eta, threepoint, Ufree)
        Hc = self.get_Hc()
        rayleigh_ritz(Hproj - Ufree.T @ Hc @ Ufree, gamma, P, v0=v0,
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
    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        H0: np.ndarray = None,
        iterative_stepper: int = 0,
        auto_find_internals: bool = True,
        **kwargs
    ):
        self.int_orig = internals
        new_int = internals.copy()
        if auto_find_internals:
            new_int.find_all_bonds()
            new_int.find_all_angles()
            new_int.find_all_dihedrals()
        new_int.validate_basis()

        PES.__init__(
            self,
            atoms,
            *args,
            constraints=new_int.cons,
            H0=None,
            proj_trans=False,
            proj_rot=False,
            **kwargs
        )

        self.int = new_int
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof
        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.jacobian()
            P = B @ np.linalg.pinv(B)
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper

    dpos = property(lambda self: self.dummies.positions.copy())

    def _set_x_iterative(self, target):
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        pos1 = None
        dpos1 = None
        x0 = self.get_x()
        dx_initial = target - x0
        g0 = np.linalg.lstsq(
            self.int.jacobian(),
            self.curr.get('g', np.zeros_like(dx_initial)),
            rcond=None,
        )[0]
        for _ in range(10):
            dx = np.linalg.lstsq(
                self.int.jacobian(),
                self.wrap_dx(target - self.get_x()),
                rcond=None,
            )[0].reshape((-1, 3))
            if np.sqrt((dx**2).sum() / len(dx)) < 1e-6:
                break
            self.atoms.positions += dx[:len(self.atoms)]
            self.dummies.positions += dx[len(self.atoms):]
            if pos1 is None:
                pos1 = self.atoms.positions.copy()
                dpos1 = self.dummies.positions.copy()
        else:
            print('Iterative stepper failed!')
            if self.iterative_stepper == 2:
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                return
            self.atoms.positions = pos1
            self.dummies.positions = dpos1
        dx_final = self.get_x() - x0
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    # Position getter/setter
    def set_x(self, target):
        if self.iterative_stepper:
            res = self._set_x_iterative(target)
            if res is not None:
                return res
        dx = target - self.get_x()

        t0 = 0.
        Binv = np.linalg.pinv(self.int.jacobian())
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ self.curr.get('g', np.zeros_like(dx))))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
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
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    def get_x(self):
        return self.int.calc()

    # Hessian of the constraints
    def get_Hc(self):
        D_cons = self.cons.hessian().ldot(self.curr['L'])
        B_int = self.int.jacobian()
        Binv_int = np.linalg.pinv(B_int)
        B_cons = self.cons.jacobian()
        L_int = self.curr['L'] @ B_cons @ Binv_int
        D_int = self.int.hessian().ldot(L_int)
        Hc = Binv_int.T @ (D_cons - D_int) @ Binv_int
        return Hc

    def get_drdx(self):
        # dr/dq = dr/dx dx/dq
        return PES.get_drdx(self) @ np.linalg.pinv(self.int.jacobian())

    def _calc_basis(self, internal=None, cons=None):
        if internal is None:
            internal = self.int
        if cons is None:
            cons = self.cons
        B = internal.jacobian()
        Ui, Si, VTi = np.linalg.svd(B)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Vnred = VTi[:nnred].T
        Siinv = np.diag(1 / Si[:nnred])
        drdxnred = cons.jacobian() @ Vnred @ Siinv
        drdx = drdxnred @ Unred.T
        Uc, Sc, VTc = np.linalg.svd(drdxnred)
        ncons = np.sum(Sc > 1e-6)
        Ucons = Unred @ VTc[:ncons].T
        Ufree = Unred @ VTc[ncons:].T
        return drdx, Ucons, Unred, Ufree

    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = np.linalg.pinv(self.int.jacobian())
        return f, g_cart @ Binv[:len(g_cart)]

    def update_internals(self, dx):
        self._update(True)

        nold = 3 * (len(self.atoms) + len(self.dummies))

        # FIXME: Testing to see if disabling this works
        #if self.bad_int is not None:
        #    for bond in self.bad_int['bonds']:
        #        self.int_orig.forbid_bond(bond)
        #    for angle in self.bad_int['angles']:
        #        self.int_orig.forbid_angle(angle)

        # Find new internals, constraints, and dummies
        new_int = self.int_orig.copy()
        new_int.find_all_bonds()
        new_int.find_all_angles()
        new_int.find_all_dihedrals()
        new_int.validate_basis()
        new_cons = new_int.cons

        # Calculate B matrix and its inverse for new and old internals
        Blast = self.int.jacobian()
        B = new_int.jacobian()
        Binv = np.linalg.pinv(B)
        Dlast = self.int.hessian()
        D = new_int.hessian()

        # # Projection matrices
        # P2 = B[:, nold:] @ Binv[nold:, :]

        # Update the info in self.curr
        x = new_int.calc()
        g = -self.atoms.get_forces().ravel() @ Binv[:3*len(self.atoms)]
        drdx, Ucons, Unred, Ufree = self._calc_basis(
            internal=new_int,
            cons=new_cons,
        )
        L = np.linalg.lstsq(drdx.T, g, rcond=None)[0]

        # Update H using old data where possible. For new (dummy) atoms,
        # use the guess hessian info.
        H = self.get_H().asarray()
        Hcart = Blast.T @ H @ Blast
        Hcart += Dlast.ldot(self.curr['g'])
        Hnew = Binv.T[:, :nold] @ (Hcart - D.ldot(g)) @ Binv
        self.dim = len(x)
        self.set_H(Hnew)

        self.int = new_int
        self.cons = new_cons

        self.curr.update(x=x, g=g, drdx=drdx, Ufree=Ufree,
                         Unred=Unred, Ucons=Ucons, L=L, B=B, Binv=Binv)

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        # dx_r = self.wrap_dx(dx) @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        B = self.int.jacobian()
        return -((Ufree @ Ufree.T) @ g @ B).reshape((-1, 3))

    def wrap_dx(self, dx):
        return self.int.wrap(dx)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        D = self.int.hessian()
        Binv = np.linalg.pinv(self.int.jacobian())
        D_tmp = -Binv @ D.rdot(dxdt)
        dydt[1] = D_tmp @ dxdt
        dydt[2] = D_tmp @ g

        return dydt.ravel()

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        # FIXME: Testing to see if this works
        #if self.bad_int is not None:
        #    self.update_internals(dx)
        #    self.bad_int = None

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

        B = self.int.jacobian()
        Binv = np.linalg.pinv(B)
        self.curr.update(B=B, Binv=Binv)
        return True
