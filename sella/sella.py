# Proof of concept MinMode implementation for Atoms which explicitly
# removes constraints from the degrees of freedom exposed to the
# optimizer

import warnings

import numpy as np
from scipy.linalg import null_space, eigh, lstsq

from ase.io import Trajectory

from .cython_routines import modified_gram_schmidt
from .internal_cython import cart_to_internal
from .eigensolvers import davidson, NumericalHessian, ProjectedMatrix, project_rotation
from .hessian_update import update_H, symmetrize_Y

def _sort_indices(indices):
    if indices[0] > indices[-1]:
        indices = tuple(reversed(indices))
    return indices

class MinModeAtoms(object):
    def __init__(self, atoms, calc, eigensolver=davidson,
                 project_translations=True, project_rotations=None,
                 constraints=None, trajectory=None, shift=1000,
                 v0=None, maxres=1e-5):
        self.atoms = atoms.copy()
        if self.atoms.constraints:
            warnings.warn('ASE Atoms object has attached constraints, '
                          'but these will be ignored! Please provide '
                          'constraints to the MinMode object '
                          'initializer instead!')
            self.atoms.set_constraint()

        self.atoms.set_calculator(calc)

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
        self._initialize_constraints(constraints, project_translations, project_rotations)

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
        self.H = None
        self.calls = 0
        self._basis_xlast = None

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, target):
        if target is None:
            self._H = None
            self._Hred = None
            self._lams = None
            self._vecs = None
            return
        self._H = target
        self._Hred = (self.Tm.T @ self.Tfree) @ target @ (self.Tfree.T @ self.Tm)
        lams, vecs = eigh(self._Hred)
        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
        self._lams = lams[indices]
        self._vecs = vecs[:, indices]

    @property
    def Hred(self):
        return self._Hred

    @property
    def lams(self):
        return self._lams

    @property
    def vecs(self):
        return self._vecs

    @property
    def x(self):
        # The current atomic positions in a flattened array.
        # Full dimensional.
        return self.atoms.positions.ravel().copy()

    @x.setter
    def x(self, target):
        self.atoms.set_positions(target.reshape((-1, 3)))

    @property
    def x_m(self):
        # The current atomic positions in the subspace orthogonal to
        # any constraints.
        return self.Tm.T @ self.x

    @property
    def res(self):
        # The vector of residuals, indicating how much our current
        # coordinate deviates from the constraint targets
        self._basis_update()
        return self._res

    @property
    def drdx(self):
        self._basis_update()
        return self._drdx

    @property
    def Tc(self):
        # An orthonormal basis of vectors spanning the subspace of the
        # constraints.
        self._basis_update()
        return self._Tc

    @property
    def Tm(self):
        # The null space of Tc, or an orthonormal basis of vectors
        # spanning the subspace orthogonal to the constraints.
        self._basis_update()
        return self._Tm

    @property
    def Tfree(self):
        # The subspace orthogonal to any fixed Atoms constraints.
        # This is the subspace in which self.H is stored.
        # Not orthogonal to all constraints (use Tm if this is what
        # you are looking for)
        self._basis_update()
        return self._Tfree

    @property
    def d_free(self):
        self._basis_update()
        return self._Tfree.shape[1]

    def _basis_update(self):
        if self._basis_xlast is None or np.any(self.x != self._basis_xlast):
            self._calc_constr_basis()
            self._basis_xlast = self.x.copy()

    def kick(self, dx_m, minmode=False, **kwargs):
        if np.linalg.norm(dx_m) == 0 and self.last['f'] is not None:
            return self.last['f'], self.last['g_m'], dx_m

        res_orig = self.res.copy()

        #dx_c = -self.Tc @ self.res
        dx_c, _, _, _ = lstsq(-self.drdx.T, self.res)

        dx = self.Tm @ dx_m + dx_c
        f1, g1 = self.f_update(self.x + dx)

        if minmode:
            self.f_minmode(**kwargs)

        return f1, self.last['g_m'], dx_m

    def xpolate(self, alpha):
        if alpha != 1.:
            raise NotImplementedError
        return self.last['x']

    def _calc_constr_basis(self):
        # If we don't have any constraints, then this is a very
        # easy task.
        if self.nconstraints == 0:
            self._res = np.empty(0)
            self._Tm = np.eye(self.d)
            self._Tfree = np.eye(self.d)
            self._Tc = np.empty((self.d, 0), dtype=np.float64)
            return

        # Calculate Tc and Tm (see above)
        self._res = np.zeros(self.nconstraints)  # The constraint residuals
        self._drdx = np.zeros((self.d, self.nconstraints))
        
        n = 0
        # For now we ignore the target value contained in the "fix"
        # constraints dict.
        del_indices = []
        for index, dim in self.constraints.get('fix', dict()).keys():
            # The residual for fixed atom constraints will always be
            # zero, because no motion that would break those constraints
            # is allowed during dynamics.
            # self._res[n] = 0  # <-- not necessary to state explicitly
            idx_flat = 3 * index + dim  # Flattened coordinate index
            # Add the index to the list of indices to delete
            del_indices.append(idx_flat)
            self._drdx[idx_flat, n] = 1.
            n += 1

        free_indices = [i for i in range(self.d) if i not in del_indices]

        self._Tfree = np.eye(self.d)[:, free_indices]

        # Now consider translation
        tvec = np.zeros_like(self.atoms.positions)
        tvec[:, 0] = 1. / len(self.atoms)
        for dim, target in self.constraints.get('translations', dict()).items():
            self._res[n] = np.average(self.atoms.positions[:, dim]) - target
            self._drdx[:, n] = np.roll(tvec, dim, axis=1).ravel()
            n += 1

        # And rotations
        if self.constraints.get('rotations', False):
            drdx_rot = project_rotation(self.x)
            _, nrot = drdx_rot.shape
            # TODO: Figure out how to use these constraints to push the
            # structure into a consistent orientation in space. That
            # might involve reworking how rotational constraints are
            # enforced completely.
            # self._res[n : n+nrot] = ???
            self._drdx[:, n : n+nrot] = drdx_rot
            n += nrot

        # Bonds
        b_indices = np.array(list(self.constraints.get('bonds',
                dict()).keys()), dtype=np.int32).reshape((-1, 2))
        # Angles
        a_indices = np.array(list(self.constraints.get('angles',
                dict()).keys()), dtype=np.int32).reshape((-1, 3))
        # Dihedrals
        d_indices = np.array(list(self.constraints.get('dihedrals',
                dict()).keys()), dtype=np.int32).reshape((-1, 4))

        mask = np.ones(self.nconstraints - n, dtype=np.uint8)
        r_int, drdx_int, _ = cart_to_internal(self.atoms.positions,
                                              b_indices,
                                              a_indices,
                                              d_indices,
                                              mask,
                                              gradient=True)

        m = 0
        for indices in b_indices:
            r_int[m] -= self.constraints['bonds'][tuple(indices)]
            m += 1

        for indices in a_indices:
            r_int[m] -= self.constraints['angles'][tuple(indices)]
            m += 1

        for indices in d_indices:
            val = r_int[m]
            target = self.constraints['dihedrals'][tuple(indices)]
            r_int[m] = (val - target + np.pi) % (2 * np.pi) - np.pi
            m += 1

        self._res[n:] = r_int
        self._drdx[:, n:] = drdx_int.T

        # If we've made it this far and there's only one constraint,
        # we don't need to work any harder to get a good orthonormal basis.
        if self._drdx.shape[1] == 1:
            self._Tc = self._drdx / np.linalg.norm(self._drdx)
        # Otherwise, we need to orthonormalize everything
        else:
            self._Tc = modified_gram_schmidt(self._drdx)

        self._Tm = null_space(self._Tc.T)

    def _initialize_constraints(self, constraints, p_t, p_r):
        self.nconstraints = 0

        # Make a copy, because we are going to be popping entries
        # off the object to ensure that we've parsed everything,
        # and we don't want to modify the user-provided object,
        # since they might have other plans for it!
        if constraints is None:
            con = dict()
        else:
            con = constraints.copy()

        # First, consider fixed atoms
        # Note that unlike other constraints, we do not allow the user
        # to specify a target for fixed atom constraints. If the user
        # wants to fix atoms to a position other than their initial
        # value, they must manually move the atoms first.
        con_f = con.pop('fix', tuple())
        fix = dict()
        # We also keep track of which dimensions any atom has been
        # fixed in for projecting out translations. E.g. if any atom
        # has been fixed in the X direction, then we do not project
        # out overall translation in that direction.
        fixed_dims = [p_t, p_t, p_t]
        # The user can provide fixed atoms constraints either as a
        # list of atom indices, or as a list of (index, dimension)
        # tuples, or they can mix the two approaches.
        for arg in con_f:
            if isinstance(arg, int):
                index = arg
                for dim, ipos in enumerate(self.pos0[index]):
                    fix[(index, dim)] = ipos
                fixed_dims = [False, False, False]
            else:
                index, dim = arg
                fix[(index, dim)] = ipos
                fixed_dim[dim] = False

        self.constraints['fix'] = fix
        self.nconstraints += len(fix)

        # Second, we consider translational/rotational motion.
        self.constraints['translations'] = dict()
        for dim, fixed in enumerate(fixed_dims):
            if fixed:
                self.constraints['translations'][dim] = np.average(self.atoms.positions[:, dim])
        #self.constraints['translations'] = fixed_dims
        self.constraints['rotations'] = p_r
        self.nconstraints += np.sum(fixed_dims) + 3 * p_r

        # Next we look at the bonds
        con_b = con.pop('bonds', tuple())
        # The constraints will be temporarily held in the 'bonds' dict.
        bonds = dict()
        # Assume user provided constraints in the form
        # ((index_a, index_b), target_distance),
        for indices, target in con_b:
            # If "indices" is actually an int (rather than a tuple),
            # then they didn't provide a target distance, so we will
            # use the current value.
            if isinstance(indices, int):
                indices = (indices, target)
                target = self.atoms.get_distance(*indices)
            bonds[_sort_indices(indices)] = target

        self.constraints['bonds'] = bonds
        self.nconstraints += len(bonds)

        # Then angles and dihedrals, which are very similar
        con_a = con.pop('angles', tuple())
        angles = dict()
        for arg in con_a:
            if len(arg) == 2:
                indices = _sort_indices(arg[0])
                target = arg[1]
            else:
                indices = _sort_indices(arg)
                target = self.atoms.get_angle(*indices) * np.pi / 180.
            angles[_sort_indices(indices)] = target

        self.constraints['angles'] = angles
        self.nconstraints += len(angles)

        con_d = con.pop('dihedrals', tuple())
        dihedrals = dict()
        for arg in con_d:
            if len(arg) == 2:
                indices = _sort_indices(arg[0])
                target = arg[1]
            else:
                indices = _sort_indices(arg)
                target = self.atoms.get_dihedral(*indices) * np.pi / 180.
            dihedrals[_sort_indices(indices)] = target

        self.constraints['dihedrals'] = dihedrals
        self.nconstraints += len(dihedrals)

        # If con is still non-empty, then we didn't process all of
        # the user-provided constraints, so throw an error.
        if con:
            raise ValueError("Don't know what to do with constraint types: {}".format(con.keys()))

    def calc_eg(self, x=None):
        if x is not None:
            self.x = x

        e = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        self.calls += 1

        if self.trajectory is not None:
            self.trajectory.write(self.atoms)
        return e, g

    def f_update(self, x):
        if self.last['f'] is not None and np.all(x == self.x):
            return self.last['f'], self.last['g']

        f, g = self.calc_eg(x)
        #xlast = self.last['x']
        h = g - self.drdx @ self.Tc.T @ g

        if self.last['f'] is not None and self.H is not None:
            # Calculate predicted vs actual change in energy
            self.df = f - self.last['f']
            dx = self.x - self.last['x']

            dx_free = self.Tfree.T @ dx
            self.df_pred = self.last['g'].T @ dx + (dx_free.T @ self.H @ dx_free) / 2.
            self.ratio = self.df_pred / self.df

            # Update Hessian matrix
            dh_free = self.Tfree.T @ (h - self.last['h'])
            self.H = update_H(self.H, dx_free, dh_free)
            #dg_free = self.Tfree.T @ (g - self.last['g'])
            #self.H = update_H(self.H, dx_free, dg_free)

        g_m = self.Tm.T @ g
        if self.H is not None:
            g_m -= (self.Tm.T @ self.Tfree) @ self.H @ (self.Tfree.T @ (self.Tc @ self.res))

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
        #Htrue = NumericalHessian(self.calc_eg, x, g, dxL, threepoint)

        # If we don't have an approximate Hessian yet, then
        H = self.H
        if H is None:
            v = self.v0
            if v is None:
                v = self.last['g']
            v = self.Tfree.T @ v
            H = np.eye(self.d_free) - 2 * np.outer(v, v) / (v @ v)

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
        #self.H = update_H(self.H, self.Tfree.T @ Vs, self.Tfree.T @ AVs)

    def converged(self, ftol):
        return ((np.linalg.norm(self.last['g_m']) < ftol)
                and np.all(np.abs(self.res) < self.maxres))

