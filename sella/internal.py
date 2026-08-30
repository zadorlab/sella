from typing import (
    Tuple, Callable, Iterator, Union, TypeVar, Optional, List, Dict, Type,
    NamedTuple,
)
from itertools import product, combinations, permutations
from functools import partialmethod
import warnings

from scipy import sparse
from scipy.linalg import svdvals
import numpy as np
from ase import Atom, Atoms, units
from ase.cell import Cell
from ase.geometry import complete_cell, minkowski_reduce
from ase.data import covalent_radii
from ase.constraints import (
    FixConstraint, FixAtoms, FixCom, FixBondLengths, FixCartesian, FixInternals
)

import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, vmap, jvp, device_get

from sella.linalg import (
    SparseInternalJacobian, SparseInternalHessian, SparseInternalHessians,
    SparseInternalHessiansSkeleton
)


# =============================================================================
# Lightweight atoms-like wrapper for efficient coordinate calculations
# =============================================================================
# Creating ASE Atoms objects has significant overhead (Atoms.__init__ validates
# positions, sets up constraints, etc.). This lightweight wrapper provides just
# the positions and cell attributes needed for coordinate calculations, reducing
# Atoms.__init__ calls from ~1258 to ~266 per optimization run (~79% reduction).
# =============================================================================

class LightAtoms:
    """Lightweight wrapper providing positions and cell without Atoms overhead."""
    __slots__ = ('positions', 'cell')

    def __init__(self, positions: np.ndarray, cell: np.ndarray) -> None:
        self.positions = positions
        self.cell = cell


class _DisjointSet:
    __slots__ = ('parent', 'n_active')

    def __init__(self, size: int, n_active: int = None) -> None:
        self.parent = list(range(size))
        self.n_active = size if n_active is None else n_active

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_a] = root_b
        self.n_active -= 1
        return True


# =============================================================================
# Vectorized (batched) internal coordinate functions using jax.vmap
# =============================================================================
# These compute gradients/hessians for ALL coordinates of a given type at once,
# avoiding Python loop overhead. JAX's vmap automatically vectorizes over the
# batch dimension, providing significant speedup for coordinate calculations.
# =============================================================================

def _bond_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Bond length: pos shape (2, 3), tvec shape (1, 3)"""
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])


def _angle_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Angle value: pos shape (3, 3), tvec shape (2, 3)"""
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    # Clamp to avoid NaN from arccos
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


def _dihedral_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Dihedral angle: pos shape (4, 3), tvec shape (3, 3)"""
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


# Batched gradient functions: input shapes (n_coords, n_atoms, 3), (n_coords, n_vecs, 3)
# Output shapes: (n_coords, n_atoms, 3)
_bond_grad_batched = jit(vmap(grad(_bond_value, argnums=0), in_axes=(0, 0)))
_angle_grad_batched = jit(vmap(grad(_angle_value, argnums=0), in_axes=(0, 0)))
_dihedral_grad_batched = jit(vmap(grad(_dihedral_value, argnums=0), in_axes=(0, 0)))

# Batched value functions
_bond_value_batched = jit(vmap(_bond_value, in_axes=(0, 0)))
_angle_value_batched = jit(vmap(_angle_value, in_axes=(0, 0)))
_dihedral_value_batched = jit(vmap(_dihedral_value, in_axes=(0, 0)))

# Batched hessian functions: output shapes (n_coords, n_atoms, 3, n_atoms, 3)
_bond_hess_batched = jit(vmap(jacfwd(grad(_bond_value, argnums=0), argnums=0), in_axes=(0, 0)))
_angle_hess_batched = jit(vmap(jacfwd(grad(_angle_value, argnums=0), argnums=0), in_axes=(0, 0)))
_dihedral_hess_batched = jit(vmap(jacfwd(grad(_dihedral_value, argnums=0), argnums=0), in_axes=(0, 0)))

# =============================================================================
# Hessian-vector product (HVP) functions using forward-over-reverse mode
# =============================================================================
# These compute H @ v directly without materializing the full Hessian matrix.
# Uses jvp(grad(f), x, v) which is O(n) instead of O(n²) for forming full Hessian.
# =============================================================================

def _bond_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single bond without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_bond_value, argnums=0), primals, tangents)
    return hvp_result


def _angle_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single angle without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_angle_value, argnums=0), primals, tangents)
    return hvp_result


def _dihedral_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single dihedral without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_dihedral_value, argnums=0), primals, tangents)
    return hvp_result


# Batched HVP functions: compute H @ v for all coords at once
# Input shapes: pos (n_coords, n_atoms, 3), tvec (n_coords, n_vecs, 3), tangent (n_coords, n_atoms, 3)
# Output shapes: (n_coords, n_atoms, 3)
_bond_hvp_batched = jit(vmap(_bond_hvp_single, in_axes=(0, 0, 0)))
_angle_hvp_batched = jit(vmap(_angle_hvp_single, in_axes=(0, 0, 0)))
_dihedral_hvp_batched = jit(vmap(_dihedral_hvp_single, in_axes=(0, 0, 0)))


# =============================================================================
# Cell-derivative functions for unit cell optimization
# =============================================================================
# These compute derivatives of internal coordinates with respect to cell matrix.
# Used for coupled atomic + cell optimization in periodic systems.
#
# The chain rule is: d(coord)/d(cell) = d(coord)/d(tvec) @ d(tvec)/d(cell)
# Since tvec = ncvec @ cell, we have d(tvec)/d(cell) = ncvec (Kronecker structure)
# =============================================================================

def _bond_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Bond length with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (1, 3) @ (3, 3) -> (1, 3)
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])


def _angle_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Angle with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (2, 3) @ (3, 3) -> (2, 3)
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


def _dihedral_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Dihedral angle with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (3, 3) @ (3, 3) -> (3, 3)
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


# Single-coordinate cell gradients: output shape (3, 3) for d(coord)/d(cell)
_bond_cell_grad_single = jit(grad(_bond_with_cell, argnums=2))
_angle_cell_grad_single = jit(grad(_angle_with_cell, argnums=2))
_dihedral_cell_grad_single = jit(grad(_dihedral_with_cell, argnums=2))

# Batched cell gradients: input (n_coords, n_atoms, 3), (n_coords, n_vecs, 3), (3, 3)
# Output: (n_coords, 3, 3)
# Note: cell is NOT batched (same cell for all coords), so in_axes=(0, 0, None)
_bond_cell_grad_batched = jit(vmap(_bond_cell_grad_single, in_axes=(0, 0, None)))
_angle_cell_grad_batched = jit(vmap(_angle_cell_grad_single, in_axes=(0, 0, None)))
_dihedral_cell_grad_batched = jit(vmap(_dihedral_cell_grad_single, in_axes=(0, 0, None)))


# =============================================================================
# Block size for GPU/SIMD efficiency
# =============================================================================
# Padding arrays to multiples of BLOCK_SIZE improves GPU performance through
# better warp-level parallelism and memory coalescing. Also reduces JAX JIT
# recompilation when array sizes change.
# =============================================================================
BLOCK_SIZE = 64

# Maximum working set for one periodic pair-image distance chunk. The full
# all-pairs table scales as n_pairs * n_images * 3 and can collide with MLIP
# memory on smaller GPUs; chunking bounds that temporary without changing the
# selected bonds.
PERIODIC_PAIR_CHUNK_BYTES = 16 * 1024 * 1024


class _BatchedCoordFamily(NamedTuple):
    key: str
    n_atoms: int
    n_tvecs: int
    width: int
    value_fn: Callable
    grad_fn: Callable
    hess_fn: Callable
    hvp_fn: Callable
    cell_grad_fn: Callable


class _BatchedCoordArrays(NamedTuple):
    indices: np.ndarray
    ncvecs: np.ndarray
    indices_padded: np.ndarray
    ncvecs_padded: np.ndarray
    n_actual: int
    flat_cols: np.ndarray
    csr_offset: int = 0


_BATCHED_COORD_FAMILIES: Tuple[_BatchedCoordFamily, ...] = (
    _BatchedCoordFamily(
        key='bonds',
        n_atoms=2,
        n_tvecs=1,
        width=6,
        value_fn=_bond_value_batched,
        grad_fn=_bond_grad_batched,
        hess_fn=_bond_hess_batched,
        hvp_fn=_bond_hvp_batched,
        cell_grad_fn=_bond_cell_grad_batched,
    ),
    _BatchedCoordFamily(
        key='angles',
        n_atoms=3,
        n_tvecs=2,
        width=9,
        value_fn=_angle_value_batched,
        grad_fn=_angle_grad_batched,
        hess_fn=_angle_hess_batched,
        hvp_fn=_angle_hvp_batched,
        cell_grad_fn=_angle_cell_grad_batched,
    ),
    _BatchedCoordFamily(
        key='dihedrals',
        n_atoms=4,
        n_tvecs=3,
        width=12,
        value_fn=_dihedral_value_batched,
        grad_fn=_dihedral_grad_batched,
        hess_fn=_dihedral_hess_batched,
        hvp_fn=_dihedral_hvp_batched,
        cell_grad_fn=_dihedral_cell_grad_batched,
    ),
)


IVec = Tuple[int, int, int]


class NoValidInternalError(ValueError):
    pass


class DuplicateInternalError(ValueError):
    pass


class DuplicateConstraintError(DuplicateInternalError):
    pass


def _gradient(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(grad(func, argnums=0))


def _hessian(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(jacfwd(jacrev(func, argnums=0), argnums=0))


class Coordinate:
    nindices = None
    kwargs = None

    def __init__(
        self,
        indices: Tuple[int, ...],
    ) -> None:
        if self.nindices is not None:
            assert len(indices) == self.nindices
        self.indices = np.array(indices, dtype=np.int32)
        self.kwargs = dict()

    def reverse(self) -> 'Coordinate':
        raise NotImplementedError

    def __eq__(self, other: 'Coordinate') -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self.indices) != len(other.indices):
            return False
        if np.all(self.indices == other.indices):
            return True
        return False

    def __add__(self, other: 'Coordinate') -> 'Coordinate':
        raise NotImplementedError

    def split(self) -> Tuple['Coordinate', 'Coordinate']:
        raise NotImplementedError

    def __repr__(self) -> str:
        out = [f'indices={self.indices}']
        out += [f'{key}={val}' for key, val in self.kwargs.items()]
        str_out = ', '.join(out)
        return f'{self.__class__.__name__}({str_out})'

    def copy(self) -> 'Coordinate':
        new = self.__class__.__new__(self.__class__)
        new.indices = self.indices.copy()
        new.kwargs = {
            key: val.copy() if hasattr(val, 'copy') else val
            for key, val in self.kwargs.items()
        }
        if hasattr(self, 'q_prev'):
            q_prev = self.q_prev
            new.q_prev = None if q_prev is None else q_prev.copy()
        return new

    @staticmethod
    def _eval0(pos: jnp.ndarray, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        return float(self._eval0(
            atoms.positions[self.indices], **self.kwargs
        ))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval1(
            atoms.positions[self.indices], **self.kwargs
        ))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        return np.array(self._eval2(
            atoms.positions[self.indices], **self.kwargs
        ))

    def _check_derivative(
        self, atoms: Atoms, delta: float, atol: float, order: int
    ) -> bool:
        """Compare an analytic derivative against central finite differences.

        ``order=1`` checks the gradient against differences of the value;
        ``order=2`` checks the Hessian against differences of the gradient.
        Warns and returns False if the max abs mismatch exceeds ``atol``.
        """
        if order == 1:
            derivative = 'Gradient'
            f0 = self.calc
            f1 = self.calc_gradient
        elif order == 2:
            derivative = 'Hessian'
            f0 = self.calc_gradient
            f1 = self.calc_hessian
        else:
            raise ValueError(f'Order {order} gradients are not implemented')

        atoms0 = atoms.copy()
        g_ref = f1(atoms0)
        g_numer = np.zeros_like(g_ref)
        atoms = atoms0.copy()
        for i, idx in enumerate(self.indices):
            for j in range(3):
                atoms.positions[idx, j] = atoms0.positions[idx, j] + delta
                fplus = f0(atoms)
                atoms.positions[idx, j] = atoms0.positions[idx, j] - delta
                fminus = f0(atoms)
                g_numer[i, j] = (fplus - fminus) / (2 * delta)
                atoms.positions[idx, j] = atoms0.positions[idx, j]
        if np.max(np.abs(g_numer - g_ref)) > atol:
            warnings.warn(f'{derivative}s for {self} failed numerical test!')
            return False
        return True

    def check_gradient(
        self, atoms: Atoms, delta: float = 1e-4, atol: float = 1e-6
    ) -> bool:
        """Finite-difference check of this coordinate's analytic gradient."""
        return self._check_derivative(atoms, delta, atol, order=1)

    def check_hessian(
        self, atoms: Atoms, delta: float = 1e-4, atol: float = 1e-6
    ) -> bool:
        """Finite-difference check of this coordinate's analytic Hessian."""
        return self._check_derivative(atoms, delta, atol, order=2)


class Internal(Coordinate):
    union = None
    diff = None

    def __init__(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None
    ) -> None:
        Coordinate.__init__(self, indices)

        if self.nindices is not None:
            if ncvecs is None:
                ncvecs = np.zeros((self.nindices - 1, 3), dtype=np.int32)
            else:
                ncvecs = np.asarray(ncvecs).reshape((self.nindices - 1, 3))
        else:
            if ncvecs is not None:
                raise ValueError(
                    "{} does not support ncvecs"
                    .format(self.__class__.__name__)
                )
            ncvecs = np.empty((0, 3), dtype=np.int32)
        self.kwargs['ncvecs'] = ncvecs

    def reverse(self) -> 'Internal':
        return self.__class__(self.indices[::-1], -self.kwargs['ncvecs'][::-1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        # Coordinates are direction-agnostic: equal when indices+ncvecs match
        # in the given order, OR when the reversed coordinate matches. Each
        # branch must pair an index match with the corresponding ncvecs match
        # (a forward index match with forward ncvecs, reverse with reverse) --
        # mixing them across branches lets distinct coordinates compare equal.
        if (
            Coordinate.__eq__(self, other)
            and np.all(self.kwargs['ncvecs'] == other.kwargs['ncvecs'])
        ):
            return True
        srev = self.reverse()
        if (
            Coordinate.__eq__(srev, other)
            and np.all(srev.kwargs['ncvecs'] == other.kwargs['ncvecs'])
        ):
            return True
        return False

    def __add__(self, other: object) -> 'Internal':
        if self.union is None:
            return NotImplemented
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            raise NoValidInternalError(
                'Cannot add {} object to itself.'
                .format(self.__class__.__name__)
            )

        for s, o in product([self, self.reverse()], [other, other.reverse()]):
            if (
                np.all(s.indices[1:] == o.indices[:-1])
                and np.all(s.kwargs['ncvecs'][1:] == o.kwargs['ncvecs'][:-1])
            ):
                new_indices = [*s.indices, o.indices[-1]]
                new_ncvecs = [*s.kwargs['ncvecs'], o.kwargs['ncvecs'][-1]]
                return self.union(new_indices, new_ncvecs)
        raise NoValidInternalError(
            '{} indices do not overlap!'.format(self.__class__.__name__)
        )

    def split(self) -> Tuple['Internal', 'Internal']:
        if self.diff is None:
            raise RuntimeError(
                "Don't know how to split a {}!".format(self.__class__.__name__)
            )
        return (
            self.diff(self.indices[:-1], self.kwargs['ncvecs'][:-1]),
            self.diff(self.indices[1:], self.kwargs['ncvecs'][1:])
        )

    @staticmethod
    def _eval0(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return float(self._eval0(atoms.positions[self.indices], tvecs))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return np.array(self._eval1(atoms.positions[self.indices], tvecs))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return np.array(self._eval2(atoms.positions[self.indices], tvecs))

    @staticmethod
    def _eval_cell_grad(
        pos: jnp.ndarray, ncvecs: jnp.ndarray, cell: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute gradient of coordinate with respect to cell matrix.

        Must be overridden in subclasses (Bond, Angle, Dihedral).
        Returns shape (3, 3) for d(coord)/d(cell).
        """
        raise NotImplementedError

    def calc_cell_gradient(self, atoms: Atoms) -> np.ndarray:
        """Compute gradient of this coordinate w.r.t. cell matrix.

        Returns:
            np.ndarray: Shape (3, 3) array of d(coord)/d(cell[i,j])
        """
        ncvecs = jnp.asarray(self.kwargs['ncvecs'], dtype=np.float64)
        cell = jnp.asarray(
            atoms.cell.array,
            dtype=np.float64
        )
        pos = jnp.asarray(atoms.positions[self.indices], dtype=np.float64)
        return np.array(self._eval_cell_grad(pos, ncvecs, cell))


def _translation(
    pos: jnp.ndarray,
    dim: int,
) -> float:
    return pos[:, dim].mean()


class Translation(Coordinate):
    def __init__(
        self,
        indices: Tuple[int, ...],
        dim: int,
    ) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs['dim'] = dim

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs['dim'] != other.kwargs['dim']:
            return False
        if set(self.indices) != set(other.indices):
            return False
        return True

    _eval0 = staticmethod(jit(_translation))
    _eval1 = staticmethod(_gradient(_translation))
    _eval2 = staticmethod(_hessian(_translation))


# Nominally, jax.numpy.linalg.eigh supports auto-differentiation,
# but if any of the eigenvalues are degenerate, the derivatives
# of *all* eigenvectors will be NaN. Worryingly, this seems to be
# the case when the molecule in question is sufficiently high-symmetry
# (e.g. methane) and has not been rotated.
#
# We are assuming here that the eigenvector of interest corresponds
# to a simple (non-degenerate) eigenvalue (though we permit the
# possibility of there being other degenerate eigenvalues).


# Numerical tolerances for the quaternion / F-matrix rotation coordinates.
# Eigenvalue-gap floor: when an F-matrix eigenvalue is within this of the top
# eigenvalue (a degenerate top eigenspace, e.g. 2-atom / linear fragments), the
# gap is treated as zero so the 1/gap term in the eigenvector-derivative
# pseudoinverse is dropped rather than blowing up.
_ROT_EIG_GAP_TOL = 1e-10
# Cartesian displacement used only for finite-difference second derivatives
# in a genuinely degenerate quaternion eigenspace (normally a diatomic).
_ROT_DEGENERATE_FD_STEP = 2e-6
# |q0 - 1| below which the rotation is treated as ~identity: asinc =
# arccos(x)/sqrt(1-x^2) and its derivatives switch to their Taylor expansion
# instead of the (removable) singularity at x = 1.
_ROT_NEAR_IDENTITY_TOL = 1e-8


def _rotation_hessian_np(pos, axis, refpos, q_stable=None):
    """Closed-form Hessian of the rotation coordinate w.r.t. positions.

    Uses an analytic eigenvector second derivative that handles degenerate
    eigenvalues (linear molecules) via the Moore-Penrose pseudoinverse,
    avoiding the NaN that JAX autodiff produces in that case.

    Parameters
    ----------
    pos : ndarray (N, 3)
    axis : int (0, 1, or 2)
    refpos : ndarray (N, 3), already centered
    q_stable : ndarray (4,), optional stabilized quaternion

    Returns
    -------
    hessian : ndarray (N, 3, N, 3)
    """
    return _rotation_hessian_single(
        np.asarray(pos, dtype=np.float64),
        axis,
        np.asarray(refpos, dtype=np.float64),
        q_stable=q_stable,
    )


def _build_F_matrix_np(dx, refpos):
    """Build the 4x4 quaternion F-matrix in numpy.

    Parameters
    ----------
    dx : ndarray (N, 3), centered positions (pos - centroid)
    refpos : ndarray (N, 3), centered reference positions
    """
    R = dx.T @ refpos
    Rtr = np.trace(R)
    Ftop = np.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
    F = np.empty((4, 4))
    F[0, 0] = Rtr
    F[0, 1:] = Ftop
    F[1:, 0] = Ftop
    F[1:, 1:] = -Rtr * np.eye(3) + R + R.T
    return F


def _build_F_matrices_np(pos_group, ref_group):
    """Build batched 4x4 quaternion F-matrices for same-sized fragments."""
    dx = pos_group - pos_group.mean(axis=1, keepdims=True)
    R = np.matmul(dx.swapaxes(1, 2), ref_group)
    Rtr = np.trace(R, axis1=1, axis2=2)
    Ftop = np.stack([
        R[:, 1, 2] - R[:, 2, 1],
        R[:, 2, 0] - R[:, 0, 2],
        R[:, 0, 1] - R[:, 1, 0],
    ], axis=1)
    n_batch = len(pos_group)
    F = np.zeros((n_batch, 4, 4))
    F[:, 0, 0] = Rtr
    F[:, 0, 1:] = Ftop
    F[:, 1:, 0] = Ftop
    for axis in range(3):
        F[:, 1 + axis, 1 + axis] = -Rtr
    F[:, 1:, 1:] += R + R.transpose(0, 2, 1)
    return F


def _stabilize_quaternion(F, q_prev):
    """Compute branch-stable quaternion from F-matrix eigendecomposition.

    Projects q_prev onto the top eigenspace of F and normalizes.
    For non-degenerate cases (1D top eigenspace), this is equivalent
    to picking the rightmost eigenvector with consistent sign.
    For degenerate cases (2-atom / linear fragments with 2D+ top
    eigenspace), this picks the linear combination closest to q_prev,
    ensuring continuity across geometry steps.
    """
    ws, vecs = np.linalg.eigh(F)
    return _stabilize_quaternion_from_eigh(ws, vecs, q_prev)


def _stabilize_quaternion_from_eigh(ws, vecs, q_prev):
    """Compute branch-stable quaternion from pre-computed eigendecomposition."""
    if q_prev is None:
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])
    top_mask = (ws[-1] - ws) < _ROT_EIG_GAP_TOL
    top_vecs = vecs[:, top_mask]
    coeffs = top_vecs.T @ q_prev
    q = top_vecs @ coeffs
    norm = np.linalg.norm(q)
    if norm < 1e-14:
        q = vecs[:, -1].copy()
    else:
        q /= norm
    if q[0] < 0:
        q = -q
    return q


def _asinc_np(x):
    """Inverse sinc: arccos(x) / sqrt(1 - x^2), with Taylor branch near x=1."""
    if x < 0.97:
        return np.arccos(x) / np.sqrt(1.0 - x * x)
    y = x - 1.0
    return (1.0 - y / 3 + 2 * y**2 / 15 - 2 * y**3 / 35
            + 8 * y**4 / 315 - 8 * y**5 / 693 + 16 * y**6 / 3003
            - 16 * y**7 / 6435 + 128 * y**8 / 109395
            - 128 * y**9 / 230945)


def _expmap_np(q):
    """Convert unit quaternion to rotation vector (3,)."""
    a = _asinc_np(q[0])
    return 2.0 * q[1:4] * a


def _rotation_3axis_jacobian_np(pos, refpos, q, ws=None, vecs=None):
    """Jacobian of all 3 rotation values w.r.t. positions, using quaternion q.

    Parameters
    ----------
    pos    : (N, 3)
    refpos : (N, 3), already centered
    q      : (4,), stabilized quaternion
    ws/vecs: optional eigendecomposition of the current F matrix

    Returns
    -------
    jac : (3, N, 3) — Jacobian[axis, atom, xyz]
    """
    N = len(pos)
    if ws is None or vecs is None:
        dx = pos - pos.mean(0)
        F = _build_F_matrix_np(dx, refpos)
        ws, vecs = np.linalg.eigh(F)

    c = q
    gaps = ws - ws[-1]
    safe_inv = np.where(np.abs(gaps) > _ROT_EIG_GAP_TOL,
                        1.0 / np.where(np.abs(gaps) > _ROT_EIG_GAP_TOL, gaps, 1.0),
                        0.0)

    Prefpos = refpos  # refpos is already centered at construction
    dFc = _apply_dF(Prefpos, c, N)  # (N, 3, 4)
    dFc_flat = dFc.reshape(N * 3, 4)
    dc_flat = -(vecs @ (safe_inv[:, None] * (vecs.T @ dFc_flat.T))).T  # (N*3, 4)

    q0 = c[0]
    asinc_val = _asinc_np(q0)
    if abs(q0 - 1.0) < _ROT_NEAR_IDENTITY_TOL:
        y = q0 - 1.0
        dasinc = -1.0 / 3 + 4 * y / 15
    elif abs(q0) < 1.0 - 1e-12:
        s2 = 1 - q0**2
        s = np.sqrt(s2)
        ac = np.arccos(q0)
        dasinc = -1.0 / s2 + q0 * ac / (s * s2)
    else:
        dasinc = 0.0

    jac = np.zeros((3, N, 3))
    for k in range(3):
        a = k + 1
        jac_flat = 2 * (dc_flat[:, a] * asinc_val + c[a] * dasinc * dc_flat[:, 0])
        jac[k] = jac_flat.reshape(N, 3)
    return jac


def _apply_dF(Prefpos, vec, N):
    """Compute dF_{k,d} @ vec for all (k,d), batched over fragments.

    Prefpos : (B, N, 3) or (N, 3)
    vec     : (B, 4) or (4,)

    Returns : (B, N, 3, 4) or (N, 3, 4)
    """
    single = Prefpos.ndim == 2
    if single:
        Prefpos = Prefpos[None]
        vec = vec[None]
    B = Prefpos.shape[0]

    v0 = vec[:, 0]        # (B,)
    v3 = vec[:, 1:]       # (B, 3)
    Pv3 = np.einsum('bni,bi->bn', Prefpos, v3)  # (B, N) = Prefpos @ v3

    result = np.zeros((B, N, 3, 4))
    for d in range(3):
        dRtr = Prefpos[:, :, d]  # (B, N)
        # dFtop for this d
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        # dR[d1,d2]-dR[d2,d1] etc., with dR[i,j]=Prefpos[k,j]*delta_{i,d}
        dFtop = np.zeros((B, N, 3))
        # Antisymmetric part of dR: dR[i,j]-dR[j,i]
        # Only nonzero entries: dR[d,j] = Pref[k,j], dR[j,d] = 0 for j!=d
        # So: component 0 = dR[1,2]-dR[2,1]:
        #   if d==1: Pref[k,2]; if d==2: -Pref[k,1]; else 0
        # Simpler pattern: cross-product-like
        dFtop[:, :, d1] = -Prefpos[:, :, d2]
        dFtop[:, :, d2] = Prefpos[:, :, d1]
        # dFtop[d] = 0 (already)

        # result[:, :, d, 0] = dRtr * v0 + dFtop @ v3
        result[:, :, d, 0] = (
            dRtr * v0[:, None]
            + np.einsum('bni,bi->bn', dFtop, v3)
        )

        # result[:, :, d, 1:] = dFtop * v0 + (-dRtr*I + dR + dR.T) @ v3
        for i_ax in range(3):
            val = -dRtr * v3[:, i_ax:i_ax+1]  # (B, 1) broadcast with (B, N) -> (B, N)
            if i_ax == d:
                val = val + Pv3  # (B, N)
            val = val + Prefpos[:, :, i_ax] * v3[:, d:d+1]  # (B, N)
            result[:, :, d, 1 + i_ax] = dFtop[:, :, i_ax] * v0[:, None] + val

    if single:
        return result[0]
    return result


def _rotation_expmap_jacobian_from_dq(q, dq):
    """Map quaternion derivatives to three rotation-vector derivatives."""
    q0 = q[:, 0]
    asinc_val = np.empty_like(q0)
    regular = q0 < 0.97
    if np.any(regular):
        q0r = q0[regular]
        asinc_val[regular] = (
            np.arccos(q0r) / np.sqrt(np.maximum(1.0 - q0r**2, 1e-30))
        )
    if np.any(~regular):
        y = q0[~regular] - 1.0
        asinc_val[~regular] = (
            1.0 - y / 3 + 2 * y**2 / 15 - 2 * y**3 / 35
            + 8 * y**4 / 315 - 8 * y**5 / 693
            + 16 * y**6 / 3003 - 16 * y**7 / 6435
            + 128 * y**8 / 109395 - 128 * y**9 / 230945
        )

    dasinc = np.zeros_like(q0)
    near_identity = np.abs(q0 - 1.0) < _ROT_NEAR_IDENTITY_TOL
    ynear = q0[near_identity] - 1.0
    dasinc[near_identity] = -1.0 / 3 + 4 * ynear / 15
    regular_deriv = (~near_identity) & (np.abs(q0) < 1.0 - 1e-12)
    if np.any(regular_deriv):
        q0r = q0[regular_deriv]
        s2 = 1.0 - q0r**2
        s = np.sqrt(s2)
        ac = np.arccos(q0r)
        dasinc[regular_deriv] = -1.0 / s2 + q0r * ac / (s * s2)

    jac = np.empty((len(q), 3, dq.shape[1]))
    for axis in range(3):
        jac[:, axis] = 2 * (
            dq[:, :, axis + 1] * asinc_val[:, None]
            + q[:, axis + 1, None] * dasinc[:, None] * dq[:, :, 0]
        )
    return jac


def _rotation_jacobian_fixed_gauge_batched(
    pos, refpos, q_reference, top_count,
):
    """Jacobian of the fixed-gauge top-eigenspace projection.

    The usual simple-eigenvector derivative is insufficient in a degenerate
    top eigenspace: away from the base point, the fixed reference quaternion
    need not lie in the new eigenspace. Differentiating the spectral projector
    first keeps the value, Jacobian, and Hessian on one local gauge chart.
    """
    pos = np.asarray(pos, dtype=np.float64)
    refpos = np.asarray(refpos, dtype=np.float64)
    q_reference = np.asarray(q_reference, dtype=np.float64)
    n_batch, n, _ = pos.shape
    ws, vecs = np.linalg.eigh(_build_F_matrices_np(pos, refpos))
    top = np.arange(4 - top_count, 4)
    bottom = np.arange(0, 4 - top_count)

    top_vecs = vecs[:, :, top]
    projected = np.squeeze(
        top_vecs @ (top_vecs.swapaxes(1, 2) @ q_reference[:, :, None]), -1
    )
    projected_norm = np.linalg.norm(projected, axis=1)
    q = projected / projected_norm[:, None]

    pref = refpos - refpos.mean(axis=1, keepdims=True)
    dz = np.zeros((n_batch, 3 * n, 4))
    for a in top:
        ua = vecs[:, :, a]
        dFua = _apply_dF(pref, ua, n).reshape(n_batch, 3 * n, 4)
        ua_ref = np.sum(ua * q_reference, axis=1)
        for b in bottom:
            ub = vecs[:, :, b]
            coefficient = (
                np.einsum('bmi,bi->bm', dFua, ub)
                / (ws[:, a] - ws[:, b])[:, None]
            )
            direction = (
                ub * ua_ref[:, None]
                + ua * np.sum(ub * q_reference, axis=1)[:, None]
            )
            dz += coefficient[:, :, None] * direction[:, None, :]

    dq = dz - np.sum(dz * q[:, None, :], axis=2)[:, :, None] * q[:, None, :]
    dq /= projected_norm[:, None, None]
    jac = _rotation_expmap_jacobian_from_dq(q, dq)
    return jac.reshape(n_batch, 3, n, 3)


def _rotation_jacobian_with_fixed_gauge(
    pos, refpos, q_reference, top_count,
):
    return _rotation_jacobian_fixed_gauge_batched(
        pos[None], refpos[None], q_reference[None], top_count,
    )[0]


def _rotation_hvp_degenerate_fd_batched(
    pos, refpos, tangent, q_reference, top_count,
):
    """Fixed-gauge degenerate HVP, batched over equal-sized fragments."""
    tangent_norm = np.linalg.norm(tangent.reshape(len(tangent), -1), axis=1)
    out = np.zeros((len(pos), 3) + pos.shape[1:])
    moving = tangent_norm > 0
    if not np.any(moving):
        return out

    pos_m = pos[moving]
    ref_m = refpos[moving]
    tangent_m = tangent[moving]
    q_m = q_reference[moving]
    norm_m = tangent_norm[moving]
    scale = np.maximum(
        np.sqrt(np.mean(ref_m * ref_m, axis=(1, 2))), 1.0
    )
    delta_t = _ROT_DEGENERATE_FD_STEP * scale / norm_m
    displacement = delta_t[:, None, None] * tangent_m
    jac_plus = _rotation_jacobian_fixed_gauge_batched(
        pos_m + displacement, ref_m, q_m, top_count,
    )
    jac_minus = _rotation_jacobian_fixed_gauge_batched(
        pos_m - displacement, ref_m, q_m, top_count,
    )
    out[moving] = (
        (jac_plus - jac_minus) / (2 * delta_t[:, None, None, None])
    )
    return out


def _rotation_hvp_degenerate_fd(
    pos, refpos, tangent, q_reference, top_count,
):
    return _rotation_hvp_degenerate_fd_batched(
        pos[None], refpos[None], tangent[None], q_reference[None], top_count,
    )[0]


def _rotation_3axis_hessian_degenerate_fd(
    pos, refpos, q_reference, top_count,
):
    """Symmetric fixed-gauge Hessians for all three rotation components."""
    n = len(pos)
    ncart = 3 * n
    scale = max(np.sqrt(np.mean(refpos * refpos)), 1.0)
    delta = _ROT_DEGENERATE_FD_STEP * scale
    pos_batch = np.repeat(pos[None], 2 * ncart, axis=0)
    flat = pos_batch.reshape(2 * ncart, ncart)
    columns = np.arange(ncart)
    flat[columns, columns] += delta
    flat[ncart + columns, columns] -= delta
    ref_batch = np.repeat(refpos[None], 2 * ncart, axis=0)
    q_batch = np.repeat(q_reference[None], 2 * ncart, axis=0)
    jac = _rotation_jacobian_fixed_gauge_batched(
        pos_batch, ref_batch, q_batch, top_count,
    )
    column_derivatives = (jac[:ncart] - jac[ncart:]) / (2 * delta)
    hessian = column_derivatives.reshape(ncart, 3, ncart).transpose(1, 2, 0)
    hessian = 0.5 * (hessian + hessian.swapaxes(1, 2))
    return hessian.reshape(3, n, 3, n, 3)


def _rotation_hessian_degenerate_fd(
    pos, axis, refpos, q_reference, top_count,
):
    return _rotation_3axis_hessian_degenerate_fd(
        pos, refpos, q_reference, top_count,
    )[axis]



def _rotation_hessian_single(pos, axis, refpos, q_stable=None,
                             ws=None, vecs=None):
    """Closed-form Hessian for a single rotation on a single fragment.

    pos    : (N, 3)
    axis   : int
    refpos : (N, 3), already centered
    q_stable : (4,), optional stabilized quaternion

    Returns (N, 3, N, 3)
    """
    N = len(pos)
    a = axis + 1

    # Eigendecomposition + safe pseudoinverse.  Batched callers already
    # compute this while stabilizing the quaternion; reuse it when available.
    if ws is None or vecs is None:
        dx = pos - pos.mean(0)
        F = _build_F_matrix_np(dx, refpos)
        ws, vecs = np.linalg.eigh(F)
    if q_stable is not None:
        c = q_stable
    else:
        c = vecs[:, -1]
        if c[0] < 0:
            c = -c
    if ws[-1] - ws[-2] <= _ROT_EIG_GAP_TOL:
        top_count = int(np.sum(ws[-1] - ws <= _ROT_EIG_GAP_TOL))
        return _rotation_hessian_degenerate_fd(
            pos, axis, refpos, c, top_count
        )
    gaps = ws - ws[-1]
    safe_inv = np.where(np.abs(gaps) > _ROT_EIG_GAP_TOL, 1.0 / np.where(np.abs(gaps) > _ROT_EIG_GAP_TOL, gaps, 1.0), 0.0)

    def M_inv_mat(mat):
        return vecs @ (safe_inv[:, None] * (vecs.T @ mat))

    # Prefpos and dFc
    P = np.eye(N) - 1.0 / N
    Prefpos = P @ refpos  # (N, 3)
    dFc = _apply_dF(Prefpos, c, N)  # (N, 3, 4)
    dFc_flat = dFc.reshape(N * 3, 4)

    dE_flat = dFc_flat @ c  # (N*3,)
    dc_flat = -M_inv_mat(dFc_flat.T).T  # (N*3, 4)

    # asinc derivatives
    q0 = c[0]
    qa = c[a]
    if abs(q0 - 1.0) < _ROT_NEAR_IDENTITY_TOL:
        y = q0 - 1.0
        asinc_val = 1 - y / 3 + 2 * y**2 / 15
        dasinc = -1.0 / 3 + 4 * y / 15
        d2asinc = 4.0 / 15
    elif abs(q0) < 1.0 - 1e-12:
        s2 = 1 - q0**2
        s = np.sqrt(s2)
        ac = np.arccos(q0)
        asinc_val = ac / s
        dasinc = -1.0 / s2 + q0 * ac / (s * s2)
        d2asinc = (3 * q0 / s2 - (1 + 2 * q0**2) * ac / (s * s2)) * (-1.0 / s2)
    else:
        asinc_val = np.pi / 2 if q0 > 0 else -np.pi / 2
        dasinc = 0.0
        d2asinc = 0.0

    df_dq = np.zeros(4)
    df_dq[0] = 2 * qa * dasinc
    df_dq[a] = 2 * asinc_val

    d2f_dq2 = np.zeros((4, 4))
    d2f_dq2[0, 0] = 2 * qa * d2asinc
    d2f_dq2[0, a] = 2 * dasinc
    d2f_dq2[a, 0] = 2 * dasinc

    # Term 1: quadratic in first derivatives
    hess_flat = dc_flat @ d2f_dq2 @ dc_flat.T

    # Term 2: df_dq contracted with d2c
    w = vecs @ (safe_inv * (vecs.T @ df_dq))
    wc = w @ c
    w_dc = dc_flat @ w
    fdq_c = df_dq @ c

    dFw = _apply_dF(Prefpos, w, N)
    dFw_flat = dFw.reshape(N * 3, 4)
    wdFdc = dFw_flat @ dc_flat.T

    d2E_mat = 2 * dFc_flat @ dc_flat.T
    dc_dot = dc_flat @ dc_flat.T

    term2 = (dE_flat[:, None] * w_dc[None, :]
             + dE_flat[None, :] * w_dc[:, None]
             + d2E_mat * wc
             - wdFdc - wdFdc.T
             - fdq_c * dc_dot)

    hess_flat += term2
    return hess_flat.reshape(N, 3, N, 3)


def _rotation_hvp_closed(pos, axis, refpos, tangent, q_stable=None):
    """HVP for a single rotation using the closed-form Hessian."""
    dx = pos - pos.mean(0)
    ws, vecs = np.linalg.eigh(_build_F_matrix_np(dx, refpos))
    if q_stable is None:
        q_stable = _stabilize_quaternion_from_eigh(ws, vecs, None)
    if ws[-1] - ws[-2] <= _ROT_EIG_GAP_TOL:
        top_count = int(np.sum(ws[-1] - ws <= _ROT_EIG_GAP_TOL))
        return _rotation_hvp_degenerate_fd(
            pos, refpos, tangent, q_stable, top_count
        )[axis]
    hess = _rotation_hessian_single(
        pos, axis, refpos, q_stable=q_stable, ws=ws, vecs=vecs
    )
    return np.einsum('aibj,bj->ai', hess, tangent)




def _build_dF_vec_batched(Pref, vec, n_batch, nr):
    """Compute dF_{k,d} @ vec for all (k,d) in a batched fragment group.

    Parameters
    ----------
    Pref : (n_batch, nr, 3), centered reference positions
    vec  : (n_batch, 4), quaternion-space vector

    Returns
    -------
    dF_vec : (n_batch, nr*3, 4)
    """
    v0 = vec[:, 0:1]       # (n_batch, 1)
    v3 = vec[:, 1:]         # (n_batch, 3)
    Pv3 = np.squeeze(Pref @ v3[:, :, None], -1)  # (n_batch, nr)

    result = np.empty((n_batch, nr, 3, 4))
    for d in range(3):
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        dRtr = Pref[:, :, d]  # (n_batch, nr)

        dFtop_d1 = -Pref[:, :, d2]  # (n_batch, nr)
        dFtop_d2 = Pref[:, :, d1]   # (n_batch, nr)

        result[:, :, d, 0] = (dRtr * v0
                              + dFtop_d1 * v3[:, d1:d1+1]
                              + dFtop_d2 * v3[:, d2:d2+1])

        vd = v3[:, d:d+1]  # (n_batch, 1)
        for i_ax in range(3):
            val = -dRtr * v3[:, i_ax:i_ax+1]
            if i_ax == d:
                val = val + Pv3
            val = val + Pref[:, :, i_ax] * vd
            if i_ax == d1:
                dFtop_iax = dFtop_d1
            elif i_ax == d2:
                dFtop_iax = dFtop_d2
            else:
                dFtop_iax = 0.0
            result[:, :, d, 1 + i_ax] = dFtop_iax * v0 + val

    return result.reshape(n_batch, nr * 3, 4)


def _build_dF_vec_batched_many(Pref, vec, n_batch, nr):
    """Compute dF_{k,d} @ vec for several quaternion-space vectors."""
    n_vec = vec.shape[1]
    v0 = vec[:, :, 0]
    v3 = vec[:, :, 1:]
    Pv3 = (Pref[:, None, :, 0] * v3[:, :, None, 0]
           + Pref[:, None, :, 1] * v3[:, :, None, 1]
           + Pref[:, None, :, 2] * v3[:, :, None, 2])

    result = np.empty((n_batch, n_vec, nr, 3, 4))
    for d in range(3):
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        dRtr = Pref[:, :, d]

        dFtop_d1 = -Pref[:, :, d2]
        dFtop_d2 = Pref[:, :, d1]

        result[:, :, :, d, 0] = (
            dRtr[:, None, :] * v0[:, :, None]
            + dFtop_d1[:, None, :] * v3[:, :, d1, None]
            + dFtop_d2[:, None, :] * v3[:, :, d2, None]
        )

        vd = v3[:, :, d]
        for i_ax in range(3):
            val = -dRtr[:, None, :] * v3[:, :, i_ax, None]
            if i_ax == d:
                val = val + Pv3
            val = val + Pref[:, None, :, i_ax] * vd[:, :, None]
            if i_ax == d1:
                dFtop_iax = dFtop_d1[:, None, :]
            elif i_ax == d2:
                dFtop_iax = dFtop_d2[:, None, :]
            else:
                dFtop_iax = 0.0
            result[:, :, :, d, 1 + i_ax] = dFtop_iax * v0[:, :, None] + val

    return result.reshape(n_batch, n_vec, nr * 3, 4)


def _rotation_3axis_jacobian_batched_np(pos_pad, ref_pad, mask,
                                        q_stable_all=None,
                                        ws_all=None, vecs_all=None):
    """Batched Jacobian of all 3 rotation values for multiple fragments."""
    n_frag, n_max, _ = pos_pad.shape
    n_real = np.sum(mask, axis=1).astype(int)
    jac = np.zeros((n_frag, 3, n_max, 3))

    size_groups = {}
    for fi, nr in enumerate(n_real):
        size_groups.setdefault(int(nr), []).append(fi)

    for nr, frag_indices in size_groups.items():
        n_batch = len(frag_indices)
        idx = np.array(frag_indices)
        pos_group = pos_pad[idx, :nr]
        ref_group = ref_pad[idx, :nr]

        if ws_all is not None and vecs_all is not None:
            ws = ws_all[idx]
            vecs = vecs_all[idx]
            if q_stable_all is not None:
                c = q_stable_all[idx]
            else:
                c = vecs[:, :, -1]
                sign = np.where(c[:, 0] >= 0, 1.0, -1.0)
                c *= sign[:, None]
        else:
            ws, vecs = np.linalg.eigh(
                _build_F_matrices_np(pos_group, ref_group)
            )
            if q_stable_all is not None:
                c = q_stable_all[idx]
            else:
                c = vecs[:, :, -1]
                sign = np.where(c[:, 0] >= 0, 1.0, -1.0)
                c *= sign[:, None]

        gaps = ws - ws[:, -1:]
        large_gap = np.abs(gaps) > _ROT_EIG_GAP_TOL
        safe_inv = np.zeros_like(gaps)
        safe_inv[large_gap] = 1.0 / gaps[large_gap]

        dFc_flat = _build_dF_vec_batched(ref_group, c, n_batch, nr)
        proj = np.matmul(dFc_flat, vecs)
        dc_flat = -np.matmul(proj * safe_inv[:, None, :],
                             vecs.swapaxes(1, 2))

        q0 = c[:, 0]
        asinc_val = np.empty_like(q0)
        regular_asinc = q0 < 0.97
        s2 = 1.0 - q0 * q0
        asinc_val[regular_asinc] = (
            np.arccos(q0[regular_asinc])
            / np.sqrt(s2[regular_asinc])
        )
        y = q0[~regular_asinc] - 1.0
        asinc_val[~regular_asinc] = (
            1.0 - y / 3 + 2 * y**2 / 15 - 2 * y**3 / 35
            + 8 * y**4 / 315 - 8 * y**5 / 693 + 16 * y**6 / 3003
            - 16 * y**7 / 6435 + 128 * y**8 / 109395
            - 128 * y**9 / 230945
        )

        dasinc = np.zeros_like(q0)
        near_identity = np.abs(q0 - 1.0) < _ROT_NEAR_IDENTITY_TOL
        ynear = q0[near_identity] - 1.0
        dasinc[near_identity] = -1.0 / 3 + 4 * ynear / 15
        regular_deriv = (~near_identity) & (np.abs(q0) < 1.0 - 1e-12)
        if np.any(regular_deriv):
            q0r = q0[regular_deriv]
            s2r = 1.0 - q0r**2
            sr = np.sqrt(s2r)
            acr = np.arccos(q0r)
            dasinc[regular_deriv] = -1.0 / s2r + q0r * acr / (sr * s2r)

        for axis in range(3):
            a = axis + 1
            jac_flat = 2 * (
                dc_flat[:, :, a] * asinc_val[:, None]
                + c[:, a:a + 1] * dasinc[:, None] * dc_flat[:, :, 0]
            )
            jac[idx, axis, :nr, :] = jac_flat.reshape(n_batch, nr, 3)

    return jac


def _rotation_3axis_hvp_batched_closed(pos_pad, ref_pad, mask, v_pad,
                                       q_stable_all=None,
                                       ws_all=None, vecs_all=None):
    """Batched HVP for multiple fragments using closed-form Hessians.

    Parameters
    ----------
    pos_pad : (B, N_max, 3)
    ref_pad : (B, N_max, 3)
    mask : (B, N_max)
    v_pad : (B, N_max, 3)
    q_stable_all : (B, 4), optional stabilized quaternions per fragment
    ws_all : (B, 4), optional cached eigenvalues per fragment
    vecs_all : (B, 4, 4), optional cached eigenvectors per fragment

    Returns
    -------
    hvp : (B, 3, N_max, 3)
    """
    B, N_max, _ = pos_pad.shape
    n_real = np.sum(mask, axis=1).astype(int)
    hvp = np.zeros((B, 3, N_max, 3))

    size_groups = {}
    for fi in range(B):
        nr = n_real[fi]
        size_groups.setdefault(nr, []).append(fi)

    for nr, frag_indices in size_groups.items():
        n_batch = len(frag_indices)
        idx = np.array(frag_indices)

        pos_group = pos_pad[idx, :nr]    # (n_batch, nr, 3)
        ref_group = ref_pad[idx, :nr]    # (n_batch, nr, 3)
        v_group = v_pad[idx, :nr]        # (n_batch, nr, 3)

        if ws_all is not None and vecs_all is not None:
            ws = ws_all[idx]
            vecs = vecs_all[idx]
            if q_stable_all is not None:
                c = q_stable_all[idx]
            else:
                c = vecs[:, :, -1]
                sign = np.where(c[:, 0] >= 0, 1.0, -1.0)
                c *= sign[:, None]
        else:
            ws, vecs = np.linalg.eigh(
                _build_F_matrices_np(pos_group, ref_group)
            )
            if q_stable_all is not None:
                c = q_stable_all[idx]
            else:
                c = vecs[:, :, -1]
                sign = np.where(c[:, 0] >= 0, 1.0, -1.0)
                c *= sign[:, None]

        degenerate = ws[:, -1] - ws[:, -2] <= _ROT_EIG_GAP_TOL
        if np.any(degenerate):
            top_counts = np.ones(n_batch, dtype=int)
            top_counts[degenerate] = np.sum(
                ws[degenerate, -1, None] - ws[degenerate]
                <= _ROT_EIG_GAP_TOL,
                axis=1,
            )
            for top_count in np.unique(top_counts[degenerate]):
                local = np.flatnonzero(
                    degenerate & (top_counts == top_count)
                )
                fragment_indices = idx[local]
                hvp[fragment_indices, :, :nr, :] = (
                    _rotation_hvp_degenerate_fd_batched(
                        pos_group[local], ref_group[local], v_group[local],
                        c[local], int(top_count),
                    )
                )

            nondegenerate = np.flatnonzero(~degenerate)
            if len(nondegenerate) == 0:
                continue
            idx = idx[nondegenerate]
            pos_group = pos_group[nondegenerate]
            ref_group = ref_group[nondegenerate]
            v_group = v_group[nondegenerate]
            ws = ws[nondegenerate]
            vecs = vecs[nondegenerate]
            c = c[nondegenerate]
            n_batch = len(nondegenerate)

        gaps = ws - ws[:, -1:]
        safe_inv = np.where(
            np.abs(gaps) > _ROT_EIG_GAP_TOL,
            1.0 / np.where(np.abs(gaps) > _ROT_EIG_GAP_TOL, gaps, 1.0),
            0.0,
        )

        # refpos is already centered at construction
        Pref = ref_group

        # dFc: dF @ c for all (k,d)
        dFc_flat = _build_dF_vec_batched(Pref, c, n_batch, nr)  # (n_batch, M, 4)
        M = nr * 3

        dE_flat = np.squeeze(dFc_flat @ c[:, :, None], -1)  # (n_batch, M)
        # dc_flat = -vecs @ (safe_inv * (vecs^T @ dFc_flat^T))^T
        proj = np.matmul(dFc_flat, vecs)  # (n_batch, M, 4)
        dc_flat = -np.matmul(proj * safe_inv[:, None, :], vecs.swapaxes(1, 2))  # (n_batch, M, 4)

        # Axis-independent computations (hoisted from axis loop)
        v_flat = v_group.reshape(n_batch, M)
        dc_v = np.squeeze(dc_flat.swapaxes(1, 2) @ v_flat[:, :, None], -1)  # (n_batch, 4)
        dE_v = (dE_flat * v_flat).sum(axis=1)  # (n_batch,)
        d2E_v = 2 * np.squeeze(dFc_flat @ dc_v[:, :, None], -1)  # (n_batch, M)
        dc_dot_v = np.squeeze(dc_flat @ dc_v[:, :, None], -1)  # (n_batch, M)

        q0 = c[:, 0]
        s2 = np.maximum(1 - q0**2, 1e-30)
        s = np.sqrt(s2)
        ac = np.arccos(np.clip(q0, -1+1e-15, 1-1e-15))
        near_one = np.abs(q0 - 1.0) < _ROT_NEAR_IDENTITY_TOL
        y = q0 - 1.0
        asinc_val = np.where(near_one, 1 - y/3 + 2*y**2/15, ac/s)
        dasinc = np.where(near_one, -1.0/3 + 4*y/15, -1.0/s2 + q0*ac/(s*s2))
        d2asinc = np.where(near_one, 4.0/15,
                           (3*q0/s2 - (1+2*q0**2)*ac/(s*s2)) * (-1.0/s2))

        axis_terms = []
        w_all = np.empty((n_batch, 3, 4))
        for axis in range(3):
            a = axis + 1
            qa = c[:, a]

            df_dq = np.zeros((n_batch, 4))
            df_dq[:, 0] = 2 * qa * dasinc
            df_dq[:, a] = 2 * asinc_val

            d2f_dq2 = np.zeros((n_batch, 4, 4))
            d2f_dq2[:, 0, 0] = 2 * qa * d2asinc
            d2f_dq2[:, 0, a] = 2 * dasinc
            d2f_dq2[:, a, 0] = 2 * dasinc

            # term1: dc @ d2f @ dc^T @ v = dc @ d2f @ dc_v
            t1_hvp = np.squeeze(
                dc_flat @ (d2f_dq2 @ dc_v[:, :, None]), -1
            )  # (n_batch, M)

            # term2: w = M_inv(df_dq)
            proj_w = np.squeeze(vecs.swapaxes(1, 2) @ df_dq[:, :, None], -1)
            w = np.squeeze(vecs @ (safe_inv * proj_w)[:, :, None], -1)  # (n_batch, 4)
            wc = (w * c).sum(axis=1)  # (n_batch,)
            w_dc = np.squeeze(dc_flat @ w[:, :, None], -1)  # (n_batch, M)
            fdq_c = (df_dq * c).sum(axis=1)  # (n_batch,)

            w_dc_v = (w_dc * v_flat).sum(axis=1)  # (n_batch,)

            w_all[:, axis] = w
            axis_terms.append((t1_hvp, wc, w_dc, fdq_c, w_dc_v))

        # dFw: dF @ w for all axes and all (k,d)
        dFw_all = _build_dF_vec_batched_many(Pref, w_all, n_batch, nr)

        for axis, (t1_hvp, wc, w_dc, fdq_c, w_dc_v) in enumerate(axis_terms):
            dFw_flat = dFw_all[:, axis]  # (n_batch, M, 4)

            # wdFdc @ v = dFw_flat @ dc_v
            wdFdc_v = np.squeeze(dFw_flat @ dc_v[:, :, None], -1)

            # wdFdc^T @ v = dc_flat @ (dFw_flat^T @ v)
            dFw_v = np.squeeze(dFw_flat.swapaxes(1, 2) @ v_flat[:, :, None], -1)
            wdFdcT_v = np.squeeze(dc_flat @ dFw_v[:, :, None], -1)

            t2_hvp = (dE_flat * w_dc_v[:, None]
                      + dE_v[:, None] * w_dc
                      + wc[:, None] * d2E_v
                      - wdFdc_v - wdFdcT_v
                      - fdq_c[:, None] * dc_dot_v)

            hvp_axis = (t1_hvp + t2_hvp).reshape(n_batch, nr, 3)
            hvp[idx, axis, :nr, :] = hvp_axis

    return hvp


class Rotation(Coordinate):
    def __init__(
        self,
        indices: Tuple[int, ...],
        axis: int,
        refpos: np.ndarray,
    ) -> None:
        assert len(indices) >= 2
        Coordinate.__init__(self, indices)
        self.kwargs['axis'] = axis
        self.kwargs['refpos'] = refpos.copy() - refpos.mean(0)
        self.q_prev = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs['axis'] != other.kwargs['axis']:
            return False
        if len(self.indices) != len(other.indices):
            return False
        if set(self.indices) != set(other.indices):
            return False
        if not np.allclose(self.kwargs['refpos'], other.kwargs['refpos']):
            return False
        return True

    def _stabilized_quaternion(self, pos: np.ndarray) -> np.ndarray:
        dx = pos - pos.mean(0)
        refpos = self.kwargs['refpos']
        F = _build_F_matrix_np(dx, refpos)
        q = _stabilize_quaternion(F, self.q_prev)
        self.q_prev = q
        return q

    def calc(self, atoms: Atoms) -> float:
        pos = np.asarray(atoms.positions[self.indices], dtype=np.float64)
        q = self._stabilized_quaternion(pos)
        axis = self.kwargs['axis']
        return float(2.0 * q[axis + 1] * _asinc_np(q[0]))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        pos = np.asarray(atoms.positions[self.indices], dtype=np.float64)
        refpos = self.kwargs['refpos']
        q = self._stabilized_quaternion(pos)
        jac = _rotation_3axis_jacobian_np(pos, refpos, q)
        return jac[self.kwargs['axis']]

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        pos = np.asarray(atoms.positions[self.indices], dtype=np.float64)
        q = self._stabilized_quaternion(pos)
        return _rotation_hessian_np(
            pos,
            self.kwargs['axis'],
            self.kwargs['refpos'],
            q_stable=q,
        )


def _displacement(
    pos: jnp.ndarray,
    refpos: jnp.ndarray,
    W: jnp.ndarray
) -> float:
    dx = (pos - refpos).ravel()
    return dx @ W @ dx


class Displacement(Coordinate):
    def __init__(
        self,
        indices: np.ndarray,
        refpos: np.ndarray,
        W: np.ndarray,
    ) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs['refpos'] = refpos.copy()
        self.kwargs['W'] = W.copy()

    def __eq__(self, other: Coordinate) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if not Coordinate.__eq__(self, other):
            return False
        return (
            np.allclose(self.kwargs['refpos'], other.kwargs['refpos'])
            and np.array_equal(self.kwargs['W'].shape, other.kwargs['W'].shape)
            and np.allclose(self.kwargs['W'], other.kwargs['W'])
        )

    _eval0 = staticmethod(jit(_displacement))
    _eval1 = staticmethod(jit(_gradient(_displacement)))
    _eval2 = staticmethod(jit(_hessian(_displacement)))


def _bond(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    return jnp.linalg.norm(
        pos[1] - pos[0] + tvecs[0]
    )


class Bond(Internal):
    nindices = 2
    _eval0 = staticmethod(jit(_bond))
    _eval1 = staticmethod(_gradient(_bond))
    _eval2 = staticmethod(_hessian(_bond))
    _eval_cell_grad = staticmethod(_bond_cell_grad_single)

    def calc_vec(self, atoms: Atoms) -> np.ndarray:
        tvecs = np.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        i, j = self.indices
        return atoms.positions[j] - atoms.positions[i] + tvecs[0]


def _angle(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = -(pos[1] - pos[0] + tvecs[0])
    dx2 = pos[2] - pos[1] + tvecs[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    # Clamp to avoid NaN from arccos due to floating-point errors
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


class Angle(Internal):
    nindices = 3
    _eval0 = staticmethod(jit(_angle))
    _eval1 = staticmethod(_gradient(_angle))
    _eval2 = staticmethod(_hessian(_angle))
    _eval_cell_grad = staticmethod(_angle_cell_grad_single)


def _dihedral(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = pos[1] - pos[0] + tvecs[0]
    dx2 = pos[2] - pos[1] + tvecs[1]
    dx3 = pos[3] - pos[2] + tvecs[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


class Dihedral(Internal):
    nindices = 4
    _eval0 = staticmethod(jit(_dihedral))
    _eval1 = staticmethod(_gradient(_dihedral))
    _eval2 = staticmethod(_hessian(_dihedral))
    _eval_cell_grad = staticmethod(_dihedral_cell_grad_single)


Bond.union = Angle
Angle.union = Dihedral
Angle.diff = Bond
Dihedral.diff = Angle


def make_internal(
    name: str,
    fun: Callable[..., float],
    nindices: int,
    use_jit: bool = True,
    jac: Callable[..., jnp.ndarray] = None,
    hess: Callable[..., jnp.ndarray] = None,
    **kwargs,
) -> Type[Coordinate]:
    if jac is None:
        jac = _gradient(fun)
    if hess is None:
        hess = _hessian(fun)

    if use_jit:
        fun = jit(fun)
        jac = jit(jac)
        hess = jit(hess)

    def __init__(self, indices):
        # Coordinate.__init__ resets self.kwargs to {}, so install the fixed
        # eval kwargs afterwards. Copy per-instance to avoid a shared dict.
        Coordinate.__init__(self, indices)
        self.kwargs = dict(kwargs)

    return type(name, (Coordinate,), dict(
        nindices=nindices,
        __init__=__init__,
        _eval0=staticmethod(fun),
        _eval1=staticmethod(jac),
        _eval2=staticmethod(hess),
    ))


class BaseInternals:
    _names = (
        'translations', 'bonds', 'angles', 'dihedrals', 'other', 'rotations'
    )

    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None
    ) -> None:
        self.atoms = atoms

        self._lastpos = None
        self._lastcell = None
        self._lastactive = None
        self._cache = dict()

        if dummies is None:
            if dinds is not None:
                raise ValueError('"dinds" provided, but no "dummies"!')
            dummies = Atoms()
            dinds = -np.ones(len(self.atoms), dtype=np.int32)
        else:
            if dinds is None:
                raise ValueError('"dummies" provided, but no "dinds"!')
            dinds = np.asarray(dinds, dtype=np.int32)
            ndum = len(dummies)
            ndind = np.sum(dinds >= 0)
            if ndum != ndind:
                raise ValueError(
                    '{} dummy atoms were provided, but only {} dummy indices!'
                    .format(ndum, ndind)
                )
            dummy_indices = np.sort(dinds[dinds >= 0])
            expected = np.arange(
                len(self.atoms),
                len(self.atoms) + ndum,
                dtype=np.int32,
            )
            if not np.array_equal(dummy_indices, expected):
                raise ValueError(
                    'Dummy indices must refer to the appended dummy block '
                    'natoms:natoms+ndummies.'
                )
        self.dummies = dummies
        self.dinds = dinds

        # Cache atom count (doesn't change during optimization)
        self._natoms = len(atoms)

        self.internals = {key: [] for key in self._names}
        self._internals_set = {key: set() for key in self._names}
        self._active = {key: [] for key in self._names}
        self.cell = None
        self.rcell = None
        self._rcell_reciprocal_T = None
        self.op = None
        self._hessian_skeleton = None

        # Batched arrays for vectorized computation (built lazily)
        self._batched_arrays_valid = False
        self._batched_family_arrays: Dict[str, _BatchedCoordArrays] = {}

        # Lazy caches.
        self._tvecs_cache = None  # set to {'cell_hash': ..., 'tvecs': ...} on first build
        self._hvp_buf = None  # reusable buffer for hessian_rdot output

    @staticmethod
    def _ignore_duplicate(adder, *args, **kwargs) -> bool:
        """Call an internal/constraint adder and ignore duplicate coordinates."""
        try:
            adder(*args, **kwargs)
        except DuplicateInternalError:
            return False
        return True

    @property
    def natoms(self) -> int:
        return self._natoms

    @property
    def ndummies(self) -> int:
        return len(self.dummies)

    @property
    def ndof(self) -> int:
        return 3 * (self._natoms + len(self.dummies))

    @property
    def ntrans(self) -> int:
        return sum(self._active['translations'])

    @property
    def nbonds(self) -> int:
        return sum(self._active['bonds'])

    @property
    def nangles(self) -> int:
        return sum(self._active['angles'])

    @property
    def ndihedrals(self) -> int:
        return sum(self._active['dihedrals'])

    @property
    def nother(self) -> int:
        return sum(self._active['other'])

    @property
    def nrotations(self) -> int:
        return sum(self._active['rotations'])

    @property
    def _active_mask(self) -> List[bool]:
        active = []
        for name in self._names:
            active += self._active[name]
        return active

    @property
    def _active_indices(self) -> List[int]:
        return [idx for idx, active in enumerate(self._active_mask) if active]

    def _split_active_mask(self, n_trans, n_bonds, n_angles,
                           n_dihedrals, n_other, n_rot):
        """Slice ``_active_mask`` into the six per-family sub-masks.

        Families are returned in canonical ``_names`` order (translations,
        bonds, angles, dihedrals, other, rotations). Callers pass their own
        per-family counts so the split stays in lockstep with however they
        enumerated the coordinates.
        """
        active_mask = self._active_mask
        masks = []
        start = 0
        for n in (n_trans, n_bonds, n_angles, n_dihedrals, n_other, n_rot):
            masks.append(active_mask[start:start + n])
            start += n
        return masks

    @property
    def nint(self) -> int:
        return len(self._active_indices)

    @property
    def all_positions(self) -> np.ndarray:
        """Get combined positions without creating an Atoms object.

        Cached on ``self._cache['all_positions']`` so repeated reads
        within a single position evaluation reuse the same vstack.
        ``_cache_check`` clears the cache whenever positions change.
        """
        if self.ndummies == 0:
            return self.atoms.positions
        cached = self._cache.get('all_positions')
        if cached is not None:
            return cached
        merged = np.vstack([self.atoms.positions, self.dummies.positions])
        self._cache['all_positions'] = merged
        return merged

    @property
    def all_atoms(self) -> Atoms:
        return self.atoms + self.dummies

    @property
    def light_atoms(self) -> LightAtoms:
        """Get lightweight atoms-like object for coordinate calculations."""
        cell = self.atoms.cell.array
        return LightAtoms(self.all_positions, cell)

    def _cache_check(self) -> None:
        # we are comparing the current atomic positions to what they were
        # the last time a property was calculated. These positions are floats,
        # but we use a strict equality check to compare to avoid subtle bugs
        # that might occur during fine-resolution geodesic steps.
        if self.ndummies == 0:
            current_pos = self.atoms.positions
        else:
            current_pos = np.vstack([self.atoms.positions, self.dummies.positions])
        current_cell = self.atoms.cell.array
        # Cached calc()/jacobian()/cell_jacobian()/hessian() outputs depend on
        # the cell (bonds/angles/dihedrals fold in ncvecs @ cell), so a cell
        # change with fixed positions must invalidate them too.
        cell_changed = (
            self._lastcell is None
            or np.any(current_cell != self._lastcell)
        )
        # jacobian_B and hessian_result are cached *after* active-mask
        # filtering, so toggling an inequality constraint (which flips
        # self._active) must invalidate them as well.
        current_active = tuple(self._active_mask)
        active_changed = current_active != self._lastactive
        if (
            self._lastpos is None
            or np.any(current_pos != self._lastpos)
            or cell_changed
            or active_changed
        ):
            self._cache = dict()
            self._lastpos = current_pos.copy()
            self._lastcell = current_cell.copy()
            self._lastactive = current_active
        # Park the freshly-merged positions in the cache so the next
        # all_positions access doesn't redo the vstack.
        if self.ndummies > 0:
            self._cache.setdefault('all_positions', self._lastpos)

    @staticmethod
    def _pad_to_block(n: int) -> int:
        """Round up to a BLOCK_SIZE multiple for stable JAX/GPU batch shapes."""
        return ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    @staticmethod
    def _flat_cols(indices: np.ndarray, n_atoms: int, width: int) -> np.ndarray:
        """Flat Cartesian columns touched by each sparse coordinate Hessian.

        For a bond (a, b), the non-zero columns in the (ndof,) output are
        [3a, 3a+1, 3a+2, 3b, 3b+1, 3b+2]. Angles and dihedrals follow the same
        pattern with 9 and 12 columns. These columns depend only on topology and
        are invalidated with the rest of the batched arrays.
        """
        if len(indices) == 0:
            return np.empty((0, width), dtype=np.intp)
        offsets = np.arange(3)
        return np.concatenate(
            [indices[:, k:k + 1] * 3 + offsets for k in range(n_atoms)],
            axis=1,
        )

    def _build_batched_family_arrays(
        self, spec: _BatchedCoordFamily
    ) -> _BatchedCoordArrays:
        """Build padded and unpadded arrays for one vectorized coordinate family.

        Unpadded arrays are used for result indexing and sparse scatter. Padded
        arrays are used for JAX batch calls so bonds, angles, and dihedrals keep
        consistent shapes and avoid recompilation. Padding masks are not stored
        because valid rows are always the leading ``n_actual`` prefix.
        """
        coords = self.internals[spec.key]
        n_coords = len(coords)
        n_atoms = spec.n_atoms
        n_tvecs = spec.n_tvecs

        if n_coords > 0:
            n_padded = self._pad_to_block(n_coords)
            indices = np.array([c.indices for c in coords], dtype=np.int32)
            ncvecs = np.array(
                [c.kwargs['ncvecs'] for c in coords], dtype=np.int32
            )
            indices_padded = np.zeros((n_padded, n_atoms), dtype=np.int32)
            ncvecs_padded = np.zeros((n_padded, n_tvecs, 3), dtype=np.int32)
            indices_padded[:n_coords] = indices
            ncvecs_padded[:n_coords] = ncvecs
        else:
            indices = np.empty((0, n_atoms), dtype=np.int32)
            ncvecs = np.empty((0, n_tvecs, 3), dtype=np.int32)
            indices_padded = indices.copy()
            ncvecs_padded = ncvecs.copy()

        return _BatchedCoordArrays(
            indices=indices,
            ncvecs=ncvecs,
            indices_padded=indices_padded,
            ncvecs_padded=ncvecs_padded,
            n_actual=n_coords,
            flat_cols=self._flat_cols(indices, n_atoms, spec.width),
        )

    def _build_hvp_csr_structure(self) -> None:
        """Build the fixed CSR layout for sparse ``hessian_rdot`` output.

        Bonds, angles, and dihedrals have fixed nnz per row (6/9/12).
        Translations have zero rows. Rotations and ``other`` coordinates are
        dense rows over all Cartesian DOF.
        """
        ndof = self.ndof
        n_trans = len(self.internals['translations'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])
        n_active = n_trans + n_other + n_rot
        families = self._batched_family_arrays
        for spec in _BATCHED_COORD_FAMILIES:
            n_active += families[spec.key].n_actual

        col_blocks = []
        nnz_per_row = [0] * n_trans
        data_offset = 0
        for spec in _BATCHED_COORD_FAMILIES:
            family = families[spec.key]
            n_coords = family.n_actual
            families[spec.key] = family._replace(csr_offset=data_offset)
            if n_coords > 0:
                width = spec.width
                col_blocks.append(family.flat_cols.ravel())
                nnz_per_row.extend([width] * n_coords)
                data_offset += n_coords * width

        self._csr_other_offset = data_offset
        for _ in range(n_other + n_rot):
            col_blocks.append(np.arange(ndof))
            nnz_per_row.append(ndof)

        self._csr_indptr = np.zeros(n_active + 1, dtype=np.int32)
        np.cumsum(nnz_per_row, out=self._csr_indptr[1:])
        self._csr_indices = (
            np.concatenate(col_blocks).astype(np.int32)
            if col_blocks else np.empty(0, dtype=np.int32)
        )
        self._csr_data = np.zeros(len(self._csr_indices), dtype=np.float64)
        self._csr_n_active = n_active

    def _build_batched_arrays(self) -> None:
        """Build batched index arrays for vectorized computation.

        Arrays are padded to multiples of BLOCK_SIZE for GPU/SIMD efficiency.
        """
        if self._batched_arrays_valid:
            return

        self._batched_family_arrays = {
            spec.key: self._build_batched_family_arrays(spec)
            for spec in _BATCHED_COORD_FAMILIES
        }
        self._build_hvp_csr_structure()
        self._batched_arrays_valid = True

    @staticmethod
    def _tvec_or_empty(indices, ncvecs, cell, n_tvec):
        """Translation vectors ``ncvecs @ cell``, or a correctly-shaped
        ``(0, n_tvec, 3)`` empty when there are no coordinates of this type."""
        if len(indices) > 0:
            return ncvecs @ cell
        return np.empty((0, n_tvec, 3), dtype=np.float64)

    def _get_cached_tvecs(self, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Get cached translation vectors for cell, computing if necessary.

        The tvecs (ncvecs @ cell) are constant for a given cell, so we cache
        them to avoid redundant matrix multiplications during ODE integration.

        Returns both unpadded tvecs (for indexing) and padded tvecs (for batch ops).
        """
        cell_hash = cell.tobytes()
        if self._tvecs_cache is not None and self._tvecs_cache['cell_hash'] == cell_hash:
            return self._tvecs_cache['tvecs']

        self._build_batched_arrays()

        tvecs = {}
        # n_tvec = 1/2/3 for bonds/angles/dihedrals. Unpadded entries are used
        # for result indexing; padded entries are used for GPU-friendly batch ops.
        families = self._batched_family_arrays
        for spec in _BATCHED_COORD_FAMILIES:
            key = spec.key
            family = families[key]
            n_tvecs = spec.n_tvecs
            tvecs[key] = self._tvec_or_empty(
                family.indices,
                family.ncvecs,
                cell,
                n_tvecs,
            )
            tvecs[f'{key}_padded'] = self._tvec_or_empty(
                family.indices_padded,
                family.ncvecs_padded,
                cell,
                n_tvecs,
            )

        self._tvecs_cache = {'cell_hash': cell_hash, 'tvecs': tvecs}
        return tvecs

    def _invalidate_structure(self) -> None:
        """Invalidate all caches that depend on the set of internals.

        Call from every path that adds/removes a coordinate, dummy atom, or
        ncvec. Unlike ``_cache_check`` (which keys on positions/cell/active
        mask), a topology change can leave those keys unchanged -- e.g.
        forbidding one bond and adding another keeps the active mask and
        positions identical -- so the position-keyed ``_cache`` must be
        cleared explicitly alongside the batched arrays and tvecs.
        """
        self._batched_arrays_valid = False
        self._batched_family_arrays = {}
        self._tvecs_cache = None
        self._hessian_skeleton = None
        self._hvp_buf = None
        self._cache = dict()
        # Force _cache_check to treat the next evaluation as fresh.
        self._lastpos = None
        self._lastcell = None
        self._lastactive = None

    def _invalidate_batched_arrays(self) -> None:
        """Invalidate batched arrays (call when internals change)."""
        self._invalidate_structure()

    def _compute_batched_value_family(self, spec: _BatchedCoordFamily,
                                      family: _BatchedCoordArrays,
                                      positions, tvecs):
        """Compute values for one batched family, then slice off padding."""
        if family.n_actual == 0:
            return np.empty(0)
        pos = positions[family.indices_padded]
        values = spec.value_fn(pos, tvecs[f"{spec.key}_padded"])
        return np.asarray(device_get(values))[:family.n_actual]

    def _compute_batched_tensor_family(self, spec: _BatchedCoordFamily,
                                       family: _BatchedCoordArrays,
                                       positions, tvecs, fn, empty_tail):
        """Compute gradient/Hessian tensors for one family with padded batches."""
        if family.n_actual == 0:
            return family.indices, np.empty((0,) + empty_tail)
        pos = positions[family.indices_padded]
        padded = fn(pos, tvecs[f"{spec.key}_padded"])
        return family.indices, np.asarray(device_get(padded))[:family.n_actual]

    def _compute_batched_values(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all internal coordinate values using vectorized operations.

        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        families = self._batched_family_arrays
        return {
            spec.key: self._compute_batched_value_family(
                spec, families[spec.key], positions, tvecs
            )
            for spec in _BATCHED_COORD_FAMILIES
        }

    def _compute_batched_gradients(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute all internal coordinate gradients using vectorized operations.

        Returns dict mapping coord type to (indices, gradients) tuples.
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        families = self._batched_family_arrays
        return {
            spec.key: self._compute_batched_tensor_family(
                spec, families[spec.key], positions, tvecs, spec.grad_fn,
                (spec.n_atoms, 3),
            )
            for spec in _BATCHED_COORD_FAMILIES
        }

    def _compute_batched_hessians(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute all internal coordinate Hessians using vectorized operations.

        Returns dict mapping coord type to (indices, Hessians) tuples.
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        families = self._batched_family_arrays
        return {
            spec.key: self._compute_batched_tensor_family(
                spec, families[spec.key], positions, tvecs, spec.hess_fn,
                (spec.n_atoms, 3, spec.n_atoms, 3),
            )
            for spec in _BATCHED_COORD_FAMILIES
        }

    def _compute_batched_cell_gradients(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all internal coordinate cell gradients.

        Only periodic bond/angle/dihedral coordinates with non-zero ncvecs
        depend on the cell. The JAX cell object is created lazily so non-periodic
        batches stay on the fast zero-return path.
        """
        self._build_batched_arrays()
        cell_jax = None
        result = {}
        families = self._batched_family_arrays

        for spec in _BATCHED_COORD_FAMILIES:
            key = spec.key
            family = families[key]
            n_actual = family.n_actual
            ncvecs = family.ncvecs
            if n_actual == 0:
                result[key] = np.empty((0, 3, 3))
                continue
            if not np.any(ncvecs):
                result[key] = np.zeros((n_actual, 3, 3))
                continue

            if cell_jax is None:
                cell_jax = jnp.asarray(cell, dtype=np.float64)
            pos = jnp.asarray(
                positions[family.indices_padded],
                dtype=np.float64,
            )
            ncvecs_padded = jnp.asarray(
                family.ncvecs_padded,
                dtype=np.float64,
            )
            grads = spec.cell_grad_fn(pos, ncvecs_padded, cell_jax)
            result[key] = np.asarray(device_get(grads))[:n_actual]

        return result

    def copy(self) -> 'BaseInternals':
        raise NotImplementedError

    def calc(self) -> np.ndarray:
        """Calculates the internal coordinate vector using vectorized operations."""
        self._cache_check()
        if 'coords' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_vals = self._compute_batched_values(positions, cell)

            # Build full coords list in order
            all_coords = []

            # Translations are simple coordinate means; compute them directly
            # instead of dispatching one tiny JAX call per TRIC axis.
            for coord in self.internals['translations']:
                idx = coord.indices
                dim = coord.kwargs['dim']
                all_coords.append(float(positions[idx, dim].mean()))

            # Bonds (batched)
            all_coords.extend(batched_vals['bonds'].tolist())

            # Angles (batched)
            all_coords.extend(batched_vals['angles'].tolist())

            # Dihedrals (batched)
            all_coords.extend(batched_vals['dihedrals'].tolist())

            # Other (not batched - heterogeneous)
            atoms = self.light_atoms
            for coord in self.internals['other']:
                all_coords.append(coord.calc(atoms))

            # Rotations (batched if all 3 axes present per fragment)
            rot_vals = self._batched_rotation_values(positions)
            if rot_vals is None:
                for coord in self.internals['rotations']:
                    all_coords.append(coord.calc(atoms))
            else:
                all_coords.extend(rot_vals)

            self._cache['coords'] = np.array(all_coords)

        coords = self._cache['coords']
        active_mask = self._active_mask
        if all(active_mask):
            return coords.copy()
        return coords[np.asarray(active_mask, dtype=bool)]

    def jacobian(self) -> np.ndarray:
        """Calculates the internal coordinate Jacobian matrix using vectorized operations."""
        self._cache_check()

        # If a fully-built B was cached, return it directly. The cache is
        # invalidated by _cache_check whenever positions change, and the active
        # mask is stable within a single position evaluation.
        cached_B = self._cache.get('jacobian_B')
        if cached_B is not None:
            return cached_B

        if 'jacobian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_grads = self._compute_batched_gradients(positions, cell)

            # Non-batched coords use lightweight atoms
            atoms = self.light_atoms
            trans_data = []
            for coord in self.internals['translations']:
                idx = coord.indices
                jac = np.zeros((len(idx), 3))
                jac[:, coord.kwargs['dim']] = 1.0 / len(idx)
                trans_data.append((idx, jac))
            other_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                          for coord in self.internals['other']]
            rot_data = self._batched_rotation_gradients(positions)
            if rot_data is None:
                rot_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                            for coord in self.internals['rotations']]

            self._cache['jacobian_batched'] = batched_grads
            self._cache['jacobian_nonbatched'] = (trans_data, other_data, rot_data)
            # Store a unique object (not a singleton) for cache identity
            self._cache['jacobian'] = object()

        # Get cached data
        batched = self._cache['jacobian_batched']
        trans_data, other_data, rot_data = self._cache['jacobian_nonbatched']

        # Get counts for each type
        n_trans = len(trans_data)
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(other_data)
        n_rot = len(rot_data)

        # Build active masks per type
        (trans_active, bonds_active, angles_active, dihedrals_active,
         other_active, rot_active) = self._split_active_mask(
            n_trans, n_bonds, n_angles, n_dihedrals, n_other, n_rot)

        n_active = sum(self._active_mask)
        n_atoms = self.natoms + self.ndummies
        B = np.zeros((n_active, n_atoms, 3))
        row = 0

        # Translations (not batched)
        for i, (idx, jac) in enumerate(trans_data):
            if trans_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        for spec, active in zip(
            _BATCHED_COORD_FAMILIES,
            (bonds_active, angles_active, dihedrals_active),
        ):
            indices, grads = batched[spec.key]
            row = self._scatter_batched_jacobian_family(
                B, row, indices, grads, active
            )

        # Other (not batched)
        for i, (idx, jac) in enumerate(other_data):
            if other_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        # Rotations (not batched)
        for i, (idx, jac) in enumerate(rot_data):
            if rot_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        result = B.reshape((n_active, 3 * n_atoms))
        self._cache['jacobian_B'] = result
        return result

    def cell_jacobian(self) -> np.ndarray:
        """Compute Jacobian of internal coordinates with respect to cell matrix.

        Returns:
            np.ndarray: Shape (n_active_coords, 9) matrix where each row is
                        the flattened d(coord)/d(cell) gradient.

        Note:
            - Translations, rotations, and other non-periodic coordinates have
              zero cell derivatives (they don't depend on the cell).
            - Only bonds, angles, and dihedrals with non-zero ncvecs have
              non-zero cell derivatives.

        Raises:
            ValueError: If the system does not have periodic boundary conditions.
        """
        if not np.any(self.atoms.pbc):
            raise ValueError(
                "cell_jacobian() requires periodic boundary conditions. "
                "Set atoms.pbc = True for periodic systems."
            )

        self._cache_check()

        if 'cell_jacobian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Compute batched cell gradients for bonds, angles, dihedrals
            cell_grads = self._compute_batched_cell_gradients(positions, cell)
            self._cache['cell_jacobian_batched'] = cell_grads
            self._cache['cell_jacobian'] = object()

        cell_grads = self._cache['cell_jacobian_batched']

        # Get counts for each type
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        # Build active masks per type
        (trans_active, bonds_active, angles_active, dihedrals_active,
         other_active, rot_active) = self._split_active_mask(
            n_trans, n_bonds, n_angles, n_dihedrals, n_other, n_rot)

        n_active = sum(self._active_mask)
        B_cell = np.zeros((n_active, 3, 3))
        row = 0

        # Translations have zero cell derivatives (they're CoM positions)
        row += sum(trans_active)

        for spec, active in zip(
            _BATCHED_COORD_FAMILIES,
            (bonds_active, angles_active, dihedrals_active),
        ):
            row = self._scatter_batched_cell_gradient_family(
                B_cell, row, cell_grads[spec.key], active
            )

        # Other has zero cell derivatives (custom coordinates, not periodic)
        row += sum(other_active)

        # Rotations have zero cell derivatives
        row += sum(rot_active)

        # Flatten cell matrix to 9-element vector (row-major order)
        return B_cell.reshape((n_active, 9))

    def _rotation_padded_inputs(self, positions: np.ndarray):
        """Build padded (pos, refpos, mask) batches grouped by fragment.

        Returns ``(pos_pad, ref_pad, mask, frag_indices, frag_axis_slots,
        valid)`` where:
          pos_pad/ref_pad shape (n_frags, N_max, 3),
          mask shape (n_frags, N_max),
          frag_indices: list of np.array per fragment,
          frag_axis_slots: list of [axis0_idx, axis1_idx, axis2_idx]
            per fragment (each entry is the original Rotation index),
          valid: True if all fragments have all 3 axes.
        Cached per geometry on ``self._cache``.
        """
        cached = self._cache.get('rotation_pad')
        if cached is not None:
            return cached
        rotations = self.internals['rotations']
        if not rotations:
            out = (None, None, None, [], [], True)
            self._cache['rotation_pad'] = out
            return out
        groups = {}
        duplicate_slot = False
        for i, r in enumerate(rotations):
            key = (tuple(r.indices), r.kwargs['refpos'].tobytes())
            slot = groups.setdefault(key, [None, None, None])
            if slot[r.kwargs['axis']] is not None:
                duplicate_slot = True
            slot[r.kwargs['axis']] = i
        if duplicate_slot or any(None in slot for slot in groups.values()):
            out = (None, None, None, [], [], False)
            self._cache['rotation_pad'] = out
            return out
        n_frags = len(groups)
        n_max = max(len(r.indices) for r in rotations)
        pos_pad = np.zeros((n_frags, n_max, 3), dtype=np.float64)
        ref_pad = np.zeros((n_frags, n_max, 3), dtype=np.float64)
        mask = np.zeros((n_frags, n_max), dtype=np.float64)
        frag_indices = []
        frag_axis_slots = []
        for fi, slot in enumerate(groups.values()):
            r0 = rotations[slot[0]]
            n = len(r0.indices)
            pos_pad[fi, :n] = positions[r0.indices]
            ref_pad[fi, :n] = r0.kwargs['refpos']
            mask[fi, :n] = 1.0
            frag_indices.append(np.asarray(r0.indices))
            frag_axis_slots.append(slot)
        out = (pos_pad, ref_pad, mask, frag_indices, frag_axis_slots, True)
        self._cache['rotation_pad'] = out
        return out

    def _get_stabilized_quaternions(self, positions: np.ndarray):
        """Return cached stabilized quaternions, recomputing if needed.

        Returns a list of (4,) numpy arrays, one per fragment, or None
        if the batched path is invalid.  Also caches per-fragment
        eigenvalues/eigenvectors in ``self._cache['stabilized_q_eigh']``
        for reuse in the HVP path.
        """
        cached = self._cache.get('stabilized_q')
        if cached is not None:
            return cached
        rotations = self.internals['rotations']
        if not rotations:
            self._cache['stabilized_q'] = []
            self._cache['stabilized_q_eigh'] = (None, None)
            return []
        pos_pad, ref_pad, mask, frag_indices, slots, valid = (
            self._rotation_padded_inputs(positions)
        )
        if not valid:
            self._cache['stabilized_q'] = None
            self._cache['stabilized_q_eigh'] = (None, None)
            return None
        n_frags = len(slots)
        ws_all = np.empty((n_frags, 4))
        vecs_all = np.empty((n_frags, 4, 4))

        size_groups = {}
        for fi, indices in enumerate(frag_indices):
            size_groups.setdefault(len(indices), []).append(fi)

        for nr, group in size_groups.items():
            idx = np.array(group)
            pos_group = pos_pad[idx, :nr]
            ref_group = ref_pad[idx, :nr]
            ws, vecs = np.linalg.eigh(
                _build_F_matrices_np(pos_group, ref_group)
            )
            ws_all[idx] = ws
            vecs_all[idx] = vecs

        qs = []
        for fi, slot in enumerate(slots):
            q_prev = rotations[slot[0]].q_prev
            q = _stabilize_quaternion_from_eigh(
                ws_all[fi], vecs_all[fi], q_prev
            )
            for axis in range(3):
                rotations[slot[axis]].q_prev = q
            qs.append(q)
        self._cache['stabilized_q'] = qs
        self._cache['stabilized_q_eigh'] = (ws_all, vecs_all)
        return qs

    def _batched_rotation_values(self, positions: np.ndarray):
        """Per-Rotation values with projective quaternion stabilization.

        Returns a length-N_rotations list of floats in original order,
        or None when the heterogeneous fall-back is required.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        qs = self._get_stabilized_quaternions(positions)
        if qs is None:
            return None
        _, _, _, _, slots, _ = self._rotation_padded_inputs(positions)
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            vals = _expmap_np(qs[fi])
            for axis, rot_idx in enumerate(slot):
                out[rot_idx] = float(vals[axis])
        return out

    def _batched_rotation_gradients(self, positions: np.ndarray):
        """Per-Rotation gradients using stabilized quaternion.

        Returns a list of ``(indices, grad)`` tuples in original order,
        or None when the heterogeneous fall-back is required.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        qs = self._get_stabilized_quaternions(positions)
        if qs is None:
            return None
        pos_pad, ref_pad, mask, frag_indices, slots, _ = (
            self._rotation_padded_inputs(positions)
        )
        ws_all, vecs_all = self._cache.get('stabilized_q_eigh', (None, None))
        jac_all = _rotation_3axis_jacobian_batched_np(
            pos_pad, ref_pad, mask,
            q_stable_all=np.array(qs), ws_all=ws_all, vecs_all=vecs_all,
        )
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            for axis, rot_idx in enumerate(slot):
                out[rot_idx] = (frag_indices[fi], jac_all[fi, axis, :n])
        return out

    def _batched_rotation_hessians(self, positions: np.ndarray):
        """Compute per-Rotation Hessians using stabilized quaternion.

        Returns a list of ``(indices, hess)`` tuples in the original
        per-Rotation order.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        qs = self._get_stabilized_quaternions(positions)
        if qs is None:
            return [(r.indices, np.array(r.calc_hessian(
                self.light_atoms))) for r in rotations]
        pos_pad, ref_pad, _, frag_indices, slots, _ = (
            self._rotation_padded_inputs(positions)
        )
        ws_all, vecs_all = self._cache.get('stabilized_q_eigh', (None, None))
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            pos_frag = np.asarray(pos_pad[fi, :n], dtype=np.float64)
            ref_frag = np.asarray(ref_pad[fi, :n], dtype=np.float64)
            if ws_all is not None and vecs_all is not None:
                ws = ws_all[fi]
                vecs = vecs_all[fi]
            else:
                ws, vecs = np.linalg.eigh(
                    _build_F_matrix_np(
                        pos_frag - pos_frag.mean(0), ref_frag
                    )
                )
            top_count = int(np.sum(
                ws[-1] - ws <= _ROT_EIG_GAP_TOL
            ))
            if top_count > 1:
                hessians = _rotation_3axis_hessian_degenerate_fd(
                    pos_frag, ref_frag, qs[fi], top_count,
                )
                for axis, rot_idx in enumerate(slot):
                    out[rot_idx] = (
                        frag_indices[fi], hessians[axis]
                    )
                continue
            for axis, rot_idx in enumerate(slot):
                h = _rotation_hessian_single(
                    pos_frag, axis, ref_frag, q_stable=qs[fi],
                    ws=ws, vecs=vecs,
                )
                out[rot_idx] = (frag_indices[fi], h)
        return out

    def _get_hessian_skeleton(self, hessians):
        """Return a cached SparseInternalHessiansSkeleton for ``hessians``.

        The skeleton holds index-derived data (per-size groupings, scatter
        indices) that depend only on which coordinates exist and which
        atom indices they touch — not on positions or Hessian values. We
        invalidate by total coord count + active mask, which jointly
        cover the mutation paths: ``add_dummy_to_internals`` /
        ``find_all_*`` / ``check_for_bad_internals`` regenerations grow
        ``self.internals``, while ``apply_inequalities`` /
        ``validate_inequalities`` flip ``self._active``.
        """
        key = (len(hessians), self.natoms + self.ndummies,
               tuple(self._active_mask))
        cached = self._hessian_skeleton
        if cached is not None and cached[0] == key:
            return cached[1]
        skeleton = SparseInternalHessiansSkeleton(hessians,
                                                  self.natoms + self.ndummies)
        self._hessian_skeleton = (key, skeleton)
        return skeleton

    def hessian(self) -> np.ndarray:
        """Calculates the Hessian matrix for each internal coordinate using vectorized operations."""
        self._cache_check()

        # Return cached SparseInternalHessians object if available
        if 'hessian_result' in self._cache:
            return self._cache['hessian_result']

        if 'hessian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_hess = self._compute_batched_hessians(positions, cell)

            # Non-batched coords use lightweight atoms. Translation hessians are
            # identically zero (translations are linear in positions), so cache
            # one zero array per (n,) and reuse — avoids 24+ JAX calls per
            # hessian rebuild on systems with TRICs.
            atoms = self.light_atoms
            trans_data = []
            zero_cache = {}
            for coord in self.internals['translations']:
                n = len(coord.indices)
                z = zero_cache.get(n)
                if z is None:
                    z = np.zeros((n, 3, n, 3))
                    zero_cache[n] = z
                trans_data.append((coord.indices, z))
            other_data = [(coord.indices, np.array(coord.calc_hessian(atoms)))
                          for coord in self.internals['other']]
            rot_data = self._batched_rotation_hessians(positions)

            self._cache['hessian_batched'] = batched_hess
            self._cache['hessian_nonbatched'] = (trans_data, other_data, rot_data)
            # Store a unique object (not a singleton) for cache identity
            self._cache['hessian'] = object()

        # Get cached data
        batched = self._cache['hessian_batched']
        trans_data, other_data, rot_data = self._cache['hessian_nonbatched']

        # Get counts for each type
        n_trans = len(trans_data)
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(other_data)
        n_rot = len(rot_data)

        # Build active masks per type
        (trans_active, bonds_active, angles_active, dihedrals_active,
         other_active, rot_active) = self._split_active_mask(
            n_trans, n_bonds, n_angles, n_dihedrals, n_other, n_rot)

        n_atoms = self.natoms + self.ndummies
        hessians = []

        # Translations (not batched). Hessian rows are stored in cached
        # nonbatched data; SparseInternalHessian only reads .vals so views are
        # safe.
        for i, (idx, hess) in enumerate(trans_data):
            if trans_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        for spec, active in zip(
            _BATCHED_COORD_FAMILIES,
            (bonds_active, angles_active, dihedrals_active),
        ):
            indices, hess = batched[spec.key]
            self._append_batched_hessian_family(
                hessians, n_atoms, indices, hess, active
            )

        # Other (not batched)
        for i, (idx, hess) in enumerate(other_data):
            if other_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        # Rotations (not batched)
        for i, (idx, hess) in enumerate(rot_data):
            if rot_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        result = SparseInternalHessians(hessians, self.ndof,
                                        skeleton=self._get_hessian_skeleton(hessians))
        self._cache['hessian_result'] = result
        return result

    def _scatter_batched_family(self, jax_result, active, n_actual,
                                all_flat_cols, csr_offset, width,
                                use_sparse, data, out, row):
        """Scatter one batched bond/angle/dihedral HVP result into the output.

        Extracted verbatim from the three per-family blocks in hessian_rdot;
        returns the advanced output row.
        """
        if jax_result is None:
            return row
        hvp = np.asarray(device_get(jax_result))
        if active.all():
            hvp = hvp[:n_actual]
            n_coords = n_actual
            flat_cols = all_flat_cols
        else:
            n_coords = int(active.sum())
            flat_cols = all_flat_cols[active]
        if use_sparse:
            data[csr_offset:csr_offset + n_coords * width] = hvp.reshape(-1)
        else:
            out[row:row + n_coords, :] = 0
            out[np.arange(row, row + n_coords)[:, None], flat_cols] = \
                hvp.reshape(n_coords, -1)
        return row + n_coords

    @staticmethod
    def _active_array(active):
        return np.asarray(active, dtype=bool)

    def _launch_batched_hvp_family(self, spec: _BatchedCoordFamily,
                                   family: _BatchedCoordArrays,
                                   positions, tvecs, v_atoms, active):
        """Launch one batched HVP kernel for active coordinates.

        If every coordinate in the family is active, use padded arrays to keep the
        compiled shape stable. If inequality constraints deactivate some rows,
        slice to the active unpadded coordinates so the kernel does no work for
        inactive rows.
        """
        active = self._active_array(active)
        if family.n_actual == 0 or not active.any():
            return None

        if active.all():
            indices = family.indices_padded
            pos = positions[indices]
            tvec = tvecs[f"{spec.key}_padded"]
            v_sub = v_atoms[indices]
        else:
            indices = family.indices[active]
            pos = positions[indices]
            tvec = tvecs[spec.key][active]
            v_sub = v_atoms[indices]
        return spec.hvp_fn(pos, tvec, v_sub)

    def _scatter_batched_jacobian_family(self, B, row, indices, grads,
                                         active):
        active = self._active_array(active)
        n_active = int(active.sum())
        if n_active == 0:
            return row
        rows_idx = np.arange(row, row + n_active)[:, None]
        B[rows_idx, indices[active]] = grads[active]
        return row + n_active

    def _append_batched_hessian_family(self, hessians, n_atoms, indices, hess,
                                       active):
        active = self._active_array(active)
        if not active.any():
            return
        active_idx = indices[active]
        active_hess = hess[active]
        for i in range(len(active_idx)):
            hessians.append(
                SparseInternalHessian(n_atoms, active_idx[i], active_hess[i])
            )

    def _scatter_batched_cell_gradient_family(self, B_cell, row, grads,
                                              active):
        active = self._active_array(active)
        n_active = int(active.sum())
        if n_active > 0:
            B_cell[row:row + n_active] = grads[active]
        return row + n_active

    def _scatter_full_row(self, hvp, idx, use_sparse, data, out, off, row,
                          ndof):
        """Scatter one dense HVP row (other/rotation coords) into the output.

        Extracted verbatim from the three full-row blocks in hessian_rdot;
        returns the advanced (off, row).
        """
        if use_sparse:
            dense_row = np.zeros(ndof)
            dense_row.reshape((-1, 3))[idx] = hvp
            data[off:off + ndof] = dense_row
            off += ndof
        else:
            out_row = out[row].reshape((-1, 3))
            out_row[idx] = hvp
        return off, row + 1

    def _contract_batched_hvp_family(self, jax_result, active, n_actual,
                                     indices, mat_atoms, out, row):
        """Contract one batched HVP family with dense Cartesian columns."""
        if jax_result is None:
            return row
        hvp = np.asarray(device_get(jax_result))
        if active.all():
            hvp = hvp[:n_actual]
            idx = indices
            n_coords = n_actual
        else:
            idx = indices[active]
            n_coords = int(active.sum())
        out[row:row + n_coords] = np.einsum('cij,cijk->ck',
                                            hvp, mat_atoms[idx])
        return row + n_coords

    @staticmethod
    def _contract_full_hvp_row(hvp, idx, mat_atoms):
        """Contract one dense HVP row with dense Cartesian columns."""
        return np.einsum('ij,ijk->k', hvp, mat_atoms[idx])

    def hessian_rdot_mat(self, v: np.ndarray, mat: np.ndarray):
        """Compute ``hessian_rdot(v) @ mat`` without forming hessian_rdot.

        ``_q_ode`` only needs contractions of each coordinate Hessian-vector
        product with a few Cartesian vectors.  Computing those contractions
        directly avoids expanding compact per-coordinate HVPs into the sparse
        ``(n_active, ndof)`` matrix and then multiplying it back down.
        """
        self._cache_check()
        positions = self.all_positions
        cell = self.atoms.cell.array
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)

        mat = np.asarray(mat)
        if mat.ndim == 1:
            mat = mat[:, None]
            squeeze = True
        else:
            squeeze = False

        v_atoms = v.reshape((-1, 3))
        mat_atoms = mat.reshape((-1, 3, mat.shape[1]))

        active_mask = self._active_mask
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        (trans_active, bonds_active, angles_active, dihedrals_active,
         other_active, rot_active) = self._split_active_mask(
            n_trans, n_bonds, n_angles, n_dihedrals, n_other, n_rot)
        bonds_active = np.asarray(bonds_active, dtype=bool)
        angles_active = np.asarray(angles_active, dtype=bool)
        dihedrals_active = np.asarray(dihedrals_active, dtype=bool)

        out = np.zeros((sum(active_mask), mat.shape[1]), dtype=np.float64)
        row = sum(trans_active)  # Translation Hessians are zero.

        batched_active = (bonds_active, angles_active, dihedrals_active)
        families = self._batched_family_arrays
        batched_hvp = {
            spec.key: self._launch_batched_hvp_family(
                spec, families[spec.key], positions, tvecs, v_atoms, active
            )
            for spec, active in zip(_BATCHED_COORD_FAMILIES, batched_active)
        }

        rot_closed_results = []
        rot_batched_slots = None
        rot_batched_frag_indices = None
        rot_batched_hvp = None
        all_rot_active = bool(np.asarray(rot_active, dtype=bool).all())
        if all_rot_active and self.internals['rotations']:
            pos_pad, ref_pad, mask, frag_indices, slots, valid = (
                self._rotation_padded_inputs(positions)
            )
        else:
            valid = False
        if valid:
            qs = self._get_stabilized_quaternions(positions)
            q_stable_all = np.array(qs) if qs is not None else None
            cached_eigh = self._cache.get('stabilized_q_eigh', (None, None))
            ws_cached, vecs_cached = cached_eigh
            n_max = mask.shape[1]
            v_pad = np.zeros((len(frag_indices), n_max, 3), dtype=np.float64)
            for fi, fi_idx in enumerate(frag_indices):
                v_pad[fi, :len(fi_idx)] = v_atoms[fi_idx]
            rot_batched_hvp = _rotation_3axis_hvp_batched_closed(
                pos_pad, ref_pad, mask, v_pad,
                q_stable_all=q_stable_all,
                ws_all=ws_cached, vecs_all=vecs_cached,
            )
            rot_batched_slots = slots
            rot_batched_frag_indices = frag_indices
        else:
            for i, coord in enumerate(self.internals['rotations']):
                if rot_active[i]:
                    idx = np.array(coord.indices)
                    pos = positions[idx]
                    v_sub = v_atoms[idx]
                    axis = coord.kwargs['axis']
                    refpos = coord.kwargs['refpos']
                    q = coord._stabilized_quaternion(pos)
                    hvp = _rotation_hvp_closed(pos, axis, refpos, v_sub,
                                               q_stable=q)
                    rot_closed_results.append((hvp, idx))

        for spec, active in zip(_BATCHED_COORD_FAMILIES, batched_active):
            family = families[spec.key]
            row = self._contract_batched_hvp_family(
                batched_hvp[spec.key], active,
                family.n_actual,
                family.indices,
                mat_atoms, out, row,
            )

        atoms = self.light_atoms
        for i, coord in enumerate(self.internals['other']):
            if other_active[i]:
                hess = np.array(coord.calc_hessian(atoms))
                idx = np.array(coord.indices)
                v_sub = v_atoms[idx]
                hvp = np.einsum('aibj,bj->ai', hess, v_sub)
                out[row] = self._contract_full_hvp_row(hvp, idx, mat_atoms)
                row += 1

        if rot_batched_hvp is not None:
            for fi, slot in enumerate(rot_batched_slots):
                n = len(rot_batched_frag_indices[fi])
                idx = rot_batched_frag_indices[fi]
                contracted = np.einsum(
                    'anj,njk->ak', rot_batched_hvp[fi, :, :n, :],
                    mat_atoms[idx],
                )
                for axis, rot_idx in enumerate(slot):
                    out[row + rot_idx] = contracted[axis]
            row += len(self.internals['rotations'])
        else:
            for hvp, idx in rot_closed_results:
                out[row] = self._contract_full_hvp_row(hvp, idx, mat_atoms)
                row += 1

        if squeeze:
            return out[:, 0]
        return out

    def hessian_rdot(self, v: np.ndarray):
        """Compute Hessian @ v for all internal coordinates using direct HVP.

        This computes the same result as hessian().rdot(v) but uses forward-over-reverse
        mode autodiff (jvp(grad(f))) to compute Hessian-vector products directly,
        avoiding the O(n²) cost of materializing full Hessian matrices.

        Args:
            v: Vector of shape (ndof,) to multiply with each coordinate's Hessian

        Returns:
            Sparse CSR matrix of shape (n_active_coords, ndof) where each row
            is H_i @ v. Returns dense ndarray as fallback when not all
            coordinates are active.
        """
        self._cache_check()
        positions = self.all_positions
        cell = self.atoms.cell.array
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)

        # Reshape v for easy indexing
        v_atoms = v.reshape((-1, 3))  # (n_atoms, 3)
        ndof = self.ndof  # Cache to avoid repeated property lookups

        # Get active mask and counts
        active_mask = self._active_mask
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        (trans_active, bonds_active, angles_active, dihedrals_active,
         other_active, rot_active) = self._split_active_mask(
            n_trans, n_bonds, n_angles, n_dihedrals, n_other, n_rot)
        bonds_active = np.array(bonds_active, dtype=bool)
        angles_active = np.array(angles_active, dtype=bool)
        dihedrals_active = np.array(dihedrals_active, dtype=bool)

        n_active = sum(active_mask)

        # Fast path: when all coords are active, use pre-built CSR structure
        use_sparse = (n_active == self._csr_n_active)

        data = None
        out = None
        if use_sparse:
            data = self._csr_data
            data[:] = 0
        else:
            if (self._hvp_buf is None
                    or self._hvp_buf.shape != (n_active, ndof)):
                self._hvp_buf = np.zeros((n_active, ndof))
            out = self._hvp_buf
            out[:] = 0

        row = 0  # Current write position in output

        # Translations - Hessian is zero
        n_active_trans = sum(trans_active)
        # out[row:row+n_active_trans] is already zero from the clear
        row += n_active_trans

        # Launch all JAX HVP computations, deferring device_get
        # This allows JAX to pipeline the computations before we block on transfer

        batched_active = (bonds_active, angles_active, dihedrals_active)
        families = self._batched_family_arrays
        batched_hvp = {
            spec.key: self._launch_batched_hvp_family(
                spec, families[spec.key], positions, tvecs, v_atoms, active
            )
            for spec, active in zip(_BATCHED_COORD_FAMILIES, batched_active)
        }

        # Compute rotation HVPs using closed-form Hessian (handles
        # degenerate eigenvalues for linear/near-linear fragments).
        rot_closed_results = []
        rot_batched_slots = None
        rot_batched_frag_indices = None
        rot_batched_hvp = None
        all_rot_active = bool(np.asarray(rot_active, dtype=bool).all())
        if all_rot_active and self.internals['rotations']:
            pos_pad, ref_pad, mask, frag_indices, slots, valid = (
                self._rotation_padded_inputs(positions)
            )
        else:
            valid = False
        if valid:
            qs = self._get_stabilized_quaternions(positions)
            q_stable_all = np.array(qs) if qs is not None else None
            cached_eigh = self._cache.get('stabilized_q_eigh', (None, None))
            ws_cached, vecs_cached = cached_eigh
            n_max = mask.shape[1]
            v_pad = np.zeros((len(frag_indices), n_max, 3), dtype=np.float64)
            for fi, fi_idx in enumerate(frag_indices):
                v_pad[fi, :len(fi_idx)] = v_atoms[fi_idx]
            rot_batched_hvp = _rotation_3axis_hvp_batched_closed(
                pos_pad, ref_pad, mask, v_pad,
                q_stable_all=q_stable_all,
                ws_all=ws_cached, vecs_all=vecs_cached,
            )
            rot_batched_slots = slots
            rot_batched_frag_indices = frag_indices
        else:
            for i, coord in enumerate(self.internals['rotations']):
                if rot_active[i]:
                    idx = np.array(coord.indices)
                    pos = positions[idx]
                    v_sub = v_atoms[idx]
                    axis = coord.kwargs['axis']
                    refpos = coord.kwargs['refpos']
                    q = coord._stabilized_quaternion(pos)
                    hvp = _rotation_hvp_closed(pos, axis, refpos, v_sub,
                                               q_stable=q)
                    rot_closed_results.append((hvp, idx))

        # Now collect results with device_get and scatter into output

        for spec, active in zip(_BATCHED_COORD_FAMILIES, batched_active):
            family = families[spec.key]
            row = self._scatter_batched_family(
                batched_hvp[spec.key], active,
                family.n_actual,
                family.flat_cols,
                family.csr_offset,
                spec.width, use_sparse, data, out, row,
            )

        # Other - use existing hessian computation (typically few coords, loop is fine)
        atoms = self.light_atoms
        off = self._csr_other_offset if use_sparse else 0
        for i, coord in enumerate(self.internals['other']):
            if other_active[i]:
                hess = np.array(coord.calc_hessian(atoms))
                idx = np.array(coord.indices)
                v_sub = v_atoms[idx]
                hvp = np.einsum('aibj,bj->ai', hess, v_sub)
                off, row = self._scatter_full_row(
                    hvp, idx, use_sparse, data, out, off, row, ndof)

        # Rotations - collect results from closed-form Hessian (no NaN
        # for degenerate eigenvalues)
        if rot_batched_hvp is not None:
            hvp_padded = rot_batched_hvp
            # hvp_padded.shape == (n_frags, 3, N_max, 3)
            ordered = [None] * len(self.internals['rotations'])
            for fi, slot in enumerate(rot_batched_slots):
                n = len(rot_batched_frag_indices[fi])
                for axis, rot_idx in enumerate(slot):
                    ordered[rot_idx] = (
                        hvp_padded[fi, axis, :n, :],
                        rot_batched_frag_indices[fi],
                    )
            for i, coord in enumerate(self.internals['rotations']):
                if not rot_active[i]:
                    continue
                hvp, idx = ordered[i]
                off, row = self._scatter_full_row(
                    hvp, idx, use_sparse, data, out, off, row, ndof)
        else:
            for hvp, idx in rot_closed_results:
                off, row = self._scatter_full_row(
                    hvp, idx, use_sparse, data, out, off, row, ndof)

        if use_sparse:
            return sparse.csr_matrix(
                (data, self._csr_indices, self._csr_indptr),
                shape=(self._csr_n_active, ndof), copy=False,
            )
        return out[:row]

    def wrap(self, vec: np.ndarray, origin: np.ndarray = None) -> np.ndarray:
        """Wraps an internal coord. displacement vector into a valid domain."""
        start = 0
        for name in self._names:
            active = self._active[name]
            n = sum(active)
            if name == 'dihedrals':
                vec[start:start + n] = (vec[start:start + n] + np.pi) % (2 * np.pi) - np.pi
            elif name == 'rotations' and n > 0:
                self._wrap_rotation_diff(vec, start, active, origin=origin)
            start += n
        return vec

    def _wrap_rotation_diff(self, vec, rot_start, active=None, origin=None):
        """Wrap rotation differences using an equivalent target rotation.

        A rotation vector ``target`` is equivalent to
        ``target + 2π k target/|target|``.  Select the equivalent target that
        is closest to ``origin``.  Wrapping along the displacement direction
        is only valid when the two rotations share an axis and can change the
        represented orientation for general rotations.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return
        if active is None:
            active = [True] * len(rotations)
        active = np.asarray(active, dtype=bool)
        local_index = {}
        n_active = 0
        for i, is_active in enumerate(active):
            if is_active:
                local_index[i] = n_active
                n_active += 1
        # Group rotations by fragment (same indices and refpos)
        groups = {}
        for i, r in enumerate(rotations):
            key = (tuple(r.indices), r.kwargs['refpos'].tobytes())
            groups.setdefault(key, []).append(i)

        for key, indices in groups.items():
            active_indices = [i for i in indices if active[i]]
            if len(active_indices) != 3:
                continue
            # Get the 3-component rotation difference vector
            idx = [rot_start + local_index[i] for i in active_indices]
            v = vec[idx].copy()
            base = np.zeros(3) if origin is None else np.asarray(origin)[idx]
            target = base + v
            target_norm = np.linalg.norm(target)
            if target_norm < 1e-10:
                continue
            axis = target / target_norm
            k0 = int(np.rint(-np.dot(v, axis) / (2 * np.pi)))
            candidates = [v + 2 * np.pi * k * axis
                          for k in (k0 - 1, k0, k0 + 1)]
            best_v = min(candidates, key=lambda candidate: candidate @ candidate)
            vec[idx] = best_v

    def __iter__(self) -> Iterator[Coordinate]:
        for name in self._names:
            for coord in self.internals[name]:
                yield coord

    def _get_neighbors(self, dx: np.ndarray) -> Iterator[np.ndarray]:
        pbc = self.atoms.pbc
        if self.cell is None or not np.allclose(self.cell, self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(
                complete_cell(self.cell), pbc=pbc
            )
            self.rcell = Cell(rcell)
            self._rcell_reciprocal_T = self.rcell.reciprocal().T
        dx_sc = dx @ self._rcell_reciprocal_T
        offset = np.zeros(3, dtype=np.int32)
        for _ in range(2):
            offset += pbc * ((dx_sc - offset) // 1.).astype(np.int32)

        for ts in product(*[np.arange(-1 * p, p + 1) for p in pbc]):
            yield (np.array(ts) - offset) @ self.op

    def _find_mic(self, indices: Tuple[int, ...]) -> np.ndarray:
        ncvecs = np.zeros((len(indices) - 1, 3), dtype=np.int32)
        if not np.any(self.atoms.pbc):
            return ncvecs

        pos = self.all_positions
        dxs = np.array([
            pos[i] - pos[j] for i, j in zip(indices[1:], indices[:-1])
        ])

        for dx, ncvec in zip(dxs, ncvecs):
            vlen = np.inf
            for neighbor in self._get_neighbors(dx):
                trial = np.linalg.norm(dx + neighbor @ self.atoms.cell)
                if trial < vlen:
                    vlen = trial
                    ncvec[:] = neighbor
        return ncvecs

    def _get_ncvecs(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None
    ) -> np.ndarray:
        if ncvecs is None:
            if mic is None or not mic:
                return np.zeros((len(indices) - 1, 3), dtype=np.int32)
            else:
                return self._find_mic(indices)
        else:
            if mic:
                raise ValueError(
                    "Minimum image convention (mic) requested, but explicit "
                    "periodic vectors (ncvecs) were also provided! These "
                    "keyword arguments are mutually exclusive."
                )
            return np.asarray(
                ncvecs,
                dtype=np.int32
            ).reshape((len(indices) - 1, 3))

    def get_principal_rotation_axes(
        self,
        indices: Tuple[int, ...]
    ) -> jnp.ndarray:
        """Calculates the principal axes of rotation of a cluster of atoms."""
        indices = np.asarray(indices, dtype=np.int32)
        pos = self.all_positions
        dx = pos[indices] - pos[indices].mean(0)
        Inertia = (
            (dx * dx).sum() * jnp.eye(3)
            - (dx[:, None, :] * dx[:, :, None]).sum(0)
        )
        _, rvecs = jnp.linalg.eigh(Inertia)
        return rvecs

    def add_dummy_to_internals(
        self,
        idx: int
    ) -> None:
        didx = self.dinds[idx]
        assert didx >= 0
        npos = len(self.all_positions)

        def dedupe(name, coords, active):
            targets_by_name = getattr(self, '_targets', None)
            kinds_by_name = getattr(self, '_kind', None)
            has_metadata = targets_by_name is not None
            if has_metadata:
                targets = targets_by_name[name]
                kinds = kinds_by_name[name]
            else:
                targets = kinds = None

            new_coords = []
            new_active = []
            new_targets = []
            new_kinds = []
            changed = len(coords) != len(active)
            for i, (coord, is_active) in enumerate(zip(coords, active)):
                try:
                    existing = new_coords.index(coord)
                except ValueError:
                    new_coords.append(coord)
                    new_active.append(is_active)
                    if has_metadata:
                        new_targets.append(targets[i])
                        new_kinds.append(kinds[i])
                else:
                    if has_metadata and (
                        new_kinds[existing] != kinds[i]
                        or new_targets[existing] != targets[i]
                    ):
                        raise DuplicateConstraintError(
                            'Dummy expansion produced duplicate coordinates '
                            'with conflicting constraints.'
                        )
                    new_active[existing] = new_active[existing] or is_active
                    changed = True
            if has_metadata:
                targets_by_name[name] = new_targets
                kinds_by_name[name] = new_kinds
            return new_coords, new_active, changed

        changed = False
        translations = []
        for trans in self.internals['translations']:
            if idx in trans.indices and didx not in trans.indices:
                new_indices = (*trans.indices, didx)
                translations.append(
                    Translation(new_indices, trans.kwargs['dim'])
                )
                changed = True
            else:
                translations.append(trans)
        translations, trans_active, trans_deduped = dedupe(
            'translations', translations, self._active['translations']
        )
        self.internals['translations'] = translations
        self._active['translations'] = trans_active
        changed = changed or trans_deduped

        rotations = []
        for rot in self.internals['rotations']:
            if idx in rot.indices and didx not in rot.indices:
                new_indices = np.array((*rot.indices, didx), dtype=np.int32)
                if np.all(new_indices < npos):
                    rotations.append(Rotation(
                        new_indices, rot.kwargs['axis'],
                        self.all_positions[new_indices]
                    ))
                    changed = True
                    continue
            rotations.append(rot)
        rotations, rot_active, rot_deduped = dedupe(
            'rotations', rotations, self._active['rotations']
        )
        self.internals['rotations'] = rotations
        self._active['rotations'] = rot_active
        changed = changed or rot_deduped

        if changed:
            self._invalidate_structure()

    def check_all_gradients(
        self, delta: float = 1e-4, atol: float = 1e-6
    ) -> bool:
        """Run check_gradient on every internal coordinate; True iff all pass."""
        success = True
        for coord in self:
            success &= coord.check_gradient(self.all_atoms, delta, atol)
        return success

    def check_all_hessians(
        self, delta: float = 1e-4, atol: float = 1e-6,
    ) -> bool:
        """Run check_hessian on every internal coordinate; True iff all pass."""
        success = True
        for coord in self:
            success &= coord.check_hessian(self.all_atoms, delta, atol)
        return success


class Constraints(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None,
        ignore_rotation: bool = True,
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self._targets = {key: [] for key in self._names}
        self._kind = {key: [] for key in self._names}
        self.ignore_rotation = ignore_rotation
        for ase_cons in atoms.constraints:
            self.merge_ase_constraint(ase_cons)

    def copy(
        self,
        _coord_memo=None,
        _dummies: Atoms = None,
        _dinds: np.ndarray = None,
    ) -> 'Constraints':
        if _coord_memo is None:
            _coord_memo = {}
        if _dummies is None:
            _dummies = self.dummies.copy()
        if _dinds is None:
            _dinds = self.dinds.copy()

        def clone(coord):
            key = id(coord)
            if key not in _coord_memo:
                _coord_memo[key] = coord.copy()
            return _coord_memo[key]

        new = self.__class__(
            self.atoms, _dummies, _dinds, self.ignore_rotation
        )
        for name in self._names:
            new.internals[name] = [
                clone(coord) for coord in self.internals[name]
            ]
            new._targets[name] = self._targets[name].copy()
            new._active[name] = self._active[name].copy()
            new._kind[name] = self._kind[name].copy()
        new._invalidate_structure()
        return new

    @property
    def targets(self) -> np.ndarray:
        vec = []
        for key in self._names:
            vec += self._targets[key]
        return np.array(vec, dtype=np.float64)[self._active_indices]

    def residual(self) -> np.ndarray:
        """Calculates the constraint residual vector."""
        targets = self.targets
        res = self.wrap(self.calc() - targets, origin=targets)
        if self.ignore_rotation and self.nrotations:
            res[-self.nrotations:] = 0.
        return res

    def has_inequalities(self) -> bool:
        """Check if any inequality constraints (lt/gt) exist."""
        for name in self._names:
            for kind in self._kind[name]:
                if kind in ('lt', 'gt'):
                    return True
        return False

    def disable_satisfied_inequalities(self) -> None:
        for name in self._names:
            for i, (coord, kind, target) in enumerate(zip(
                self.internals[name], self._kind[name], self._targets[name]
            )):
                if kind == 'lt' and coord.calc(self.all_atoms) <= target:
                    active = False
                elif kind == 'gt' and coord.calc(self.all_atoms) >= target:
                    active = False
                else:
                    active = True
                self._active[name][i] = active

    def validate_inequalities(self) -> bool:
        all_valid = True
        for name in self._names:
            for i, (coord, kind, target) in enumerate(zip(
                self.internals[name], self._kind[name], self._targets[name]
            )):
                if self._active[name][i]:
                    continue
                if kind == 'lt' and coord.calc(self.all_atoms) > target:
                    self._active[name][i] = True
                    all_valid = False
                elif kind == 'gt' and coord.calc(self.all_atoms) < target:
                    self._active[name][i] = True
                    all_valid = False
        return all_valid

    def fix_rotation(
        self,
        indices: Union[Tuple[int, ...], Rotation] = None,
        axis: int = None,
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError(
                    "'axis' keyword cannot be used with explicit Rotation"
                )
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.asarray(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.fix_rotation(indices, axis)
                return
            new = Rotation(
                indices,
                axis,
                self.all_positions[indices]
            )
        try:
            _ = self.internals['rotations'].index(new)
        except ValueError:
            self.internals['rotations'].append(new)
            self._targets['rotations'].append(0.)
            self._active['rotations'].append(True)
            self._kind['rotations'].append('eq')
            self._invalidate_structure()
        else:
            raise DuplicateConstraintError(
                "This rotation has already been constrained!"
            )

    def fix_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None,
        target: float = None,
        replace_ok: bool = True,
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Translation'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            if np.isscalar(index):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                if target is not None:
                    raise ValueError(
                        '"target" keyword requires explicit "dim"!'
                    )
                for dim in range(3):
                    self.fix_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if target is None:
            target = new.calc(self.all_atoms)
        try:
            idx = self.internals['translations'].index(new)
        except ValueError:
            self.internals['translations'].append(new)
            self._targets['translations'].append(target)
            self._active['translations'].append(True)
            self._kind['translations'].append('eq')
            self._invalidate_structure()
        else:
            if replace_ok:
                self._targets['translations'][idx] = target
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(new, self._targets['translations'][idx])
            )

    def _fix_internal(
        self,
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        conv: float,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
        target: float = None,
        comparator: str = 'eq',
        replace_ok: bool = True,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        if target is None:
            target = new.calc(self.all_atoms)
        else:
            target *= conv
        try:
            idx = self.internals[name].index(new)
        except ValueError:
            self.internals[name].append(new)
            self._targets[name].append(target)
            self._active[name].append(True)
            self._kind[name].append(comparator)
            self._invalidate_structure()
        else:
            if replace_ok:
                self._targets[name][idx] = target
                self._kind[name][idx] = comparator
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(new, self._targets[name][idx] / conv)
            )

    fix_bond = partialmethod(_fix_internal, Bond, 'bonds', 1.)
    fix_angle = partialmethod(_fix_internal, Angle, 'angles', np.pi / 180.)
    fix_dihedral = partialmethod(
        _fix_internal, Dihedral, 'dihedrals', np.pi / 180.
    )

    def fix_other(
        self,
        coord: Coordinate,
        target: float = None,
        comparator: str = 'eq',
        replace_ok: bool = True,
    ) -> None:
        if target is None:
            target = coord.calc(self.all_atoms)
        try:
            idx = self.internals['other'].index(coord)
        except ValueError:
            self.internals['other'].append(coord)
            self._targets['other'].append(target)
            self._active['other'].append(True)
            self._kind['other'].append(comparator)
            self._invalidate_structure()
        else:
            if replace_ok:
                self._targets['other'][idx] = target
                self._kind['other'][idx] = comparator
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(coord, self._targets['other'][idx])
            )

    def _merge_fix_atoms(self, ase_cons):
        for index in ase_cons.index:
            self._ignore_duplicate(self.fix_translation, index)

    def _merge_fix_com(self, ase_cons):
        self._ignore_duplicate(self.fix_translation)

    def _merge_fix_bond_lengths(self, ase_cons):
        for i, indices in enumerate(ase_cons.pairs):
            target = None if ase_cons.bondlengths is None else ase_cons.bondlengths[i]
            self._ignore_duplicate(
                self.fix_bond, indices, mic=True, target=target
            )

    @staticmethod
    def _fix_cartesian_indices_and_mask(ase_cons):
        # ASE's FixCartesian API changed around 3.23: older versions store a
        # scalar atom in .a and invert the mask at construction (stored mask
        # True = *free*); newer versions store .index with mask True = fixed.
        raw_mask = np.asarray(ase_cons.mask, dtype=bool)
        if hasattr(ase_cons, 'index'):
            return np.atleast_1d(ase_cons.index), raw_mask
        return np.atleast_1d(ase_cons.a), ~raw_mask

    def _merge_fix_cartesian(self, ase_cons):
        indices, fixed_mask = self._fix_cartesian_indices_and_mask(ase_cons)
        for atom in indices:
            for dim in np.flatnonzero(fixed_mask):
                self._ignore_duplicate(
                    self.fix_translation, int(atom), dim=int(dim)
                )

    def _merge_fix_internals(self, ase_cons):
        """Merge ASE FixInternals entries, preserving ASE target values."""
        for ase_cons_list, adder in zip(
            (ase_cons.bonds, ase_cons.angles, ase_cons.dihedrals),
            (self.fix_bond, self.fix_angle, self.fix_dihedral),
        ):
            for target, indices in ase_cons_list:
                self._ignore_duplicate(adder, indices, target=target)
        if ase_cons.bondcombos:
            raise RuntimeError(
                "Sella currently does not support combination constraints."
            )

    def merge_ase_constraint(self, ase_cons: FixConstraint) -> None:
        handlers = (
            (FixAtoms, self._merge_fix_atoms),
            (FixCom, self._merge_fix_com),
            (FixBondLengths, self._merge_fix_bond_lengths),
            (FixCartesian, self._merge_fix_cartesian),
            (FixInternals, self._merge_fix_internals),
        )
        for ase_type, handler in handlers:
            if isinstance(ase_cons, ase_type):
                handler(ase_cons)
                return
        raise RuntimeError(
            "Sella does not currently implement the ASE {} Constraint "
            "class.".format(ase_cons.__class__.__name__)
        )


class Internals(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        atol: float = 15.,
        dinds: np.ndarray = None,
        cons: Constraints = None,
        allow_fragments: bool = False,
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self.atol = atol * np.pi / 180.
        # Fragment-welding redundancy (see _mst_welding_bonds). Augmentation is
        # ON by default: it closes fragment rings, symmetrizes extended contact
        # interfaces, and makes welding permutation-invariant, at the cost of
        # redundant internals (which Sella handles). It is provably inert on
        # generic asymmetric systems -- the physical gate means only genuine
        # near-degenerate contacts qualify -- so it only acts where the bare
        # spanning tree would be biased.
        self._weld_augment = True
        self._weld_tol = 0.15    # add contacts within 15% of an interface min
        self._weld_gate = 1.4    # ... and <= 1.4 x sum(covalent radii)
        self.forbidden = {key: [] for key in self._names}
        if cons is None:
            cons = Constraints(self.atoms, self.dummies, self.dinds)
        else:
            if (
                (dummies is not None and dummies is not cons.dummies)
                or (dinds is not None and dinds is not cons.dinds)
            ):
                raise RuntimeError(
                    "Constraints has inconsistent dummy atom definitions!"
                )
            self.dummies = cons.dummies
            self.dinds = cons.dinds
        self.cons = cons

        for kind, adder in zip(self._names, (
            self.add_translation, self.add_bond, self.add_angle,
            self.add_dihedral, self.add_other, self.add_rotation
        )):
            for coord in self.cons.internals[kind]:
                adder(coord)
        self.allow_fragments = allow_fragments
        self.fragment_atom_groups = None

    def copy(self) -> 'Internals':
        coord_memo = {}
        dummies = self.dummies.copy()
        dinds = self.dinds.copy()

        def clone(coord):
            key = id(coord)
            if key not in coord_memo:
                coord_memo[key] = coord.copy()
            return coord_memo[key]

        new = self.__class__(
            self.atoms,
            dummies,
            self.atol * 180. / np.pi,
            dinds,
            self.cons.copy(
                _coord_memo=coord_memo,
                _dummies=dummies,
                _dinds=dinds,
            ),
            self.allow_fragments,
        )
        for name in self._names:
            new.internals[name] = [
                clone(coord) for coord in self.internals[name]
            ]
            if name in ('bonds', 'angles', 'dihedrals'):
                new._internals_set[name] = {
                    new._internal_key(coord)
                    for coord in new.internals[name]
                }
            else:
                new._internals_set[name] = self._internals_set[name].copy()
            new.forbidden[name] = [
                clone(coord) for coord in self.forbidden[name]
            ]
            new._active[name] = self._active[name].copy()
        if self.fragment_atom_groups is not None:
            new.fragment_atom_groups = [
                g.copy() for g in self.fragment_atom_groups
            ]
        new._invalidate_structure()
        return new

    def add_rotation(
        self,
        indices: Union[Tuple[int, ...], Rotation] = None,
        axis: int = None,
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError(
                    "'axis' keyword cannot be used with explicit Rotation"
                )
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.array(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.add_rotation(indices, axis)
                return
            new = Rotation(
                indices,
                axis,
                self.all_positions[indices]
            )
        if (
            new in self.internals['rotations']
            or new in self.forbidden['rotations']
        ):
            raise DuplicateInternalError
        self.internals['rotations'].append(new)
        self._active['rotations'].append(True)
        self._invalidate_structure()

    def add_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Translation'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.add_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if (
            new in self.internals['translations']
            or new in self.forbidden['translations']
        ):
            raise DuplicateInternalError
        self.internals['translations'].append(new)
        self._active['translations'].append(True)
        self._invalidate_structure()

    def _add_fragment_coords(
        self, group, with_rotation: bool = True
    ) -> None:
        """Add fragment translation (and optionally rotation) coordinates.

        Idempotent: ``find_all_bonds()`` may run more than once on the same
        Internals (e.g. a pre-populated fragment Internals passed to
        InternalPES with auto_find_internals=True), so silently skip fragment
        TRICs that already exist instead of raising DuplicateInternalError --
        mirroring the duplicate handling on fragment-welding bonds.
        """
        for dim in range(3):
            self._ignore_duplicate(self.add_translation, group, dim)
        if with_rotation and len(group) >= 2:
            for axis in range(3):
                self._ignore_duplicate(self.add_rotation, group, axis)

    @staticmethod
    def _internal_key(coord: 'Internal') -> tuple:
        """Orientation-independent dedup key for an Internal coordinate.

        ``Internal.__eq__`` treats a coordinate and its reverse as equal, so
        the ``_internals_set`` key must be canonical across orientation --
        otherwise ``add_bond((1, 0))`` slips past a prior ``add_bond((0, 1))``.
        Take the lexicographically smaller of the forward and reversed keys.
        """
        def raw(c: 'Internal') -> tuple:
            return (tuple(int(i) for i in c.indices),
                    tuple(map(tuple, c.kwargs['ncvecs'])))
        return min(raw(coord), raw(coord.reverse()))

    def _add_internal(
        self,
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        key = self._internal_key(new)
        if (
            key in self._internals_set[name]
            or new in self.forbidden[name]
        ):
            raise DuplicateInternalError
        self.internals[name].append(new)
        self._internals_set[name].add(key)
        self._active[name].append(True)
        self._invalidate_structure()

    add_bond = partialmethod(_add_internal, Bond, 'bonds')
    add_angle = partialmethod(_add_internal, Angle, 'angles')
    add_dihedral = partialmethod(_add_internal, Dihedral, 'dihedrals')

    def add_other(
        self,
        coord: Coordinate,
    ) -> None:
        try:
            self.internals['other'].index(coord)
        except ValueError:
            self.internals['other'].append(coord)
            self._active['other'].append(True)
            self._invalidate_structure()
        else:
            raise DuplicateInternalError()

    def forbid_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Translation'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.forbid_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        try:
            idx = self.internals['translations'].index(new)
        except ValueError:
            pass
        else:
            self.internals['translations'].pop(idx)
            self._active['translations'].pop(idx)
            self._invalidate_structure()
        if new not in self.forbidden['translations']:
            self.forbidden['translations'].append(new)

    def _forbid_internal(
        self,
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        try:
            idx = self.internals[name].index(new)
        except ValueError:
            pass
        else:
            removed = self.internals[name].pop(idx)
            self._active[name].pop(idx)
            self._internals_set[name].discard(self._internal_key(removed))
            self._invalidate_structure()
        if new not in self.forbidden[name]:
            self.forbidden[name].append(new)

    forbid_bond = partialmethod(_forbid_internal, Bond, 'bonds')
    forbid_angle = partialmethod(_forbid_internal, Angle, 'angles')
    forbid_dihedral = partialmethod(_forbid_internal, Dihedral, 'dihedrals')

    @staticmethod
    def flood_fill(
        index: int,
        nbonds: np.ndarray,
        c10y: np.ndarray,
        labels: np.ndarray,
        label: int
    ) -> None:
        for j in c10y[index, :nbonds[index]]:
            if labels[j] != label:
                labels[j] = label
                Internals.flood_fill(j, nbonds, c10y, labels, label)

    @staticmethod
    def _component_labels(labels, natoms):
        comp = labels.copy()
        next_label = int(comp.max()) + 1 if comp.size else 0
        for atom in range(natoms):
            if comp[atom] < 0:
                comp[atom] = next_label
                next_label += 1
        return comp, next_label

    @staticmethod
    def _component_pair_key(comp_a, comp_b):
        return frozenset((int(comp_a), int(comp_b)))

    @staticmethod
    def _record_link(links, have, a, b, ts):
        pair = (min(a, b), max(a, b))
        if pair in have:
            return False
        links.append((a, b, ts))
        have.add(pair)
        return True

    def _ensure_reduced_cell_cache(self):
        if self.cell is not None and np.allclose(self.cell, self.atoms.cell):
            return
        self.cell = self.atoms.cell.array.copy()
        rcell, self.op = minkowski_reduce(
            complete_cell(self.cell), pbc=self.atoms.pbc
        )
        self.rcell = Cell(rcell)
        self._rcell_reciprocal_T = self.rcell.reciprocal().T

    def _periodic_pair_chunk_size(self, n_images):
        # Per pair-image, the peak contains integer translations, Cartesian
        # translations, the displaced vectors used by norm(), and distances.
        # Use a deliberately conservative estimate to keep real peaks bounded.
        bytes_per_pair_image = (
            3 * np.dtype(np.int32).itemsize
            + 7 * np.dtype(np.float64).itemsize
        )
        bytes_per_pair = max(1, n_images * bytes_per_pair_image)
        return max(1, PERIODIC_PAIR_CHUNK_BYTES // bytes_per_pair)

    def _iter_periodic_pair_distances(self, ii, jj):
        self._ensure_reduced_cell_cache()

        ranges = [np.arange(-1 * p, p + 1) for p in self.atoms.pbc]
        base_ts = np.array(list(product(*ranges)), dtype=np.int32)
        chunk_size = self._periodic_pair_chunk_size(len(base_ts))
        positions = self.atoms.positions
        cell = self.atoms.cell.array

        for start in range(0, len(ii), chunk_size):
            stop = min(start + chunk_size, len(ii))
            dx = positions[jj[start:stop]] - positions[ii[start:stop]]
            dx_sc = dx @ self._rcell_reciprocal_T

            offset = np.zeros(dx_sc.shape, dtype=np.int32)
            for _ in range(2):
                offset += (
                    self.atoms.pbc * ((dx_sc - offset) // 1.)
                ).astype(np.int32)

            translations = (base_ts[None, :, :] - offset[:, None, :]) @ self.op
            tvecs_cart = translations @ cell
            dists = np.linalg.norm(dx[:, None, :] + tvecs_cart, axis=2)
            yield start, stop, dists, translations

    def _periodic_pair_distances(self, ii, jj):
        chunks = list(self._iter_periodic_pair_distances(ii, jj))
        if not chunks:
            return np.empty((0, 0)), np.empty((0, 0, 3), dtype=np.int32)
        dists = np.concatenate([chunk[2] for chunk in chunks])
        translations = np.concatenate([chunk[3] for chunk in chunks])
        return dists, translations

    def _find_bonds_vectorized(self, labels, scale, rcov):
        """Vectorized bond search across all candidate atom pairs.

        Returns a list of (i, j, ts) tuples for bonds that pass the
        distance threshold, where ts is the integer translation vector.
        """
        natoms = self.natoms

        ii, jj = np.triu_indices(natoms, k=0)
        same_frag = (labels[ii] == labels[jj]) & (labels[ii] != -1)
        keep = ~same_frag
        ii, jj = ii[keep], jj[keep]
        if len(ii) == 0:
            return []

        results = []
        for start, stop, dists, translations in (
            self._iter_periodic_pair_distances(ii, jj)
        ):
            ii_chunk = ii[start:stop]
            jj_chunk = jj[start:stop]
            thresholds = scale * (rcov[ii_chunk] + rcov[jj_chunk])
            bond_mask = dists <= thresholds[:, None]

            self_bond = (ii_chunk == jj_chunk)
            zero_ts = np.all(translations == 0, axis=2)
            bond_mask &= ~(self_bond[:, None] & zero_ts)

            pair_idx, ts_idx = np.nonzero(bond_mask)
            for k in range(len(pair_idx)):
                p = pair_idx[k]
                t = ts_idx[k]
                ts = translations[p, t].astype(np.int32)
                results.append((int(ii_chunk[p]), int(jj_chunk[p]), ts))
        return results

    def _mst_welding_bonds(self, labels):
        """Connect disconnected fragments with a minimum-spanning-tree of the
        shortest inter-fragment atom contacts.

        Returns a list of ``(i, j, ts)`` links (``ts`` = integer cell
        translation) that merge all fragments into one connected graph, adding
        exactly ``n_components - 1`` bonds -- each the shortest contact joining
        two still-separate components.

        This replaces the old "inflate the covalent-radius multiplier until the
        graph connects" welding, which grew the cutoff to ~2-3x and then added
        *every* inter-fragment pair under it -- manufacturing grossly stretched
        (4-6 A) and transannular bonds.  Shortest-first linking never prefers a
        transannular contact over the genuine short link, and adds only the
        minimal set needed for a complete internal-coordinate system.

        ``labels`` follows the ``find_all_bonds`` convention: ``>= 0`` is a
        fragment id and ``-1`` marks a lone (unbonded) atom, treated here as its
        own singleton component.
        """
        comp, next_label = self._component_labels(labels, self.natoms)
        n_components = len({int(c) for c in comp})

        ii, jj = np.triu_indices(self.natoms, k=1)
        keep = comp[ii] != comp[jj]
        ii, jj = ii[keep], jj[keep]
        if len(ii) == 0:
            return []

        best_dist = np.empty(len(ii), dtype=np.float64)
        best_ts = np.empty((len(ii), 3), dtype=np.int32)
        for start, stop, dists, translations in (
            self._iter_periodic_pair_distances(ii, jj)
        ):
            best = dists.argmin(axis=1)
            rows = np.arange(stop - start)
            best_dist[start:stop] = dists[rows, best]
            best_ts[start:stop] = translations[rows, best].astype(np.int32)

        # Connect components. A bare minimum-spanning *tree* is minimal but
        # biased: it under-determines extended contact interfaces (one weld
        # where several comparable contacts exist), turns a ring of fragments
        # into a chain (C- not O-shaped), and breaks symmetry on exact distance
        # ties (tie-broken by atom index -> not permutation invariant). The
        # optional augmentation pass first adds *all* physically-plausible
        # near-minimum contacts per interface (redundant internals, which Sella
        # handles via the pseudoinverse) to close rings and symmetrize; the MST
        # pass then guarantees connectivity for whatever is still separate. The
        # physical gate (<= `_weld_gate` x sum of covalent radii) ensures
        # genuinely-separated fragments still get only a single minimal weld, so
        # we never re-introduce the stretched/transannular bonds MST replaced.
        order = np.argsort(best_dist, kind='stable')
        links = []
        have = set()
        weld_set = _DisjointSet(next_label, n_components)

        if self._weld_augment:
            tol = self._weld_tol
            gate = self._weld_gate
            # Connection scale: the largest edge in a minimum spanning tree over
            # the raw components -- the smallest possible "max contact distance"
            # that connects everything, i.e. exactly where the old flooding
            # welding stopped inflating its cutoff. Augmenting up to this scale
            # (and never beyond) restores flooding's redundant-but-bounded
            # contact set: it can never add a contact LONGER than the
            # unavoidable minimal weld, yet it recovers the extra near-min
            # contacts a bare tree drops. This is geometry-adaptive, so it
            # handles weakly-bonded / large-vdW fragments (e.g. noble-gas
            # clusters) whose real contacts sit far outside a covalent-radius
            # gate, without loosening that gate for tightly-bonded organics.
            raw_set = _DisjointSet(next_label, n_components)
            dmst_max = 0.0
            for k in order:
                if raw_set.n_active == 1:
                    break
                if raw_set.union(int(comp[ii[k]]), int(comp[jj[k]])):
                    if best_dist[k] > dmst_max:
                        dmst_max = best_dist[k]

            # Shortest contact per component-pair (order is sorted, so the first
            # time a pair is seen is its minimum).
            pair_min = {}
            for k in order:
                key = self._component_pair_key(comp[ii[k]], comp[jj[k]])
                if key not in pair_min:
                    pair_min[key] = best_dist[k]

            for k in order:
                a, b = int(ii[k]), int(jj[k])
                dmin = pair_min[self._component_pair_key(comp[a], comp[b])]
                rcov_sum = (covalent_radii[self.atoms.numbers[a]]
                            + covalent_radii[self.atoms.numbers[b]])
                # Near this interface's own minimum (never a transannular
                # contact within a tight interface) AND within either the
                # covalent gate or the geometry-adaptive connection scale.
                if best_dist[k] <= dmin * (1. + tol) and (
                    best_dist[k] <= gate * rcov_sum
                    or best_dist[k] <= dmst_max * (1. + tol)
                ):
                    self._record_link(links, have, a, b, best_ts[k])
                    weld_set.union(int(comp[a]), int(comp[b]))

        # MST pass: add the shortest contact that still merges two separate
        # components until the whole graph is connected.
        for k in order:
            if weld_set.n_active == 1:
                break
            a, b = int(ii[k]), int(jj[k])
            if weld_set.union(int(comp[a]), int(comp[b])):
                self._record_link(links, have, a, b, best_ts[k])
        return links

    def _wrap_fragment_positions(self, group, cumshifts):
        """Shift atom positions so fragment atoms are contiguous across PBC.

        BFS from first atom in group, using bond ncvecs to bring each
        bonded neighbor into the same periodic image. Accumulates shifts
        along bond chains so molecules spanning multiple cell boundaries
        are fully contracted. Records cumulative shifts in cumshifts dict
        for subsequent ncvec correction.
        """
        group_set = set(group)
        cell = np.asarray(self.atoms.cell)

        adj = {i: [] for i in group}
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i in group_set and j in group_set:
                ncvec = bond.kwargs['ncvecs'][0]
                adj[i].append((j, ncvec))
                adj[j].append((i, -ncvec))

        anchor = group[0]
        cumshifts[anchor] = np.zeros(3, dtype=int)
        queue = [anchor]
        while queue:
            i = queue.pop(0)
            for j, ncvec in adj[i]:
                if j in cumshifts:
                    continue
                cumshifts[j] = ncvec + cumshifts[i]
                self.atoms.positions[j] += cumshifts[j] @ cell
                queue.append(j)

    def _remap_ncvecs_after_fragment_shifts(self, cumshifts):
        """Keep periodic internal coordinates consistent after unwrapping.

        Fragment unwrapping changes absolute Cartesian positions by integer
        cell vectors. Every stored ncvec between consecutive atoms must be
        shifted by the corresponding endpoint image changes so the represented
        physical bond/angle/dihedral is unchanged.
        """
        zero = np.zeros(3, dtype=np.int32)
        seen = set()
        changed_names = set()

        for name in ('bonds', 'angles', 'dihedrals'):
            for collection in (self.internals[name], self.forbidden[name]):
                for coord in collection:
                    if id(coord) in seen:
                        continue
                    seen.add(id(coord))

                    old = coord.kwargs['ncvecs']
                    new = np.asarray(old, dtype=np.int32).copy()
                    changed = False
                    for k in range(len(new)):
                        i = int(coord.indices[k])
                        j = int(coord.indices[k + 1])
                        shift_i = cumshifts.get(i, zero)
                        shift_j = cumshifts.get(j, zero)
                        if np.any(shift_i != 0) or np.any(shift_j != 0):
                            new[k] = old[k] - shift_j + shift_i
                            changed = True
                    if changed and not np.array_equal(new, old):
                        coord.kwargs['ncvecs'] = new
                        changed_names.add(name)

        for name in changed_names:
            self._internals_set[name] = {
                self._internal_key(coord) for coord in self.internals[name]
            }
        if changed_names:
            self._invalidate_structure()
            self.cons._invalidate_structure()

    def _initial_bond_adjacency(self, max_bonds):
        nbonds = np.zeros(self.natoms, dtype=np.int32)
        c10y = -np.ones((self.natoms, max_bonds), dtype=np.int32)
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i >= self.natoms or j >= self.natoms:
                continue
            c10y[i, nbonds[i]] = j
            nbonds[i] += 1
            c10y[j, nbonds[j]] = i
            nbonds[j] += 1
        return nbonds, c10y

    def _connected_fragment_labels(self, nbonds, c10y, labels):
        labels[:] = -1
        nlabels = 0
        for i in range(self.natoms):
            if labels[i] == -1:
                labels[i] = nlabels
                self.flood_fill(i, nbonds, c10y, labels, nlabels)
                nlabels += 1
        return nlabels

    def _add_bond_candidates(self, candidates, nbonds, c10y, max_bonds):
        for i, j, ts in candidates:
            if not self._ignore_duplicate(self.add_bond, (i, j), ts):
                continue
            if nbonds[i] < max_bonds and nbonds[j] < max_bonds:
                c10y[i, nbonds[i]] = j
                nbonds[i] += 1
                c10y[j, nbonds[j]] = i
                nbonds[j] += 1

    def _add_disconnected_fragment_coords(self, labels, nlabels):
        assert nlabels > 1
        groups = [[] for _ in range(nlabels)]
        singletons = []
        for i, label in enumerate(labels):
            if label == -1:
                self._add_fragment_coords([i], with_rotation=False)
                singletons.append(i)
            else:
                groups[label].append(i)

        cumshifts = {}
        self.fragment_atom_groups = []
        for group in groups:
            if not group:
                continue
            self._wrap_fragment_positions(group, cumshifts)
            self.fragment_atom_groups.append(np.array(group, dtype=np.int32))
            self._add_fragment_coords(group)

        for i in singletons:
            self.fragment_atom_groups.append(np.array([i], dtype=np.int32))
        return cumshifts

    def _add_single_fragment_pbc_coords(self):
        group = list(range(self.natoms))
        cumshifts = {}
        self._wrap_fragment_positions(group, cumshifts)
        self.fragment_atom_groups = [np.array(group, dtype=np.int32)]
        self._add_fragment_coords(group)
        return cumshifts

    def _add_improper_dihedral_for_linear_angle(self, j, jbonds, b1, b2):
        for b3 in jbonds:
            if b3 in (b1, b2):
                continue

            ordered = (
                (int(b1.indices[1]), b1.kwargs['ncvecs'][0]),
                (int(b3.indices[1]), b3.kwargs['ncvecs'][0]),
                (int(b2.indices[1]), b2.kwargs['ncvecs'][0]),
            )
            if not self._improper_dihedral_well_defined(j, ordered):
                continue

            indices = (b1.indices[1], j, b3.indices[1], b2.indices[1])
            ncvecs = (
                -b1.kwargs['ncvecs'][0],
                b3.kwargs['ncvecs'][0],
                b2.kwargs['ncvecs'][0] - b3.kwargs['ncvecs'][0]
            )
            self._ignore_duplicate(self.add_dihedral, indices, ncvecs)
            return True
        return False

    def _add_linear_angle_replacements(self, j, jbonds, linear):
        if len(jbonds) == 2:
            self._add_linear_bend_dummy(j, jbonds, *jbonds)
            return

        dummy_added = False
        for b1, b2 in linear:
            if self._add_improper_dihedral_for_linear_angle(j, jbonds, b1, b2):
                continue

            # No existing third bond gives two well-defined dihedral planes.
            # Fall back to the dummy linear bend machinery instead of adding an
            # undefined improper with NaN derivatives.
            if not dummy_added:
                self._add_linear_bend_dummy(j, jbonds, b1, b2)
                dummy_added = True

    def find_all_bonds(
        self,
        nbond_cart_thr: int = 6,
        max_bonds: int = 20,
        scale: float = 1.25,
    ) -> None:
        rcov = covalent_radii[self.atoms.numbers]
        nbonds, c10y = self._initial_bond_adjacency(max_bonds)
        labels = -np.ones(self.natoms, dtype=np.int32)

        first_run = True
        while True:
            nlabels = self._connected_fragment_labels(nbonds, c10y, labels)
            if nlabels == 1:
                break

            # Remove labels from atoms with no bonding partners.
            # This must happen BEFORE the allow_fragments break, otherwise
            # single atoms will retain fragment labels and cause rotation ICs
            # to be incorrectly added to single-atom groups.
            labels[nbonds == 0] = -1

            if self.allow_fragments and not first_run:
                break

            if first_run:
                candidates = self._find_bonds_vectorized(labels, scale, rcov)
                first_run = False
                weld_pass = False
            else:
                # Fragments remain and allow_fragments is False: weld them with
                # a minimum-spanning-tree of the shortest inter-fragment
                # contacts, rather than inflating the covalent-radius multiplier
                # (which manufactures stretched/transannular bonds). This single
                # pass connects every component.
                candidates = self._mst_welding_bonds(labels)
                weld_pass = True

            self._add_bond_candidates(candidates, nbonds, c10y, max_bonds)
            if weld_pass:
                # MST spans all components in one pass; connectivity is
                # guaranteed by construction, so we are done welding.
                break

        if self.allow_fragments and nlabels != 1:
            cumshifts = self._add_disconnected_fragment_coords(labels, nlabels)
        elif (self.allow_fragments and nlabels == 1
              and np.any(self.atoms.pbc)
              and len(self.internals['bonds']) > 0):
            cumshifts = self._add_single_fragment_pbc_coords()
        else:
            cumshifts = {}

        if cumshifts:
            self._remap_ncvecs_after_fragment_shifts(cumshifts)

    def find_all_angles(
        self,
    ) -> None:
        bonds = [[] for _ in range(self.natoms)]
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i >= self.natoms or j >= self.natoms:
                continue
            bonds[i].append(bond)
            bonds[j].append(bond.reverse())

        for j, jbonds in enumerate(bonds):
            linear = []
            for b1, b2 in combinations(jbonds, 2):
                new = Angle(
                    (b1.indices[1], j, b2.indices[1]),
                    (-b1.kwargs['ncvecs'][0], b2.kwargs['ncvecs'][0]),
                )
                # Angles inside the linear window are kept as ordinary bends;
                # near-linear ones get dummy/dihedral treatment.
                if self.atol < new.calc(self.atoms) < np.pi - self.atol:
                    self._ignore_duplicate(self.add_angle, new)
                else:
                    self.forbid_angle(new)
                    linear.append((b1, b2))
            if linear:
                self._add_linear_angle_replacements(j, jbonds, linear)

    def _add_linear_bend_dummy(self, j, jbonds, b1, b2):
        """Add the dummy-coordinate representation for a linear bend."""
        # Sort the defining bonds from shortest to longest to keep the dummy
        # orientation deterministic when both sides are equivalent.
        b1, b2 = sorted((b1, b2), key=lambda x: x.calc(self.atoms))
        if self.dinds[j] < 0:
            self.dinds[j] = self.natoms + self.ndummies
            dx1 = -b1.calc_vec(self.atoms)
            dx1 /= np.linalg.norm(dx1)
            dx2 = b2.calc_vec(self.atoms)
            dx2 /= np.linalg.norm(dx2)
            dpos = np.cross(dx1, dx2)
            dpos_norm = np.linalg.norm(dpos)
            if dpos_norm < 1e-4:
                # Pick the cartesian basis vector that is maximally orthogonal
                # with the shorter of the two displacement vectors.
                dim = np.argmin(np.abs(dx1))
                dpos[:] = 0.
                dpos[dim] = 1.
                dpos -= dx1 * (dpos @ dx1)
                dpos /= np.linalg.norm(dpos)
            else:
                dpos /= dpos_norm
            self.dummies += Atom('X', self.atoms.positions[j] + dpos)
            self._invalidate_structure()

        dbond = Bond((j, self.dinds[j]))
        self._ignore_duplicate(self.cons.fix_bond, dbond, replace_ok=False)
        self._ignore_duplicate(self.add_bond, dbond)

        # Fix one dummy angle. For linear O1-C-O2, the two dummy angles are
        # supplementary, so constraining both over-constrains real atoms.
        dangle1 = b1 + dbond
        self._ignore_duplicate(self.cons.fix_angle, dangle1, replace_ok=False)

        if b2.indices[1] == j:
            b2 = b2.reverse()
        dbond2 = Bond(
            (self.dinds[j], b2.indices[1]), b2.kwargs['ncvecs']
        )
        dangle3 = dbond + dbond2
        self._ignore_duplicate(self.add_dihedral, dangle1 + dangle3)
        self.add_dummy_to_internals(j)
        self.cons.add_dummy_to_internals(j)

        for bond in jbonds:
            new = bond + dbond
            assert new.indices[1] == j
            angle = new.calc(self.all_atoms)
            if self.atol < angle < np.pi - self.atol:
                self._ignore_duplicate(self.add_angle, new)
            else:
                self.forbid_angle(new)

    def _angles_by_bond_edge(self):
        """Group angles by bond edge so only edge-sharing pairs are combined.

        Proper dihedrals are formed from two angles that share a central bond;
        ``Angle.__add__`` only succeeds for those pairs. Grouping keeps the loop
        focused on valid candidates instead of trying every angle pair.
        """
        edge_to_angles = {}
        for angle in self.internals['angles']:
            i, j, k = angle.indices
            for edge_key in ((min(i, j), max(i, j)), (min(j, k), max(j, k))):
                edge_to_angles.setdefault(edge_key, []).append(angle)
        return edge_to_angles

    @staticmethod
    def _dihedral_is_self_loop(dihedral):
        """True when the first and last atom are the same exact periodic image."""
        return (
            dihedral.indices[0] == dihedral.indices[3]
            and np.all(
                np.sum(dihedral.kwargs['ncvecs'], axis=0)
                == np.array((0, 0, 0))
            )
        )

    def _add_proper_dihedrals(self):
        """Add proper dihedrals and return the atoms they pass through."""
        edge_to_angles = self._angles_by_bond_edge()
        seen_pairs = set()
        centers = set()
        active_keys = {
            self._internal_key(dihedral)
            for dihedral, active in zip(
                self.internals['dihedrals'], self._active['dihedrals']
            )
            if active
        }
        for angles_on_edge in edge_to_angles.values():
            for a1, a2 in combinations(angles_on_edge, 2):
                pair_key = (id(a1), id(a2))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                try:
                    new = a1 + a2
                except NoValidInternalError:
                    continue
                if self._dihedral_is_self_loop(new):
                    continue
                key = self._internal_key(new)
                if self._ignore_duplicate(self.add_dihedral, new):
                    active_keys.add(key)
                if key in active_keys:
                    centers.add(int(new.indices[1]))
                    centers.add(int(new.indices[2]))
        return centers

    def _bond_neighbor_list(self):
        """Build per-atom bond neighbors with ncvecs pointing away from center."""
        neighbors = [[] for _ in range(self.natoms)]
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i < self.natoms:
                neighbors[i].append((int(j), bond.kwargs['ncvecs'][0]))
            if j < self.natoms:
                neighbors[j].append((int(i), -bond.kwargs['ncvecs'][0]))
        return neighbors

    @staticmethod
    def _improper_dihedral_args(center, ordered_neighbors):
        """Return improper dihedral indices and ncvecs for one neighbor order."""
        n0, n1, n2 = ordered_neighbors
        i0, ncvec0 = n0
        i1, ncvec1 = n1
        i2, ncvec2 = n2
        # Improper dihedral indices: (i0, center, i1, i2).
        # The ncvecs connect consecutive atoms in the dihedral.
        return (
            (i0, center, i1, i2),
            (-ncvec0, ncvec1, ncvec2 - ncvec1),
        )

    @staticmethod
    def _neighbor_key(neighbor):
        atom, ncvec = neighbor
        return int(atom), tuple(int(x) for x in ncvec)

    def _existing_improper_triples(self, center, center_neighbors):
        """Return neighbor triples already used by impropers at ``center``."""
        neighbor_keys = {
            self._neighbor_key(neighbor) for neighbor in center_neighbors
        }
        triples = set()
        for dihedral, active in zip(
            self.internals['dihedrals'], self._active['dihedrals']
        ):
            if not active:
                continue

            offsets = np.vstack((
                np.zeros(3, dtype=np.int32),
                np.cumsum(dihedral.kwargs['ncvecs'], axis=0),
            ))
            for center_pos in (1, 2):
                if dihedral.indices[center_pos] != center:
                    continue
                keys = []
                for pos, atom in enumerate(dihedral.indices):
                    if pos == center_pos:
                        continue
                    keys.append((
                        int(atom),
                        tuple(int(x) for x in (
                            offsets[pos] - offsets[center_pos]
                        )),
                    ))
                triple = frozenset(keys)
                if len(triple) == 3 and triple <= neighbor_keys:
                    triples.add(triple)
        return triples

    def _ordered_improper_neighbors(self, center, triple):
        """Choose a numerically valid ordering for one topology-defined triple."""
        for ordered in permutations(triple):
            if not self._improper_dihedral_well_defined(center, ordered):
                continue
            return ordered
        return None

    def _improper_fan(self, center, center_neighbors, existing):
        """Build a minimal ``degree - 2`` fan of local neighbor triples."""
        pair_scores = {}
        for triple in existing:
            for pair in combinations(triple, 2):
                pair = frozenset(pair)
                pair_scores[pair] = pair_scores.get(pair, 0) + 1

        anchor_pairs = list(combinations(range(len(center_neighbors)), 2))
        anchor_pairs.sort(
            key=lambda anchors: -pair_scores.get(
                frozenset(
                    self._neighbor_key(center_neighbors[i]) for i in anchors
                ),
                0,
            )
        )
        for anchors in anchor_pairs:
            fan = []
            for other in range(len(center_neighbors)):
                if other in anchors:
                    continue
                triple = tuple(
                    center_neighbors[i] for i in (*anchors, other)
                )
                ordered = self._ordered_improper_neighbors(center, triple)
                if ordered is None:
                    break
                key = frozenset(self._neighbor_key(n) for n in triple)
                fan.append((key, ordered))
            else:
                return fan
        return None

    def _add_improper_dihedrals(self, proper_centers) -> None:
        """Add fallback improper dihedrals for under-constrained centers.

        Improper dihedrals are needed because:
        1. At planar geometries, bond/angle derivatives vanish for
           out-of-plane motion.
        2. Even if the initial geometry is non-planar, it may planarize during
           optimization.
        3. Improper dihedrals capture the out-of-plane umbrella modes.

        A degree-k planar star has k - 2 internal out-of-plane modes that bond
        and angle derivatives cannot span. Add exactly that many impropers as
        a fan. The count depends only on the bond graph, so a non-planar star
        retains the coordinates it would need if it later becomes planar.

        This adds some redundancy away from planarity, but keeps the Jacobian
        well-conditioned for planar systems such as nitrate. To avoid adding
        unnecessary coordinates, the fan is omitted when generated proper
        dihedrals pass through the center, and existing local impropers are
        reused when completing it. With complete angle enumeration, one
        proper-capable incident edge generates at least k - 2 distinct proper
        torsions, so the boolean proper-center test is sufficient here.
        """
        neighbors = self._bond_neighbor_list()
        for center in range(self.natoms):
            center_neighbors = neighbors[center]
            if len(center_neighbors) < 3 or center in proper_centers:
                continue

            existing = self._existing_improper_triples(
                center, center_neighbors
            )
            fan = self._improper_fan(center, center_neighbors, existing)
            if fan is None:
                continue
            for key, ordered in fan:
                if key in existing:
                    continue
                indices, ncvecs = self._improper_dihedral_args(
                    center, ordered
                )
                if self._ignore_duplicate(
                    self.add_dihedral, indices, ncvecs
                ):
                    existing.add(key)

    def find_all_dihedrals(self) -> None:
        """Find proper dihedrals first, then add necessary improper fallbacks."""
        proper_centers = self._add_proper_dihedrals()
        self._add_improper_dihedrals(proper_centers)

    def _improper_dihedral_well_defined(
        self,
        center: int,
        ordered_neighbors,
        rel_tol: float = 1e-8,
    ) -> bool:
        """Return True if an improper-dihedral ordering has two valid planes."""
        pos = self.all_positions
        cell = self.atoms.cell.array
        vecs = []
        for neighbor, ncvec in ordered_neighbors:
            vecs.append(pos[neighbor] - pos[center] + ncvec @ cell)

        def noncollinear(a, b):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom <= 1e-12:
                return False
            return np.linalg.norm(np.cross(a, b)) > rel_tol * denom

        return (
            noncollinear(vecs[0], vecs[1])
            and noncollinear(vecs[1], vecs[2])
        )

    def validate_basis(self) -> None:
        jac = self.jacobian()
        S = svdvals(jac)
        ndeloc = np.sum(S > 1e-8)

        # If TRICs (translations/rotations) are present, they span the full
        # 3N DOF. Otherwise, 6 DOF are removed for global translation/rotation.
        has_trics = (len(self.internals['translations']) > 0 or
                     len(self.internals['rotations']) > 0)
        if has_trics:
            ndof = 3 * (self.natoms + self.ndummies)
        else:
            ntot = self.natoms + self.ndummies
            has_periodic_bonds = any(
                np.any(bond.kwargs['ncvecs'] != 0)
                for bond in self.internals['bonds']
            )
            if has_periodic_bonds:
                ndof = 3 * ntot
            elif ntot <= 1:
                ndof = 0
            elif ntot == 2:
                ndof = 1
            else:
                ndof = 3 * ntot - 6

        if ndeloc != ndof:
            warnings.warn(
                f'Internal coordinates ({ndeloc} DOF) do not span the '
                f'full coordinate space ({ndof} DOF). This is expected '
                f'when using user-specified internals or constraint-only '
                f'coordinates, but may indicate missing coordinates if '
                f'auto-detection was used.'
            )

    def _rotation_fragment_is_linear(self, indices, rel_tol=1e-3):
        """True if a fragment's atoms are (near-)collinear.

        A collinear rigid fragment has only two well-defined rotational DOF;
        rotation about the molecular axis is a null mode, so its quaternion
        F-matrix always has a (near-)degenerate top eigenpair. That degeneracy
        is expected and already handled continuously by ``_stabilize_quaternion``
        (value/branch) and by the redundant-coordinate SVD (rank), so it must
        NOT be treated as a "bad internal" -- doing so triggers a PES rebuild
        and geodesic abort on every step, freezing the optimizer. Compact
        fragments keep the degeneracy guard, where a small gap signals a genuine
        orientational ambiguity. Single-atom fragments never get a rotation
        (see ``add_rotation`` callers, ``len(group) >= 2``) so are never checked.
        """
        idx = np.asarray(indices, dtype=int)
        if len(idx) < 3:
            return True  # a 2-atom fragment is always collinear
        dx = self.all_positions[idx]
        dx = dx - dx.mean(0)
        w = np.linalg.eigvalsh(dx.T @ dx)  # ascending: w[0] <= w[1] <= w[2]
        # Collinear iff the second-largest spatial extent is negligible relative
        # to the largest (rank-1 gyration tensor).
        return w[2] <= 0 or w[1] <= rel_tol * w[2]

    @staticmethod
    def _rotation_gap_is_bad(ws, threshold=0.02):
        """True when the top quaternion F-matrix eigenpair is near-degenerate."""
        gap = ws[-1] - ws[-2]
        spread = ws[-1] - ws[0]
        return spread > 0 and gap / spread < threshold

    def _bad_angles(self):
        """Return angle coordinates near 0 or pi using the padded JAX batch."""
        self._build_batched_arrays()
        angles = self._batched_family_arrays['angles']
        if angles.n_actual == 0:
            return []
        tvecs = self._get_cached_tvecs(self.atoms.cell.array)
        angle_pos = self.all_positions[angles.indices_padded]
        angle_vals_padded = np.asarray(
            _angle_value_batched(angle_pos, tvecs['angles_padded'])
        )
        angle_vals = angle_vals_padded[:angles.n_actual]
        bad_mask = ~((self.atol < angle_vals)
                     & (angle_vals < np.pi - self.atol))
        return [self.internals['angles'][idx] for idx in np.where(bad_mask)[0]]

    def _bad_rotation_from_cached_eigh(self, rotations, cached_eigh):
        """Check rotation F-matrix gaps using eigenvalues cached by Jacobian eval."""
        ws_all, _ = cached_eigh
        if ws_all is None:
            return None
        _, _, _, _, slots, valid = self._rotation_padded_inputs(
            self.all_positions
        )
        if not valid:
            return None
        for fi, slot in enumerate(slots):
            if not self._rotation_gap_is_bad(ws_all[fi]):
                continue
            candidate = rotations[slot[0]]
            if self._rotation_fragment_is_linear(candidate.indices):
                continue
            return candidate
        return None

    def _bad_rotation_from_direct_eigh(self, rotations):
        """Compute rotation F-matrix eigenvalues directly on the first check."""
        positions = self.all_positions
        seen_fragments = set()
        for rot in rotations:
            frag_key = tuple(rot.indices)
            if frag_key in seen_fragments:
                continue
            seen_fragments.add(frag_key)
            idx = np.array(rot.indices)
            pos = positions[idx]
            dx = pos - pos.mean(0)
            ws = np.linalg.eigvalsh(_build_F_matrix_np(dx, rot.kwargs['refpos']))
            if not self._rotation_gap_is_bad(ws):
                continue
            if self._rotation_fragment_is_linear(rot.indices):
                continue
            return rot
        return None

    def _bad_rotation(self):
        """Return the first near-degenerate non-linear rotation coordinate."""
        rotations = self.internals['rotations']
        if not rotations:
            return None
        cached_eigh = self._cache.get('stabilized_q_eigh')
        if cached_eigh is not None:
            return self._bad_rotation_from_cached_eigh(rotations, cached_eigh)
        return self._bad_rotation_from_direct_eigh(rotations)

    def check_for_bad_internals(self) -> Optional[Dict[str, List[Coordinate]]]:
        """Check for angles near 0/pi or near-degenerate rotation F-matrices.

        Uses vectorized computation for efficiency.
        """
        bad = {'bonds': [], 'angles': []}

        angles = self.internals['angles']
        if not angles:
            return None

        bad['angles'].extend(self._bad_angles())
        bad_rotation = self._bad_rotation()
        if bad_rotation is not None:
            bad['angles'].append(bad_rotation)

        for ints in bad.values():
            if ints:
                return bad
        return None

    def _h0_bond(
        self,
        bond: Bond,
        Ab: float = 0.3601,
        Bb: float = 1.944,
    ) -> float:
        idx = np.asarray(bond.indices, dtype=np.int32)
        rcov = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rij = bond.calc(self.all_atoms)
        h0 = Ab * np.exp(-Bb * (rij - rcov) / units.Bohr)
        return h0 * units.Hartree / units.Bohr**2

    def _h0_angle(
        self,
        angle: Angle,
        Aa: float = 0.089,
        Ba: float = 0.11,
        Ca: float = 0.44,
        Da: float = -0.42,
    ) -> float:
        bab, bbc = angle.split()
        idxab = np.asarray(bab.indices, dtype=np.int32)
        idxbc = np.asarray(bbc.indices, dtype=np.int32)
        rcovab = covalent_radii[self.all_atoms.numbers[idxab]].sum()
        rcovbc = covalent_radii[self.all_atoms.numbers[idxbc]].sum()
        rab = bab.calc(self.all_atoms)
        rbc = bbc.calc(self.all_atoms)
        h0 = (
            Aa + Ba * np.exp(-Ca * (rab + rbc - rcovab - rcovbc) / units.Bohr)
            / (rcovab * rcovbc / units.Bohr**2)**Da
        )
        return h0 * units.Hartree

    def _h0_dihedral(
        self,
        dihedral: Dihedral,
        nbonds: np.ndarray,
        At: float = 0.0015,
        Bt: float = 14.0,
        Ct: float = 2.85,
        Dt: float = 0.57,
        Et: float = 4.00,
    ) -> float:
        _, bbc = dihedral.split()[0].split()
        idx = np.asarray(bbc.indices, dtype=np.int32)
        rcovbc = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rbc = bbc.calc(self.all_atoms)
        L = nbonds[idx].sum() - 2
        h0 = (
            At + Bt * L**Dt * np.exp(-Ct * (rbc - rcovbc) / units.Bohr)
            / (rbc * rcovbc / units.Bohr**2)**Et
        )
        return h0 * units.Hartree

    def guess_hessian(self, h0cart=70.) -> np.ndarray:
        # Bond count per atom (molecular topology) feeds the dihedral
        # curvature heuristic; count over all bonds, independent of which
        # coordinates are currently active.
        nbonds = np.zeros(len(self.all_atoms), dtype=np.int32)
        for bond in self.internals['bonds']:
            i, j = bond.indices
            nbonds[i] += 1
            nbonds[j] += 1

        h0_tr = 0.05 * units.Hartree
        periodic = np.any(self.atoms.pbc)
        if periodic and self.allow_fragments:
            h0_trans = 5.0
            h0_rot = 15.0
        else:
            h0_trans = h0_tr
            h0_rot = h0_tr
        dummy_set = set(range(self.natoms, self.natoms + self.ndummies))

        def _h0_value(name, coord):
            if name == 'translations':
                return h0_trans if self.allow_fragments else h0cart
            if name == 'bonds':
                return self._h0_bond(coord)
            if name == 'angles':
                return self._h0_angle(coord)
            if name == 'dihedrals':
                if any(k in dummy_set for k in coord.indices):
                    return 0.5 * units.Hartree
                return self._h0_dihedral(coord, nbonds)
            if name == 'rotations':
                return h0_rot if self.allow_fragments else h0cart
            # 'other': generic coordinate, use the default Cartesian curvature.
            return h0cart

        # Walk the coordinate families in canonical order, active coords only,
        # so the diagonal stays aligned with jacobian()/hessian() row ordering.
        h0 = np.zeros(self.nint, dtype=np.float64)
        idx = 0
        for name in self._names:
            for coord, active in zip(self.internals[name],
                                     self._active[name]):
                if not active:
                    continue
                h0[idx] = _h0_value(name, coord)
                idx += 1
        return np.diag(np.abs(h0))
