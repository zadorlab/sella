"""Shared numeric constants for Sella.

Single source of truth for tolerances that were previously duplicated across
modules.
"""

# rcond passed to np.linalg.lstsq throughout the optimizer (Jacobian
# pseudo-inverse solves, constraint least-squares, projected steps).
_LSTSQ_RCOND = 1e-10
