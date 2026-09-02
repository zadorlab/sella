# Changelog

## 2.6.0

### Packaging

* Sella is now pure Python. The Cython extension modules (`force_match`,
  `utilities.blas`, `utilities.math`) have been replaced with portable
  Python implementations, and releases now ship a single
  `py3-none-any` wheel instead of per-platform compiled wheels.
  Installing no longer requires a C compiler.

### Requirements

* Minimum Python is now 3.9 (was 3.8).
* Minimum ASE is now 3.26.0 (was 3.18.0). Sella's IRC logging passes a
  flat gradient array to `Optimizer.log`, which older ASE releases do
  not accept, so ASE < 3.26 is no longer supported.

### Added

* `configure_compute` and `set_cpu_threads` for controlling thread and
  compute settings.
* Optional GPU acceleration for large linear-algebra operations when
  PyTorch with CUDA is present. PyTorch is not required; Sella falls
  back to CPU when it is unavailable.
* `hessian_progress` option to log Hessian diagonalization progress.
* Accepted trajectory frame IDs are now recorded in the log.

### Fixed

* Constrained IRC force projection.
* Guess Hessian projector; added `exact_geodesic`.
* The JAX compilation cache is no longer created in the home directory
  when another path is configured, and an unwritable home no longer
  breaks import.

## 1.0.0

* Internal coordinate optimization implemented
  * Invoke with `Sella(..., internal=True)`
  * Automatically constructs dummy atoms when necessary
