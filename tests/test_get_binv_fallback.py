import warnings

import numpy as np
from ase.build import molecule
from ase.calculators.lj import LennardJones

from sella.internal import Internals
from sella.peswrapper import InternalPES


def test_get_binv_rank_deficient_fallback_matches_primary():
    """The _get_Binv cache-miss fallback must reproduce the SAME truncated
    pseudoinverse as the primary QR path.

    In the rank-deficient branch _get_jacobian_qr caches a pseudoinverse
    rank-truncated at 1e-6*sigma_max. If the 2-entry LRU evicts it, _get_Binv
    rebuilds it as pinv(R) @ Q.T from the cached truncated (Q, R) -- matching
    the primary path -- rather than a plain np.linalg.pinv(B), whose default
    (machine-eps) rcond would keep the near-null directions the rank cut
    discarded and yield an inconsistent, potentially blown-up Binv.
    """
    warnings.simplefilter("ignore")
    atoms = molecule("CH3CH2OH")  # redundant internals -> rank-deficient B
    atoms.calc = LennardJones()
    pes = InternalPES(atoms, Internals(atoms))

    B = pes.int.jacobian()
    Q, R = pes._get_jacobian_qr()
    assert R.shape[0] != R.shape[1], "expected the rank-deficient branch to fire"

    Binv_primary = pes._get_Binv()  # cached by _get_jacobian_qr

    # Force the cache-miss and exercise the recompute fallback.
    pes._pinv_cache = pes._pinv_cache.__class__()
    Binv_fallback = pes._get_Binv()

    np.testing.assert_allclose(Binv_fallback, Binv_primary, atol=1e-10)
    # ...and it is a genuine pseudoinverse of B.
    np.testing.assert_allclose(B @ Binv_fallback @ B, B, atol=1e-8)
