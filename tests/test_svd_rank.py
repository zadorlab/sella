import numpy as np

from sella.peswrapper import _svd_rank


def test_svd_rank_well_scaled():
    # O(1) spectrum: relative and absolute agree.
    s = np.array([2.0, 1.0, 1e-9])
    assert _svd_rank(s) == 2


def test_svd_rank_relative_not_absolute():
    # Poorly-scaled spectrum where a relative cutoff (1e-6 * s[0] = 1e-3)
    # and an absolute 1e-6 cutoff disagree. The relative rank is what the
    # convention across the other SVD sites uses.
    s = np.array([1e3, 1e-2, 1e-5])
    assert _svd_rank(s) == 2
    # An absolute 1e-6 threshold would have wrongly kept the 1e-5 mode.
    assert int(np.sum(s > 1e-6)) == 3


def test_svd_rank_empty():
    assert _svd_rank(np.array([])) == 0
