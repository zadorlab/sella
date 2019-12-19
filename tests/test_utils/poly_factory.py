import numpy as np

from itertools import permutations


def poly_factory(dim, order, rng=None):
    """Generates a random multi-dimensional polynomial function."""
    if rng is None:
        rng = np.random.RandomState(1)

    coeffs = []
    for i in range(order + 1):
        tmp = rng.normal(size=(dim,) * i)
        coeff = np.zeros_like(tmp)
        for n, permute in enumerate(permutations(range(i))):
            coeff += np.transpose(tmp, permute)
        coeffs.append(coeff / ((n + 1) * np.math.factorial(i)))

    def poly(x):
        res = 0
        grad = np.zeros_like(x)
        hess = np.zeros((dim, dim))
        for i, coeff in enumerate(coeffs):
            lastlast = None
            last = None
            for j in range(i):
                lastlast = last
                last = coeff
                coeff = coeff @ x
            if last is not None:
                grad += i * last
            if lastlast is not None:
                hess += i * (i - 1) * lastlast
            res += coeff
        return res, grad, hess

    return poly
