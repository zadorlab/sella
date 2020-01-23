from sella.utilities.math cimport normalize, vec_sum, symmetrize, skew, mgs

def wrap_normalize(x):
    return normalize(x)

def wrap_vec_sum(x, y, z, scale):
    return vec_sum(x, y, z, scale)

def wrap_symmetrize(X_np):
    cdef double[:, :] X = memoryview(X_np)
    cdef size_t n = len(X)
    cdef size_t lda = X.strides[0] >> 3
    return symmetrize(&X[0, 0], n, lda)

def wrap_skew(x, Y, scale):
    return skew(x, Y, scale)

def wrap_mgs(X, Y, eps1=1e-15, eps2=1e-6, maxiter=1000):
    return mgs(X, Y=Y, eps1=eps1, eps2=eps2, maxiter=maxiter)


wrappers = {'normalize': wrap_normalize,
            'vec_sum': wrap_vec_sum,
            'symmetrize': wrap_symmetrize,
            'skew': wrap_skew,
            'mgs': wrap_mgs}

