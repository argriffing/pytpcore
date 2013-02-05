"""
Faster-than-pure-Python truncated Taylor polynomial arithmetic.

The central data type is a numpy float64 ndarray of shape (D, N)
where D is the number of coefficients that we are tracking
and N is related to the number of ndarray-valued variables we are tracking
and the number of entries in each variable.
The reason for the ordering (D, N) rather than (N, D) is for
better compatibilty with AlgoPy.
If we care only about gradients, then D will be 1.
If we care about hessians then D will be 2.
If we are doing strange experimental things then we may want D bigger than 2.
"""

from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef long tp_exp(
        np.float64_t [:, :] x,
        np.float64_t [:, :] tmp,
        np.float64_t [:, :] out,
        ) nogil:
    """
    @param x: the data ndarray
    @param tmp: a meaningless buffer shaped like x
    @param out: output goes into this buffer also shaped like x
    @return: None
    """
    cdef long D = x.shape[0]
    cdef long N = x.shape[1]
    cdef long n, d, i

    for n in range(N):

        # take care of a base case
        out[0, n] = exp(x[0, n])

        # the tmp array will have the coefficients of the derivative of x
        for d in range(1, D):
            tmp[d-1, n] = x[d, n] * d

        # do a thing that is kind of like a convolution
        for d in range(1, D):
            out[d, n] = 0
            for i in range(d):
                out[d, n] += out[d-1-i, n] * tmp[i, n]
            out[d, n] /= d

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef long tp_reciprocal(
        np.float64_t [:, :] x,
        np.float64_t [:, :] out,
        ) nogil:
    """
    @param x: the data ndarray
    @param out: output goes into this buffer also shaped like x
    @return: None
    """
    cdef long D = x.shape[0]
    cdef long N = x.shape[1]
    cdef long n, d, i

    for n in range(N):

        # compute the base case
        out[0, n] = 1.0 / x[0, n]

        # compute the rest of the coefficients
        for d in range(1, D):
            out[d, n] = 0
            for i in range(d):
                out[d, n] -= out[i, n] * x[d-i, n]
            out[d, n] *= out[0, n]

    return 0

