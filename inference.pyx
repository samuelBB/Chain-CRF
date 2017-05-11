#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t DT
from scipy.misc import logsumexp


cpdef double Z_DP_concat(self,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] x,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] W_u,
    np.ndarray[DT, ndim=3, negative_indices=False, mode='c'] W_b
):
    cdef int i, n
    cdef int xl = len(x)
    cdef int Ll = len(self.L)
    cdef int W_u_shape = W_u.shape[0]
    cdef np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] z = np.vstack([-np.dot(W_u, x[0]), np.zeros(W_u_shape)])
    for i in xrange(1, xl):
        for n in xrange(Ll):
            z[1, n] = logsumexp(z[0] - np.dot(x[i], W_u[n]) - np.dot(
                W_b[:, n], np.concatenate((x[i - 1], x[i]))))
        z[0], z[1] = z[1], 0.
    return logsumexp(z[0])


cpdef double Z_concat(self,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] x,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] W_u,
    np.ndarray[DT, ndim=3, negative_indices=False, mode='c'] W_b
):
    cdef int i
    cdef int xl = len(x)
    cdef np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] z = -np.dot(W_u, x[0])
    for i in xrange(1, xl):
        z = logsumexp((z - np.dot(W_u, x[i, :, None]) - np.dot(W_b,
            np.concatenate((x[i - 1], x[i]))[:, None]))[self.L, :, self.L].T, 1)
    return logsumexp(z)


cpdef double Z_DP_scalar(self,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] x,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] W_u,
    np.ndarray[DT, ndim=3, negative_indices=False, mode='c'] W_b
):
    cdef int i, n
    cdef int xl = len(x)
    cdef int Ll = len(self.L)
    cdef np.ndarray[DT, ndim=3, negative_indices=False, mode='c'] pb = self.phi_B
    cdef int W_u_shape = W_u.shape[0]
    cdef np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] z = np.vstack([-np.dot(W_u, x[0]), np.zeros(W_u_shape)])
    for i in xrange(1, xl):
        for n in xrange(Ll):
            z[1, n] = logsumexp(z[0]-np.dot(x[i],W_u[n])-np.multiply(W_b[:,n],pb[:,n]))
        z[0], z[1] = z[1], 0.
    return logsumexp(z[0])


cpdef double Z_scalar(self,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] x,
    np.ndarray[DT, ndim=2, negative_indices=False, mode='c'] W_u,
    np.ndarray[DT, ndim=3, negative_indices=False, mode='c'] W_b
):
    cdef int i
    cdef int xl = len(x)
    cdef np.ndarray[DT, ndim=3, negative_indices=False, mode='c'] pb = self.phi_B
    cdef  z = -np.dot(W_u, x[0])
    for i in xrange(1, xl):
        z = logsumexp(z - np.dot(W_u, x[i, :, None]) - np.multiply(W_b, pb))
    return logsumexp(z)