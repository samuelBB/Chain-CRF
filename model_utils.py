import numpy as np


def potts(n_labels, a=1., b=0.):
    A = np.full((n_labels, n_labels), b)
    np.fill_diagonal(A, a)
    return A


def E((phi_u, phi_b), (W_u, W_b)):
    return (phi_u * W_u).sum() + (phi_b * W_b).sum()


def regularize(f, reg):
    def _regularize(W):
        return f(W) + reg * np.linalg.norm(W)**2
    return _regularize


def split_weights(W, n_U, U_shape, B_shape):
    return W[:n_U].reshape(U_shape), W[n_U:].reshape(B_shape)

