from itertools import izip
from types import MethodType

import numpy as np
from scipy.misc import logsumexp

import inference


class ChainCRF:
    def __init__(self, X, Y, L, phi_B=None):
        self.X, self.Y, self.L, self.phi_B = X, Y, L, phi_B
        self.concat = self.phi_B is None
        self._make_dims()
        self.phis = [self.featurize(x, y) for x, y in izip(self.X, self.Y)]
        for f in filter(lambda x: 'Z' in x, dir(inference)):
            setattr(self, f, MethodType(getattr(inference, f), self))
        self.Z = self.Z_concat if self.concat else self.Z_scalar

    def _make_dims(self):
        self.n_labels, self.n_feats = len(self.L), len(self.X[0][0])
        self.n_U = self.n_labels * self.n_feats
        self.U_shape = (self.n_labels, self.n_feats)
        self.B_shape = (self.n_labels, self.n_labels, 2*self.n_feats if self.concat else 1)
        self.n_W = self.n_U + self.n_labels**2 * (2*self.n_feats if self.concat else 1)
        self.dims = self.n_U, self.U_shape, self.B_shape

    def featurize(self, x, y):
        """
        :Input:
        x:
         A 2D python list, where axis 1 runs over each variable in the crf chain, and 
         axis 2 runs over each variable's features
        y:
         A 1D python list which runs over the crf variable labels
        n_labels:
         number of labels in the variable label space
        n_features:
         number of unary features per variable
        concat:
         binary features are concatenated unary features [default]
        phi_B:
         binary features are given by potentials for label combinations; this should be 
         of shape `n_labels * n_labels` (e.g. potential=potts(n_labels, a, b))

        :Output:
        U:
         A numpy array of shape (l, n_labels, n_features) where l is the chain length. 
         U[i] for variable i is nonzero only in entry y--for y is the label of i--where
         it equals the feature vector of i. 
         i=(v,w) is nonzero only in entry (label_id(v), label_id(w)), where the value is
         the concatenation of binary features for i and the unary features of v the 
         unary features is so we can store all weights in one array; hence, we avoid 
         duplication by only adding the unary terms of the first node in each edge.
        B:
         A numpy array of shape (l-1, n_labels, n_labels). B[i,y,z] for edge i=(v,w)
         is nonzero only in entry (y,z)--for (y,z) the label pair of i--where it equals
         the concatenated feature vector (x_v, x_w) if `concat` else potential[y,z]  
        """
        U = np.zeros((len(x), self.n_labels, self.n_feats))
        for i, (f, y_i) in enumerate(izip(x, y)):
            U[i, y_i] = f
        B = np.zeros((len(x) - 1, self.n_labels, self.n_labels,
            2 * self.n_feats if self.concat else 1))
        xy = zip(x, y)
        for i, ((f1, y1), (f2, y2)) in enumerate(izip(xy, xy[1:])):
            if self.concat:
                B[i, y1, y2] = np.concatenate((f1, f2))
            else:
                B[i, y1, y2, 0]= self.phi_B[y1, y2]
        return U, B

    def Z_s(self, x, W_u, W_b):
        z = -np.dot(W_u, x[0])
        for i in xrange(1, len(x)):
            z = logsumexp(z - np.dot(W_u, x[i, :, None]) - W_b * self.phi_B)
        return z

    def Z_c(self, x, W_u, W_b):
        z = -np.dot(W_u, x[0])
        for i in xrange(1, len(x)):
            z = logsumexp((z - np.dot(W_u, x[i, :, None]) - np.dot(W_b,
                np.concatenate((x[i - 1], x[i]))[:, None]))[self.L, :, self.L].T, 1)
        return logsumexp(z)