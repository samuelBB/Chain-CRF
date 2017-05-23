from itertools import izip, product

import numpy as np
from sklearn.model_selection import train_test_split

from inference import Inference, SlowInference, Other


class ChainCRF(Inference, SlowInference, Other):
    def __init__(self, X, Y, L, phi_B=None, test_pct=None, val_pct=.4, seed=42):
        """ 
        :param X:
         training examples (n_features = number of unary features per variable)
        :param Y:
         training labels
        :param L:
         label space (n_labels = number of labels in L)
        :param phi_B:
         binary features are given by potentials for label combinations; this should be of shape
         `n_labels * n_labels` (e.g. potential=potts(n_labels, a, b)); if this is None, binary 
         features are concatenated unary features [default]
        :test_pct: (default None => test_pct = 25%)
         #train-examples = (1-test_pct) * |X|
        :val_pct:
         #val-examples = test_pct * val_pct * |X|
         #test-examples = test_pct * (1-val_pct) * |X|
        :seed:
         fixed random seed for "random" train/test/val split reproducibility 
        """
        self.X, X_tv, self.Y, Y_tv = train_test_split(
            X, Y, test_size=test_pct, random_state=seed)
        self.X_t, self.X_v, self.Y_t, self.Y_v = train_test_split(
            X_tv, Y_tv, test_size=val_pct, random_state=seed)
        self.Ns = self.N_tr, self.N_v, self.N_t = map(len, (self.X, self.X_v, self.X_t))
        self.L = L
        self.n_labels = len(self.L)
        self.phi_B = phi_B
        self.concat = self.phi_B is None
        self.n_feats = len(self.X[0][0])
        self.n_U = self.n_labels * self.n_feats
        self.n_W = self.n_U + self.n_labels**2 * (2*self.n_feats if self.concat else 1)
        self.U_shape = (self.n_labels, self.n_feats)
        self.B_shape = (self.n_labels,) * 2 + ((2 * self.n_feats if self.concat else 1),)
        self.B_term = self.B_concat if self.concat else self.B_scalar
        self.edge_feat = self.edge_feat_concat if self.concat else self.edge_feat_scalar

    def E(self, x, y, (W_u, W_b)):
        """
        x: "raw" features, y: label
        XXX old impl:
        if y is not None:
            x = self.featurize(x,y)
        return (x[0]*W[0]).sum() + (x[1]*W[1]).sum()
        """
        s = sum(self.B_term(W_b,x,i+1,(y1,y2)) + np.dot(x1,W_u[y1]) for i,((x1,y1),(x2,y2))
            in enumerate(self.xy(x, y))) + np.dot(x[-1], W_u[y[-1]])
        return s[0] if hasattr(s, '__len__') else s

    def XY(self, fold=''):
        X, Y = map(lambda s: getattr(self, s + fold), 'XY')
        return izip(X, Y)

    def xy(self, x, y):
        z = zip(x, y); return izip(z, z[1:])

    def split_W(self, W):
        return W[:self.n_U].reshape(self.U_shape), W[self.n_U:].reshape(self.B_shape)

    def configs(self, n):
        """
        all length-n labelings over label space self.L  
        """
        return product(*[self.L] * n)

    def set_idxs(self, c, I, V):
        c[I] = V; return c

    def B_scalar(self, W_b, _, __, I=None):
        return W_b[I]   * self.phi_B[I]   if hasattr(I, '__len__') else (
               W_b[:,I] * self.phi_B[:,I] if I else
               W_b      * self.phi_B)

    def B_concat(self, W_b, x, i, I=None):
        concat = np.concatenate((x[i - 1], x[i]))
        return np.dot(W_b[I],   concat) if hasattr(I, '__len__') else (
               np.dot(W_b[:,I], concat) if I else
               np.dot(W_b,      concat[:, None]))

    def edge_feat_concat(self, x, i, j, *_):
        return np.concatenate((x[i], x[j]))

    def edge_feat_scalar(self, x, _, __, m, n):
        return self.phi_B[m, n]