from itertools import izip
from random import choice

import numpy as np
from scipy.misc import logsumexp as lse

from utils import softmax


class Inference:
    def MAP(self, x, (W_u, W_b), only_label=True):
        t = np.zeros((2, self.n_labels))
        c = np.zeros((len(x), self.n_labels), int)
        for i in xrange(len(x)):
           for n in self.L:
               t[1, n], c[i, n] = min((np.dot(x[i],W_u[m]) + (self.B_term(W_b,x,i+1,(m,n))
                   if i+1<len(x) else 0.) + t[0, m], m) for m in self.L)
           t[0], t[1] = t[1], 0.
        s = t[0].argmin()
        v = t[0, s]
        ms = np.zeros(len(x), int)
        for i in xrange(len(x)-1, -1, -1):
            s = c[i, s]
            ms[i] = s
        return ms if only_label else (v, ms)

    def Z(self, x, W_u, W_b):
        z = -np.dot(W_u, x[0])
        for i in xrange(1, len(x)):
            z = lse((z - np.dot(W_u,x[i,:,None]) - self.B_term(W_b,x,i))[self.L,:,self.L].T, 1)
        return lse(z)

    def E_PL(self, x, y, W, s, l):
        return self.E(x, self.set_idxs(y, s, l), W)

    def Z_PL(self, x, y, W, s):
        return lse([-self.E_PL(x, y, W, s, l) for l in self.L])

    def marginal(self, x, (W_u, W_b), pairs):
        """
        unnormalized fast node/edge marginals
        pairs: 
         1) node marginal - [(i,v)] where i=node-index and v=clamped-value - P(y_i=v)
         2) edge marginal - [(i,u),(j,v)] for P(y_i=u, y_j=v)
        """
        if len(pairs) == 1:
            (i, u), (j, v) = pairs * 2
            prob = -np.dot(x[i], W_u[u])
        else:
            (i, u), (j, v) = pairs
            prob = -sum(np.dot(x[k],W_u[w]) for k,w in pairs) - self.B_term(W_b, x, j, (u,v))
        if i > 0:
            L = np.full((2, self.n_labels), 1.)
            for k in xrange(i - 1):
                for n in self.L:
                    L[1, n] = sum(np.exp(-np.dot(x[k], W_u[m]) - self.B_term(
                        W_b, x, k+1, (m, n))) * L[0, m] for m in self.L)
                L[0], L[1] = L[1], 0.
            prob += lse([(-np.dot(x[i - 1], W_u[m]) - self.B_term(W_b, x, i, (m, u))) +
                np.log(L[0, m]) for m in self.L])
        if j < len(x) - 1:
            R = np.full((2, self.n_labels), 1.)
            for k in xrange(len(x)-1, j+1, -1):
                for m in self.L:
                    R[1, m] = sum(np.exp(-np.dot(x[k], W_u[n]) - self.B_term(
                        W_b, x, k, (m, n))) * R[0, n] for n in self.L)
                R[0], R[1] = R[1], 0.
            prob += lse([(-np.dot(x[j + 1], W_u[n]) - self.B_term(W_b, x, j + 1, (v, n)) +
                np.log(R[0, n])) for n in self.L])
        return prob[0] if hasattr(prob, '__len__') else prob

    def MPM(self, x, W):
        return np.array([max((self.marginal(x, W, [(i, l)]), l) for l in self.L)[1]
            for i in xrange(len(x))])

    def Gibbs(self, x, W, n_samples=1, burn=100, interval=1):
        samples, y_old = [], np.array([choice(self.L) for _ in xrange(len(x))])
        for j in xrange(burn + n_samples * interval):
            y_new = y_old.copy()
            for i in xrange(len(x)):
                y_new[i] = np.random.choice(self.L, p=softmax(np.array(
                    [-self.E(x, self.set_idxs(y_new, i, l), W) for l in self.L])))
            if j >= burn and j % interval == 0:
                samples.append(y_new)
        return samples


class SlowInference:
    def MAP_slow(self, x, W):
        return min((self.E(x, y, W), y) for y in self.configs(len(x)))

    def Z_slow(self, x, *W):
        return lse([-self.E(x, y, W) for y in self.configs(len(x))])

    def marginal_slow(self, x, W, pairs):
        """
        unnormalized
        """
        I, V = map(list, zip(*pairs))
        J, y = sorted(set(xrange(len(x))) - set(I)), self.set_idxs(np.zeros(len(x), int), I, V)
        return lse([-self.E(x, self.set_idxs(y, J, c), W) for c in self.configs(len(J))])


class Other:
    def Z_full_DP_scalar(self, x, W_u, W_b):
        z = np.vstack([-np.dot(W_u, x[0]), np.zeros(W_u.shape[0])])
        for i in xrange(1, len(x)):
            for n in xrange(self.n_labels):
                q = np.zeros(self.n_labels)
                for m in xrange(self.n_labels):
                    q[m] = z[0, m] - np.dot(x[i], W_u[n]) - W_b[n, m] * self.phi_B[n, m]
                z[1, n] = lse(q)
            z[0], z[1] = z[1], 0.
        return lse(z[0])

    def Z_DP_scalar(self, x, W_u, W_b):
        z = np.vstack([-np.dot(W_u, x[0]), np.zeros(W_u.shape[0])])
        for i in xrange(1, len(x)):
            for n in xrange(self.n_labels):
                z[1, n] = lse(z[0] - np.dot(x[i],W_u[n]) - np.squeeze(W_b[:,n]*self.phi_B[:,n]))
            z[0], z[1] = z[1], 0.
        return lse(z[0])

    def Z_DP_concat(self, x, W_u, W_b):
        z = np.vstack([-np.dot(W_u, x[0]), np.zeros(W_u.shape[0])])
        for i in xrange(1, len(x)):
            for n in xrange(self.n_labels):
                z[1, n] = lse(z[0] - np.dot(x[i], W_u[n]) - np.dot(W_b[:, n],
                    np.concatenate((x[i - 1], x[i]))))
            z[0], z[1] = z[1], 0.
        return lse(z[0])

    def u_a(self, x, W, i, v):
        return sum(np.exp(-np.dot(x[i - 1], W[0][l]) - W[1][l, v] * self.phi_B[l, v]) * (
            self.u_a(x, W, i - 1, l) if i > 1 else 1.) for l in self.L)

    def u_b(self, x, W, i, v):
        return sum(np.exp(-np.dot(x[i + 1], W[0][l]) - W[1][v, l] * self.phi_B[v, l]) * (
            self.u_b(x, W, i + 1, l) if i < len(x) - 2 else 1.) for l in self.L)

    def marginal_rec_scalar(self, x, W, pairs):
        """
        fast node/edge marginals 
        """
        if len(pairs) == 1:
            (i, u), (j, v) = pairs * 2
            prob = -np.dot(x[i], W[0][u])
        else:
            (i, u), (j, v) = pairs
            prob = -sum(np.dot(x[k], W[0][w]) for k, w in pairs) - W[1][u, v] * self.phi_B[u, v]
        if i > 0:
            prob += np.log(self.u_a(x, W, i, u))
        if j < len(x)-1:
            prob += np.log(self.u_b(x, W, j, v))
        return prob

    def u_a_c(self, x, W, i, v):
        return sum(np.exp(-np.dot(x[i - 1], W[0][l]) - self.B_term(W[1],x,i,(l,v))) * (
            self.u_a_c(x, W, i - 1, l) if i > 1 else 1.) for l in self.L)

    def u_b_c(self, x, W, i, v):
        return sum(np.exp(-np.dot(x[i + 1], W[0][l]) - self.B_term(W[1],x,i+1,(v,l))) * (
            self.u_b_c(x, W, i + 1, l) if i < len(x) - 2 else 1.) for l in self.L)

    def marginal_rec_concat(self, x, W, pairs):
        """
        fast node/edge marginals 
        """
        if len(pairs) == 1:
            (i, u), (j, v) = pairs * 2
            prob = -np.dot(x[i], W[0][u])
        else:
            (i, u), (j, v) = pairs
            prob = -sum(np.dot(x[k], W[0][w]) for k, w in pairs) - self.B_term(W[1],x,j,(u,v))
        if i > 0:
            prob += np.log(self.u_a_c(x, W, i, u))
        if j < len(x)-1:
            prob += np.log(self.u_b_c(x, W, j, v))
        return prob

    def featurize(self, x, y):
        """
        :Input:
        x:
         A 2D python list, where axis 1 runs over each variable in the crf chain, and 
         axis 2 runs over each variable's features
        y:
         A 1D python list which runs over the crf variable labels

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
        U = np.zeros((len(x),) + self.U_shape)
        for i, (f, y_i) in enumerate(izip(x, y)):
            U[i, y_i] = f
        B = np.zeros((len(x) - 1,) + self.B_shape)
        for i, ((f1, y1), (f2, y2)) in enumerate(self.xy(x, y)):
            if self.concat:
                B[i, y1, y2] = np.concatenate((f1, f2))
            else:
                B[i, y1, y2, 0] = self.phi_B[y1, y2]
        return U, B