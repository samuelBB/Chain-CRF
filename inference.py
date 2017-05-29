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

    def marginal(self, x, (W_u, W_b), pairs, normalize=False, logscale=True):
        """
        fast log-scale node/edge marginals (optional: normalize with Z)
        pairs: 
         1) node marginal - [(i,v)] where i=node-index and v=clamped-value - P(y_i=v)
         2) edge marginal - [(i,u),(j,v)] for P(y_i=u, y_j=v)
        """
        if len(pairs) == 1:
            (i, u), (j, u2) = pairs * 2
            prob = -np.dot(x[i], W_u[u])
        else:
            (i, u), (j, u2) = pairs
            prob = -sum(np.dot(x[k],W_u[w]) for k,w in pairs) - self.B_term(W_b,x,j,(u,u2))
        if i > 0:
            left = np.zeros((2, self.n_labels))
            if i > 1:
                for m in self.L:
                    left[0, m] = lse([-np.dot(x[0], W_u[l]) - self.B_term(W_b, x, 1, (l, m))
                        for l in self.L])
            for v in range(2, i):
                for m in self.L:
                    left[1, m] = lse([left[0, l] - self.B_term(W_b, x, v, (l, m))
                        - np.dot(x[v - 1], W_u[l]) for l in self.L])
                left[0], left[1] = left[1], 0.
            prob += lse([left[0, l] - np.dot(x[i - 1], W_u[l]) - self.B_term(W_b, x, i, (l, u))
                for l in self.L])
        if j < len(x) - 1:
            right = np.zeros((2, self.n_labels))
            if j < len(x) - 2:
                for m in self.L:
                    right[0, m] = lse([-np.dot(x[-1], W_u[l]) - self.B_term(W_b, x, -1, (m, l))
                        for l in self.L])
            for v in range(len(x) - 3, j, -1):
                for m in self.L:
                    right[1, m] = lse([right[0, l] - self.B_term(W_b, x, v + 1, (m, l))
                        - np.dot(x[v + 1], W_u[l]) for l in self.L])
                right[0], right[1] = right[1], 0.
            prob += lse([right[0, l] - self.B_term(W_b, x, j + 1, (u2, l))
                - np.dot(x[j + 1], W_u[l]) for l in self.L])
        if normalize:
            prob -= self.Z(x, W_u, W_b)
        if not logscale:
            prob = np.exp(prob)
        return prob

    def MPM(self, x, W):
        return np.array([max((self.marginal(x, W, [(i, l)]), l) for l in self.L)[1]
            for i in xrange(len(x))])

    def Gibbs(self, x, W, n_samples=1, burn=0, interval=1, init=None):
        y_old = np.array(init) if init is not None else np.array([choice(self.L)
                                                                 for _ in xrange(len(x))])
        samples = []
        for j in xrange(burn + n_samples * interval):
            y_new = np.array(y_old)
            for i in xrange(len(x)):
                Es = np.array([-self.E(x, self.set_idxs(y_new, i, l), W) for l in self.L])
                y_new[i] = np.random.choice(self.L, p=softmax(Es))
            if j >= burn and j % interval == 0:
                samples.append(y_new)
        return samples


class SlowInference:
    def MAP_slow(self, x, W):
        return min((self.E(x, y, W), y) for y in self.configs(len(x)))

    def Z_slow(self, x, *W):
        return lse([-self.E(x, y, W) for y in self.configs(len(x))])

    def marginal_slow(self, x, W, pairs, normalize=False, logscale=True):
        """
        unnormalized
        """
        I, V = map(list, zip(*pairs))
        J, y = sorted(set(xrange(len(x))) - set(I)), self.set_idxs(np.zeros(len(x), int), I, V)
        prob = lse([-self.E(x, self.set_idxs(y, J, c), W) for c in self.configs(len(J))])
        if normalize:
            prob -= self.Z(x, *W)
        if not logscale:
            prob = np.exp(prob)
        return prob


class Other:
    """
    "semi"-efficient inference and misc. helper funcs
    """
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

    def marg_helper(self, x, W_u, W_b, i, val, up=False):
        """ helper """
        if i in (0, len(x) - 1):
            return 0.
        j = i+1 if up else i-1
        k = i+1 if up else i
        pair = lambda m: (val, m) if up else (m, val)
        msgs = []
        for l in self.L:
            msg = -np.dot(x[j], W_u[l]) - self.B_term(W_b, x, k, pair(l))
            if 0 < j < len(x) - 1:
                msg += self.marg_helper(x, W_u, W_b, j, l, up)
            msgs.append(msg)
        return lse(msgs)

    def marginal_rec(self, x, (W_u, W_b), pairs, normalize=False, logscale=True):
        """
        fast node/edge marginals
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
        prob += self.marg_helper(x,W_u,W_b,i,u) + self.marg_helper(x,W_u,W_b,j,v,True)
        if normalize:
            prob -= self.Z(x, W_u, W_b)
        if not logscale:
            prob = np.exp(prob)