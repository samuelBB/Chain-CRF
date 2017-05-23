from random import randint
from itertools import izip

import numpy as np
from sklearn import svm
from scipy.optimize import minimize, check_grad as cg, approx_fprime

from utils import timed

# TODO pseudo-likelihood, scikit linear-kernel SVM
# TODO if-time: cross-val/tune
# TODO train/val/test splitting

class Learner:
    def __init__(self, crf, gibbs=False, n_samples=1, burn=0, interval=1, shuffle=False):
        self.crf = crf
        self.E_f = self.exp_feat_gibbs if gibbs else self.exp_feat
        self.n_samples = n_samples
        self.burn = burn
        self.interval = interval
        self.shuffle = shuffle

    def feat(self, x, y):
        node_feats = np.zeros(self.crf.U_shape)
        edge_feats = np.zeros(self.crf.B_shape)
        V = range(len(x))
        for i in V:
            node_feats[y[i]] += x[i]
        for i, j in izip(V, V[1:]):
            edge_feats[y[i], y[j]] += self.crf.edge_feat(x, i, j, y[i], y[j])
        return np.concatenate((node_feats.flatten(), edge_feats.flatten()))

    def exp_feat(self, x, W):
        node_feats = np.zeros(self.crf.U_shape)
        edge_feats = np.zeros(self.crf.B_shape)
        V, Z = range(len(x)), self.crf.Z(x, *W)
        for i in V:
            for l in self.crf.L:
                node_feats[l] += np.exp(self.crf.marginal(x, W, [(i, l)]) - Z) * x[i]
        for i, j in izip(V, V[1:]):
            for m, n in self.crf.configs(2):
                ef = self.crf.edge_feat(x, i, j, m, n)
                edge_feats[m, n] += np.exp(self.crf.marginal(x,W,[(i,m),(j,n)]) - Z) * ef
        return np.concatenate((node_feats.flatten(), edge_feats.flatten()))

    def exp_feat_gibbs(self, x, W):
        S = self.crf.Gibbs(x, W, self.n_samples, self.burn, self.interval, self.shuffle)
        feats = np.zeros(self.crf.n_W)
        for s in S:
            feats += self.feat(x, s)
        return feats / self.n_samples

    def regularize(self, reg):
        def obj(W):
            return self.obj(W) + reg * np.linalg.norm(W)**2
        def grad(W, *a):
            return self.grad(W, *a) + 2. * reg * W
        return obj, grad

    def check_grad(self, N=1):
        Ws = [np.random.rand(self.crf.n_W) for _ in range(N)]
        return [cg(self.obj, self.grad, W) for W in Ws]

    def grad_apx(self, W):
        return approx_fprime(W, self.obj, np.sqrt(np.finfo(float).eps))

    def __call__(self, method='L-BFGS-B', disp=True, reg=0.):
        """
        if implementing self.{obj(W),grad(W)} to be used with scipy optimization 
        """
        obj, grad = self.regularize(reg) if reg > 0. else (self.obj, self.grad)
        init = np.zeros(self.crf.n_W)
        self.opt = minimize(obj, init, method=method, jac=grad, options={'disp': disp})
        return self.opt.x


class ML(Learner):
    """
    Maximum Likelihood
    """
    def obj(self, W):
        Ws = self.crf.split_W(W)
        return sum(self.crf.E(x, Ws, y) + self.crf.Z(x, *Ws) for x, y in self.crf.XY())

    def grad(self, W):
        Ws, feats = self.crf.split_W(W), np.zeros(self.crf.n_W)
        for x, y in self.crf.XY():
            feats += self.feat(x, y) - self.E_f(x, Ws)
        return feats


class PL(Learner):
    """
    Pseudo-Likelihood
    """
    # TODO FIXME
    def exp_feat_PL(self, x, y, W, s):
        node_feats = np.zeros(self.crf.U_shape)
        edge_feats = np.zeros(self.crf.B_shape)
        Z = self.crf.Z_PL(x, y, W, s)
        for l in self.crf.L:
            node_feats[l] += np.exp(self.crf.E_PL(x, y, W, s, l) - Z) * x[s]
            if s > 0:
                ef = self.crf.edge_feat(x, s-1, s, y[s-1], l)
                edge_feats[y[s-1], l] += np.exp(self.crf.E_PL(x, y, W, s, l) - Z) * ef
            if s < len(x) - 1:
                ef = self.crf.edge_feat(x, s, s+1, l, y[s+1])
                edge_feats[l, y[s+1]] += np.exp(self.crf.E_PL(x, y, W, s, l) - Z) * ef
        # return node_feats.flatten()
        return np.concatenate((node_feats.flatten(), edge_feats.flatten()))

    def obj(self, W):
        Ws = self.crf.split_W(W)
        s = 0.
        for x, y in self.crf.XY():
            s += self.crf.E(x, Ws, y)
            for i in xrange(len(x)):
                 s += self.crf.Z_PL(x, y, Ws, i)
        return s

    # TODO FIXME
    def grad(self, W):
        Ws = self.crf.split_W(W)
        s = 0.
        for x, y in self.crf.XY():
            s += self.feat(x, y)
            for i in xrange(len(x)):
                s -= self.exp_feat_PL(x, y, Ws, i)
        return s


class SML(Learner):
    """
    Stochastic Maximum Likelihood
    """
    def grad(self, W, x, y):
        return self.feat(x, y) - self.E_f(W, x)

    def train(self, lr=1., step=None, n_iters=10**5, rand_init=False, reg=0.):
        N = len(self.crf.X)
        grad = self.regularize(reg)[1] if reg > 0. else self.grad
        W = (np.random.rand if rand_init else np.zeros)(self.crf.n_W)
        for i in xrange(n_iters):
            r = randint(0, N - 1)
            W -= lr * grad(W, self.crf.X[r], self.crf.Y[r])
            if step:
                lr *= np.power(.1, np.floor(i * step / n_iters))
        return W


def baseline_SVC(crf, C=1., loss='squared_hinge', penalty='l2'):
    X, Y = np.concatenate(crf.X), np.concatenate(crf.Y)
    dual = len(crf.N) <= len(crf.n_feats)
    return svm.LinearSVC(penalty, loss, dual, C=C).fit(X, Y)


if __name__ == '__main__':
    from data import potts, synthetic
    from crf import ChainCRF

    nl = 4
    data = synthetic(100, seq_len_range=(4, 7), n_feats=5, n_labels=nl)
    X, Y = zip(*data)

    # crf = ChainCRF(X, Y, range(nl))
    crf = ChainCRF(X, Y, range(nl), potts(nl))

    learner = PL(crf)

    W = crf.split_W(np.random.rand(crf.n_W))
    # with timed('g'):
    #     learner.grad(W)
    # print learner.check_grad()
    with timed('exp_feat'):
        learner.exp_feat(crf.X[0], W)
    learner.burn = 100
    with timed('exp_gibbs'):
        learner.exp_feat_gibbs(crf.X[0], W)