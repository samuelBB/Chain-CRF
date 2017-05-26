from itertools import izip

import numpy as np

from crf import ChainCRF
from data import read_ocr, synthetic, potts, ocr_bigram_freqs, synthetic_gaussian
from training import ML, SML, train_svc_multiple, train_gesture
from utils import timed


def test_E(crf):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x,y = crf.X[0], crf.Y[0]
    with timed('E'):
        s = crf.E(x, y, Ws)
    print s


def test_E_sum(crf):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    with timed('fast'):
        s = sum(crf.E(x, y, Ws) for x,y in izip(crf.X,crf.Y))
    print s


def test_Z(crf, name='concat', slow=False):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    if slow:
        with timed('Z_slow_%s' % name):
            print crf.Z_slow(x, *Ws)
    with timed('Z (Z_%s)' % name):
        print crf.Z(x, *Ws)
    if name == 'scalar':
        with timed('Z_scalar_DP'):
            print crf.Z_DP_scalar(x, *Ws)
    else:
        with timed('Z_concat_DP'):
            print crf.Z_DP_concat(x, *Ws)


def test_rmle(crf, parallel=False):
    W = np.random.rand(crf.n_W)
    def rmle(W):
        Ws = crf.split_W(W)
        return sum(crf.E(x, y, Ws) + crf.Z(x, *Ws) for x, y in crf.XY())
    with timed('rmle'):
        X = rmle(W)
    print X
    if parallel:
        from pathos.multiprocessing import ProcessPool
        def rmle_p(W):
            Ws = crf.split_W(W)
            def para((x, y)):
                return crf.E(x, y, Ws) + crf.Z(x, *Ws)
            return sum(ProcessPool().map(para, crf.XY()))
        with timed('rmle_parallel'):
            X = rmle_p(W)
        print X


def test_marginals(crf,L,e=True,v=True,slow=False,fast_only=False,const=False,norm=False):
    if const:
        np.random.seed(0)
    Ws = crf.split_W(np.random.rand(crf.n_W))
    # print Ws
    x = crf.X[0]
    import random
    r = random.choice(range(1, len(x) - 1))
    l1, l2 = random.choice(L), random.choice(L)
    D = [(r, l1), (r + 1, l2)]
    print D, '\n'
    if e:
        print '[EDGE MARG]'
        with timed('marg_fast_bottomup'):
            print crf.marginal(x, Ws, D, normalize=norm)
        if not fast_only:
            with timed('marg_rec'):
                print crf.marginal_rec(x, Ws, D, normalize=norm)
        if slow:
            with timed('marginal_slow'):
                print crf.marginal_slow(x, Ws, D, normalize=norm)
    if v:
        print '[VAR MARG]'
        with timed('marg_fast'):
            print crf.marginal(x, Ws, [D[0]], normalize=norm)
        if not fast_only:
            with timed('marg_rec'):
                print crf.marginal_rec(x, Ws, [D[0]], normalize=norm)
        if slow:
            with timed('marginal_slow'):
                print crf.marginal_slow(x, Ws, [D[0]], normalize=norm)


def test_MPM(crf, name='concat'):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    with timed('MPM_' + name):
        print crf.MPM(x,Ws)


def test_MAP(crf, name='concat', slow=False):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    with timed('MAP_' + name):
        v,y = crf.MAP(x, Ws, only_label=False)
    print v,y,'evald', crf.E(x, y, Ws)
    if slow:
        with timed('MAP_slow_' + name):
            v, y = crf.MAP_slow(x, Ws)
        print v, y, 'evald', crf.E(x, y, Ws)


def test_sample(crf, name='concat'):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    ns, burn, intvl = 2, 1, 4
    with timed('gibbs_' + name):
        samps = crf.Gibbs(x, Ws, ns, burn, intvl)
    print len(samps), samps


if __name__ == '__main__':
    # nl = 4
    # data = synthetic(100, lims=(4, 8), n_feats=8, n_labels=nl)
    # nl = 26
    # data = read_ocr()
    # X,Y = zip(*data)

    X, Y, nl = synthetic_gaussian()

    # crf = ChainCRF(X, Y, range(nl))

    crf = ChainCRF(X, Y, range(nl), potts(nl))
    # crf = ChainCRF(X, Y, range(nl), ocr_bigram_freqs() * 100.)

    ml = ML(crf, gibbs=True, n_samps=5, burn=10, interval=5)
    ml.train(rand=True, path='synth_gauss_reg1_lbfgs')

    # sml = SML(crf, gibbs=True, cd=True, n_samps=1, interval=1)
    # sml.sgd(n_iters=100,rand=True, path='OCR_SML_reg_1')

    # train_svc_multiple()

    # Ws = crf.split_W(np.random.rand(crf.n_W))
    # test_sample(crf)

    # train_gesture()

    # test_MAP(crf)
    # test_MPM(crf)
    # test_marginals(crf, range(nl), norm=True, slow=True)
    # test_E_sum(crf)
    # test_Z(crf, slow=True)
    # test_rmle(crf)

    # test_sample(crf, 'scalar')
    # test_MAP(crf,'scalar')
    # test_MPM(crf, 'scalar')
    # test_rmle(crf)
    # test_E_sum(crf)
    # test_Z(crf, 'scalar', slow=True)
