from itertools import izip

import numpy as np

from crf import ChainCRF
from data import synthetic, read_ocr, potts
from utils import timed


def test_E(crf):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x,y = crf.X[0], crf.Y[0]
    with timed('E'):
        s = crf.E(x, Ws, y)
    print s


def test_E_sum(crf):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    with timed('fast'):
        s = sum(crf.E(x, Ws, y) for x,y in izip(crf.X,crf.Y))
    with timed('loop'):
        s = sum(crf.E(phi, Ws) for phi in crf.phis)
    print s


def test_Z(crf, name='concat', slow=False):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    if slow:
        with timed('Z_slow_%s' % name):
            print crf.Z_slow(x, Ws)
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
        return sum(crf.E(x, Ws, y) + crf.Z(x, *Ws) for x, y in crf.XY())
    with timed('rmle'):
        X = rmle(W)
    print X
    if parallel:
        from pathos.multiprocessing import ProcessPool
        def rmle_p(W):
            Ws = crf.split_W(W)
            def para((x, y)):
                return crf.E(x, Ws, y) + crf.Z(x, *Ws)
            return sum(ProcessPool().map(para, crf.XY()))
        with timed('rmle_parallel'):
            X = rmle_p(W)
        print X


def test_marginals(crf, L, name='concat', e=True, v=True):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    import random
    r = random.choice(range(1, len(x) - 1))
    l1, l2 = random.choice(L), random.choice(L)
    D = [(r, l1), (r + 1, l2)]
    xs,ls = map(list,zip(*D))
    if e:
        print '[EDGE MARG]'
        with timed('marg_fast'):
            print crf.marginal(x, Ws, D)
        with timed('marg_rec'):
            if name == 'concat':
                print crf.marginal_rec_concat(x, Ws, D)
            else:
                print crf.marginal_rec_scalar(x,Ws,D)
        with timed('marginal_slow'):
            print crf.marginal_slow(x, Ws, xs, ls)
    if v:
        print '[VAR MARG]'
        with timed('marg_fast'):
            print crf.marginal(x, Ws, [D[0]])
        with timed('marg_rec'):
            if name == 'concat':
                print crf.marginal_rec_concat(x, Ws, [D[0]])
            else:
                print crf.marginal_rec_scalar(x,Ws,[D[0]])
        with timed('marginal_slow'):
            print crf.marginal_slow(x, Ws, [D[0][0]], [D[0][1]])

def test_MPM(crf, name='concat'):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    with timed('MPM_' + name):
        print crf.MPM(x,Ws)

def test_MAP(crf, name='concat'):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    with timed('MAP_' + name):
        v,y = crf.MAP(x, Ws)
    print v,y,'evald', crf.E(x,Ws,y)
    # with timed('MAP_slow_' + name):
    #     v, y = crf.MAP_slow(x, Ws)
    # print v, y, 'evald', crf.E(x, Ws, y)


def test_sample(crf, name='concat'):
    Ws = crf.split_W(np.random.rand(crf.n_W))
    x = crf.X[0]
    ns, burn, intvl, shuf = 20, 11, 8, False
    with timed('gibbs_' + name):
        samps = crf.gibbs(x, Ws, ns, burn, intvl, shuf)
    print len(samps), samps



if __name__ == '__main__':
    # nl = 4
    # data = synthetic(500, seq_len_range=(4, 7), n_feats=10, n_labels=nl)
    nl = 26
    data = read_ocr()
    X,Y = zip(*data)

    crf = ChainCRF(X, Y, range(nl))
    # test_sample(crf)
    # test_MAP(crf)
    # test_MPM(crf)
    # test_marginals(crf, range(nl))
    # test_E_sum(crf)
    # test_Z(crf, slow=True)
    test_rmle(crf)

    crf = ChainCRF(X, Y, range(nl), potts(nl))
    # test_sample(crf, 'scalar')
    # test_MAP(crf,'scalar')
    # test_MPM(crf, 'scalar')
    # test_marginals(crf, range(nl), 'scalar')
    test_rmle(crf)
    # test_E_sum(crf)
    # test_Z(crf, 'scalar', slow=False)
