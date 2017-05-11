from itertools import izip

import numpy as np

from crf import ChainCRF
from data import synthetic, read_ocr
from model_utils import E, split_weights, potts
from utils import timed

from tf_crf import RMLE_tf

def test_E(crf):
    W = np.random.rand(crf.n_W)
    Ws = Wu, Wb = split_weights(W, *crf.dims)
    UB = U, B = map(np.vstack, zip(*crf.phis))
    with timed('agg'):
        print (U * Wu).sum() + (B * Wb).sum()
    with timed('loop'):
        print sum((phi_u * Wu).sum() + (phi_b * Wb).sum() for phi_u, phi_b in crf.phis)
    RMLE_tf(crf, UB, Ws)


def test_Z(crf, name='concat'):
    Ws = split_weights(np.random.rand(crf.n_W), *crf.dims)
    x = crf.X[0]
    zdp = getattr(crf, 'Z_DP_' + name)
    with timed('Z_DP_' + name):
        print zdp(x, *Ws)
    with timed('Z (Z_%s)' % name):
        print crf.Z(x, *Ws)
    zold = getattr(crf, 'Z_' + name[0])
    with timed('Z_' + name[0]):
        print zold(x, *Ws)
    # with timed('Z_tf'):
    #     print Z_tf(x, Ws, L, len(x))


def test_rmle(crf, Z):
    W = np.random.rand(crf.n_W)
    def rmle(W):
        Ws = split_weights(W, *crf.dims)
        return sum((phi[0] * Ws[0]).sum() + (phi[1] * Ws[1]).sum() + Z(x, *Ws)
                   for phi, x in izip(crf.phis, crf.X))
    with timed('rmle'):
        rmle(W)
    # with timed('rmle_tf'):
    #     print rmle_tf(W)


if __name__ == '__main__':
    # print np.random.random((2,2))
    # exit()


    # nl = 10
    # data = synthetic(5000, seq_len_range=(4, 10), n_feats=15, n_labels=nl)
    nl = 26
    data = read_ocr()
    X,Y = zip(*data)

    # crf = ChainCRF(X, Y, range(nl))
    # test_E(crf)
    # test_Z(crf)
    # test_rmle(crf, crf.Z)

    crf = ChainCRF(X, Y, range(nl), potts(nl)[..., None])
    test_E(crf)
    # test_Z(crf, 'scalar')
    # test_rmle(crf, crf.Z)
    # test_rmle(crf, crf.Z_scalar)