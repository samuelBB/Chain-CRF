from itertools import izip

import numpy as np
from scipy.optimize import minimize

from model_utils import E, regularize, split_weights


def RMLE(crf, reg=0.):
    """
    :Input:
    X:
     A 3D python list, where axis 1 runs over each training example, axis 2 runs over 
     each variable in the chain, and axis 3 runs over each variable's features
    Y:
     A 2D python array where axis 1 runs over each training example, and axis 2 runs 
     over the label for each variable in the chain
    L:
     variable label space
    """
    def rmle(W):
        Ws = split_weights(W, *crf.dims)
        return sum(E(phi, Ws) + crf.Z(x, *Ws) for phi, x in izip(crf.phis, crf.X))
    if reg > 0:
        rmle = regularize(rmle, reg)

    # TODO
    # def grad_rmle()

    return minimize(rmle, np.zeros(crf.n_W), method='L-BFGS-B', options={'disp': True})