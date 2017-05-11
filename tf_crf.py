import numpy as np
import tensorflow as tf
from utils import timed
# from tensorflow.contrib.crf import viterbi_decode, crf_log_likelihood


# def split_weights(W, n_U, U_shape, B_shape):
#     return W[:n_U].reshape(U_shape), W[n_U:].reshape(B_shape)
#
# def E((phi_u, phi_b), (W_u, W_b)):
#     return (phi_u * W_u).sum() + (phi_b * W_b).sum()
#
# def Z_s(self, x, W_u, W_b):
#     z = -np.dot(W_u, x[0])
#     for i in xrange(1, len(x)):
#         z = logsumexp(z - np.dot(W_u, x[i, :, None]) - W_b * self.phi_B)
#     return logsumexp(z)


# W = tf.get_variable('W', [2,2,2])
# V = tf.Variable([[2,2],[3,3]])
# op = tf.variables_initializer([V])
# loss = tf.reduce_sum(tf.square(V))
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 100})


def RMLE_tf(crf, UB, Ws):
    Ut, Bt = map(tf.constant, UB)
    Wut, Wbt = map(tf.Variable, Ws)

    E = tf.reduce_sum(Ut*Wut) + tf.reduce_sum(Bt*Wbt)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        with timed('tf'):
            print session.run(E)
        # optimizer.minimize(session)

