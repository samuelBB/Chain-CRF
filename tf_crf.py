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


# XXX see example in src doc
# W = tf.get_variable('W', [2,2,2])
# V = tf.Variable([[2,2],[3,3]])
# op = tf.variables_initializer([V])
# loss = tf.reduce_sum(tf.square(V))
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 100})
# XXX in session: optimizer.minimize(session)


def RMLE_tf(crf, Ws):
    shapes = map(lambda p: p.shape[1:], crf.phis[0])
    UB = U, B = map(lambda s: tf.placeholder(tf.float64, shape=(None,)+s), shapes)
    W_u, W_b = map(tf.Variable, Ws)
    E = tf.reduce_sum(U * W_u) + tf.reduce_sum(B * W_b)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        e = 0
        with timed('tf'):
            for ub in crf.phis:
                e += session.run(E, feed_dict=dict(zip(UB, ub)))
        print e