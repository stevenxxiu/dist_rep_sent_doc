import unittest
from collections import namedtuple

import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal
from scipy.special import expit

from dist_rep_sent_doc.layers import *


class TestHierarchicalSoftmaxLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = np.array([
            [0, 0, 0], [0, 1, 2], [0, 1, 3], [0, 1, 2], [0, 1, 3],
        ]), np.array([
            [1, 0, 0], [-1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, 1, 1],
        ], dtype=np.float32), np.array([
            [True, False, False], [True, True, True], [True, True, True], [True, True, True], [True, True, True],
        ])

    def test_call_training(self):
        X = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.int32, [None])
        l = HierarchicalSoftmaxLayer(
            self.tree,
            W_initializer=tf.constant_initializer([[.01, .02], [.03, .04], [.05, .06], [.07, .08]]),
        )
        cost = l.apply([X, y], training=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res = sess.run(cost, feed_dict={
                X: np.array([[.16, .17], [.18, .19], [.20, .21]]),
                y: [2, 1, 0],
            })
            assert_array_almost_equal(res, np.array([
                (
                    np.log(1 - expit(np.dot([.16, .17], [.01, .02]))) +
                    np.log(expit(np.dot([.16, .17], [.03, .04]))) +
                    np.log(1 - expit(np.dot([.16, .17], [.07, .08])))
                ), (
                    np.log(1 - expit(np.dot([.18, .19], [.01, .02]))) +
                    np.log(1 - expit(np.dot([.18, .19], [.03, .04]))) +
                    np.log(1 - expit(np.dot([.18, .19], [.05, .06])))
                ), np.log(expit(np.dot([.20, .21], [.01, .02])))
            ]).sum(), decimal=7)

    def test_get_output_for(self):
        X = tf.placeholder(tf.float32, [None, 2])
        l = HierarchicalSoftmaxLayer(
            self.tree,
            W_initializer=tf.constant_initializer([[.01, .02], [.03, .04], [.05, .06], [.07, .08]]),
        )
        probs = l.apply(X, training=False)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res = sess.run(probs, feed_dict={
                X: np.array([[.16, .17], [.18, .19], [.20, .21]]),
            })
            assert_array_almost_equal(res[0][:2], np.array([
                np.log(expit(np.dot([.16, .17], [.01, .02]))), (
                    np.log(1 - expit(np.dot([.16, .17], [.01, .02]))) +
                    np.log(1 - expit(np.dot([.16, .17], [.03, .04]))) +
                    np.log(1 - expit(np.dot([.16, .17], [.05, .06])))
                )
            ]), decimal=7)
