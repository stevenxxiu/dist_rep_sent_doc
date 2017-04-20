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
        Node = namedtuple('Node', ['id_', 'left', 'right'])
        Node.__new__.__defaults__ = (None,) * len(Node._fields)
        cls.tree = Node(
            0,
            Node('a'),
            Node(
                1,
                Node(3, Node('e'), Node('c')),
                Node(2, Node('d'), Node('b'))
            )
        )

    def test_get_node_to_path(self):
        l = HierarchicalSoftmaxLayer(self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})
        self.assertEqual(l._get_node_to_path(self.tree), {
            0: [(0, 1)],
            1: [(0, -1), (1, -1), (2, -1)],
            2: [(0, -1), (1, 1), (3, -1)],
            3: [(0, -1), (1, -1), (2, 1)],
            4: [(0, -1), (1, 1), (3, 1)],
        })

    def test_call_training(self):
        X = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.int32, [None])
        l = HierarchicalSoftmaxLayer(
            self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
            W_initializer=tf.constant_initializer([[.01, .02], [.03, .04], [.05, .06], [.07, .08]]),
        )
        cost = l.apply([X, y], training=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res = sess.run(cost, feed_dict={
                X: np.array([[.16, .17], [.18, .19], [.20, .21]]),
                y: [2, 1, 0],
            })
            assert_array_almost_equal(res[0], -np.array([
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
            self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
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
