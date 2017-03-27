import unittest
from collections import namedtuple

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
            0: [(0, -1)],
            1: [(0, 1), (1, 1), (2, 1)],
            2: [(0, 1), (1, -1), (3, 1)],
            3: [(0, 1), (1, 1), (2, -1)],
            4: [(0, 1), (1, -1), (3, -1)],
        })

    def test_call_training(self):
        X = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.int32, [None])
        l = HierarchicalSoftmaxLayer(
            self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
            W_initializer=tf.constant_initializer([[.01, .02], [.03, .04], [.05, .06], [.07, .08], [.09, .10]]),
            b_initializer=tf.constant_initializer(np.array([.11, .12, .13, .14, .15]))
        )
        cost = l.apply(tf.tuple(X, y), training=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res = sess.run(cost, feed_dict={
                X: np.array([[.16, .17], [.18, .19], [.20, .21]]),
                y: np.array([0, 1, 2], dtype=np.int32)
            })
            assert_array_almost_equal(res, np.array([
                np.log(1 - expit(np.dot([.16, .17], [.01, .02]) + 0.11)), (
                    np.log(expit(np.dot([.18, .19], [.01, .02]) + 0.11)) +
                    np.log(expit(np.dot([.18, .19], [.03, .04]) + 0.12)) +
                    np.log(expit(np.dot([.18, .19], [.05, .06]) + 0.13))
                ), (
                    np.log(expit(np.dot([.20, .21], [.01, .02]) + 0.11)) +
                    np.log(1 - expit(np.dot([.20, .21], [.03, .04]) + 0.12)) +
                    np.log(expit(np.dot([.20, .21], [.07, .08]) + 0.14))
                )
            ]).mean(), decimal=8)

    # def test_get_output_for(self):
    #     X = tf.placeholder(tf.float32, [None, 2])
    #     l = HierarchicalSoftmaxLayer(
    #         self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
    #         W_initializer=tf.constant_initializer([[.01, .02], [.03, .04], [.05, .06], [.07, .08], [.09, .10]]),
    #         b_initializer=tf.constant_initializer(np.array([.11, .12, .13, .14, .15]))
    #     )
    #     probs = l.apply(X, training=False)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         res = l.get_hs_outputs(sess.run(probs, feed_dict={
    #             X: np.array([[.16, .17], [.18, .19], [.20, .21]]),
    #         }))
    #         assert_array_almost_equal(res[0][:2], np.array([
    #             np.log(1 - expit(np.dot([.16, .17], [.01, .02]) + 0.11)), (
    #                 np.log(expit(np.dot([.16, .17], [.01, .02]) + 0.11)) +
    #                 np.log(expit(np.dot([.16, .17], [.03, .04]) + 0.12)) +
    #                 np.log(expit(np.dot([.16, .17], [.05, .06]) + 0.13))
    #             )
    #         ]), decimal=7)
