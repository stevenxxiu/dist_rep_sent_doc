import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.special import expit

from main import *


class TestHuffman(unittest.TestCase):
    def test_build_huffman(self):
        # wikipedia's example, equivalent huffman tree
        self.assertEqual(
            build_huffman({'a': 15, 'b': 7, 'c': 6, 'd': 6, 'e': 5}),
            {0: ['a', 1], 1: [3, 2], 3: ['e', 'c'], 2: ['d', 'b']}
        )

    def test_tree_to_paths(self):
        self.assertEqual(tree_to_paths({0: ['a', 1], 1: [3, 2], 3: ['e', 'c'], 2: ['d', 'b']}), {
            'a': [(0, -1)],
            'b': [(0, 1), (1, 1), (2, 1)],
            'c': [(0, 1), (1, -1), (3, 1)],
            'd': [(0, 1), (1, 1), (2, -1)],
            'e': [(0, 1), (1, -1), (3, -1)],
        })

    def test_tree_to_mat(self):
        mask_mat, path_mat, child_mat = tree_to_mat(
            {0: ['a', 1], 1: [3, 2], 3: ['e', 'c'], 2: ['d', 'b']},
            {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        )
        assert_array_equal(mask_mat, np.array([
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]))
        assert_array_equal(path_mat, np.array([
            [0, 0, 0],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 2],
            [0, 1, 3],
        ], dtype=int))
        assert_array_equal(child_mat, np.array([
            [-1, 0, 0],
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [1, -1, -1],
        ]))


class TestHierarchicalSoftmax(unittest.TestCase):
    def test_hierarchical_softmax(self):
        context = T.matrix()
        W = T.matrix()
        b = T.vector()
        mask_mat, path_mat, child_mat = tree_to_mat(
            {0: ['a', 1], 1: [3, 2], 3: ['e', 'c'], 2: ['d', 'b']},
            {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        )
        res = theano.function(
            [context, W, b],
            hierarchical_softmax(
                context, W, b, theano.shared(mask_mat), theano.shared(path_mat), theano.shared(child_mat)
            ), on_unused_input='ignore'
        )(
            np.array([[.01, .02], [.03, .04]], dtype=np.float32),
            np.array([[.05, .06], [.07, .08], [.09, .10], [.11, .12], [.13, .14]], dtype=np.float32),
            np.array([.15, .16, .17, .18, .19], dtype=np.float32)
        )
        assert_array_almost_equal(res[0][0], np.array([
            np.log(1 - expit(np.dot([.01, .02], [.05, .06]) + 0.15))
        ]), decimal=4)

    def test_hierarchical_softmax_cross_entropy(self):
        context = T.matrix()
        W = T.matrix()
        b = T.vector()
        target = T.ivector()
        mask_mat, path_mat, child_mat = tree_to_mat(
            {0: ['a', 1], 1: [3, 2], 3: ['e', 'c'], 2: ['d', 'b']},
            {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        )
        res = theano.function(
            [context, W, b, target],
            hierarchical_softmax_cross_entropy(
                context, W, b, theano.shared(mask_mat), theano.shared(path_mat), theano.shared(child_mat), target
            )
        )(
            np.array([[.01, .02], [.03, .04]], dtype=np.float32),
            np.array([[.05, .06], [.07, .08], [.09, .10], [.11, .12], [.13, .14]], dtype=np.float32),
            np.array([.15, .16, .17, .18, .19], dtype=np.float32),
            np.array([0, 1], dtype=np.int32)
        )
        assert_array_almost_equal(res, np.array([[
            -np.log(1 - expit(np.dot([.01, .02], [.05, .06]) + 0.15)),
            -(
                np.log(expit(np.dot([.01, .02], [.05, .06]) + 0.15)) +
                np.log(expit(np.dot([.01, .02], [.07, .08]) + 0.16)) +
                np.log(expit(np.dot([.01, .02], [.09, .10]) + 0.17))
            )
        ], [
            -np.log(1 - expit(np.dot([.03, .04], [.05, .06]) + 0.15)),
            -(
                np.log(expit(np.dot([.03, .04], [.05, .06]) + 0.15)) +
                np.log(expit(np.dot([.03, .04], [.07, .08]) + 0.16)) +
                np.log(expit(np.dot([.03, .04], [.09, .10]) + 0.17))
            )
        ]]), decimal=4)
