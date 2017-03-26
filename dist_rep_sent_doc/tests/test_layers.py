import unittest
from collections import namedtuple

import lasagne
import numpy as np
import theano
import theano.tensor as T
from numpy.testing import assert_array_almost_equal, assert_array_equal
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
        l_in = lasagne.layers.InputLayer((None, 2))
        l = HierarchicalSoftmaxLayer(l_in, self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})
        self.assertEqual(l._get_node_to_path(self.tree), {
            0: [(0, -1)],
            1: [(0, 1), (1, 1), (2, 1)],
            2: [(0, 1), (1, -1), (3, 1)],
            3: [(0, 1), (1, 1), (2, -1)],
            4: [(0, 1), (1, -1), (3, -1)],
        })

    def test_get_output_for_target(self):
        hs_nodes = T.ivector('hs_nodes')
        hs_signs = T.vector('hs_signs')
        hs_indexes = T.ivector('hs_indexes')
        l_in = lasagne.layers.InputLayer((None, 2))
        l = HierarchicalSoftmaxLayer(
            l_in, self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
            W=np.array([[.01, .02], [.03, .04], [.05, .06], [.07, .08], [.09, .10]], dtype=np.float32),
            b=np.array([.11, .12, .13, .14, .15], dtype=np.float32),
        )
        res = theano.function(
            [l_in.input_var, hs_nodes, hs_signs, hs_indexes],
            lasagne.layers.get_output(l, hs_nodes=hs_nodes, hs_signs=hs_signs, hs_indexes=hs_indexes),
            on_unused_input='ignore'
        )(
            np.array([[.16, .17], [.18, .19], [.20, .21]], dtype=np.float32),
            *l.get_hs_inputs([0, 1, 2])
        )
        assert_array_almost_equal(res, np.array(
            np.log(1 - expit(np.dot([.16, .17], [.01, .02]) + 0.11)) + (
                np.log(expit(np.dot([.18, .19], [.01, .02]) + 0.11)) +
                np.log(expit(np.dot([.18, .19], [.03, .04]) + 0.12)) +
                np.log(expit(np.dot([.18, .19], [.05, .06]) + 0.13))
            ) + (
                np.log(expit(np.dot([.20, .21], [.01, .02]) + 0.11)) +
                np.log(1 - expit(np.dot([.20, .21], [.03, .04]) + 0.12)) +
                np.log(expit(np.dot([.20, .21], [.07, .08]) + 0.14))
            )
        ), decimal=8)

    def test_get_output_for(self):
        l_in = lasagne.layers.InputLayer((2, 2))
        l = HierarchicalSoftmaxLayer(
            l_in, self.tree, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
            W=np.array([[.01, .02], [.03, .04], [.05, .06], [.07, .08], [.09, .10]], dtype=np.float32),
            b=np.array([.11, .12, .13, .14, .15], dtype=np.float32),
        )
        res = l.get_hs_outputs(theano.function([l_in.input_var], lasagne.layers.get_output(l))(
            np.array([[.16, .17], [.18, .19], [.20, .21]], dtype=np.float32),
        ))
        assert_array_almost_equal(res[0][:2], np.array([
            np.log(1 - expit(np.dot([.16, .17], [.01, .02]) + 0.11)), (
                np.log(expit(np.dot([.16, .17], [.01, .02]) + 0.11)) +
                np.log(expit(np.dot([.16, .17], [.03, .04]) + 0.12)) +
                np.log(expit(np.dot([.16, .17], [.05, .06]) + 0.13))
            )
        ]), decimal=7)
