import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dist_rep_sent_doc.huffman import *


class TestHuffman(unittest.TestCase):
    def test_build_huffman(self):
        # wikipedia's example, equivalent huffman tree
        self.assertEqual(
            build_huffman({'a': 15, 'b': 7, 'c': 6, 'd': 6, 'e': 5}),
            HuffmanNode(
                3, 39,
                HuffmanNode('a', 15),
                HuffmanNode(
                    2, 24,
                    HuffmanNode(0, 11, HuffmanNode('e', 5), HuffmanNode('c', 6)),
                    HuffmanNode(1, 13, HuffmanNode('d', 6), HuffmanNode('b', 7))
                )
            )
        )

    def test_tree_to_arrays(self):
        nodes, signs, masks = tree_to_arrays(build_huffman(
            {'a': 15, 'b': 7, 'c': 6, 'd': 6, 'e': 5}),
            {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}, 5
        )
        assert_array_equal(nodes, np.array([
            [3, 0, 0], [3, 2, 1], [3, 2, 0], [3, 2, 1], [3, 2, 0],
        ]))
        assert_array_equal(signs, np.float32([
            [1, 0, 0], [-1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, 1, 1],
        ]))
        assert_array_equal(masks, np.array([
            [True, False, False], [True, True, True], [True, True, True], [True, True, True], [True, True, True],
        ]))
