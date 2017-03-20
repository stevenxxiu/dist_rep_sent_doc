import unittest
from numpy.testing import assert_array_equal

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
            'a': [(0, 0)],
            'b': [(0, 1), (1, 1), (2, 1)],
            'c': [(0, 1), (1, 0), (3, 1)],
            'd': [(0, 1), (1, 1), (2, 0)],
            'e': [(0, 1), (1, 0), (3, 0)],
        })

    def test_tree_to_mat(self):
        path_matrix, child_matrix = tree_to_mat(
            {0: ['a', 1], 1: [3, 2], 3: ['e', 'c'], 2: ['d', 'b']},
            {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        )
        assert_array_equal(path_matrix, np.array([
            [0, -1, -1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 2],
            [0, 1, 3],
        ], dtype=int))
        assert_array_equal(child_matrix, np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
        ]))


# XXX test that hierarchical softmax is correct for the huffman tree

# XXX test that the cross entropy calculation is correct for huffman trees
