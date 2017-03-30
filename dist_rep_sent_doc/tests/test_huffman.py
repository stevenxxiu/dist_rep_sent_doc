import unittest

from dist_rep_sent_doc.huffman import *


class TestHuffman(unittest.TestCase):
    def test_build_huffman(self):
        # wikipedia's example, equivalent huffman tree
        self.assertEqual(
            build_huffman({'a': 15, 'b': 7, 'c': 6, 'd': 6, 'e': 5}),
            HuffmanNode(
                0, 39,
                HuffmanNode('a', 15),
                HuffmanNode(
                    1, 24,
                    HuffmanNode(3, 11, HuffmanNode('e', 5), HuffmanNode('c', 6)),
                    HuffmanNode(2, 13, HuffmanNode('d', 6), HuffmanNode('b', 7))
                )
            )
        )
