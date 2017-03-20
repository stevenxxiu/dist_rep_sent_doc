import datetime
import heapq
import itertools
import random
from collections import Counter, namedtuple

import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *


def build_huffman(word_to_freq):
    '''
    :return: A dictionary mapping each node or word to it's children.
    '''
    node_to_children = {}
    queue = [(freq, word) for word, freq in word_to_freq.items()]
    heapq.heapify(queue)
    while len(queue) > 1:
        children = [heapq.heappop(queue), heapq.heappop(queue)]
        n = len(queue)
        node_to_children[n] = [children[0][1], children[1][1]]
        heapq.heappush(queue, (children[0][0] + children[1][0], n))
    return node_to_children


def tree_to_paths(tree):
    node_to_path = {0: []}
    for node, children in tree.items():
        path = node_to_path.pop(node)
        for i, child in enumerate(children):
            node_to_path[child] = path + [(node, i)]
    return node_to_path


def tree_to_mat(tree, word_to_index):
    '''
    Converts a huffman tree to a path matrix where each row contains the path from the root with -1 representing nop,
    a child matrix where 0 and 1 represent the left and right child respectively.
    '''
    word_to_path = tree_to_paths(tree)
    d = max(len(path) for path in word_to_path.values())
    path_matrix = np.empty((len(word_to_index), d), dtype=int)
    path_matrix.fill(-1)
    child_matrix = np.zeros((len(word_to_index), d))
    for word, path in word_to_path.items():
        path_matrix[word_to_index[word], :len(path)] = [path_item[0] for path_item in path]
        child_matrix[word_to_index[word], :len(path)] = [path_item[1] for path_item in path]
    return path_matrix, child_matrix


def words_to_mat(words, step_size, word_to_index):
    pass


def run_model():
    pass


def main():
    pass

if __name__ == '__main__':
    main()
