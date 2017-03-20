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
            node_to_path[child] = path + [(node, -1 if i == 0 else 1)]
    return node_to_path


def tree_to_mat(tree, word_to_index):
    '''
    Converts a huffman tree to a path matrix where each row contains the path from the root, a child matrix where 0
    and 1 represent the left and right child respectively.
    '''
    word_to_path = tree_to_paths(tree)
    d = max(len(path) for path in word_to_path.values())
    mask_mat = np.zeros((len(word_to_index), d))
    path_mat = np.zeros((len(word_to_index), d), dtype=int)
    child_mat = np.zeros((len(word_to_index), d))
    for word, path in word_to_path.items():
        mask_mat[word_to_index[word], :len(path)] = 1
        path_mat[word_to_index[word], :len(path)] = [path_item[0] for path_item in path]
        child_mat[word_to_index[word], :len(path)] = [path_item[1] for path_item in path]
    return mask_mat, path_mat, child_mat


def hierarchical_softmax(context, W, b, mask_mat, path_mat, child_mat):
    # sparse matrices are not implemented on the gpu yet
    return T.sum(mask_mat.dimshuffle(('x', 0, 1)) * T.log(T.nnet.sigmoid(
        child_mat.dimshuffle(('x', 0, 1)) *
        (T.dot(W[path_mat], context.T).dimshuffle((2, 0, 1)) + b.dimshuffle((0, 'x')))
    )), axis=-1)


def hierarchical_softmax_cross_entropy(context, W, b, mask_mat, path_mat, child_mat, target):
    # sparse matrices are not implemented on the gpu yet
    return -T.sum(mask_mat[target].dimshuffle(('x', 0, 1)) * T.log(T.nnet.sigmoid(
        child_mat[target].dimshuffle(('x', 0, 1)) *
        (T.dot(W[path_mat[target]], context.T).dimshuffle((2, 0, 1)) + b[target].dimshuffle((0, 'x')))
    )), axis=-1)


def words_to_mat(words, step_size, word_to_index):
    pass


def run_model():
    pass


def main():
    pass

if __name__ == '__main__':
    main()
