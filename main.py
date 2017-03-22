import heapq
from collections import Counter

import numpy as np
import theano.tensor as T

from data import preprocess_data


class HuffmanNode:
    def __init__(self, id_, freq, left=None, right=None):
        self.id_ = id_
        self.freq = freq
        self.left = left
        self.right = right

    def __eq__(self, other):
        return self.freq == other.freq and self.id_ == other.id_ and \
               self.left == other.left and self.right == other.right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman(word_to_freq):
    queue = [HuffmanNode(word, freq) for word, freq in word_to_freq.items()]
    heapq.heapify(queue)
    while len(queue) > 1:
        children = [heapq.heappop(queue), heapq.heappop(queue)]
        heapq.heappush(queue, HuffmanNode(len(queue), children[0].freq + children[1].freq, *children))
    return queue[0]


def tree_to_paths(tree):
    '''
    :return: A dictionary mapping words in the tree to it's path.
    '''
    if tree.left:
        return {
            word: [(tree.id_, -1 if i == 0 else 1)] + path
            for i, child in enumerate([tree.left, tree.right])
            for word, path in tree_to_paths(child).items()
        }
    else:
        return {tree.id_: []}


def tree_to_mat(tree, word_to_index):
    '''
    Converts a huffman tree to:
    - A mask matrix where each row is a binary mask for the other matrices.
    - A path matrix where each row contains the path from the root.
    - A child matrix where -1 and 1 represent the left and right child respectively.
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
    train, val, test = preprocess_data('../data/stanford_sentiment_treebank/class_5')
    word_to_freq = Counter(word for docs in (train, val, test) for doc in docs for word in doc[1])
    vocab_min_freq = 0
    word_to_index = {'<unk>': 0}
    for word, count in word_to_freq.items():
        if count >= vocab_min_freq:
            word_to_index[word] = len(word_to_index)
    mask_mat, path_mat, child_mat = tree_to_mat(build_huffman(word_to_freq), word_to_index)

if __name__ == '__main__':
    main()
