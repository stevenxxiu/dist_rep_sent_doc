from collections import Counter

import lasagne
import numpy as np
import theano.tensor as T

from dist_rep_sent_doc.data import preprocess_data
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer


def docs_to_mat(docs, window_size, word_to_index):
    mask, words, target = [], [], []
    for i, doc in enumerate(docs):
        word_indexes = [word_to_index[word] for word in doc[1]]
        for j in range(len(word_indexes)):
            cur_mask = np.zeros(window_size + 1, dtype=np.int32)
            cur_mask[0] = 1
            cur_mask[window_size - j + 1:] = 1
            mask.append(cur_mask)
            cur_words = np.zeros(window_size + 1, dtype=np.int32)
            cur_words[0] = len(word_to_index) + i
            cur_words[max(window_size - j, 0) + 1:] = word_indexes[max(j - window_size, 0):j]
            words.append(cur_words)
            target.append(word_indexes[j])
    return np.array(mask), np.array(words), np.array(target)


def run_model(train, val, test, window_size, embedding_size):
    # get huffman tree
    word_to_freq = Counter(word for docs in (train, val, test) for doc in docs for word in doc[1])
    vocab_min_freq = 0
    word_to_index = {'<unk>': 0}
    for word, count in word_to_freq.items():
        if count >= vocab_min_freq:
            word_to_index[word] = len(word_to_index)
    tree = build_huffman(word_to_freq)

    # convert data to index matrix
    train_mask, train_words, train_target = docs_to_mat(train, window_size, word_to_index)
    val_mask, val_words, val_target = docs_to_mat(val, window_size, word_to_index)
    test_mask, test_words, test_target = docs_to_mat(test, window_size, word_to_index)

    # training network
    mask_var = T.imatrix('mask')
    words_var = T.imatrix('words')
    l_words_in = lasagne.layers.InputLayer((None, window_size + 1), words_var)
    l_emb = lasagne.layers.EmbeddingLayer(l_words_in, len(word_to_index) + len(train), embedding_size)
    l_masked = lasagne.layers.ExpressionLayer(l_emb, lambda x: mask_var * x)
    l_concat = lasagne.layers.ConcatLayer(l_masked)
    l_out = HierarchicalSoftmaxLayer(l_concat, tree, word_to_index)

    # training outputs


def main():
    run_model(*preprocess_data('../data/stanford_sentiment_treebank/class_5'), 8, 400)

if __name__ == '__main__':
    main()
