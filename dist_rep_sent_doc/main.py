import datetime
from collections import Counter

import joblib
import lasagne
import numpy as np
import theano
import theano.tensor as T

from dist_rep_sent_doc.data import preprocess_data
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


def docs_to_mat(docs, window_size, word_to_index):
    words, target = [], []
    for i, doc in enumerate(docs):
        word_indexes = (window_size - 1) * [word_to_index['<null>']] + [word_to_index[word] for word in doc[1]]
        for j in range(len(doc[1])):
            cur_words = np.zeros(window_size, dtype=np.int32)
            cur_words[0] = len(word_to_index) + i
            cur_words[1:] = word_indexes[j:j + window_size - 1]
            words.append(cur_words)
            target.append(word_indexes[j + window_size - 1])
    return np.array(words), np.array(target)


@memory.cache
def gen_data(path, window_size):
    train, val, test = preprocess_data(path)

    # get huffman tree
    word_to_freq = Counter(word for docs in (train, val, test) for doc in docs for word in doc[1])
    vocab_min_freq = 0
    word_to_index = {'<null>': 0, '<unk>': 1}
    for word, count in word_to_freq.items():
        if count >= vocab_min_freq:
            word_to_index[word] = len(word_to_index)
    tree = build_huffman(word_to_freq)

    # convert data to index matrix
    return (
        docs_to_mat(train, window_size, word_to_index),
        docs_to_mat(val, window_size, word_to_index),
        docs_to_mat(test, window_size, word_to_index),
        tree, word_to_index
    )


def run_model(train, val, test, tree, word_to_index, window_size, embedding_size, batch_size, epoch_size):
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test

    # training network
    words_var = T.imatrix('words')
    l_in = lasagne.layers.InputLayer((None, window_size), words_var)
    l_emb = lasagne.layers.EmbeddingLayer(l_in, len(word_to_index) + len(train_y), embedding_size)
    l_flatten = lasagne.layers.FlattenLayer(l_emb)
    l_out = HierarchicalSoftmaxLayer(l_flatten, tree, word_to_index)

    # training outputs
    hs_nodes = T.ivector('hs_nodes')
    hs_signs = T.vector('hs_signs')
    hs_indexes = T.ivector('hs_indexes')
    network_output = lasagne.layers.get_output(l_out)
    cost = -lasagne.layers.get_output(l_out, hs_nodes=hs_nodes, hs_signs=hs_signs, hs_indexes=hs_indexes)
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.adadelta(cost, all_params)

    # functions
    train = theano.function([l_in.input_var, hs_nodes, hs_signs, hs_indexes], cost, updates=updates)
    compute_cost = theano.function([l_in.input_var, hs_nodes, hs_signs, hs_indexes], cost)

    for i in range(epoch_size):
        # generate minibatches
        p = np.random.permutation(len(train_y))

        # train
        train_X, train_y = train_X[p], train_y[p]
        for j in range(0, len(train_y), batch_size):
            batch_X, batch_y = train_X[j:j + batch_size], train_y[j:j + batch_size]
            cost = train(batch_X, *l_out.get_hs_inputs(batch_y)) / batch_size
            if j % 2560 == 0:
                print(datetime.datetime.now(), cost)


def main():
    run_model(
        *gen_data('../data/stanford_sentiment_treebank/class_5', window_size=8),
        window_size=8, embedding_size=400, batch_size=256, epoch_size=10
    )

if __name__ == '__main__':
    main()
