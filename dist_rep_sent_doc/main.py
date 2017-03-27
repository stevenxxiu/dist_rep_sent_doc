import datetime
from collections import Counter

import joblib
import numpy as np

import tensorflow as tf
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

    # training data
    X = tf.placeholder(tf.int32, [None, window_size])
    training = tf.placeholder(tf.bool, [])
    emb = tf.nn.embedding_lookup(tf.Variable(tf.random_normal(
        [len(word_to_index) + len(train_X), embedding_size], stddev=0.01)
    ), X)
    flatten = tf.reshape(emb, [-1, window_size * embedding_size])
    l = HierarchicalSoftmaxLayer(tree, word_to_index)
    cost = -l.apply(flatten, training=training)
    tf.train.AdadeltaOptimizer._apply_sparse = tf.train.AdadeltaOptimizer._apply_dense
    train = tf.train.AdadeltaOptimizer(1.0).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch_size):
            # generate minibatches
            p = np.random.permutation(len(train_y))

            # train
            train_X, train_y = train_X[p], train_y[p]
            for j in range(0, len(train_y), batch_size):
                batch_X, batch_y = train_X[j:j + batch_size], train_y[j:j + batch_size]
                sess.run(train, feed_dict={X: batch_X, training: True, **l.get_hs_inputs(batch_y)})
                if j % 2560 == 0:
                    print(datetime.datetime.now())


def main():
    run_model(
        *gen_data('../data/stanford_sentiment_treebank/class_5', window_size=8),
        window_size=8, embedding_size=400, batch_size=256, epoch_size=10
    )

if __name__ == '__main__':
    main()
