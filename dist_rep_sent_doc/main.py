import datetime
import os
import uuid
from collections import Counter

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from dist_rep_sent_doc.data import preprocess_data
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


def docs_to_mat(docs, window_size, word_to_index):
    doc, words, target = [], [], []
    for i, doc_ in enumerate(docs):
        word_indexes = (window_size - 1) * [word_to_index['<null>']] + [word_to_index[word] for word in doc_[1]]
        for j in range(len(doc_[1])):
            doc.append([i])
            words.append(word_indexes[j:j + window_size - 1])
            target.append(word_indexes[j + window_size - 1])
    return len(docs), np.array(doc), np.array(words), np.array(target)


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


@memory.cache(ignore=['data'])
def run_pv_dm(
    name, data, training_, tree, word_to_index, window_size, embedding_size, batch_size, epoch_size,
    train_model_name=None
):
    data_n_docs, data_X_docs, data_X_words, data_y = data

    # network
    X_docs = tf.placeholder(tf.int32, [None, 1])
    X_words = tf.placeholder(tf.int32, [None, window_size - 1])
    doc_emb = tf.Variable(tf.random_normal([data_n_docs, embedding_size]))
    word_emb = tf.Variable(tf.random_normal([len(word_to_index), embedding_size]))
    emb = tf.concat([tf.nn.embedding_lookup(doc_emb, X_docs), tf.nn.embedding_lookup(word_emb, X_words)], 1)
    flatten = tf.reshape(emb, [-1, window_size * embedding_size])
    l = HierarchicalSoftmaxLayer(tree, word_to_index, name='hs')
    cost = -l.apply(flatten, training=True)
    hs_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hs')
    opt = tf.train.AdadeltaOptimizer(1.0)
    grads_and_vars = opt.compute_gradients(cost)
    if training_:
        # don't use sparse gradients in hierarchical softmax to run them on the gpu
        for i, (grad, var) in enumerate(grads_and_vars):
            if var in hs_vars:
                grads_and_vars[i] = (ops.convert_to_tensor(grad), var)
    else:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == doc_emb]
    train = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if not training_:
            saver = tf.train.import_meta_graph(f'{train_model_name}.meta')
            saver.restore(sess, train_model_name)

        # train
        for i in range(epoch_size):
            p = np.random.permutation(len(data_y))
            data_X_docs, data_X_words, data_y = data_X_docs[p], data_X_words[p], data_y[p]
            for j in range(0, len(data_y), batch_size):
                batch_X_docs, batch_X_words, batch_y = \
                    data_X_docs[j:j + batch_size], data_X_words[j:j + batch_size], data_y[j:j + batch_size]
                feed_dict = {X_docs: batch_X_docs, X_words: batch_X_words, **l.get_hs_inputs(batch_y)}
                sess.run(train, feed_dict=feed_dict)
                if j % 256000 == 0:
                    print(datetime.datetime.now(), j, sess.run(cost, feed_dict=feed_dict))

        # save model
        name = os.path.join('__cache__', str(uuid.uuid4()))
        saver = tf.train.Saver([word_emb, doc_emb, *hs_vars])
        saver.save(sess, name)
        return name


def main():
    train, val, test, tree, word_to_index = gen_data('../data/stanford_sentiment_treebank/class_5', window_size=8)
    pv_dm_train_name = run_pv_dm(
        'train_5', train, training_=True, tree=tree, word_to_index=word_to_index, window_size=8, embedding_size=400,
        batch_size=256, epoch_size=10
    )
    pv_dm_val_name = run_pv_dm(
        'val_5', val, training_=False, tree=tree, word_to_index=word_to_index, window_size=8, embedding_size=400,
        batch_size=256, epoch_size=10, train_model_name=pv_dm_train_name
    )

if __name__ == '__main__':
    main()
