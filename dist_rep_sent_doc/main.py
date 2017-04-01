import csv
import datetime
import os
import uuid
from collections import Counter

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops

from dist_rep_sent_doc.data import sstb
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


def docs_to_mat(docs, window_size, word_to_index):
    doc, words, target = [], [], []
    for i, doc_ in enumerate(docs):
        word_indexes = (window_size - 1) * [word_to_index['<null>']] + [word_to_index[word] for word in doc_[1]]
        for j in range(len(doc_[1])):
            doc.append(i)
            words.append(word_indexes[j:j + window_size - 1])
            target.append(word_indexes[j + window_size - 1])
    return np.array(doc), np.array(words), np.array(target)


@memory.cache
def gen_data(path, window_size):
    train, val, test = sstb.load_data(path)

    # get huffman tree
    word_to_freq = Counter(word for docs in (train, val, test) for doc in docs for word in doc[1])
    vocab_min_freq = 0
    word_to_index = {'<null>': 0, '<unk>': 1}
    for word, count in word_to_freq.items():
        if count >= vocab_min_freq:
            word_to_index[word] = len(word_to_index)
    word_to_freq['<null>'] = word_to_freq['<unk>'] = 0
    tree = build_huffman(word_to_freq)

    # convert data to index matrix
    return (
        train, docs_to_mat(train, window_size, word_to_index),
        val, docs_to_mat(val, window_size, word_to_index),
        test, docs_to_mat(test, window_size, word_to_index),
        tree, word_to_index, word_to_freq
    )


def save_model(path, docs, word_to_index, word_to_freq, emb_doc, emb_word, hs_vars, sess):
    # visualize embeddings
    config = projector.ProjectorConfig()
    for emb_name, emb in [('emb_doc', emb_doc), ('emb_word', emb_word)]:
        emb_conf = config.embeddings.add()
        emb_conf.tensor_name = emb_name
        emb_conf.metadata_path = os.path.abspath(os.path.join(path, f'{emb_name}_metadata.tsv'))
        with open(emb_conf.metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['Word', 'Frequency'])
            if emb == emb_doc:
                for doc in docs:
                    writer.writerow([' '.join(doc[1]), 1])
            elif emb == emb_word:
                words = len(word_to_index) * [None]
                for word, i in word_to_index.items():
                    words[i] = word
                for word in words:
                    writer.writerow([word, word_to_freq[word]])
    summary_writer = tf.summary.FileWriter(path)
    projector.visualize_embeddings(summary_writer, config)

    # save model
    saver = tf.train.Saver({'emb_word': emb_word, 'emb_doc': emb_doc, 'hs_W': hs_vars[0], 'hs_b': hs_vars[1]})
    saver.save(sess, os.path.join(path, 'model.ckpt'))


@memory.cache(ignore=['docs', 'mats', 'tree', 'word_to_index', 'word_to_freq'])
def run_pv_dm(
    name, docs, mats, tree, word_to_index, word_to_freq, training_, window_size, embedding_size, lr, batch_size,
    epoch_size, train_model_path=None
):
    mat_X_doc, mat_X_words, mat_y = mats

    # network
    X_doc = tf.placeholder(tf.int32, [None])
    X_words = tf.placeholder(tf.int32, [None, window_size - 1])
    emb_doc = tf.Variable(tf.random_normal([len(docs), embedding_size]))
    emb_word = tf.Variable(tf.random_normal([len(word_to_index), embedding_size]))
    emb = tf.concat([
        tf.reshape(tf.nn.embedding_lookup(emb_doc, X_doc), [-1, 1, embedding_size]),
        tf.nn.embedding_lookup(emb_word, X_words)
    ], 1)
    flatten = tf.reshape(emb, [-1, window_size * embedding_size])
    l = HierarchicalSoftmaxLayer(tree, word_to_index, name='hs')
    loss = -l.apply(flatten, training=True)
    hs_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hs')
    opt = tf.train.AdadeltaOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss)
    if training_:
        # don't use sparse gradients in hierarchical softmax to run them on the gpu
        for i, (grad, var) in enumerate(grads_and_vars):
            if var in hs_vars:
                grads_and_vars[i] = (ops.convert_to_tensor(grad), var)
    else:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == emb_doc]
    train = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if not training_:
            saver = tf.train.Saver({'emb_word': emb_word, 'hs_W': hs_vars[0], 'hs_b': hs_vars[1]})
            saver.restore(sess, os.path.join(train_model_path, 'model.ckpt'))

        # train
        for i in range(epoch_size):
            p = np.random.permutation(len(mat_y))
            mat_X_doc_, mat_X_words_, mat_y_ = mat_X_doc[p], mat_X_words[p], mat_y[p]
            for j in range(0, len(mat_y), batch_size):
                batch_X_doc, batch_X_words, batch_y = \
                    mat_X_doc_[j:j + batch_size], mat_X_words_[j:j + batch_size], mat_y_[j:j + batch_size]
                feed_dict = {X_doc: batch_X_doc, X_words: batch_X_words, **l.get_hs_inputs(batch_y)}
                sess.run(train, feed_dict=feed_dict)
                if j % 256000 == 0:
                    print(datetime.datetime.now(), j, sess.run(loss, feed_dict=feed_dict))

        # save
        path = os.path.join('__cache__', 'tf', f'{name}-{uuid.uuid4()}')
        os.makedirs(path)
        save_model(path, docs, word_to_index, word_to_freq, emb_doc, emb_word, hs_vars, sess)
        return path


def run_log_reg(train_docs, val_docs, pv_dm_train_path, pv_dm_val_path, embedding_size, batch_size, epoch_size):
    X = tf.placeholder(tf.float32, [None, embedding_size])
    y = tf.placeholder(tf.int32, [None])
    dense = tf.layers.dense(X, 5, kernel_initializer=init_ops.glorot_uniform_initializer())
    pred = tf.argmax(dense, 1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense, labels=y))
    train = tf.train.AdadeltaOptimizer(1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pv_dm_train = tf.Variable(tf.zeros([len(train_docs), embedding_size]))
        pv_dm_val = tf.Variable(tf.zeros([len(val_docs), embedding_size]))
        tf.train.Saver({'emb_doc': pv_dm_train}).restore(sess, os.path.join(pv_dm_train_path, 'model.ckpt'))
        tf.train.Saver({'emb_doc': pv_dm_val}).restore(sess, os.path.join(pv_dm_val_path, 'model.ckpt'))

        train_X = pv_dm_train.eval(sess)
        train_y = np.array([doc[0] for doc in train_docs])
        for i in range(epoch_size):
            p = np.random.permutation(len(train_y))
            train_X_, train_y_ = train_X[p], train_y[p]
            for j in range(0, len(train_y), batch_size):
                batch_X, batch_y = train_X_[j:j + batch_size], train_y_[j:j + batch_size]
                feed_dict = {X: batch_X, y: batch_y}
                sess.run(train, feed_dict={X: batch_X, y: batch_y})
                if j == 0:
                    print(datetime.datetime.now(), j, sess.run(loss, feed_dict=feed_dict))

        print(Counter(sess.run(pred, {X: train_X})))
        # print(sess.run(pred, {X: pv_dm_val}))


def main():
    train_docs, train_mats, val_docs, val_mats, test_docs, test_mats, tree, word_to_index, word_to_freq = \
        gen_data('../data/stanford_sentiment_treebank/class_5', window_size=8)

    # pv dm
    pv_dm_train_path = run_pv_dm(
        'train_5', train_docs, train_mats, tree, word_to_index, word_to_freq, training_=True, window_size=8,
        embedding_size=400, lr=10, batch_size=256, epoch_size=5
    )
    pv_dm_val_path = run_pv_dm(
        'val_5', val_docs, val_mats, tree, word_to_index, word_to_freq, training_=False, window_size=8,
        embedding_size=400, lr=1000, batch_size=256, epoch_size=100, train_model_path=pv_dm_train_path
    )

    # log reg
    run_log_reg(
        train_docs, val_docs, pv_dm_train_path, pv_dm_val_path, embedding_size=400, batch_size=256, epoch_size=5
    )


if __name__ == '__main__':
    main()
