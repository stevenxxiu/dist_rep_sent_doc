import csv
import datetime
import os
import uuid
from collections import Counter

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.tensorboard.plugins import projector

from dist_rep_sent_doc.data import imdb, sstb
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


@memory.cache(ignore=['docs'])
def gen_tables(name, docs, vocab_min_freq, sample):
    # map word to counts and indexes, infrequent words are removed entirely
    word_to_freq = {}
    for word, count in Counter(word for doc in docs for word in doc[1]).items():
        if count >= vocab_min_freq:
            word_to_freq[word] = count
    word_to_index = {}
    for word in word_to_freq:
        word_to_index[word] = len(word_to_index)
    word_to_index['\0'] = len(word_to_index)

    # get huffman tree
    tree = build_huffman(word_to_freq)

    # get word sub-sampling probabilities
    word_to_prob = {}
    total_count = sum(word_to_freq.values())
    for word, count in word_to_freq.items():
        ratio = sample / (count / total_count) if count > 0 else 1
        word_to_prob[word] = min(np.sqrt(ratio), 1)

    return word_to_freq, word_to_index, tree, word_to_prob


def save_model(path, docs, word_to_index, word_to_freq, emb_doc, emb_word, hs_W, sess):
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
            elif emb_word and emb == emb_word:
                words = len(word_to_index) * [None]
                for word, i in word_to_index.items():
                    words[i] = word
                for word in words:
                    writer.writerow(['\\0', 0] if word == '\0' else [word, word_to_freq[word]])
    summary_writer = tf.summary.FileWriter(path)
    projector.visualize_embeddings(summary_writer, config)

    # save model
    saver = tf.train.Saver(
        {'emb_word': emb_word, 'emb_doc': emb_doc, 'hs_W': hs_W} if emb_word else
        {'emb_doc': emb_doc, 'hs_W': hs_W}
    )
    saver.save(sess, os.path.join(path, 'model.ckpt'))


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# noinspection PyTypeChecker
def run_pvdm(
    name, docs, word_to_freq, word_to_index, tree, word_to_prob, training_, window_size, embedding_size, start_lr, end_lr,
    batch_size, epoch_size, train_model_path=None
):
    # network
    X_doc = tf.placeholder(tf.int32, [None])
    X_words = tf.placeholder(tf.int32, [None, 2 * window_size])
    y = tf.placeholder(tf.int32, [None])
    lr = tf.placeholder(tf.float32, [])
    emb_doc = tf.Variable(tf.random_uniform(
        [len(docs), embedding_size], -0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb_word = tf.Variable(tf.random_uniform(
        [len(word_to_index), embedding_size], -0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb = tf.concat([
        tf.reshape(tf.nn.embedding_lookup(emb_doc, X_doc), [-1, 1, embedding_size]),
        tf.nn.embedding_lookup(emb_word, X_words)
    ], 1)
    flatten = tf.reshape(emb, [-1, (2 * window_size + 1) * embedding_size])
    l = HierarchicalSoftmaxLayer(tree, word_to_index)
    loss = -l.apply([flatten, y], training=True)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss)
    if not training_:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == emb_doc]
    grads_and_vars = [
        (tf.IndexedSlices(tf.clip_by_value(grad.values, -10, 10), grad.indices), var)
        for grad, var in grads_and_vars
    ]
    train_op = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if not training_:
            saver = tf.train.Saver({'emb_word': emb_word, 'hs_W': l.W})
            saver.restore(sess, os.path.join(train_model_path, 'model.ckpt'))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            X_doc_, X_words_, y_ = [], [], []
            for j, doc in enumerate(docs):
                # remove infrequent words & sample frequent words
                index = np.array([word_to_index[word] for word in doc[1] if word in word_to_index])
                probs = np.array([word_to_prob[word] for word in doc[1] if word in word_to_index])
                index = index[np.random.binomial(1, p=probs).astype(np.bool)]
                padded = np.pad(index, window_size, 'constant', constant_values=word_to_index['\0'])
                rolled = rolling_window(padded, 2 * window_size + 1)
                X_doc_.append(np.repeat(j, len(index)))
                y_.append(index)
                X_words_.append(np.delete(rolled, window_size, axis=1))
            X_doc_ = np.concatenate(X_doc_)
            X_words_ = np.vstack(X_words_)
            y_ = np.concatenate(y_)
            p = np.random.permutation(len(y_))
            cur_lr = start_lr - (start_lr - end_lr) / (epoch_size - 1) * i
            for j in range(0, len(y_), batch_size):
                k = p[j:j + batch_size]
                sess.run(train_op, feed_dict={X_doc: X_doc_[k], X_words: X_words_[k], y: y_[k], lr: cur_lr})
            print(datetime.datetime.now(), f'finished epoch {i}')

        # save
        path = os.path.join('__cache__', 'tf', f'pvdm-{name}-{uuid.uuid4()}')
        os.makedirs(path)
        save_model(path, docs, word_to_index, word_to_freq, emb_doc, emb_word, l.W, sess)
        return path


# noinspection PyTypeChecker
def run_dbow(
    name, docs, word_to_freq, word_to_index, tree, word_to_prob, training_, embedding_size, start_lr, end_lr,
    batch_size, epoch_size, train_model_path=None
):
    # network
    X_doc = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.int32, [None])
    lr = tf.placeholder(tf.float32, [])
    emb_doc = tf.Variable(tf.random_uniform(
        [len(docs), embedding_size], -0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb = tf.nn.embedding_lookup(emb_doc, X_doc)
    l = HierarchicalSoftmaxLayer(tree, word_to_index)
    loss = -l.apply([emb, y], training=True)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss)
    if not training_:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == emb_doc]
    grads_and_vars = [
        (tf.IndexedSlices(tf.clip_by_value(grad.values, -10, 10), grad.indices), var)
        for grad, var in grads_and_vars
    ]
    train_op = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if not training_:
            saver = tf.train.Saver({'hs_W': l.W})
            saver.restore(sess, os.path.join(train_model_path, 'model.ckpt'))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            X_doc_, y_ = [], []
            for j, doc in enumerate(docs):
                # remove infrequent words
                index = np.array([word_to_index[word] for word in doc[1] if word in word_to_index])
                X_doc_.append(np.repeat(j, len(index)))
                y_.append(index)
            X_doc_ = np.concatenate(X_doc_)
            y_ = np.concatenate(y_)
            p = np.random.permutation(len(y_))
            cur_lr = start_lr - (start_lr - end_lr) / (epoch_size - 1) * i
            for j in range(0, len(y_), batch_size):
                k = p[j:j + batch_size]
                sess.run(train_op, feed_dict={X_doc: X_doc_[k], y: y_[k], lr: cur_lr})
            print(datetime.datetime.now(), f'finished epoch {i}')

        # save
        path = os.path.join('__cache__', 'tf', f'dbow-{name}-{uuid.uuid4()}')
        os.makedirs(path)
        save_model(path, docs, word_to_index, word_to_freq, emb_doc, None, l.W, sess)
        return path


def load_docvecs(train, test, pvdm_train_path, pvdm_test_path, dbow_train_path, dbow_test_path):
    pvdm_train, pvdm_test = tf.Variable(0., validate_shape=False), tf.Variable(0., validate_shape=False)
    dbow_train, dbow_test = tf.Variable(0., validate_shape=False), tf.Variable(0., validate_shape=False)
    with tf.Session() as sess:
        tf.train.Saver({'emb_doc': pvdm_train}).restore(sess, os.path.join(pvdm_train_path, 'model.ckpt'))
        tf.train.Saver({'emb_doc': pvdm_test}).restore(sess, os.path.join(pvdm_test_path, 'model.ckpt'))
        tf.train.Saver({'emb_doc': dbow_train}).restore(sess, os.path.join(dbow_train_path, 'model.ckpt'))
        tf.train.Saver({'emb_doc': dbow_test}).restore(sess, os.path.join(dbow_test_path, 'model.ckpt'))
        i = [i for i, doc in enumerate(train) if doc[0] is not None]
        X_train = sess.run(tf.concat([pvdm_train, dbow_train], 1))[i]
        y_train = np.array([doc[0] for doc in train])[i]
        X_test = sess.run(tf.concat([pvdm_test, dbow_test], 1))
        y_test = np.array([doc[0] for doc in test])
        return X_train, y_train, X_test, y_test


# noinspection PyTypeChecker
def run_nn(X_train, y_train, X_test, y_test, embedding_size, layer_sizes, start_lr, end_lr, batch_size, epoch_size):
    X = tf.placeholder(tf.float32, [None, embedding_size])
    y = tf.placeholder(tf.int32, [None])
    lr = tf.placeholder(tf.int32, [])
    dense = X
    for layer_size in layer_sizes:
        dense = tf.layers.dense(dense, layer_size, kernel_initializer=init_ops.glorot_uniform_initializer())
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense, labels=y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch_size):
            p = np.random.permutation(len(y_train))
            total_loss = 0
            cur_lr = start_lr - (start_lr - end_lr) / (epoch_size - 1) * i
            for j in range(0, len(y_train), batch_size):
                k = p[j:j + batch_size]
                _, batch_loss = sess.run([train_op, loss], feed_dict={X: X_train[k], y: y_train[k], lr: cur_lr})
                total_loss += len(k) * batch_loss
            print(datetime.datetime.now(), f'finished epoch {i}, loss: {total_loss / len(y_train):f}')
        y_pred = sess.run(tf.argmax(dense, axis=1), feed_dict={X: X_test})
        corrects = sum(y_pred == y_test)
        print(f'error rate: {(len(y_pred) - corrects) / len(y_pred)}')


# noinspection PyTypeChecker
def main():
    name = 'imdb'
    if name == 'imdb':
        train, val, test = imdb.load_data('../data/imdb_sentiment')
        tables = gen_tables(name, train, 2, 1e-3)

        # pvdm
        pvdm_train_path = run_pvdm(
            f'{name}_train', train, *tables, training_=True,
            window_size=5, embedding_size=100, start_lr=0.025, end_lr=0.001, batch_size=512, epoch_size=20
        )
        pvdm_test_path = run_pvdm(
            f'{name}_test', test, *tables, training_=False,
            window_size=5, embedding_size=100, start_lr=0.1, end_lr=0.0001, batch_size=2048, epoch_size=5,
            train_model_path=pvdm_train_path
        )

        # dbow
        dbow_train_path = run_dbow(
            f'{name}_train', train, *tables, training_=True,
            embedding_size=100, start_lr=0.025, end_lr=0.001, batch_size=512, epoch_size=20
        )
        dbow_test_path = run_dbow(
            f'{name}_test', test, *tables, training_=False,
            embedding_size=100, start_lr=0.1, end_lr=0.0001, batch_size=2048, epoch_size=5,
            train_model_path=dbow_train_path
        )

        # classify
        run_nn(
            *load_docvecs(train, test, pvdm_train_path, pvdm_test_path, dbow_train_path, dbow_test_path),
            embedding_size=200, layer_sizes=[2], start_lr=5, end_lr=0.1, batch_size=2048, epoch_size=20
        )

    elif name in ('sstb_2', 'sstb_5'):
        train, val, test = sstb.load_data(
            '../data/stanford_sentiment_treebank/class_2' if name == 'sstb_2' else
            '../data/stanford_sentiment_treebank/class_5'
        )
        tables = gen_tables(name, train, 2, 1e-5)

        # pvdm
        pvdm_train_path = run_pvdm(
            f'{name}_train', train, *tables, training_=True,
            window_size=4, embedding_size=100, start_lr=0.025, end_lr=0.001, batch_size=512, epoch_size=20
        )
        pvdm_test_path = run_pvdm(
            f'{name}_test', test, *tables, training_=False,
            window_size=4, embedding_size=100, start_lr=0.1, end_lr=0.0001, batch_size=2048, epoch_size=5,
            train_model_path=pvdm_train_path
        )

        # dbow
        dbow_train_path = run_dbow(
            f'{name}_train', train, *tables, training_=True,
            embedding_size=100, start_lr=0.025, end_lr=0.001, batch_size=512, epoch_size=20
        )
        dbow_test_path = run_dbow(
            f'{name}_test', test, *tables, training_=False,
            embedding_size=100, start_lr=0.1, end_lr=0.0001, batch_size=2048, epoch_size=5,
            train_model_path=dbow_train_path
        )

        # classify
        run_nn(
            *load_docvecs(train, test, pvdm_train_path, pvdm_test_path, dbow_train_path, dbow_test_path),
            embedding_size=200, layer_sizes=[(2 if name == 'sstb_2' else 5)],
            start_lr=5, end_lr=0.1, batch_size=2048, epoch_size=10
        )

if __name__ == '__main__':
    main()
