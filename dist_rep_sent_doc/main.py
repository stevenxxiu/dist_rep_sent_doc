import argparse
import csv
import datetime
import json
import os
import shlex
import sys
from collections import Counter
from multiprocessing import Process, Queue

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import LazyAdamOptimizer
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import init_ops

from dist_rep_sent_doc.data import imdb, sstb
from dist_rep_sent_doc.huffman import build_huffman, tree_to_arrays
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


@memory.cache(ignore=['docs'])
def gen_tables(name, docs, vocab_min_freq):
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
    tree = tree_to_arrays(build_huffman(word_to_freq), word_to_index, len(word_to_freq))

    return word_to_index, word_to_freq, tree


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
    saver.save(sess, os.path.join(path, 'model.ckpt'), write_meta_graph=False)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def freq_word_subsample(docs, word_to_index, word_to_freq, sample, epoch_size, qs):
    # remove infrequent words & sample frequent words
    probs = np.empty(len(word_to_freq))
    total_count = sum(word_to_freq.values())
    for word, count in word_to_freq.items():
        ratio = sample / (count / total_count) if count > 0 else 1
        probs[word_to_index[word]] = min(np.sqrt(ratio), 1)
    for i in range(epoch_size):
        indexes = []
        for j, doc in enumerate(docs):
            index = np.array([word_to_index[word] for word in doc[1] if word in word_to_index])
            indexes.append(index[np.random.binomial(1, p=probs[index]).astype(np.bool)])
        for q in qs:
            q.put(indexes)


# noinspection PyTypeChecker
def pvdm_sample(indexes, word_to_index, window_size, epoch_size, q):
    for i in range(epoch_size):
        X_doc_, X_words_, y_ = [], [], []
        for j, index in enumerate(indexes.get()):
            padded = np.pad(index, (window_size, 0), 'constant', constant_values=word_to_index['\0'])
            rolled = rolling_window(padded, window_size + 1)
            X_doc_.append(np.repeat(j, len(index)))
            y_.append(index)
            X_words_.append(np.delete(rolled, window_size, axis=1))
        q.put([np.concatenate(X_doc_), np.vstack(X_words_), np.concatenate(y_)])


def run_pvdm(
    docs, word_to_index, word_to_freq, tree, mode, window_size, embedding_size, lr,
    sample, batch_size, epoch_size, save_path, train_path=None
):
    # network
    tf.reset_default_graph()
    X_doc = tf.placeholder(tf.int32, [None])
    X_words = tf.placeholder(tf.int32, [None, window_size])
    y = tf.placeholder(tf.int32, [None])
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
    flatten = tf.reshape(emb, [-1, (window_size + 1) * embedding_size]) if mode == 'concat' else tf.reduce_mean(emb, 1)
    l = HierarchicalSoftmaxLayer(tree)
    loss = -l.apply([flatten, y], training=True)
    opt = LazyAdamOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss)
    if train_path:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == emb_doc]
    train_op = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if train_path:
            saver = tf.train.Saver({'emb_word': emb_word, 'hs_W': l.W})
            saver.restore(sess, os.path.join(train_path, 'model.ckpt'))

        # start sampling
        q_pvdm, q = Queue(1), Queue(1)
        Process(target=freq_word_subsample, args=(docs, word_to_index, word_to_freq, sample, epoch_size, [q])).start()
        Process(target=pvdm_sample, args=(q, word_to_index, window_size, epoch_size, q_pvdm)).start()

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            X_doc_, X_words_, y_ = q_pvdm.get()
            p = np.random.permutation(len(y_))
            total_loss = 0
            for j in range(0, len(y_), batch_size):
                k = p[j:j + batch_size]
                _, batch_loss = sess.run([train_op, loss], feed_dict={X_doc: X_doc_[k], X_words: X_words_[k], y: y_[k]})
                total_loss += batch_loss
            print(datetime.datetime.now(), f'finished epoch {i}, loss: {total_loss / len(y_):f}')

        # save
        os.makedirs(save_path, exist_ok=True)
        save_model(save_path, docs, word_to_index, word_to_freq, emb_doc, emb_word, l.W, sess)


# noinspection PyTypeChecker
def dbow_sample(indexes, epoch_size, q):
    for i in range(epoch_size):
        X_doc_, y_ = [], []
        for j, index in enumerate(indexes.get()):
            X_doc_.append(np.repeat(j, len(index)))
            y_.append(index)
        q.put([np.concatenate(X_doc_), np.concatenate(y_)])


# noinspection PyTypeChecker
def sg_sample(indexes, window_size, epoch_size, q):
    for i in range(epoch_size):
        X_doc_, X_words_, y_ = [], [], []
        for j, index in enumerate(indexes.get()):
            padded = np.pad(index, window_size, 'constant', constant_values=-1)
            rolled = rolling_window(padded, 2 * window_size + 1)
            rolled = np.ravel(np.delete(rolled, window_size, axis=1))
            mask = rolled != -1
            y_.append(np.repeat(index, 2 * window_size)[mask])
            X_words_.append(rolled[mask])
        q.put([np.concatenate(X_words_), np.concatenate(y_)])


def run_dbow(
    docs, word_to_index, word_to_freq, tree, sg, embedding_size, lr,
    sample, batch_size, epoch_size, save_path, train_path=None
):
    # network
    tf.reset_default_graph()
    X_doc = tf.placeholder(tf.int32, [None])
    X_words = tf.placeholder(tf.int32, [None])
    y_doc = tf.placeholder(tf.int32, [None])
    y_words = tf.placeholder(tf.int32, [None])
    emb_doc = tf.Variable(tf.random_uniform(
        [len(docs), embedding_size], -0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb_word = tf.Variable(tf.random_uniform(
        [len(word_to_index), embedding_size], -0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb_doc_lookup = tf.nn.embedding_lookup(emb_doc, X_doc)
    emb_word_lookup = tf.nn.embedding_lookup(emb_word, X_words)
    l = HierarchicalSoftmaxLayer(tree)
    loss = -(l.apply([emb_doc_lookup, y_doc], training=True) + l.apply([emb_word_lookup, y_words], training=True))
    opt = LazyAdamOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss)
    if train_path:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == emb_doc]
    train_op = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if train_path:
            saver = tf.train.Saver({'hs_W': l.W})
            saver.restore(sess, os.path.join(train_path, 'model.ckpt'))
            sg = False

        # start sampling
        q_dbow, q_sg, qs = Queue(1), Queue(1), [Queue(1), Queue(1)] if sg else [Queue(1)]
        Process(target=freq_word_subsample, args=(docs, word_to_index, word_to_freq, sample, epoch_size, qs)).start()
        Process(target=dbow_sample, args=(qs[0], epoch_size, q_dbow)).start()
        sg and Process(target=sg_sample, args=(qs[1], sg, epoch_size, q_sg)).start()

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            X_, y_ = q_dbow.get()
            mask_ = np.ones(len(X_), dtype=np.bool)
            if sg:
                X_words_, y_words_ = q_sg.get()
                X_ = np.concatenate([X_, X_words_])
                y_ = np.concatenate([y_, y_words_])
                mask_ = np.concatenate([mask_, np.zeros(len(y_words_), dtype=np.bool)])
            p = np.random.permutation(len(y_))
            total_loss = 0
            for j in range(0, len(y_), batch_size):
                k = p[j:j + batch_size]
                cur_X, cur_y = X_[k], y_[k]
                doc_mask = mask_[k]
                word_mask = np.logical_not(doc_mask)
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc: cur_X[doc_mask], y_doc: cur_y[doc_mask], X_words: cur_X[word_mask], y_words: cur_y[word_mask]
                })
                total_loss += batch_loss
            print(datetime.datetime.now(), f'finished epoch {i}, loss: {total_loss / len(y_):f}')

        # save
        os.makedirs(save_path, exist_ok=True)
        save_model(save_path, docs, word_to_index, word_to_freq, emb_doc, emb_word if sg else None, l.W, sess)


def load_nn_data(docs, paths):
    vars_ = [tf.Variable(0., validate_shape=False) for _ in paths]
    with tf.Session() as sess:
        for var, path in zip(vars_, paths):
            tf.train.Saver({'emb_doc': var}).restore(sess, os.path.join(path, 'model.ckpt'))
        i = [i for i, doc in enumerate(docs) if doc[0] is not None]
        return sess.run(tf.concat(vars_, 1))[i], np.array([doc[0] for doc in docs])[i]


# noinspection PyTypeChecker
def run_nn(X_train, y_train, X_test, y_test, layer_sizes, lr, batch_size, epoch_size):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    y = tf.placeholder(tf.int32, [None])
    dense = X
    for i, layer_size in enumerate(layer_sizes):
        dense = tf.layers.dense(
            dense, layer_size,
            activation=tf.nn.relu if i < len(layer_sizes) else None,
            kernel_initializer=init_ops.glorot_uniform_initializer()
        )
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            p = np.random.permutation(len(y_train))
            total_loss = 0
            for j in range(0, len(y_train), batch_size):
                k = p[j:j + batch_size]
                _, batch_loss = sess.run([train_op, loss], feed_dict={X: X_train[k], y: y_train[k]})
                total_loss += len(k) * batch_loss
            y_pred = sess.run(tf.argmax(dense, axis=1), feed_dict={X: X_test})
            corrects = sum(y_pred == y_test)
            print(
                datetime.datetime.now(),
                f'finished epoch {i}, '
                f'loss: {total_loss / len(y_train):f}, '
                f'error rate: {(len(y_pred) - corrects) / len(y_pred):f}'
            )


def main():
    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset', choices=('imdb', 'sstb_2', 'sstb_5'))
    arg_parser.add_argument('val_test', choices=('val', 'test'))
    arg_parser.add_argument('method', choices=('pvdm', 'dbow', 'nn'))
    arg_parser.add_argument('hyperparams')
    args = arg_parser.parse_args()
    hyperparams = json.loads(args.hyperparams)
    train, val, test = \
        imdb.load_data('../data/imdb_sentiment') if args.dataset == 'imdb' else \
        sstb.load_data('../data/stanford_sentiment_treebank/class_2') if args.dataset == 'sstb_2' else \
        sstb.load_data('../data/stanford_sentiment_treebank/class_5')
    val_test = {'val': val, 'test': test}[args.val_test]
    if args.dataset == 'imdb' and args.val_test == 'test':
        train.extend(val)
    if args.method == 'pvdm':
        tables = gen_tables((args.dataset, args.val_test), train, hyperparams.pop('min_freq'))
        run_pvdm(train if 'train_path' not in hyperparams else val_test, *tables, **hyperparams)
    elif args.method == 'dbow':
        tables = gen_tables((args.dataset, args.val_test), train, hyperparams.pop('min_freq'))
        run_dbow(train if 'train_path' not in hyperparams else val_test, *tables, **hyperparams)
    else:
        run_nn(
            *load_nn_data(train, hyperparams.pop('train_paths')),
            *load_nn_data(val_test, hyperparams.pop('test_paths')),
            **hyperparams
        )

if __name__ == '__main__':
    main()
