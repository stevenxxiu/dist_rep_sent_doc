import csv
import datetime
import os
import threading
import uuid
from collections import Counter

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import init_ops

from dist_rep_sent_doc.data import imdb, sstb
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


@memory.cache(ignore=['docsets'])
def gen_tables(name, docsets, vocab_min_freq, sample):
    total_count = 0

    # map word to counts and indexes, infrequent words are removed entirely
    word_to_freq = {'<null>': 0}
    for word, count in Counter(word for docs in docsets for doc in docs for word in doc[1]).items():
        if count >= vocab_min_freq:
            word_to_freq[word] = count
            total_count += count
    word_to_index = {word: i for i, word in enumerate(word_to_freq)}

    # get huffman tree
    tree = build_huffman(word_to_freq)

    # get word sub-sampling probabilities
    word_to_prob = {}
    for word, count in word_to_freq.items():
        # gensim modification, add ratio after square rooting it
        ratio = sample / (count / total_count) if count > 0 else 1
        word_to_prob[word] = min(np.sqrt(ratio) + ratio, 1)

    return word_to_freq, word_to_index, tree, word_to_prob


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


def load_and_enqueue(
    docs, word_to_prob, word_to_index, window_size, epoch_size, sess, queue, X_doc_input, X_words_input, y_input, data
):
    # includes all epochs to work around https://github.com/tensorflow/tensorflow/issues/4535
    enqueue_many = queue.enqueue_many([X_doc_input, X_words_input, y_input])
    for i in range(epoch_size):
        data['cur_epoch'] = i
        p = np.random.permutation(len(docs))
        for j in p:
            # remove infrequent words & sample frequent words
            X_doc, X_words, y = [], [], []
            doc = [word for word in docs[j][1] if word in word_to_index and np.random.random() < word_to_prob[word]]
            for k, word in enumerate(doc):
                window = doc[k - window_size + 1:k]
                window = ['<null>'] * (window_size - len(window) - 1) + window
                X_doc.append(j)
                X_words.append([word_to_index[word_] for word_ in window])
                y.append(word_to_index[word])
            sess.run(enqueue_many, feed_dict={X_doc_input: X_doc, X_words_input: X_words, y_input: y})
    data['cur_epoch'] = epoch_size


@memory.cache(ignore=['docs', 'word_to_freq', 'word_to_index', 'tree', 'word_to_prob'])
def run_pv_dm(
    name, docs, word_to_freq, word_to_index, tree, word_to_prob, training_, window_size, embedding_size, cur_lr,
    batch_size, epoch_size, train_model_path=None
):
    # queue
    X_doc_input = tf.placeholder(tf.int32, [None])
    X_words_input = tf.placeholder(tf.int32, [None, window_size - 1])
    y_input = tf.placeholder(tf.int32, [None])
    vars_ = [X_doc_input, X_words_input, y_input]
    queue = tf.FIFOQueue(2 * batch_size, [var.dtype for var in vars_], shapes=[var.shape[1:] for var in vars_])

    # network
    X_doc, X_words, y = queue.dequeue_up_to(batch_size)
    lr = tf.placeholder(tf.float32, [])
    emb_doc = tf.Variable(tf.random_normal([len(docs), embedding_size]))
    emb_word = tf.Variable(tf.random_normal([len(word_to_index), embedding_size]))
    emb = tf.concat([
        tf.reshape(tf.nn.embedding_lookup(emb_doc, X_doc), [-1, 1, embedding_size]),
        tf.nn.embedding_lookup(emb_word, X_words)
    ], 1)
    flatten = tf.reshape(emb, [-1, window_size * embedding_size])
    l = HierarchicalSoftmaxLayer(tree, word_to_index, name='hs')
    loss = -l.apply([flatten, y], training=True)
    hs_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hs')
    opt = tf.train.GradientDescentOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss)
    if not training_:
        # only train document embeddings
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var == emb_doc]
    train_op = opt.apply_gradients(grads_and_vars)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load trained model if we are doing inference
        if not training_:
            saver = tf.train.Saver({'emb_word': emb_word, 'hs_W': hs_vars[0], 'hs_b': hs_vars[1]})
            saver.restore(sess, os.path.join(train_model_path, 'model.ckpt'))

        # train
        lr_delta = (cur_lr - 0.001) / len(docs)
        cur_epoch = 0
        data = {'cur_epoch': cur_epoch}
        threading.Thread(target=load_and_enqueue, args=(
            docs, word_to_prob, word_to_index, window_size, epoch_size,
            sess, queue, X_doc_input, X_words_input, y_input, data
        )).start()
        print(datetime.datetime.now(), 'started training')
        while True:
            sess.run(train_op, feed_dict={lr: cur_lr})
            if data['cur_epoch'] != cur_epoch:
                print(datetime.datetime.now(), f'finished epoch {cur_epoch}')
                cur_epoch = data['cur_epoch']
                if cur_epoch == epoch_size:
                    break
                cur_lr -= lr_delta

        # save
        path = os.path.join('__cache__', 'tf', f'{name}-{uuid.uuid4()}')
        os.makedirs(path)
        save_model(path, docs, word_to_index, word_to_freq, emb_doc, emb_word, hs_vars, sess)
        return path


def run_log_reg(train_docs, test_docs, pv_dm_train_path, pv_dm_test_path, embedding_size, batch_size, epoch_size):
    X = tf.placeholder(tf.float32, [None, embedding_size])
    y = tf.placeholder(tf.int32, [None])
    dense = tf.layers.dense(X, 5, kernel_initializer=init_ops.glorot_uniform_initializer())
    pred = tf.argmax(dense, 1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense, labels=y))
    train = tf.train.AdadeltaOptimizer(1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pv_dm_train = tf.Variable(tf.zeros([len(train_docs), embedding_size]))
        pv_dm_test = tf.Variable(tf.zeros([len(test_docs), embedding_size]))
        tf.train.Saver({'emb_doc': pv_dm_train}).restore(sess, os.path.join(pv_dm_train_path, 'model.ckpt'))
        tf.train.Saver({'emb_doc': pv_dm_test}).restore(sess, os.path.join(pv_dm_test_path, 'model.ckpt'))

        train_X = pv_dm_train.eval(sess)
        train_y = np.array([doc[0] for doc in train_docs])
        for i in range(epoch_size):
            p = np.random.permutation(len(train_y))
            train_X_, train_y_ = train_X[p], train_y[p]
            for j in range(0, len(train_y), batch_size):
                batch_X, batch_y = train_X_[j:j + batch_size], train_y_[j:j + batch_size]
                sess.run(train, feed_dict={X: batch_X, y: batch_y})

        print(Counter(sess.run(pred, {X: train_X})))
        # print(sess.run(pred, {X: pv_dm_test}))


def main():
    name = 'imdb'
    if name == 'imdb':
        train, val, test = imdb.load_data('../data/imdb_sentiment')
        tables = gen_tables(name, [train, val, test], 2, 1e-3)

        # pv dm
        pv_dm_train_path = run_pv_dm(
            f'{name}_train', train, *tables, training_=True,
            window_size=10, embedding_size=100, cur_lr=0.025, batch_size=2048, epoch_size=20
        )
        pv_dm_test_path = run_pv_dm(
            f'{name}_val', test, *tables, training_=False,
            window_size=10, embedding_size=100, cur_lr=0.025, batch_size=2048, epoch_size=20,
            train_model_path=pv_dm_train_path
        )

        # log reg
        run_log_reg(
            train, test, pv_dm_train_path, pv_dm_test_path, embedding_size=100, batch_size=256, epoch_size=5
        )

    elif name in ('sstb_2', 'sstb_5'):
        train, val, test = sstb.load_data(
            '../data/stanford_sentiment_treebank/class_2' if name == 'sstb_2' else
            '../data/stanford_sentiment_treebank/class_5'
        )
        tables = gen_tables(name, [train, val, test], 2, 1e-5)

        # pv dm
        pv_dm_train_path = run_pv_dm(
            f'{name}_train', train, *tables, training_=True,
            window_size=8, embedding_size=400, cur_lr=0.025, batch_size=256, epoch_size=5
        )
        pv_dm_test_path = run_pv_dm(
            f'{name}_val', test, *tables, training_=False,
            window_size=8, embedding_size=400, cur_lr=0.025, batch_size=256, epoch_size=100,
            train_model_path=pv_dm_train_path
        )

        # log reg
        run_log_reg(
            train, test, pv_dm_train_path, pv_dm_test_path, embedding_size=400, batch_size=256, epoch_size=5
        )


if __name__ == '__main__':
    main()
