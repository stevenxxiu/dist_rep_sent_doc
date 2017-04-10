import binascii
import csv
import datetime
import os
import threading
import uuid
from collections import Counter

import joblib
import numpy as np
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from dist_rep_sent_doc.data import imdb, sstb
from dist_rep_sent_doc.huffman import build_huffman
from dist_rep_sent_doc.layers import HierarchicalSoftmaxLayer

memory = joblib.Memory('__cache__', verbose=0)


@memory.cache(ignore=['docs'])
def gen_tables(name, docs, vocab_min_freq, sample):
    total_count = 0

    # map word to counts and indexes, infrequent words are removed entirely
    word_to_freq = {}
    words = []
    for word, count in Counter(word for doc in docs for word in doc[1]).items():
        if count >= vocab_min_freq:
            word_to_freq[word] = count
            words.append(word)
            total_count += count
    words.append('\0')
    words.sort(key=lambda word_: word_to_freq.get(word_, 1), reverse=True)

    word_to_index = {}
    for i, word in enumerate(words):
        word_to_index[word] = i

    # get huffman tree
    tree = build_huffman(word_to_freq)

    # get word sub-sampling probabilities
    word_to_prob = {}
    for word, count in word_to_freq.items():
        ratio = sample / (count / total_count) if count > 0 else 1
        word_to_prob[word] = min(np.sqrt(ratio) + ratio, 1)
        word_to_prob[word] = int(round(word_to_prob[word] * 2**32))

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
                    writer.writerow(['\\0', 0] if word == '\0' else [word, word_to_freq[word]])
    summary_writer = tf.summary.FileWriter(path)
    projector.visualize_embeddings(summary_writer, config)

    # save model
    saver = tf.train.Saver({'emb_word': emb_word, 'emb_doc': emb_doc, 'hs_W': hs_vars[0]})
    saver.save(sess, os.path.join(path, 'model.ckpt'))


def load_and_enqueue(
    docs, word_to_prob, word_to_index, window_size, sess, queue, X_doc_input, X_words_input, y_input,
    random
):
    enqueue_many_op = queue.enqueue_many([X_doc_input, X_words_input, y_input])
    p = random.permutation(len(docs))
    for j in p:
        # remove infrequent words & sample frequent words
        X_doc, X_words, y = [], [], []
        doc = [
            word for word in docs[j][1] if
            word in word_to_index and random.rand() * 2**32 < word_to_prob[word]
        ]
        for k, word in enumerate(doc):
            # window_size before word and window_size after word
            before = doc[max(k - window_size, 0):k]
            before = (window_size - len(before)) * ['\0'] + before
            after = doc[k + 1:min(k + 1 + window_size, len(doc))]
            after = after + (window_size - len(after)) * ['\0']
            window = before + after
            X_doc.append(j)
            X_words.append([word_to_index[word_] for word_ in window])
            y.append(word_to_index[word])
        sess.run(enqueue_many_op, feed_dict={X_doc_input: X_doc, X_words_input: X_words, y_input: y})
    sess.run(queue.close())


# @memory.cache(ignore=['docs', 'word_to_freq', 'word_to_index', 'tree', 'word_to_prob'])
def run_pv_dm(
    name, docs, word_to_freq, word_to_index, tree, word_to_prob, training_, window_size, embedding_size, cur_lr, min_lr,
    batch_size, epoch_size, train_model_path=None
):
    # queue per epoch since we cannot reset queues
    X_doc_input = tf.placeholder(tf.int32, [None])
    X_words_input = tf.placeholder(tf.int32, [None, 2 * window_size])
    y_input = tf.placeholder(tf.int32, [None])
    vars_ = [X_doc_input, X_words_input, y_input]
    queues = []
    for i in range(epoch_size):
        queues.append(tf.FIFOQueue(4096, [var.dtype for var in vars_], shapes=[var.shape[1:] for var in vars_]))

    # network
    cur_epoch = tf.placeholder(tf.int32, [])
    X_doc, X_words, y = tf.QueueBase.from_list(cur_epoch, queues).dequeue_up_to(batch_size)
    lr = tf.placeholder(tf.float32, [])
    emb_doc = tf.Variable(tf.random_uniform(
        [len(docs), embedding_size], - 0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb_word = tf.Variable(tf.random_uniform(
        [len(word_to_index), embedding_size], - 0.5 / embedding_size, 0.5 / embedding_size
    ))
    emb = tf.concat([
        tf.reshape(tf.nn.embedding_lookup(emb_doc, X_doc), [-1, 1, embedding_size]),
        tf.nn.embedding_lookup(emb_word, X_words)
    ], 1)
    flatten = tf.reshape(emb, [-1, (2 * window_size + 1) * embedding_size])
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

        emb_word_init = []
        for word, i in word_to_index.items():
            once = np.random.RandomState(binascii.crc32(f'{word}0'.encode('utf-8')) & 0xffffffff)
            emb_word_init.append((once.rand(embedding_size) - 0.5) / embedding_size)
        sess.run(emb_word.assign(np.array(emb_word_init)))

        emb_doc_init = []
        for i in range(len(docs)):
            once = np.random.RandomState(binascii.crc32(f'0 {i}'.encode('utf-8')) & 0xffffffff)
            emb_doc_init.append((once.rand(embedding_size) - 0.5) / embedding_size)
        sess.run(emb_doc.assign(np.array(emb_doc_init)))

        # load trained model if we are doing inference
        if not training_:
            saver = tf.train.Saver({'emb_word': emb_word, 'hs_W': hs_vars[0]})
            saver.restore(sess, os.path.join(train_model_path, 'model.ckpt'))

        # train
        lr_delta = (cur_lr - min_lr) / epoch_size
        print(datetime.datetime.now(), 'started training')
        random = np.random.RandomState(0)
        for i in range(epoch_size):
            threading.Thread(target=load_and_enqueue, args=(
                docs, word_to_prob, word_to_index, window_size, sess, queues[i], X_doc_input, X_words_input, y_input,
                random
            )).start()
            while True:
                try:
                    sess.run(train_op, feed_dict={lr: cur_lr, cur_epoch: i})
                except tf.errors.OutOfRangeError:
                    break
            print(datetime.datetime.now(), f'finished epoch {i}')
            if training_:
                cur_lr -= lr_delta
            else:
                cur_lr = ((cur_lr - min_lr) / (epoch_size - i)) + min_lr

        # save
        path = os.path.join('__cache__', 'tf', f'{name}-{uuid.uuid4()}')
        # os.makedirs(path)
        # save_model(path, docs, word_to_index, word_to_freq, emb_doc, emb_word, hs_vars, sess)
        return path


def run_log_reg(train_docs, test_docs, pv_dm_train_path, pv_dm_test_path, embedding_size):
    with tf.Session() as sess:
        pv_dm_train = tf.Variable(tf.zeros([len(train_docs), embedding_size]))
        pv_dm_test = tf.Variable(tf.zeros([len(test_docs), embedding_size]))
        tf.train.Saver({'emb_doc': pv_dm_train}).restore(sess, os.path.join(pv_dm_train_path, 'model.ckpt'))
        tf.train.Saver({'emb_doc': pv_dm_test}).restore(sess, os.path.join(pv_dm_test_path, 'model.ckpt'))

        i = [i for i, doc in enumerate(train_docs) if doc[0] is not None]
        train_X = pv_dm_train.eval(sess)[i]
        train_y = np.array([doc[0] for doc in train_docs], dtype=np.float32)[i]
        logit = sm.Logit(train_y, train_X)
        predictor = logit.fit(disp=0)

        test_X = pv_dm_test.eval(sess)
        test_pred = predictor.predict(test_X)
        corrects = sum(np.rint(test_pred) == [doc[0] for doc in test_docs])
        print(f'error rate: {(len(test_pred) - corrects) / len(test_pred)}')


def main():
    name = 'imdb'
    if name == 'imdb':
        train, val, test = imdb.load_data('../data/imdb_sentiment')
        train = train[:1]
        tables = gen_tables(name, train, 2, 1e-3)

        # pv dm
        pv_dm_train_path = run_pv_dm(
            f'{name}_train', train, *tables, training_=True,
            window_size=5, embedding_size=100, cur_lr=0.025, min_lr=0.001, batch_size=1, epoch_size=4
        )
        # pv_dm_test_path = run_pv_dm(
        #     f'{name}_val', test, *tables, training_=False,
        #     window_size=5, embedding_size=100, cur_lr=0.1, min_lr=0.0001, batch_size=1, epoch_size=3,
        #     train_model_path=pv_dm_train_path
        # )
        #
        # # log reg
        # run_log_reg(
        #     train, test, pv_dm_train_path, pv_dm_test_path, embedding_size=100
        # )

    elif name in ('sstb_2', 'sstb_5'):
        train, val, test = sstb.load_data(
            '../data/stanford_sentiment_treebank/class_2' if name == 'sstb_2' else
            '../data/stanford_sentiment_treebank/class_5'
        )
        tables = gen_tables(name, train, 2, 1e-5)

        # pv dm
        pv_dm_train_path = run_pv_dm(
            f'{name}_train', train, *tables, training_=True,
            window_size=4, embedding_size=100, cur_lr=0.025, min_lr=0.001, batch_size=64, epoch_size=20
        )
        pv_dm_test_path = run_pv_dm(
            f'{name}_val', test, *tables, training_=False,
            window_size=4, embedding_size=100, cur_lr=0.1, min_lr=0.0001, batch_size=64, epoch_size=3,
            train_model_path=pv_dm_train_path
        )

        # log reg
        run_log_reg(
            train, test, pv_dm_train_path, pv_dm_test_path, embedding_size=400
        )


if __name__ == '__main__':
    main()
