import glob
import os
import re


def normalize_data():
    # concat and normalize test/train data
    for dir_ in ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']:
        files = glob.glob(os.path.join('../data/imdb_sentiment', dir_, '*.txt'))
        docs = [None] * len(files)
        for file in files:
            id_ = int(re.search(r'(\d+)_\d+\.txt$', file).group(1))
            with open(file, 'r', encoding='utf-8') as sr:
                contents = sr.read()
                # replace control chars
                contents = contents.replace('\x85', ' ')
                # to lower case
                contents = contents.lower()
                # replace breaks with spaces
                contents = contents.replace('<br />', ' ')
                # pad punctuation with spaces on both sides
                contents = re.sub(r'([.",()!?;:])', r' \1 ', contents)
                docs[id_] = contents
        with open(os.path.join('../data/imdb_sentiment', f'{dir_.replace("/", "_")}.txt'), 'w', encoding='utf-8') as sr:
            for doc in docs:
                sr.write(doc)
                sr.write('\n')


def load_data(path):
    # The validation set is selected s.t. it is roughly 20% and no movies overlap with the training sets to prevent rare
    # words from occuring in both. This is similar to the construction of the test set.
    train_sents, val_sents, test_sents = [], [], []
    with open(os.path.join(path, 'train_pos.txt'), encoding='utf-8') as sr:
        for i, line in enumerate(sr):
            (train_sents if i < 10004 else val_sents).append((1, line.split()))
    with open(os.path.join(path, 'train_neg.txt'), encoding='utf-8') as sr:
        for i, line in enumerate(sr):
            (train_sents if i < 10001 else val_sents).append((0, line.split()))
    with open(os.path.join(path, 'train_unsup.txt'), encoding='utf-8') as sr:
        for line in sr:
            train_sents.append((None, line.split()))
    with open(os.path.join(path, 'test_pos.txt'), encoding='utf-8') as sr:
        for line in sr:
            test_sents.append((1, line.split()))
    with open(os.path.join(path, 'test_neg.txt'), encoding='utf-8') as sr:
        for line in sr:
            test_sents.append((0, line.split()))
    return train_sents, val_sents, test_sents


if __name__ == '__main__':
    normalize_data()
