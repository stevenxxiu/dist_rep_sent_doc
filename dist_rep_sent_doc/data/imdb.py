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

if __name__ == '__main__':
    normalize_data()
