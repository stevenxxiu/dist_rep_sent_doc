import os
import re
from collections import namedtuple

__all__ = ['load_data']

ParsedNode = namedtuple('ParsedNode', ['label', 'word', 'children'])
label_re = re.compile(r'\d+')
word_re = re.compile(r'[^)]+')


def read_tree(s, i=0):
    assert s.startswith('(', i)
    i += 1
    label = label_re.match(s, i).group(0)
    i += len(label)
    children = []
    while True:
        if s.startswith(' (', i):
            i += 1
            child, i = read_tree(s, i)
            children.append(child)
        elif s.startswith(')', i):
            i += 1
            return ParsedNode(int(label), None, children), i
        elif s.startswith(' ', i):
            i += 1
            word = word_re.match(s, i).group(0)
            i += len(word)
            assert s.startswith(')', i)
            i += 1
            return ParsedNode(int(label), word, None), i
        else:
            raise ValueError(s, i)


def get_phrases(tree, accum=None):
    if tree.children:
        words = []
        for child in tree.children:
            words.extend(get_phrases(child, accum)[1])
    else:
        words = [tree.word]
    res = (tree.label, words)
    if accum is not None:
        accum.append(res)
    return res


def load_data(path):
    # return train phrases, val sentences, test sentences
    train_trees, val_trees, test_trees = [], [], []
    for filename, res in zip(['train.txt', 'dev.txt', 'test.txt'], [train_trees, val_trees, test_trees]):
        with open(os.path.join(path, filename), encoding='utf-8') as sr:
            for line in sr:
                res.append(read_tree(line.rstrip())[0])
    train_phrases = []
    for tree in train_trees:
        get_phrases(tree, train_phrases)
    train_phrases = set((label, tuple(doc)) for label, doc in train_phrases)
    val_sents, test_sents = [], []
    for trees, res in zip([val_trees, test_trees], [val_sents, test_sents]):
        for tree in trees:
            res.append(get_phrases(tree))
    return train_phrases, val_sents, test_sents
