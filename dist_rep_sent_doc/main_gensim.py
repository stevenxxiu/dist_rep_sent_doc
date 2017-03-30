import logging
import os
import random
from collections import Counter

import joblib
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression

from dist_rep_sent_doc.data import preprocess_data

memory = joblib.Memory('__cache__', verbose=0)


@memory.cache
def gen_data(path):
    return sum([[
        [TaggedDocument(doc[1], [i]) for i, doc in enumerate(docs)],
        [doc[0] for doc in docs]
    ] for docs in preprocess_data(path)], [])


def main():
    # fit
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    train_X, train_y, val_X, val_y, test_X, test_y = gen_data('../data/stanford_sentiment_treebank/class_5')
    model = Doc2Vec(
        size=400, window=8, min_count=0, workers=4, dm=1, dm_concat=1, hs=1, iter=1, alpha=0.025, min_alpha=0.025
    )
    model.build_vocab(train_X + val_X + test_X)

    for epoch in range(10):
        random.shuffle(train_X)
        model.train(train_X)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    # save
    os.makedirs('__cache__/gensim', exist_ok=True)
    model.save('__cache__/gensim/pvdm.model')

    # logreg
    model = Doc2Vec.load('__cache__/gensim/pvdm.model')
    logreg = LogisticRegression()
    logreg.fit(np.array(model.docvecs), train_y)
    print(Counter(logreg.predict(np.array(model.docvecs))))

if __name__ == '__main__':
    main()
