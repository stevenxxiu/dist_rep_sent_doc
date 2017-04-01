# Distributed Representations of Sentences and Documents
## Dataset
To normalize the dataset (the stanford tool does some regex replacements to fix weird trees):

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 2 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_2/stanford_sentiment_treebank

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 5 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_5/stanford_sentiment_treebank

## Unclear parts of the paper:
The random initializations of the word embedding and softmax weights are not given.

The learning rate/gradient descent method is not specified. The objective is not specified, although from the previous word embeddings paragraph it is probably the average log probability. In fact for adadelta, training and inference require quite different learning rates to achieve good results.

Hierarchical softmax usually has a bias, it is not described `Distributed representations of words and phrases and their compositionality` paper, but the paper does say "... softmax weights U, b and paragraph vectors D` ...", so we suppose that there is a bias.

It is not described how unknown words during validation are handled.

It is not described what process of fitting the logistic regression is used, we use sklearn's defaults.

### Can be missed
A detailed explanation to window size is in the Experimental Protocols section.

### Gensim differences
- Option to remove infrequent words entirely.
- Frequent word subsampling different.
- Reduced window sampling for the pv-dm average model, which is not present in the paper.

## Code
See whether using theano boolean masks are faster or multiplies by the mask are faster.

### Cookbook
To perform inference:

    saver = tf.train.Saver({'emb_word': emb_word, 'emb_doc': emb_doc, 'hs_W': hs_vars[0], 'hs_b': hs_vars[1]})
    saver.restore(sess, '__cache__/tf/train_5-47b9f78a-08da-47a7-8d26-af2f9e4df747/model.ckpt')
    words = len(word_to_index) * [None]
    for word, i in word_to_index.items():
        words[i] = word
    print(' '.join(words[i] for i in batch_X_words[0]))
    print(words[batch_y[0]])
    print(words[np.argmax(sess.run(l.apply(flatten, training=False), feed_dict={X_doc: batch_X_doc, X_words: batch_X_words}), 1)[0]])

## Possible improvements
Since we use a gpu, it might be possible to use softmax itself instead of hierarchical softmax, since it is likely quite fast too.
