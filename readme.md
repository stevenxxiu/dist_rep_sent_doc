# Distributed Representations of Sentences and Documents
This is an attempt to reproduce the paper using tensorflow, commonly known as `doc2vec`. We cannot fully reproduce it's results, but obtain results similar to gensim.

## Dataset
To normalize the dataset (the stanford tool does some regex replacements to fix weird trees):

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 2 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_2/stanford_sentiment_treebank

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 5 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_5/stanford_sentiment_treebank

## Unclear parts of the paper:
It is possible that the model is actually predicting the center word instead of the end word since this gives better results, since predicting the center word is what `word2vec` does.

The random initializations of the word embedding and softmax weights are not specified.

Hierarchical softmax usually has a bias, it is not described in the `word2vec` paper, but there is no bias in the `word2vec` paper, so we suppose this.

The learning rate/gradient descent method is not specified. The objective is not specified, although from the previous word embeddings paragraph it is probably to maximize the total log probability. Training and inference learning rates can be quite different. It makes more sense for the gradients to not be means, as the parameters used are rarely the same (the purpose of having mean gradients is so that different batch sizes give similar gradients, which makes sense if the same parameters are used per instance).

It is not described how unknown words during validation are handled.

It is not described in detail what the logistic regression and neural network models are, we assume the most common models and use no regularization.

### Can be missed
A detailed explanation of window size is in the Experimental Protocols section.

The imdb dataset uses a neural net instead of logistic regression alone.

### Gensim differences
- Option to remove infrequent words entirely.
- Includes frequent word subsampling, which also uses a different formula to the `word2vec` paper.
- Reduced window sampling for the pv-dm average model, which is not present in the paper.
- Predicts center word instead of last word, the window sizes on the left and right are required to be the same.
- Hierarchical softmax has no bias.
- Shuffling during training is by shuffling all documents first, then training each document from left to right, and ensuring that each batch contains windows from different documents.
