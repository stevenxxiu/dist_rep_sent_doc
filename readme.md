# Distributed Representations of Sentences and Documents
## Dataset
To normalize the dataset (the stanford tool does some regex replacements to fix weird trees):

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 2 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_2/stanford_sentiment_treebank

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 5 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_5/stanford_sentiment_treebank

## Unclear parts of the paper:
The model itself is very confusing, the diagram shows prediction of the next word given previous words, but the paper also says that the only formal change from the `word2vec` paper is the addition of the paragraph vector, however the `word2vec` paper predicts the center word given surrounding words.

The random initializations of the word embedding and softmax weights are not specified.

Hierarchical softmax usually has a bias, it is not described in the `word2vec` paper, but there is no bias in the `word2vec` paper, so we suppose this.

The learning rate/gradient descent method is not specified. The objective is not specified, although from the previous word embeddings paragraph it is probably to maximize the average log probability. Training and inference learning rates can be quite different. It makes more sense for the gradients to not be means, as the input variables are rarely the same.

It is not described how unknown words during validation are handled.

It is not described what process of training the logistic regression or neural net plus logistic regression models are used, we use sklearn's defaults.

### Can be missed
A detailed explanation of window size is in the Experimental Protocols section.

The imdb dataset uses a neural net and a logistic regression, instead of logistic regression alone.

### Gensim differences
- Option to remove infrequent words entirely.
- Includes frequent word subsampling, which also uses a different formula to the `word2vec` paper.
- Reduced window sampling for the pv-dm average model, which is not present in the paper.
- Predicts center word instead of last word, the window sizes on the left and right are required to be the same.
- Hierarchical softmax has no bias.
