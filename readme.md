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

### Can be missed
A detailed explanation to window size is in the Experimental Protocols section.

## Code
See whether using theano boolean masks are faster or multiplies by the mask are faster.

## Possible improvements
Since we use a gpu, it might be possible to use softmax itself instead of hierarchical softmax, since it is likely quite fast too.
