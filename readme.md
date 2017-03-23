# Distributed Representations of Sentences and Documents
## Dataset
To normalize the dataset (the stanford tool does some regex replacements to fix weird trees):

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 2 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_2/stanford_sentiment_treebank

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 5 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_5/stanford_sentiment_treebank

## Unclear parts of the paper:
What happens when the phrase is shorter than the window size + 1? Do we simply ignore phrases longer than it? This discards some training instances, so we pad the first words with 0s.

## Code
See whether using theano boolean masks are faster or multiplies by the mask are faster.

## Possible improvements
Since we use a gpu, it might be possible to use softmax itself instead of hierarchical softmax, since it is likely quite fast too.
