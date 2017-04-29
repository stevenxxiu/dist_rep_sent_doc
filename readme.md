# Distributed Representations of Sentences and Documents
This is an attempt to reproduce the paper using tensorflow, commonly known as `doc2vec`. We cannot fully reproduce it's results, but obtain results similar to gensim.

## Dataset
To normalize the dataset (the stanford tool does some regex replacements to fix weird trees):

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 2 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_2/stanford_sentiment_treebank

    java -cp stanford-corenlp-3.7.0.jar edu.stanford.nlp.sentiment.ReadSentimentDataset -numClasses 5 -inputDir ../data/_input/stanford_sentiment_treebank  -outputDir ../data/class_5/stanford_sentiment_treebank

## Report
A report for our re-implementation can be found at https://github.com/stevenxxiu/dist_rep_sent_doc_report.
