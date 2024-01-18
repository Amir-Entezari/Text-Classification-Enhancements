import numpy as np
from nltk import word_tokenize

from src.indexing import InvertedIndex


class BaseClassifier:
    def __init__(self, indexing_model):
        self.indexing_model: InvertedIndex = indexing_model

    def train(self, **kwargs):
        raise NotImplementedError("Subclasses must override this method.")


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, indexing_model):
        super().__init__(indexing_model)
        self.prob_list = {
            'positive': {},
            'negative': {}
        }

    def train(self):
        """
        This function creates a table of probabilities using the formula from lecture 12.
        """
        # Compute B based on the slides:
        B = len(self.indexing_model.posting_list)  # Size of the vocabulary
        positive_word_size = B
        negative_word_size = B
        # Compute the Denominator of the positive class :
        for t in self.indexing_model.positive_posting_list:
            positive_word_size += t['cf']
        # Compute the Denominator of the negative class :
        for t in self.indexing_model.negative_posting_list:
            negative_word_size += t['cf']
        # Apply the formula in lecture 12. here we use logarithm to avoid roundoff
        for t in self.indexing_model.positive_posting_list:
            self.prob_list['positive'][t['word']] = np.log((t['cf'] + 1) / positive_word_size)
        for t in self.indexing_model.negative_posting_list:
            self.prob_list['negative'][t['word']] = np.log((t['cf'] + 1) / negative_word_size)

        self.positive_word_size = positive_word_size
        self.negative_word_size = negative_word_size
        return self

    def classify(self, document):
        """
        In this function I implement the naive bayes classifier. Because we apply logarithm to the probablities that we store in prob_list, we can use summotion for naive bayes.
        :param query: str
            the document we want to classify
        :return:
            return the predicted class of the document
        """
        tokenized_query = word_tokenize(document)
        neg_prob = 0
        pos_prob = 0
        for token in tokenized_query:
            try:
                neg_prob += self.prob_list['negative'][token]
            except:
                neg_prob += np.log(1 / self.positive_word_size)
            try:
                pos_prob += self.prob_list['positive'][token]
            except:
                pos_prob += np.log(1 / self.negative_word_size)
        return "positive" if pos_prob > neg_prob else "negative"

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.classify(X[i]))
        return y_pred
