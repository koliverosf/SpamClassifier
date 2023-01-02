from numpy.random import permutation
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import Utilities


class LanguageProcessor:
    def __init__(self, file):
        self.file = file
        self.train = None
        self.test = None
        self.vocabulary = None
        self.prob_df = None
        self.n_spam = 0
        self.n_ham = 0

    def split_train_test(self, train=0.999):
        df = self.file.sample(frac=1, random_state=43, ignore_index=False)
        X_train, X_test = train_test_split(df, train_size=train, shuffle=False)
        self.train = X_train
        self.test = X_test

    def train_model(self):
        vectorized = CountVectorizer()
        model = vectorized.fit_transform(self.train['SMS']).toarray()

        self.vocabulary = vectorized.get_feature_names_out()
        df = pd.DataFrame(model, columns=self.vocabulary)

        return self.train.join(df), vectorized

    def get_probabilities(self, df, a_laplace):
        n_spam = len(self.train.where(self.train['Target'] == 'spam').dropna()['SMS'].sum().split())
        n_ham = len(self.train.where(self.train['Target'] == 'ham').dropna()['SMS'].sum().split())
        n_vocab = len(self.vocabulary)

        matrix = []
        n_word_spam = df.where(df['Target'] == 'spam').dropna().sum().drop('Target').drop('SMS')
        n_word_ham = df.where(df['Target'] == 'ham').dropna().sum().drop('Target').drop('SMS')
        for word in self.vocabulary:
            matrix.append(
                [word,
                 round((n_word_spam[word] + a_laplace) / (n_spam + a_laplace * n_vocab), 7),
                 round((n_word_ham[word] + a_laplace) / (n_ham + a_laplace * n_vocab), 7)
                 ]
            )
        prob_df = pd.DataFrame(matrix, columns=['index', 'Spam Probability', 'Ham Probability']).set_index('index')
        prob_df.index.name = None
        self.n_spam = n_spam
        self.n_ham = n_ham
        self.prob_df = prob_df
        return prob_df

    def predict(self, sentence):
        p_spam = self.n_spam / (self.n_spam + self.n_ham)
        p_ham = self.n_ham / (self.n_spam + self.n_ham)

        prediction = 'unknown'

        p_is_spam = p_spam
        p_is_ham = p_ham
        for word in sentence.split():
            if word in self.prob_df.index:
                p_is_ham = p_is_ham * self.prob_df['Ham Probability'][word]
                p_is_spam = p_is_spam * self.prob_df['Spam Probability'][word]

        if p_is_spam > p_is_ham:
            prediction = 'spam'
        elif p_is_spam < p_is_ham:
            prediction = 'ham'

        return prediction

    def predict_df(self, type='homemade', bowmodel=None):
        if type == 'homemade':
            prediction = pd.concat([self.test['Target'], self.test['SMS'].apply(self.predict)], axis=1)
            prediction.rename(columns={'Target': 'Predicted', 'SMS': 'Actual'}, inplace=True)
        else:
            model = MultinomialNB(alpha=2)
            bow = bowmodel.transform(self.train['SMS']).toarray()
            model.fit(bow, self.train['Target'].apply(Utilities.translate_target))
            arr_pred = model.predict(bowmodel.transform(self.test['SMS']).toarray())
            prediction = pd.DataFrame({'Predicted': [Utilities.translate_pred(x) for x in arr_pred], 'Actual': self.test['Target']})
        return prediction

    def prediction_statistics(self, prediction):

        negative = prediction.where(prediction['Predicted'] == 'spam').dropna()
        positive = prediction.where(prediction['Predicted'] == 'ham').dropna()

        TP = positive.where(positive['Predicted'] == positive['Actual']).dropna().shape[0]
        TN = positive.where(positive['Predicted'] != positive['Actual']).dropna().shape[0]
        FP = negative.where(negative['Predicted'] == negative['Actual']).dropna().shape[0]
        FN = negative.where(negative['Predicted'] != negative['Actual']).dropna().shape[0]

        prec = TP / (TP + FP)
        recall = TP / (FN + TP)
        stats = {'Accuracy': (TP + TN) / (TP + TN + FP + FN), 'Recall': recall,
                 'Precision': prec, 'F1': (2 * recall * prec) / (prec + recall)}
        return stats

