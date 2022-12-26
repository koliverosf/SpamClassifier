from numpy.random import permutation
import nltk
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class LanguageProcessor:
    def __init__(self, file):
        self.file = file
        self.train = None
        self.test = None
        self.vocabulary = None

    def split_train_test(self, train=0.8):
        df = self.file.sample(frac=1, random_state=43, ignore_index=True)
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
                 round((n_word_spam[word] + a_laplace)/(n_spam + a_laplace*n_vocab), 7),
                 round((n_word_ham[word] + a_laplace)/(n_ham + a_laplace*n_vocab), 7)
                 ]
            )
        prob_df = pd.DataFrame(matrix, columns=['index', 'Spam Probability', 'Ham Probability']).set_index('index')
        prob_df.index.name = None
        return prob_df

    def predict_model(self, prob_df):
        prob_df
