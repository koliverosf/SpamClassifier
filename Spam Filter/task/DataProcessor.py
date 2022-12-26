import string
import pandas as pd
import spacy as sp
import Utilities
import re


# python -m spacy download en_core_web_sm-3.4.1 --direct


class DataProcessor:

    def __init__(self, file_name):
        self.file_name = file_name
        self.text = ''
        self.file = None
        self.processed_file = None
        self.tokens = []

    def load_csv(self):
        full_file = pd.read_csv(self.file_name, encoding='iso-8859-1')
        file = full_file[['v1', 'v2']]
        self.file = file.rename(columns={'v1': 'Target', 'v2': 'SMS'}, inplace=False)



    def process_data(self):
        en = sp.load('en_core_web_sm')

        self.file['SMS'] = self.file['SMS'].str.lower()

        # Lemmatization
        self.file["SMS"] = self.file['SMS'].apply(lambda x: " ".join([y.lemma_ for y in en(x)]))


        self.file["Lemmatized"] = self.file['SMS'].apply(lambda x: x.split())

        self.file["Lemmatized"] = self.file["Lemmatized"].apply(Utilities.remove_punct)
        self.file["Lemmatized"] = self.file["Lemmatized"].apply(Utilities.aanumbers)
        self.file["Lemmatized"] = self.file["Lemmatized"].apply(Utilities.stopwords_remover)
        self.file["Lemmatized"] = self.file["Lemmatized"].apply(Utilities.singletterword)

        # Remove punctuations, substitute something with numbers to aanumbers
        # new_data = []
        # for i, row in self.file.iterrows():
        #     lemmatized = [re.sub(
        #         r"[a-zA-z]*[0-9]+[a-zA-z]*[0-9]*",
        #         'aanumbers',
        #         token.lemma_.translate(str.maketrans('', '', string.punctuation))
        #     )
        #         for token in en(row[['SMS']][0]) if len(token) > 1]
        #
        #     clean = []
        #     for token in lemmatized:
        #          if token not in en.Defaults.stop_words:
        #              clean.append(token)
        #     new_data.append(clean)

        # df = pd.DataFrame({'Lemmatized': new_data}) #new_data if above uncommented
        # self.file = self.file.join(df)
        # #  Set processed file
        self.processed_file = self.file[['Target']]
        self.processed_file = self.processed_file.join(self.file['Lemmatized'].apply(lambda x: ' '.join(x)))
        self.processed_file = self.processed_file.rename(columns={'Target': 'Target', 'Lemmatized': 'SMS'},
                                                         inplace=False)

    def display_data(self, columns=None, n_max=200):
        pd.options.display.max_rows = n_max

        if columns is None:
            pd.options.display.max_columns = self.processed_file.shape[1]
        else:
            pd.options.display.max_columns = columns

        print(self.processed_file.head(200))

