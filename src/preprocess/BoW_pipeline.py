import sys
import os
import pdb
import argparse

sys.path.append("../..")
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
import datetime, pickle, codecs
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import numpy as np
import heapq
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from logging import getLogger, DEBUG, INFO, WARNING, ERROR


from joblib import dump, load
from sklearn.linear_model import LogisticRegression


class BoW_Pipeline(object):
    def __init__(self):
        self.most_freq = []
        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

    def clean_text(self, text, hashtag=False):
        """
		This utility function cleans incoming text
		Arg:
				text (str): input text
				hashtag (bool): whether or not to clean hashtags
		Returns:
				clean_words (list): List of strings of cleaned words

		What will be covered in the cleaning process:
		1. Remove punctuation
		2. Remove stopwords
		3. Lower case (on lower_case == True)
		4. Lemmatize
		"""
        if hashtag:
            punctuations = "!\"$%&'()*+,-./:;<=>?@[\\]^_`{|}~"  # except hashtag
        else:
            punctuations = string.punctuation

        # 1
        nopunc = [char for char in text if char not in punctuations]
        nopunc = "".join(nopunc)

        # 2
        clean_words = [
            word for word in nopunc.split() if word.lower() not in stopwords.words("english")
        ]

        # 3
        clean_words = [word.lower() for word in clean_words]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        clean_words = [
            " ".join(lemmatizer.lemmatize(word) for word in sent.split()) for sent in clean_words
        ]

        return clean_words

    def BoW_sent_vec(self, text):
        """
		This utility function returns Bag Of Words features given an input text
		"""
        sent_vec = []
        for token in self.most_freq:
            if token in text:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        return np.array(sent_vec)


class BoW_Train(BoW_Pipeline):
    def __init__(self):
        super().__init__()

    def preprocess_train(
        self,
        data_load_path,
        load_frac,
        BoW_save_path=None,
        data_save_path=None,
        random_seed=69,
        n_features=500,
    ):
        df = pd.read_csv(data_load_path, usecols=["site", "text", "label"]).sample(
            frac=load_frac, random_state=random_seed
        )
        df = self.filter_sources(df)
        wordfreq = self.build_word_frequency(df)
        df = self.BoW(df, wordfreq, n_features, BoW_save_path)
        if data_save_path != None:
            df.to_csv(data_save_path, index=False)
        self.data = df
        self.logger.info("Done Loading and Preprocessing training data.")

    def build_word_frequency(self, data, train_split=0.8):
        wordfreq = {}
        data["text"] = data["text"].apply(self.clean_text)
        texts = data["text"].values.tolist()

        n_train = int(train_split * len(data))

        for text in tqdm(texts[:n_train], "# examples"):
            for token in text:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
        return wordfreq

    def filter_sources(self, df):
        # Removing obvious giveaway source words
        df["text"] = df["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        sources = list(df["site"].unique())
        sources_no_dot = [i.replace(".", "") for i in sources]
        sources_no_dotcom = [i.replace(".com", "") for i in sources]
        sources_no_news = [i.replace("news", "") for i in sources_no_dotcom]
        for i in sources_no_dot:
            sources.append(i)
        for i in sources_no_dotcom:
            sources.append(i)
        for i in sources_no_news:
            sources.append(i)
        sources = list(dict.fromkeys(sources)) + ["(cnn)", "cnn", "fox"]
        df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sources))
        return df

    def BoW(self, data, wordfreq, n, BoW_save_path):

        self.most_freq = heapq.nlargest(n, wordfreq, key=wordfreq.get)
        if BoW_save_path != None:
            with open(BoW_save_path, "wb") as fp:
                pickle.dump(self.most_freq, fp, protocol=pickle.HIGHEST_PROTOCOL)
            msg = "Dumpled BoW to {}".format(BoW_save_path)
            self.logger.info(msg)
        data["features"] = data["text"].apply(lambda x: self.BoW_sent_vec(x))
        return data


class BoW_Test(BoW_Pipeline):
    def __init__(self, tokens_path):
        super().__init__()
        self.most_freq = pickle.load(open(tokens_path, "rb"))

    def featurize(self, text):
        x = self.clean_text(text)
        features = np.array(self.BoW_sent_vec(x))
        return features
