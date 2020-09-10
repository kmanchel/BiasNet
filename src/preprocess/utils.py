#IMPORT STUFF

import datetime, pickle, os, codecs, re, string
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

import string
from spacy.lang.en import English
import gensim, nltk, logging

from nltk.corpus import stopwords
from nltk import tokenize
from nltk.stem import SnowballStemmer

try:
    import cPickle as pickle
except ImportError:
    import pickle

import en_core_web_sm

#GENERAL PURPOSE UTILS
class Params():
    """A class to load hyperparameters from a json file"""
    
    def __init__(self, json_path):
        self.update = json_path
        
        #load parameters onto members
        self.load(self.update)
    
    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as jf:
            json.dump(self.__dict__, jf, indent=4)
            
    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as jf:
            params = json.load(jf)
            self.__dict__.update(params)
            
    @property
    def dict(self):
        """give the class dictionary-like access (e.g. params.dict['learning_rate']) """
        return self.__dict__
    

def pickle_object(obj, fname):
    """Saves and serializes object as pickle file
        Args:
            obj: (any) Any Object
            fname: (str) path to save the object
        Return:
            N/A
    """
    with open(fname, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(fname):
    """Loads pickle file as object
        Args:
            fname: (str) path to save the object
        Return:
            obj: (any)
    """
    pickle_in = open(fname,"rb")
    obj = pickle.load(pickle_in)
    return obj

#NLP UTILS
def encode_texts(params, texts, tokenizer):
    """Encodes texts as they come from the data pipeline to go into model
        Args:
            params: (Params) Params used here are MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH
            texts: (tf.string) path to the pretrained word embedding
            tokenizer: (tf.keras.utils.Preprocessing.Tokenizer) Tokenizer which has been built on the dataset
        Return:
            embedding_matrix: (np.array, shape = (N, like word embedding), dtype = float32) 
    """
    texts = np.char.decode(texts.numpy().astype(np.bytes_), 'UTF-8')
    texts = [texts.tolist()]
    encoded_texts = np.zeros((len(texts), params.MAX_SENTENCE_COUNT, params.MAX_SENTENCE_LENGTH))
    for i, text in enumerate(texts):
        encoded_text = np.array(tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences(text), 
            maxlen=params.MAX_SENTENCE_LENGTH))[:params.MAX_SENTENCE_COUNT]
        encoded_texts[i][-len(encoded_text):] = encoded_text
    return encoded_texts

#TRAIN-VAL SPLIT
def _train_validate_test_split(df, train_percent=.8, validate_percent=.2, seed=243):
    """Helper function for build_train_test()"""
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    return train, validate


def build_train_test(main_csv_path, dataset_subset_ratio, seed=243):
    """This function creates the train, validation, and test csv files"""
    """The created csvs are be what should be passed into input_fn() to create the tf.data.dataset object"""
    df = pd.read_csv(main_csv_path)
    #Shrink dataset by the ratio provided
    df = df.sample(frac=dataset_subset_ratio, random_state=seed).reset_index(drop=True)
    train, val = _train_validate_test_split(df, train_percent=.8, validate_percent=.2)
    train.to_csv("train.csv", index=False)
    print("Created train.csv in current directory")
    val.to_csv("val.csv", index=False)
    print("Created val.csv in current directory")
   