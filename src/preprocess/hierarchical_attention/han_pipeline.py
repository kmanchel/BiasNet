"""This script creates a tf.data.Dataset data loader pipeline"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

sys.path.append("..")
sys.path

import tensorflow as tf
import pandas as pd
import numpy as np
import string
import tensorflow_datasets as tfds
from utils import Params, load_pickle

def get_dataset(file_path,batch_size, **kwargs):
    """Data loader from dataset csv file
        Args:
            params: (Params) Model Parameters (dataset path is used here)
        Return:
            dataset: (tf.tfds.data.make_csv_dataset) 
    """
    
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,select_columns=['text','label'],batch_size=batch_size,label_name='label',
        ignore_errors=True,shuffle=True, **kwargs)
    return dataset


def fix_fn(features, label):
    """Transforms (tf.tfds.data.make_csv_dataset) to (tf.tfds.data.dataset) 
        Args:
            features: (oDict) From csv dataset object
            labels: (tf.int32) From csv dataset object
        Return:
            features: (tf.string) Basically maps this from oDict to string
            label: (tf.int32)
    """
    def _fix(text,label):
        text = text.numpy()
        return text,label
    
    return tf.py_function(_fix, inp=[features['text'], label], Tout=(tf.string, tf.int32))

def _py_fn(fn):
    def apply_tf_pyfunc(*args, **kwargs):
        return tf.py_function(fn, list(args), **kwargs)
    return apply_tf_pyfunc


@_py_fn
def _encode_texts(text_, label, __tokenizer, params):
    texts = np.char.decode(text_.numpy().astype(np.bytes_), 'UTF-8')
    texts = [eval(j) for j in texts]
    encoded_texts = np.zeros((len(texts), params.MAX_SENTENCE_COUNT, params.MAX_SENTENCE_LENGTH))
    for i, text in enumerate(texts):
        encoded_text = np.array(tf.keras.preprocessing.sequence.pad_sequences(
            __tokenizer.texts_to_sequences(text), 
            maxlen=params.MAX_SENTENCE_LENGTH))[:params.MAX_SENTENCE_COUNT]
        encoded_texts[i][-len(encoded_text):] = encoded_text
    return encoded_texts, label





class HAN_Pipeline:
    """Pipeline for Hierarchical Attention Network Model
        Args:
            params: (Params) Model Parameters (dataset paths are expected here) 
    """
    def __init__(self,params):
        print("Loaded Training Articles from %s"%params.clean_dataset_train)
        self.train = get_dataset(params.clean_dataset_train,params.batch_size).map(fix_fn)
        print("Loaded Validation Articles from %s"%params.clean_dataset_val)
        self.val = get_dataset(params.clean_dataset_val,params.batch_size).map(fix_fn)
        self.tokenizer = load_pickle(params.tokenizer_path)
        self.embedding_matrix = np.load(params.embedding_path)

        self.encode_texts_fn = lambda text, label: _encode_texts(text, label, self.tokenizer, params, Tout=[tf.float32, tf.int32])
    
    def train_loader(self):
        return self.train.map(self.encode_texts_fn)

    def val_loader(self):
        return self.val.map(self.encode_texts_fn)

