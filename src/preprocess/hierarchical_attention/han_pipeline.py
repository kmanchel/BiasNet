"""This script creates a tf.data.Dataset data loader pipeline"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import datetime, pickle, os, codecs, re, string
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import string
from preprocess.utils import Params, get_dataset, fix_fn, _py_fn, load_subword_embedding, normalize
import pdb


class HAN_DataLoader:
    """
    Pipeline for Hierarchical Attention Network Model
    """

    def __init__(self, pipeline_params, train=True):
        """
        Args:
            self: Class variable init. its "self" explanatory!
            pipeline_params (dict):
                        These must contain atleast the following info:
                            - train_path: absolute path to training dataset.
                            - val_path: absolute path to validation dataset.
                            - batch_size: batch size for fitting model down the road.
                            - MAX_SENTENCE_LENGTH: integer to indicate sentence length to encode to.
                            - MAX_SENTENCE_COUNT: integer indicating max number of sentences to consider in article.
                            - tokenizer_save_path: absolute path to where tokenizer will be saved for later use.
                            - pretrained_embedding_path: absolute path to where pretrained embeddings can be found.
                            - embeddings_save_path: absolute path to where embedding matrix is to be saved for later use.
        """

        self.params = pipeline_params

        if train:
            self._init_train()

    def _init_train(self):
        print("Loaded Training Articles from %s" % self.params.train_path)
        self.train = get_dataset(self.params.train_path, self.params.batch_size).map(fix_fn)
        print("Loaded Validation Articles from %s" % self.params.val_path)
        self.val = get_dataset(self.params.val_path, self.params.batch_size).map(fix_fn)

    def _train_tokenizer(self, max_iterations=100):
        if os.path.exists(self.params.tokenizer_save_path):
            pickle_in = open(self.params.tokenizer_save_path, "rb")
            tokenizer = pickle.load(pickle_in)
            return tokenizer
        else:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(
                filters='"()*,-/;[\]^_`{|}~', oov_token="UNK", char_level=False
            )
            train_iterator = iter(self.train)

            iteration_count = 0

            for batch, _ in tqdm(train_iterator, "fitting tokenizer"):
                batch = np.char.decode(batch.numpy().astype(np.bytes_), "UTF-8")
                tokenizer.fit_on_texts(batch)

                if save_path != None:
                    if iteration_count % (max_iterations // 4) == 0:
                        with open(self.params.tokenizer_save_path, "wb") as fp:
                            pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
                        print("Tokenizer saved as %s" % self.params.tokenizer_save_path)
                iteration_count += 1
                if iteration_count == max_iterations:
                    break

            print(">> Done Building Tokenizer")

            return tokenizer

    def _create_embedding_matrix(self, tokenizer):
        if not os.path.exists(self.params.embeddings_save_path):
            embeddings_matrix = load_subword_embedding(
                tokenizer.word_index,
                self.params.pretrained_embedding_path,
                self.params.embeddings_save_path,
            )
        else:
            embeddings_matrix = np.load(self.params.embeddings_save_path)
        return embeddings_matrix

    def _load_tokenizer(self):
        pickle_in = open(self.params.tokenizer_save_path, "rb")
        tokenizer = pickle.load(pickle_in)
        return tokenizer

    def _load_embedding_matrix(self):
        embeddings_matrix = np.load(self.params.embeddings_save_path)
        return embeddings_matrix

    def _encoder_fn(self, __tokenizer):
        params = self.params

        @_py_fn
        def encode_texts(text_, label):
            texts = np.char.decode(text_.numpy().astype(np.bytes_), "UTF-8")  # len = batch_size

            # texts = [eval(j) for j in texts]
            encoded_texts = np.zeros(
                (len(texts), params.MAX_SENTENCE_COUNT, params.MAX_SENTENCE_LENGTH)
            )
            for i, text in tqdm(enumerate(texts), "Encoding texts in batch"):
                text = normalize(
                    str(text)
                )  # Preprocessing steps applied here: Punct removal, Stopword removal.
                encoded_text = np.array(
                    tf.keras.preprocessing.sequence.pad_sequences(
                        __tokenizer.texts_to_sequences(text), maxlen=params.MAX_SENTENCE_LENGTH
                    )
                )[: params.MAX_SENTENCE_COUNT]
                encoded_texts[i][-len(encoded_text) :] = encoded_text
            return encoded_texts, label

        encode_texts_fn = lambda text, label: encode_texts(text, label, Tout=[tf.float32, tf.int32])

        return encode_texts_fn

    def build(self):
        tokenizer = self._train_tokenizer()
        embedding_matrix = self._create_embedding_matrix(tokenizer)
        encoder_fn = self._encoder_fn(tokenizer)

        train_loader = self.train.map(encoder_fn)
        val_loader = self.val.map(encoder_fn)

        return train_loader, val_loader, tokenizer, embedding_matrix

    def load(self):
        tokenizer = self._load_tokenizer()
        embedding_matrix = self._load_embedding_matrix()

        return tokenizer, embedding_matrix


if __name__ == "__main__":
    pipeline_params = Params(
        "/home/kmanchel/Documents/GitHub/BiasNet/results/hierarchical_attention/params.json"
    )

    pipeline = HAN_DataLoader(pipeline_params)

    train, val, tokenizer, embedding_matrix = pipeline.build()

    x, y = next(iter(train))
    pdb.set_trace()
