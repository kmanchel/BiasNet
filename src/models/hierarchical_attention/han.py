from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../preprocess/")
# sys.path.append(os.getcwd() + "/../")
# sys.path.append(os.getcwd() + "/preprocess")

import datetime, pickle, os, codecs, re, string
import tensorflow as tf
import pandas as pd
import numpy as np
import string
from tqdm import tqdm
import pdb

from utils import Params

import logging
from logging import getLogger, DEBUG, INFO, WARNING, ERROR

logging.basicConfig()
logging.root.setLevel(INFO)

from hierarchical_attention.han_pipeline import HAN_DataLoader

import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import ReduceLROnPlateau, LambdaCallback, ModelCheckpoint

from attention import Attention


class HAN_Model:
    def __init__(self, params, train_mode=True):

        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

        self.params = params
        self.train_mode = train_mode
        if self.train_mode:
            print("Building Train Pipeline")
            self._init_Train_pipeline(HAN_DataLoader)
        else:
            print("Building Prediction Pipeline")
            self._init_Pred_pipeline(HAN_DataLoader)

    def _init_Train_pipeline(self, DataLoader):
        self.train_set, self.val_set, self.tokenizer, self.embedding_matrix = DataLoader(
            self.params, self.train_mode
        ).build()

    def _init_Pred_pipeline(self, DataLoader):
        pipeline = DataLoader(self.params, self.train_mode)
        self.tokenizer, self.embedding_matrix = pipeline.load()
        self.featurize = pipeline._encoder_fn(self.tokenizer)

    def load_ckpt(self, compile_params):
        model = self.build_model(compile_params)
        model.load_weights(self.params.model_save_path)
        return model

    def build_model(
        self,
        compile_params,
        n_classes=3,
        embedding_dim=300,
    ):

        l2_reg = regularizers.l2(0.004)

        # embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        embedding_weights = self.embedding_matrix
        assert embedding_weights.shape[1] == embedding_dim

        sentence_in = tf.keras.layers.Input(
            shape=(self.params.MAX_SENTENCE_LENGTH,), dtype="int32", name="input_1"
        )

        embedded_word_seq = tf.keras.layers.Embedding(
            embedding_weights.shape[0],
            embedding_dim,
            weights=[embedding_weights],
            input_length=self.params.MAX_SENTENCE_LENGTH,
            mask_zero=False,
            trainable=False,
            name="word_embeddings",
        )(sentence_in)

        dropout = tf.keras.layers.Dropout(compile_params["dropout"])(embedded_word_seq)

        if self.params.model_type == "CHAN":
            self.logger.info("USING CHAN MODEL")
            filter_sizes = [3, 4, 5]
            convs = []
            for filter_size in filter_sizes:
                conv = tf.keras.layers.Conv1D(
                    filters=64, kernel_size=filter_size, padding="same", activation=tf.nn.leaky_relu
                )(dropout)
                pool = tf.keras.layers.MaxPool1D(filter_size)(conv)
                convs.append(pool)

            concatenate = tf.keras.layers.Concatenate(axis=1)(convs)
            self.logger.info("CONCATENATED")
            dropout = tf.keras.layers.Dropout(0.1)(concatenate)
        else:
            dropout = tf.keras.layers.Dropout(0.1)(embedded_word_seq)

        if self.params.rnn_type is "GRU":
            self.logger.info("GRU MODEL")
            word_encoder = tf.keras.layers.Bidirectional(
                tf.compat.v1.keras.layers.CuDNNGRU(50, return_sequences=True)
            )(dropout)
        else:
            self.logger.info("LSTM MODEL")
            word_encoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(50, return_sequences=True, dropout=compile_params["dropout"])
            )(embedded_word_seq)

        dense_transform_word = tf.keras.layers.Dense(
            100, activation=tf.nn.leaky_relu, name="dense_transform_word", kernel_regularizer=l2_reg
        )(word_encoder)

        # word attention
        attention_weighted_sentence = tf.keras.Model(
            sentence_in, Attention(name="word_attention")(dense_transform_word), name="Word_Level"
        )

        # __word_attention_model = attention_weighted_sentence

        attention_weighted_sentence.summary()

        # sentence-attention-weighted document scores

        texts_in = tf.keras.layers.Input(
            shape=(self.params.MAX_SENTENCE_COUNT, self.params.MAX_SENTENCE_LENGTH),
            dtype="int32",
            name="input_2",
        )

        attention_weighted_sentences = tf.keras.layers.TimeDistributed(attention_weighted_sentence)(
            texts_in
        )

        if self.params.rnn_type is "GRU":
            dropout = tf.keras.layers.Dropout(compile_params["dropout"] / 2)(
                attention_weighted_sentences
            )
            sentence_encoder = tf.keras.layers.Bidirectional(
                tf.compat.v1.keras.layers.CuDNNGRU(50, return_sequences=True)
            )(dropout)
        else:
            sentence_encoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    50,
                    return_sequences=True,
                    dropout=compile_params["dropout"] / 2,
                    recurrent_dropout=0.2,
                )
            )(attention_weighted_sentences)

        dense_transform_sentence = tf.keras.layers.Dense(
            100,
            activation=tf.nn.leaky_relu,
            name="dense_transform_sentence",
            kernel_regularizer=l2_reg,
        )(sentence_encoder)

        # sentence attention
        attention_weighted_text = Attention(name="sentence_attention")(dense_transform_sentence)

        prediction = tf.keras.layers.Dense(n_classes, activation="softmax")(attention_weighted_text)

        model = tf.keras.Model(texts_in, prediction, name="Sentence_Level")
        model.summary()

        # optimizer=tf.keras.optimizers.Adam(lr=compile_params['learning_rate'], decay=0.0001)

        # model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])

        return model

    def train(self, compile_params, model=None):
        """
        Simple training loop for use with models defined using tf.keras. It trains
        a model for one epoch on the training set and periodically checks
        accuracy on the validation set.

        Args:
            - self: looks for self.params, self.tokenizer, self.embedding_matrix
            - compile_params: (dict) hyperparameters for the training process
            - model: (tf.model) initial model to iterate over

        Returns:
            -Model: (tf.model) trained model
            -History: (dict) performance metrics
        """
        with tf.device(compile_params["device"]):
            if compile_params["initiate_model"]:
                model = self.build_model(compile_params)

            # Specify Optimizer
            optimizer = tf.keras.optimizers.Adam(
                lr=compile_params["learning_rate"], decay=compile_params["decay"]
            )

            # Calculate Loss
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            train_loss = tf.keras.metrics.Mean(name="train_loss")
            val_loss = tf.keras.metrics.Mean(name="val_loss")

            # Train Accuracy
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
            val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

            # Logging all the above for plots later on
            train_loss_log = []
            val_loss_log = []
            train_accuracy_log = []
            val_accuracy_log = []

            for epoch in range(compile_params["num_epochs"]):
                t = 0
                msg = "|||train||| Epoch: {}".format(epoch + 1)
                self.logger.info(msg)
                self.logger.debug(msg)
                train_loss.reset_states()
                train_accuracy.reset_states()

                train_dset = iter(self.train_set)

                for x_np, y_np in train_dset:
                    print("Batch")
                    x_np = x_np.numpy()

                    y_np = y_np.numpy()
                    y_np = tf.keras.utils.to_categorical(y_np, num_classes=3, dtype="int32")
                    with tf.GradientTape() as tape:
                        self.logger.debug("Checkpoint 1")

                        # Use the model function to build the forward pass.
                        scores = model(x_np, training=True)
                        self.logger.debug("Checkpoint: Scores Obtained")

                        loss = loss_fn(y_np, scores)
                        self.logger.debug("Debug Checkpoint: Loss Calculated")

                        gradients = tape.gradient(loss, model.trainable_variables)
                        self.logger.debug("Debug Checkpoint: Grads Calculated")

                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        self.logger.debug("Debug Checkpoint: Grads Applied")

                        # Update the metrics
                        train_loss.update_state(loss)
                        train_accuracy.update_state(y_np, scores)
                        msg_loss = "Train Loss: {}".format(train_loss.result())
                        msg_acc = "Train Accuracy: {}".format(train_accuracy.result())
                        self.logger.debug(msg_loss)
                        self.logger.debug(msg_acc)
                        self.logger.debug("Debug Checkpoint: Loss and Accuracy Updated")

                        msg = "We are on val iteration {}".format(t)
                        self.logger.debug(msg)

                        if t % compile_params["check_every"] == 0:
                            tt = 0
                            self.logger.debug("Checking Validation Status")

                            val_loss.reset_states()
                            val_accuracy.reset_states()

                            val_dset = iter(self.val_set)

                            for test_x, test_y in val_dset:
                                test_x = test_x.numpy()

                                test_y = test_y.numpy()
                                test_y = tf.keras.utils.to_categorical(
                                    test_y, num_classes=3, dtype="int32"
                                )
                                prediction = model(test_x, training=False)
                                self.logger.debug("Debug Checkpoint: Made Predictions")

                                t_loss = loss_fn(test_y, prediction)

                                val_loss.update_state(t_loss)
                                val_accuracy.update_state(test_y, prediction)
                                msg_loss = "Val Loss: {}".format(val_loss.result())
                                msg_acc = "Val Accuracy: {}".format(val_accuracy.result())
                                self.logger.debug(msg_loss)
                                self.logger.debug(msg_acc)
                                tt += 1

                                if tt == compile_params["val_batch_num"]:
                                    break

                            template = "|||train||| Iteration {}, Epoch {}, Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}"
                            msg = template.format(
                                t,
                                epoch + 1,
                                train_loss.result(),
                                train_accuracy.result(),
                                val_loss.result(),
                                val_accuracy.result(),
                            )
                            self.logger.info(msg)
                            self.logger.debug(msg)
                        t += 1
                        if t == 1000 // self.params.batch_size:
                            break

                    train_loss_log.append(train_loss.result())
                    val_loss_log.append(val_loss.result())
                    train_accuracy_log.append(train_accuracy.result())
                    val_accuracy_log.append(val_accuracy.result())
                model.save_weights(self.params.model_save_path)
                history = {
                    "loss": {"train": train_loss_log, "validation": val_loss_log},
                    "accuracy": {"train": train_accuracy_log, "validation": val_accuracy_log},
                }
                with open(self.params.model_history_path, "wb") as fp:
                    pickle.dump(history, fp, protocol=pickle.HIGHEST_PROTOCOL)
            return (model, history)
