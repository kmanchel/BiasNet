import unittest
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src/")
from file_utils import is_tf_available
from testing_utils import require_tf


if is_tf_available():

    import datetime, pickle, codecs, re, string
    from tqdm import tqdm
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import string
    from preprocess.utils import (
        Params,
        get_dataset,
        fix_fn,
        _py_fn,
        load_subword_embedding,
        normalize,
    )
    from hierarchical_attention.han_pipeline import HAN_DataLoader


@require_tf
class TestHANLoader(unittest.TestCase):
    def setUp(self):
        """
        Loads
        """
        self.tmp_path = "~/BiasNet/test/tmp/"
        build_params = {
            "train_path": "~/BiasNet/test/fixtures/test.csv",
            "val_path": "~/BiasNet/test/fixtures/test.csv",
            "batch_size": 512,
            "MAX_SENTENCE_LENGTH": 50,
            "MAX_SENTENCE_COUNT": 15,
            "tokenizer_save_path": self.tmp + "tokenizer.pkl",
            "embedding_dims": 300,
            "pretrained_embedding_path": self.tmp + "glove.840B.300d.txt",
            "embeddings_save_path": self.tmp + "hierarchical_attention/embedding_matrix.npy",
            "model_save_path": self.tmp + "hierarchical_attention/model_checkpoints/weights",
            "model_history_path": self.tmp + "/hierarchical_attention/history.pkl",
            "learning_rate": 1e-5,
            "num_epochs": 8,
            "reg_scale": 1e-4,
            "epochs": 1,
            "rnn_type": "GRU",
            "word_embedding_type": "pre_trained",
            "model_type": "HAN",
        }
        self.pipeline_params = Params(build_params)

    def tearDown(self):
        """
        Clears up tmp directory
        """
        cmd1 = "cd {}".format(self.tmp)
        os.system(cmd1)
        os.system("rm -rf *")

    def test_build(self):
        """
        Tests train mode initialization
        """
        pipeline = HAN_DataLoader(self.pipeline_params)
        train, val, tokenizer, embedding_matrix = pipeline.build()

        ##CHECK DATA TYPES
        self.assertEqual(type(train) == "tf.data.Dataset")
        self.assertEqual(type(val) == "tf.data.Dataset")
        self.assertEqual(type(tokenizer) == "tf.keras.preprocessing.text.Tokenizer")

        ##CHECK EMBEDDING SHAPE
        self.assertEqual(embedding_matrix.shape[0] == self.pipeline_params.embedding_dims)

    def test_load(self):
        """
        Tests test mode initialization
        """
        pass

    def test_init(self):
        pass

    def test_train_tokenizer(self):
        pass

    def test_create_embedding_matrix(self):
        pass

    def test_encoder_fn(self):
        pass
