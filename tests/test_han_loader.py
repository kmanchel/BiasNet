import unittest
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")
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
    @classmethod
    def setUpClass(cls):
        """
        Loads
        """
        self.tmp_path = os.path.dirname(os.path.abspath(__file__)) + "/tmp/"
        print("tmp path:", self.tmp_path)
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)

        train_path = os.path.dirname(os.path.abspath(__file__)) + "/fixtures/test.csv"
        print("train path:", train_path)
        val_path = os.path.dirname(os.path.abspath(__file__)) + "/fixtures/test.csv"
        checkpoint_path = (
            os.path.dirname(os.path.abspath(__file__)) + "/fixtures/model_checkpoints/weights"
        )
        build_config = {
            "train_path": train_path,
            "val_path": val_path,
            "batch_size": 5,
            "MAX_SENTENCE_LENGTH": 50,
            "MAX_SENTENCE_COUNT": 15,
            "tokenizer_save_path": self.tmp_path + "tokenizer.pkl",
            "embedding_dims": 300,
            "pretrained_embedding_path": os.path.dirname(os.path.abspath(__file__))
            + "/fixtures/glove.840B.300d.txt",
            "embeddings_save_path": self.tmp_path + "hierarchical_attention/embedding_matrix.npy",
            "model_save_path": checkpoint_path,
            "model_history_path": self.tmp_path + "hierarchical_attention/history.pkl",
            "epochs": 1,
            "rnn_type": "GRU",
        }
        with open(self.tmp_path + "config.json", "w") as fp:
            json.dump(build_config, fp)

        self.pipeline_params = Params(self.tmp_path + "config.json")
        device = tf.config.experimental.list_physical_devices("GPU")[0].name
        train_params = {
            "initiate_model": True,
            "device": device,
            "learning_rate": 0.001,
            "decay": 0.0001,
            "dropout": 0.25,
            "num_epochs": 3,
            "metric": "Accuracy",
            "check_every": 50,
            "val_batch_num": 1,
        }
        self.compile_params = {
            "initiate_model": True,
            "device": device,
            "learning_rate": 0.001,
            "decay": 0.0001,
            "dropout": 0.25,
            "num_epochs": 1,
            "metric": "Accuracy",
            "check_every": 50,
            "val_batch_num": 1,
        }

        print("SETUP DONE")

    @classmethod
    def tearDownClass(cls):
        """
        Clears up tmp directory
        """
        cmd1 = "cd {}".format(self.tmp_path)
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


if __name__ == "__main__":
    unittest.main()