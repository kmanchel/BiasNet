import unittest
import sys, os
import pdb

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")
from file_utils import is_tf_available
from testing_utils import require_tf
import json


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
    from models.hierarchical_attention.han import HAN_Model


@require_tf
class TestHANLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Loads
        """
        cls.tmp_path = os.path.dirname(os.path.abspath(__file__)) + "/tmp/"
        print("tmp path:", cls.tmp_path)
        if not os.path.exists(cls.tmp_path):
            os.mkdir(cls.tmp_path)

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
            "tokenizer_save_path": cls.tmp_path + "tokenizer.pkl",
            "embedding_dims": 300,
            "pretrained_embedding_path": os.path.dirname(os.path.abspath(__file__))
            + "/fixtures/glove.840B.300d.txt",
            "embeddings_save_path": os.path.dirname(os.path.abspath(__file__))
            + "/fixtures/embedding_matrix.npy",
            "model_save_path": checkpoint_path,
            "model_type": "HAN",
            "model_history_path": cls.tmp_path + "hierarchical_attention/history.pkl",
            "epochs": 1,
            "rnn_type": "GRU",
        }
        with open(cls.tmp_path + "config.json", "w") as fp:
            json.dump(build_config, fp)

        cls.pipeline_params = Params(cls.tmp_path + "config.json")
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
        cls.compile_params = {
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
        os.system("rm -rf tmp")

        print("TEARDOWN DONE")

    def test_train_mode(self):
        """
        Tests train mode initialization
        """
        train_mode = True
        model = HAN_Model(self.pipeline_params, train_mode)
        self.assertTrue(hasattr(model, "train_set"))
        self.assertTrue(hasattr(model, "val_set"))
        self.assertTrue(hasattr(model, "tokenizer"))
        self.assertTrue(hasattr(model, "embedding_matrix"))

    def test_predict_mode(self):
        """
        Tests test mode initialization
        """
        train_mode = False
        model = HAN_Model(self.pipeline_params, train_mode)
        # print(model.__dict__)
        self.assertTrue(hasattr(model, "featurize"))
        self.assertTrue(hasattr(model, "tokenizer"))
        self.assertTrue(hasattr(model, "embedding_matrix"))

    def test_load_ckpt(self):
        train_mode = False
        model = HAN_Model(self.pipeline_params, train_mode)
        han_model = model.load_ckpt(self.compile_params)
        # Check Number of Layers
        self.assertEqual(len(han_model.weights), 25)
        # Sanity Check on Tensor type
        self.assertTrue(tf.is_tensor(han_model.weights[3]))
        # Check Final layer shape
        self.assertEqual(han_model.weights[-1].numpy().shape, (3,))
        # Check First layer shape
        self.assertEqual(han_model.weights[0].numpy().shape, (300, 150))

    def test_build_model(self):
        self.pipeline_params.rnn_type = "GRU"
        model = HAN_Model(self.pipeline_params)
        han_model = model.build_model(self.compile_params)
        print(han_model.weights)
        # Check Number of Layers
        self.assertEqual(len(han_model.weights), 25)
        # Sanity Check on Tensor type
        self.assertTrue(tf.is_tensor(han_model.weights[3]))
        # Check Final layer shape
        self.assertEqual(han_model.weights[-1].numpy().shape, (3,))
        # Check First layer shape
        self.assertEqual(han_model.weights[0].numpy().shape, (300, 150))

    def test_train(self):
        pass


if __name__ == "__main__":
    unittest.main()