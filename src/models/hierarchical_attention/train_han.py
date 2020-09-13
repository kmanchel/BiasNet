from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys

sys.path.append(os.getcwd() + "/../")
sys.path.append(os.getcwd() + "/preprocess")

import logging
from logging import getLogger, DEBUG, INFO, WARNING, ERROR

logging.basicConfig()
logging.root.setLevel(INFO)
import pdb

from utils import Params
from han import HAN_Model

pipeline_params = Params(
    "/home/kmanchel/Documents/GitHub/BiasNet/results/hierarchical_attention/params.json"
)
train_mode = True
device = "/device:GPU:3"
model = HAN_Model(pipeline_params, train_mode)
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

tf_model, history = model.train(train_params)