import sys

sys.path.append("../..")

from __future__ import absolute_import, division, print_function, unicode_literals

import datetime, pickle, os, codecs, re, string
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import tensorflow_datasets as tfds
from utils import (
    pickle_object,
    load_pickle,
    Params,
    encode_texts,
    build_train_test,
    load_subword_embedding_300d,
)
from tqdm import tqdm
from preprocess.han_pipeline import HAN_Pipeline

import tensorflow as tf
from tf.keras import regularizers, initializers
from tf.keras import backend as K
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.utils import CustomObjectScope
from tf.keras.layers import Layer, InputSpec
from tf.keras.callbacks import ReduceLROnPlateau, LambdaCallback, ModelCheckpoint

