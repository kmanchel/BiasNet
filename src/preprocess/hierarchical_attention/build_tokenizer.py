"""This script builds tf.keras Tokenizer and stores as pickle for future use in model pipeline"""

import sys

sys.path.append("..")
sys.path

import tensorflow as tf
import pandas as pd
import numpy as np
import string
import tensorflow_datasets as tfds
from utils import pickle_object, load_pickle, Params
from data_loader import Corpus
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save_batch_every", default=1)

args = parser.parse_args()
    
if __name__ == "__main__":

    params = Params('params.json')
    data = Corpus(params)
    iterator = iter(data.dataset)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='"()*,-/;[\]^_`{|}~', oov_token='UNK',char_level=False)
    necessary_its = (48000//params.batch_size)
    print("Batch iterations: %d"%necessary_its)
    i = 0
    print("\nTokenizer saved as %s"%params.tokenizer_file)
    while i<=necessary_its:
        
        for batch, _ in tqdm(iterator):
            batch = np.char.decode(batch.numpy().astype(np.bytes_), 'UTF-8')
            tokenizer.fit_on_texts(batch)
            if i%args.save_batch_every == 0:
                pickle_object(tokenizer,params.tokenizer_file)

            i+=1
        if necessary_its//i == 2:
            print("HALFWAY DONE")
    print("DONE BUILDING TOKENIZER")
        


