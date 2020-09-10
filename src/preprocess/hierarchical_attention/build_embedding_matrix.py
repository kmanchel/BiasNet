"""This script builds and saves the embedding matrix from pretrained Word Embeddings"""

import sys

sys.path.append("..")
sys.path

import datetime, pickle, os, codecs, re, string
import json
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import tensorflow_datasets as tfds
from utils import Params, pickle_object, load_pickle
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--emb_path", default="/home/jupyter/nlp/glove.840B.300d.txt")

args = parser.parse_args()

def load_subword_embedding(word_index, emb_path):
    """Creates an embedding matrix from a pretrained word embedding
        Args:
            word_index: (tf.keras.utils.preprocessing.Tokenizer.word_index) 
            emb_path: (str) path to the pretrained word embedding
        Return:
            embedding_matrix: (np.array, shape = (N, like word embedding), dtype = float32) 
    """

    print('load_subword_embedding...')
    
    embeddings_index = {}
    f = codecs.open(emb_path, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    
    return embedding_matrix

    
if __name__ == "__main__":

    params = Params('params.json')
    tokenizer = load_pickle(params.tokenizer_file)
    embeddings_matrix = load_subword_embedding(tokenizer.word_index,args.emb_path)
    pickle_object(embeddings_matrix,params.embedding_matrix)
    print("Created and Saved Embedding Matrix @ %s"%params.embedding_matrix)
    


