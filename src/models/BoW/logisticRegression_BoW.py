import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append("..")
import pdb
import argparse

import datetime, pickle
import numpy as np
import heapq
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

from preprocess.BoW_pipeline import BoW_Train

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="training data location")
    parser.add_argument("--BoW_path", type=str, help="location to save BoW Tokens")
    parser.add_argument("--model_path", type=int, help="location to save serialized model")
    args = parser.parse_args()

    pipeline = BoW_Train()
    pipeline.preprocess_train(args.train_path, 0.8, args.BoW_path)
    y = pipeline.data["label"].values
    X = np.stack(pipeline.data["features"].values.tolist())
    print("Feature Extraction Done. Now training Model")

    clf = LogisticRegression(random_state=0).fit(X, y)

    dump(clf, args.model_path)
