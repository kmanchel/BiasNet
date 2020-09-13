import sys
import os
import pdb

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import tensorflow as tf
from preprocess.utils import Params
from preprocess.BoW.BoW_pipeline import BoW_Test
from preprocess.hierarchical_attention.han_pipeline import HAN_DataLoader
from models.hierarchical_attention.han import HAN_Model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_UI", methods=["POST"])
def predict_UI():
    """
    For rendering results on HTML GUI
    """
    model_id = request.form["model_id"]
    text = request.form["text"]

    model_output = parse_and_predict(model_id, text)
    msg = "The Provided Article {}.".format(model_output)
    return render_template(
        "index.html",
        prediction_text=msg,
    )


@app.route("/predict_API", methods=["POST"])
def predict_api():
    """
    For direct API calls trought request
    """
    data = request.get_json(force=True)
    model_id = data["model_id"]
    text = data["text"]
    model_output = parse_and_predict(model_id, text)

    output = {"prediction": model_output}

    return jsonify(output)


def parse_and_predict(model_id, text):
    """
    Args:
        model_id: (int) Identification number to call a given model.
                        1. Bag of Words
                        2. Hierarchical Attention for Document Classification
                        3. BERT for Document Classification
        text: (str) Input article to make predictions on.
    Returns:
        model_output: (str) Political Leaning (for now)
    """

    output_mapping = {0: "leans Left", 1: "is Neutral", 2: "leans Right"}

    if int(model_id) == 1:
        pipeline = BoW_Test(
            tokens_path="/home/kmanchel/Documents/GitHub/BiasNet/results/BoW/BoW_tokens.pkl"
        )
        model = joblib.load(
            "/home/kmanchel/Documents/GitHub/BiasNet/results/BoW/LR_BoW_test.joblib"
        )
        features = [pipeline.featurize(x) for x in text]
        prediction = model.predict(features)[0]

    elif int(model_id) == 2:
        pipeline_params = Params(
            "/home/kmanchel/Documents/GitHub/BiasNet/results/hierarchical_attention/params.json"
        )
        train_mode = False

        model = HAN_Model(pipeline_params, train_mode)
        features, _ = model.featurize([text], [0])

        han_model = model.load_ckpt({"dropout": 0.25})

        prediction = han_model.predict([features])[0].argmax()

    else:
        prediction = "Prediction Not Found"

    return output_mapping[prediction]


if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True)
