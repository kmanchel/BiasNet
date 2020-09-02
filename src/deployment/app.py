import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
from preprocess.BoW_pipeline import BoW_Test

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    features = [pipeline.featurize(x) for x in request.form.values()]
    prediction = model.predict(features)[0]

    output_mapping = {0: "leans Right", 1: "is Neutral", 2: "leans Left"}

    return render_template(
        "index.html", prediction_text="The provided text {}".format(output_mapping[prediction])
    )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    For direct API calls trought request
    """
    data = request.get_json(force=True)
    features = [pipeline.featurize(x) for x in list(data.values())]
    prediction = model.predict(features)[0]

    output_mapping = {0: "leans Right", 1: "is Neutral", 2: "leans Left"}
    output = {"prediction": output_mapping[prediction]}
    print(output)

    return jsonify(output)


if __name__ == "__main__":
    pipeline = BoW_Test(
        tokens_path="/Users/kmanchel/Documents/GitHub/BiasNet/results/models/BoW_tokens.pkl"
    )
    model = joblib.load(
        "/Users/kmanchel/Documents/GitHub/BiasNet/results/models/LR_BoW_test.joblib"
    )
    app.run(debug=True)
