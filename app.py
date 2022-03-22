from flask import Flask, jsonify, request
from flask_cors import CORS
import dill

import pandas as pd

app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods=["GET"])
def main():
    response = jsonify({'message': 'Please user /predict or /predict-multi with text as request.'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        with open('model/model_biner.pkl', 'rb') as f:
            pipeline = dill.load(f)
            pred = pipeline.predict(pd.DataFrame({'text': [request.json['text']]}))
            res = pipeline.encoder.classes_[pred[0]]
            response = jsonify({'label': res})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
@app.route("/predict-multi", methods=["POST"])
def predictx():
    if request.method == "POST":
        with open('model/model_multi.pkl', 'rb') as f:
            pipeline = dill.load(f)
            pred = pipeline.predict(pd.DataFrame({'text': [request.json['text']]}))
            res = pipeline.encoder.classes_[pred[0]]
            response = jsonify({'label': res})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

