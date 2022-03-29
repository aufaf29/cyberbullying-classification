from flask import Flask, jsonify, request
from flask_cors import CORS
import dill
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

dill._dill._reverse_typemap['ClassType'] = type


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
        text = request.json['text']
        lang = request.json['lang']
        label = ""
        score = 0.0
        notes = ""
        
        if (lang == "en"):
            file_biner_en = open('model/model_biner.pkl', 'rb')
            model_biner_en = dill.load(file_biner_en)
            file_biner_en.close()
            
            pred_proba_biner = model_biner_en.predict_proba(pd.DataFrame({'text': [text]}))[0]

            file_multi_en = open('model/model_multi.pkl', 'rb')
            model_multi_en = dill.load(file_multi_en)
            file_multi_en.close()
            
            pred_multi = model_multi_en.predict(pd.DataFrame({'text': [text]}))
            pred_multi = model_multi_en.encoder.classes_[pred_multi[0]]
            
            if (pred_proba_biner >= 0.5):
                label = "Not Cyberbullying"
                score = int(pred_proba_biner * 100)
            else:
                label = "Cyberbullying"
                score = 100 - int(pred_proba_biner * 100)
                if (pred_multi == "not_cyberbullying"):
                    notes = "Other Cyberbullying"
                else:
                    notes = " ".join(pred_multi.split("_")).title()
            
        elif(lang == "id"):
            file_biner_id = open('model/model.pkl', 'rb')
            model_biner_id = dill.load(file_biner_id)
            file_biner_id.close()
            
            pred_proba_biner = model_biner_id.predict_proba(pd.DataFrame({'text': [text]}))[0]
            
            if (pred_proba_biner >= 0.5):
                label = "Not Cyberbullying"
                score = int(pred_proba_biner * 100)
            else:
                label = "Cyberbullying"
                score = 100 - int(pred_proba_biner * 100)
        elif(lang == "id-BiLSTM"):
            f = open("model/model_keras_id/tokenizer.pkl", "rb")
            tokenizer = dill.load(f)
            f.close()

            model = load_model('model/model_keras_id/rnn_fasttext.h5')

            data = tokenizer.texts_to_sequences([text])
            data  = sequence.pad_sequences(data, maxlen=50)
            
            with tf.device('/cpu:0'):
                pred_proba_biner = model.predict(data)[0][0]
                
                if (pred_proba_biner >= 0.5):
                    label = "Not Cyberbullying"
                    score = int(pred_proba_biner * 100)
                else:
                    label = "Cyberbullying"
                    score = 100 - int(pred_proba_biner * 100)
                
        
        response = jsonify({
            'text': text,
            'lang': "English" if lang == "en" else "Indonesia",
            'label': label,
            'score': score,
            'notes': notes
        })
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

