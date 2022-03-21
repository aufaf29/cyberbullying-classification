from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
cors = CORS(app)


import re
import string
import nltk
import pickle
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
from wordcloud import STOPWORDS

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS.update(['did', 'rt', 'will', 'im', 'thing'])

def show_wordcloud(data):
    words = ''
     
    for sentence in data:
     
        tokens = str(sentence).split()
         
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
         
        words += " ".join(tokens) + " "
     
    wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 12).generate(words)
     
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
     
    plt.show()
    
def show_top_ngram(df_column):
    
    vectorizer = TfidfVectorizer(ngram_range=(2,2))

    ngrams = vectorizer.fit_transform(df_column)
    count_values = ngrams.toarray().sum(axis=0)
    vocab = vectorizer.vocabulary_
    df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'bigram/trigram'})

    return df_ngram

def delete_url(text):
    links = re.findall(re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL), text)
    for link in links:
        text = text.replace(link[0], ' ')    
    return text

def delete_mention_tag(text):
               
    # filter kata yang mengandung penanda mention dan hashtag
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in ['@','#']:
                words.append(word)
    return ' '.join(words)

def preprocessing(text):
    text = text.lower()            
    text = delete_url(text)           
    text = delete_mention_tag(text)  
    text = text.strip()
    text = " ".join([word for word in text.split() if not word in set(STOPWORDS)]) 
    text = re.sub(r" \d+ ", " ", text)   
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"  ", " ", text)
    return text

class ModelPipeline:
    def __init__(self, imbalance_handler, vectorizer, encoder, classifier):
        self.imbalance_handler = imbalance_handler
        self.vectorizer = vectorizer
        self.encoder = encoder
        self.classifier = classifier
        
    def encoder(self):
        return self.encoder
    
    def fit(self, X_train, y_train):
        X_train, y_train = self.imbalance_handler.fit_resample(X_train, y_train)
        
        self.vectorizer.fit(X_train['text'].values)
        X_train = self.vectorizer.transform(X_train['text'].values)
        
        self.encoder.fit(y_train)
        y_train = pd.DataFrame(self.encoder.transform(y_train.values.ravel()))
    
        return self.classifier.fit(X_train, y_train.values.ravel())
    
    def predict(self, X_test):
        X_test['text'] = X_test.apply(lambda row: preprocessing(row.text), axis=1)
        X_test = self.vectorizer.transform(X_test['text'].values)
        return self.classifier.predict(X_test)
    
    def predict_proba(self, X_test):
        X_test['text'] = X_test.apply(lambda row: preprocessing(row.text), axis=1)
        X_test = self.vectorizer.transform(X_test['text'].values)
        return self.classifier.predict_proba(X_test)[:, 1]
    
    def classification_report(self, y_test, y_pred):
        y_test = pd.DataFrame(self.encoder.transform(y_test))
        return classification_report(y_test, y_pred, target_names=self.encoder.classes_)
    
    def precision_recall_curve(self, y_test, y_pred):
        y_test = pd.DataFrame(self.encoder.transform(y_test))
        display = PrecisionRecallDisplay.from_predictions(
            y_test, y_pred, name="NaiveBayes", ax = plt.gca()
        )
        res = display.ax_.set_title("Precision-Recall Curve")
        return res
    
    def confusion_matrix_display(self, y_test, y_pred):
        y_test = pd.DataFrame(self.encoder.transform(y_test))
        
        fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16, 8))
        
        display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axn[0], cmap=plt.cm.Blues, normalize=None, display_labels=self.encoder.classes_, xticks_rotation='vertical')
        display.ax_.set_title("Confusion Matrix Non Normalized")
        
        display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axn[1], cmap=plt.cm.Blues, normalize='true', display_labels=self.encoder.classes_, xticks_rotation='vertical')
        display.ax_.set_title("Confusion Matrix Normalized")
        
        plt.show();

@app.route("/", methods=["GET"])
def main():
    response = jsonify({'some': 'data', 'seme': 'datu'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/predict", methods=["POST"])
def predict():
    print("DCDCCDCD")
    if request.method == "POST":
        text = request.form.get("text", False) 
        print(text)
        
        f = open("model/model_biner.pkl", "rb")
        pipeline = pickle.load(f)
        f.close()
        
        pred = pipeline.predict(pd.DataFrame({'text': [text]}))
        
        res = pipeline.encoder.classes_[pred[0]]
        
        response = jsonify({'label': res})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response