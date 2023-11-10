from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import json
import os
import sys
import logging
import pickle
model_filename="./heart_attack"
import numpy as np
# Initalise the Flask app
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# instantiate index page
@app.route("/")
def index():
    print(os.getcwd())
    return render_template("./index.html")

# return model predictions
@app.route("/api/predict", methods=["GET"])
def predict():
    msg_data={}
    for k in request.args.keys():
    	val=request.args.get(k)
    	msg_data[k]=val
    f = open("models/X_test.json","r")
    X_test = json.load(f)
    f.close()
    all_cols=X_test
    input_df=pd.DataFrame(msg_data,columns=all_cols,index=[0])
    app.logger.warning("test log", input_df)
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    con_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    ha_encoded_data = pd.get_dummies(input_df, columns = cat_cols)
    app.logger.warning("test log", input_df)
    model=load_model(model_filename)
    arr_results = model.predict(input_df)
    treatment_likelihood=""
    if arr_results[0]==0:
    	treatment_likelihood="No"
    elif arr_results[0]==1:
    	treatment_likelihood="Yes"
    return treatment_likelihood

if __name__ == "__main_":
   app.debug = False
   from werkzeug.serving import run_simple
   run_simple("localhost", 5000, app)
