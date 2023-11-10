from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import json
import os
import sys
import logging
import pickle
from sklearn.preprocessing import StandardScaler
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
    req_cols = ["sex_0", "cp_0", "cp_2", "cp_3", "fbs_0", "restecg_0", "restecg_2", "exng_1", "slp_0", "slp_2", "caa_0", "caa_2", "caa_3", "caa_4", "thall_0", "thall_2", "thall_3", "sex_1", "cp_1", "fbs_1", "restecg_1", "exng_0", "slp_1", "caa_1", "thall_1" ]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
    con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
    ha_encoded_data = pd.get_dummies(input_df, columns = cat_cols)
    ha_encoded_data = ha_encoded_data.reindex(columns=req_cols, fill_value=0)
    # instantiating the scaler
    scaler = StandardScaler()
    # scaling the continuous features
    app.logger.warning("1:", input_df)
    app.logger.warning("2:", ha_encoded_data)
    ha_encoded_data = ha_encoded_data.join(input_df[con_cols])
    ha_encoded_data[con_cols] = scaler.fit_transform(ha_encoded_data[con_cols])
    app.logger.warning("1:", input_df)
    app.logger.warning("2:", ha_encoded_data)
    model=load_model(model_filename)
    arr_results = model.predict(ha_encoded_data)
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
