# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 08:41:17 2022

@author: Dell
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('cluster.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    exp1 = int(request.args.get('exp1'))
    exp2 = int(request.args.get('exp2'))
    exp3 = int(request.args.get('exp3'))
    exp4 = int(request.args.get('exp4'))

    prediction = model.predict([[exp1,exp2,exp3,exp4]])
    
    print("K-means prediction",prediction)
    if prediction==[0]:
      print("cluster 0")
    elif prediction==[1]:
      print("cluster 1")
    else:
      print("cluster 2")

        
    return render_template('index.html', prediction_text='k-means cluster = {}'.format(prediction))


app.run()
