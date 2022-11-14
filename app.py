#from django.shortcuts import renderfrom 
from flask import Flask,app,jsonify,url_for,render_template,request
import numpy as np
import pandas as pd
import recommd_sys as rs

app=Flask(__name__)

@app.route('/')
def home():
    print("hello")
    return render_template('home.html')

@app.route('/getRec',methods=['POST'])
def getRecommendations():
    inputData=request.json['data']
    print(inputData['name'])
    return jsonify(rs.genre_recomm(inputData['name']))
    #inputMovie=
    #print(rs.genre_recomm(inputMovie))
@app.route('/getAllTitles')
def getAllTitles():
    return jsonify(list(rs.getAllTitlesAvailable()))
if __name__=='__main__':
    app.run(debug=True)