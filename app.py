
from flask import Flask,render_template,request
import numpy as np
import joblib

model=joblib.load("knnmodel.pkl")


app=Flask(__name__)      #we have created an object of the Flask class to initiate server

@app.route('/predict' ,methods=['POST','GET'])
def predict():

    td=[float(x) for x in request.form.values()]
    #above statementis called List compreshension
    #it helps to fetch parameter from front end, copies to list td

    testdata=np.array([td])

    result=model.predict(testdata)

    msg=f"Recomended crop is: {result[0]}"
    return render_template('crop.html',res=msg)

@app.route('/')
def index():

    return render_template('crop.html')


app.run(debug=True,host="0.0.0.0")