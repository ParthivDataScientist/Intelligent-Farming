from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipline.prediction_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app = application

#Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(

            Nitrogen=request.form.get("Nitrogen"),
            Phosphorus=request.form.get("Phosphorus"),
            Potassium=request.form.get("Potassium"),
            Temperature=request.form.get("Temperature"),
            Humidity=request.form.get("Humidity"),
            pH_Value=request.form.get("pH_Value"),
            Rainfall=request.form.get("Rainfall"))
        
        features= data.get_data_as_dataframe()
        print(features)

        predict_pipline=PredictPipeline()
        result = predict_pipline.predict(features=features)
        return render_template("home.html", results = result[1])
        return 

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
