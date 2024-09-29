import sys
import pandas as pd 
import numpy as np
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        model_path = 'C://Users//Parthiv//Desktop//Farming//artifacts//model.pkl'
        model=load_object(file_path=model_path)
        preds = model.predict(features)
        return preds
        
    

        


class CustomData:
    def __init__(self,Nitrogen :int, Phosphorus :int, Potassium : int, Temperature:int,
       Humidity:int, pH_Value:int, Rainfall:int):
        
        self.Nitrogen = Nitrogen
        self.Phosphorus = Phosphorus
        self.Potassium = Potassium
        self.Temperature = Temperature
        self.Humidity = Humidity
        self.pH_Value = pH_Value
        self.Rainfall = Rainfall

    def get_data_as_dataframe(self):
        custom_data_input_dict = {
            'Nitrogen': [self.Nitrogen], 
            'Phosphorus': [self.Phosphorus],
            'Potassium': [self.Potassium],
            'Temperature': [self.Temperature], 
            'Humidity': [self.Humidity], 
            'pH_Value': [self.pH_Value], 
            'Rainfall': [self.Rainfall]}
        
        return pd.DataFrame(custom_data_input_dict)