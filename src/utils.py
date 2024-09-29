# common function that the whole project has 
import pandas as pd 
import os 
import numpy as np 
import dill

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except:
        pass


def load_object(file_path):
    with open(file_path,'rb') as file_obj:
        return dill.load(file_obj)

        

        
def evaluate_models(X_train, y_train, X_test, y_test, models):
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train,y_train_pred)

            test_model_socre = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_socre

        return report

     