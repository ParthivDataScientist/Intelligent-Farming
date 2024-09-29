import os 
import sys
import numpy as np 
import pandas as pd 

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
from src.utils import save_object,evaluate_models




@dataclass
class modeltrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerConfig()

    def intiate_model_trainer(self, train_array, test_array):
        
            # Split dataset
            X_train,y_train,X_test,y_test=(
                train_array.iloc[:, :-1],
                train_array.iloc[:,-1],
                test_array.iloc[:, :-1],
                test_array.iloc[:,-1])
            
            #MODELS 
            models ={
                "Random Forest Classifier" : RandomForestClassifier()
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test = X_test, y_test=y_test,
                                                models=models)

            #BEST model score
            best_model_score = max(sorted(model_report.values()))

            #Best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name,best_model_score



            

        

    
