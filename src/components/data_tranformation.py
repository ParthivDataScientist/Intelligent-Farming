import os 
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    

    def initiate_data_transformation(self,train_path,test_path):
        
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            target_column_name ="Crop"



            

            return(
                 train_df, test_df
            )

