# This files is use for collecting data from different resource

import os 
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig



from src.components.model_train import modeltrainerConfig
from src.components.model_train import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts",'train.csv')
    test_data_path : str = os.path.join("artifacts",'test.csv')
    raw_data_path : str = os.path.join("artifacts",'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()
        

    def intiate_data_ingestion(self):
        try:
            # connet the file with the location or database or API 
            df=pd.read_csv("Notebook\\data\\Crop_Recommendation.csv")

            #Make the folder and name it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_set,test_set = train_test_split(df, test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            print("Okay")

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data =  obj.intiate_data_ingestion()

    data_trasformation = DataTransformation()
    train_arr, test_arr = data_trasformation.initiate_data_transformation(train_data, test_data)
    
    

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.intiate_model_trainer(train_arr, test_arr))
    