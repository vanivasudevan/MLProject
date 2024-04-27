import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass   # used to create class variable

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass  # can define the class varible directly without __init__
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')
class DataIngestion:
    def __init__(self):
     self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion  component")
        try:
          df = pd.read_csv('notebook/data/stud.csv')
          logging.info("Read the dataset as dataframe")
          os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

          df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
          logging.info("Train and Test split initiated")

          train_set,test_set = train_test_split(df,test_size=20,random_state=42)
          train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
          test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
          logging.info("Ingestion of the data is completed")

          return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        
        except Exception as e:
           raise CustomException(e,sys)
        


if __name__ == "__main__":
   obj = DataIngestion()
   train_data,test_data=obj.initiate_data_ingestion()
   data_trainsformation = DataTransformation()
   train_arr,test_arr,_ = data_trainsformation.initiate_data_transformation(train_data,test_data)
   modeltrainer=ModelTrainer()
   print(modeltrainer.initiate_model_trainer(train_arr,test_arr)) 