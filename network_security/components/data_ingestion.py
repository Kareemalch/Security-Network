from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataingestionArtifact

import os
import sys
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_data_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace(to_replace=" ", value=np.nan, inplace=True)
            
            return df
        
        except Exception as e:  
            raise NetworkSecurityException(e, sys)
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio
            )
            
            logging.info("Performed train test split on the dataframe")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info("Exporting train and test file path")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, 
                index=False, 
                header=True
            )
            
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, 
                index=False, 
                header=True
            )
            
            logging.info("Exported train and test file path")
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initial_data_ingestion(self):
        try:
            # 1. Export data from MongoDB to DataFrame
            dataframe = self.export_data_as_dataframe()
            logging.info("Exported data from MongoDB as DataFrame")
            
            # 2. Save to feature store (CSV)
            dataframe = self.export_data_into_feature_store(dataframe)
            logging.info("Exported data into feature store")
            
            # 3. Split into train and test
            self.split_data_as_train_test(dataframe)
            logging.info("Train test split completed")
            
            # 4. Create artifact with the file paths
            dataingestionartifact = DataingestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info("Data ingestion artifact created successfully")
            
            return dataingestionartifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)