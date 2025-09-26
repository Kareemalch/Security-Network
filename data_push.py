import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        try:
            # Check if file exists before trying to read it
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == '__main__':
    # Fix 1: Use os.path.join for cross-platform compatibility
    # Fix 2: Use forward slashes or raw string
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")
    # Alternative: FILE_PATH = "Network_Data/phisingData.csv"
    # Alternative: FILE_PATH = r"Network_Data\phisingData.csv"
    
    DATABASE = "sample_mflix"  # Use existing database
    Collection = "NetworkData"  # This will create a new collection
    
    # Debug: Print the current working directory and file path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for file at: {os.path.abspath(FILE_PATH)}")
    print(f"File exists: {os.path.exists(FILE_PATH)}")
    
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(f"Number of records converted: {len(records)}")
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(f"Records inserted to MongoDB: {no_of_records}")