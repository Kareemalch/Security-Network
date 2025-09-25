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
            raise NetworkSecurityException(e,sys)
        

    def cv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())  # Fixed: removed extra .values()
            return records  # Fixed: added return statement
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
         try:
            self.records = records
            self.database = database
            self.collection = collection
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
         except Exception as e:
              raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    # Debug: Check current directory and file existence
    print("Current working directory:", os.getcwd())
    
    # Create directory if it doesn't exist
    os.makedirs("Network_Data", exist_ok=True)
    
    FILE_PATH = "Network_Data/phisingData.csv"
    
    # Check if file exists
    print(f"Looking for file at: {os.path.abspath(FILE_PATH)}")
    print(f"File exists: {os.path.exists(FILE_PATH)}")
    
    if not os.path.exists(FILE_PATH):
        print(f"Please place your CSV file at: {os.path.abspath(FILE_PATH)}")
        # List files in Network_Data directory to help debug
        if os.path.exists("Network_Data"):
            files = os.listdir("Network_Data")
            if files:
                print(f"Files found in Network_Data directory: {files}")
            else:
                print("Network_Data directory is empty")
        exit(1)
    
    DATABASE = "AI"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    
    # Get records from CSV
    records = networkobj.cv_to_json_convertor(file_path=FILE_PATH)
    print(f"Number of records loaded: {len(records) if records else 0}")
    print("Sample record:", records[0] if records else "No records")
    
    # Insert into MongoDB
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(f"Number of records inserted: {no_of_records}")