import os
import sys
import json
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv
from network_security.exception.exception import NetworkSecurityException

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class NetworkDataExtract():
    
    def csv_to_json(self, csv_path, json_path):
        """Convert CSV to JSON file"""
        try:
            data = pd.read_csv(csv_path)
            records = data.to_dict('records')
            with open(json_path, 'w') as f:
                json.dump(records, f, indent=4)
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def json_to_mongodb(self, json_path, database, collection):
        """Insert JSON into MongoDB"""
        try:
            with open(json_path, 'r') as f:
                records = json.load(f)
            
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
            client[database][collection].insert_many(records)
            client.close()
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == '__main__':
    extractor = NetworkDataExtract()
    
    # Step 1: CSV to JSON
    records = extractor.csv_to_json(
        "Network_Data/phisingData.csv",
        "Network_Data/phisingData.json"
    )
    print(f"✓ Saved {len(records)} records to JSON")
    
    # Step 2: JSON to MongoDB
    count = extractor.json_to_mongodb(
        "Network_Data/phisingData.json",
        "KareemAI",
        "NetworkData"
    )
    print(f"✓ Inserted {count} records to MongoDB")