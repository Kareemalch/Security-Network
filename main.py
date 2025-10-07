from network_security.components.data_ingestion import DataIngestion
from network_security.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging



 

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initial_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)