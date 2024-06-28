import sys
from aerial_image_segmentation.components.data_ingestion import DataIngestion
# from aerial_image_segmentation.components.data_transformation import DataTransformation
# from aerial_image_segmentation.components.model_training import ModelTrainer
# from aerial_image_segmentation.components.model_evaluation import ModelEvaluation
# from aerial_image_segmentation.components.model_pusher import ModelPusher
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.logger import logging

from aerial_image_segmentation.entity.artifact_entity import (
    DataIngestionArtifact,
    )

from aerial_image_segmentation.entity.config_entity import (DataIngestionConfig)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting data ingestion process...")
        
        try:
            logging.info("Initializing data ingestion module...")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            
            logging.info("Initiating data ingestion from S3...")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        
        except DataException as de:
            logging.error(f"DataException occurred during data ingestion: {de}")
            raise  # Re-raise the exception for higher-level handling
        
        except Exception as e:
            logging.error(f"Unexpected error during data ingestion: {e}")
            raise  # Re-raise the exception for higher-level handling

        finally:
            logging.info("Exiting data ingestion process.")

# Example usage
if __name__ == "__main__":
    pipeline = TrainPipeline()
    try:
        pipeline.start_data_ingestion()
    except Exception as e:
        logging.error(f"TrainPipeline failed with error: {e}")