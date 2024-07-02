import sys
from aerial_image_segmentation.components.data_ingestion import DataIngestion
from aerial_image_segmentation.components.data_transformation import DataTransformation
# from aerial_image_segmentation.components.model_training import ModelTrainer
# from aerial_image_segmentation.components.model_evaluation import ModelEvaluation
# from aerial_image_segmentation.components.model_pusher import ModelPusher
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.logger import logging

from aerial_image_segmentation.entity.artifact_entity import (DataIngestionArtifact, DataTransformationArtifact)

from aerial_image_segmentation.entity.config_entity import (DataIngestionConfig, DataTransformationConfig)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        
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
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        
        logging.info("Entered the start_data_transformation method of TrainPipeline class")

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config)

            data_transformation_artifact = data_transformation.initiate_data_transformation(artifact_path=data_ingestion_artifact.zip_file_path)

            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class")

            return data_transformation_artifact

        except Exception as e:
            raise DataException(e)
        
# Example usage
if __name__ == "__main__":
    pipeline = TrainPipeline()
    try:
        # pipeline.start_data_ingestion()
        # data_ingestion_artifact = pipeline.start_data_ingestion()
        # Create a dummy DataIngestionArtifact
        data_ingestion_artifact = DataIngestionArtifact(zip_file_path=r"D:\Self-L\deepstream\artifacts\06_28_2024_13_10_32\data_ingestion\archive.zip")
        pipeline.start_data_transformation(data_ingestion_artifact)
    except Exception as e:
        logging.error(f"TrainPipeline failed with error: {e}")