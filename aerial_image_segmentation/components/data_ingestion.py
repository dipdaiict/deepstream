import sys
from aerial_image_segmentation.cloud_storage.s3_operation import S3Operation
from aerial_image_segmentation.constant.training_pipeline import *
from aerial_image_segmentation.entity.artifact_entity import DataIngestionArtifact
from aerial_image_segmentation.entity.config_entity import DataIngestionConfig
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.logger import logging

class DataIngestion:
    """
    This class handles the process of ingesting data from Amazon S3 for training an aerial image segmentation model.
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the data ingestion configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): The configuration object containing details
                                                        relevant to data ingestion.
        """
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()  # Initialize S3 operation object

    def get_data_from_s3(self) -> None:
        """
        Downloads data from the S3 bucket specified in the configuration.

        Raises:
            DataException: If an error occurs during the download process.
        """
        logging.info("Entered the get_data_from_s3 method of Data ingestion class")

        try:
            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                bucket_folder_name=self.data_ingestion_config.s3_data_folder
            )

            logging.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise DataException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by downloading data from S3 and creating a DataIngestionArtifact.

        Returns:
            DataIngestionArtifact: An object containing the paths to the training and testing data files.

        Raises:
            DataException: If an error occurs during the data ingestion process.
        """
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            self.get_data_from_s3()

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path
            )

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            return data_ingestion_artifact

        except Exception as e:
            raise DataException(e, sys)
