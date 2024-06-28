import os
from dataclasses import dataclass
from torch import device
from aerial_image_segmentation.constant.training_pipeline import *

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER
        self.s3_data_file: str = S3_FILE_NAME   # It is a Zip File
        self.bucket_name: str = BUCKET_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        self.data_path: str = os.path.join(self.artifact_dir, "data_ingestion")
        self.zip_file_path = os.path.join(self.data_path, self.s3_data_file)