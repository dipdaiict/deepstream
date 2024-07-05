import os
from torch import device
from typing import List, Optional
from dataclasses import dataclass, field   
from aerial_image_segmentation.constant.training_pipeline import *

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER
        self.s3_file_name: str = S3_FILE_NAME   # It is a Zip File
        self.bucket_name: str = BUCKET_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        self.data_path: str = os.path.join(self.artifact_dir, "data_ingestion")
        self.zip_file_path: str = os.path.join(self.data_path, self.s3_file_name)

        # Ensure the directories exist
        os.makedirs(self.data_path, exist_ok=True)

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.train_ratio: float = TRAIN_RATIO
        self.test_ratio: float = TEST_RATIO
        self.validation_ratio: float = VALIDATION_RATIO
        self.batch_size: int = BATCH_SIZE
        self.transform_config = TRANSFORM_CONFIG
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "data_transformation")
        self.train_transforms_file: str = os.path.join(self.artifact_dir, TRAIN_TRANSFORMS_FILE)

@dataclass
class ExternalModelConfig:
    model_name: str = MODEL_NAME
    encoder_name: str = ENCODER_NAME
    encoder_weights: Optional[str] = ENCODER_WEIGHTS
    classes: int = CLASSES
    activation: Optional[str] = ACTIVATION
    encoder_depth: int = ENCODER_DEPTH
    decoder_channels: List[int] = field(default_factory=lambda: DECODER_CHANNELS)
    device: str = DEVICE

@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: int = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")
        self.trained_bentoml_model_name: str = "segmentation_model"
        self.trained_model_path: int = os.path.join(self.artifact_dir, TRAINED_MODEL_NAME)
        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY
        self.epochs: int = EPOCH
        self.optimizer_params: dict = {"lr": MAX_LR, 'weight_decay': WEIGHT_DECAY}
        self.device: device = DEVICE