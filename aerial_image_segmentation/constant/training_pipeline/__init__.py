import cv2
import torch
from typing import List
from datetime import datetime

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants:
ARTIFACT_DIR: str = "artifacts"
BUCKET_NAME: str = "aerial-image-data"
S3_DATA_FOLDER: str = "data"
S3_FILE_NAME = "archive.zip"

# Data Transformation Constants:
BATCH_SIZE: int = 3
TRAIN_RATIO: int = 0.70
TEST_RATIO: int = 0.20
VALIDATION_RATIO: int = 0.10
TRANSFORM_CONFIG = {
    "TRAIN_TRANSFORMS": {
        "RESIZE": {"HEIGHT": 704, "WIDTH": 1056, "INTERPOLATION": "CV2.INTER_NEAREST"},
        "HORIZONTAL_FLIP": True,
        "GRID_DISTORTION": {"PROBABILITY": 0.2}
    },
    "VAL_TRANSFORMS": {
        "RESIZE": {"HEIGHT": 704, "WIDTH": 1056, "INTERPOLATION": "CV2.INTER_NEAREST"},
        "HORIZONTAL_FLIP": True,
        "GRID_DISTORTION": {"PROBABILITY": 0.2}
    }
}
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"

# External Model Trainer Constants: 1
MODEL_NAME: str = "Unet"
ENCODER_NAME: str = "mobilenet_v2"
ENCODER_WEIGHTS: str = "imagenet"
CLASSES: int = 23
ACTIVATION: str = None
ENCODER_DEPTH: int = 5
DECODER_CHANNELS = [256, 128, 64, 32, 16]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
# Add Other Params Also.....

# External Model Trainer Constants: 2
EPOCH: int = 1
CLASSES: int = 23
MAX_LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
TRAINED_MODEL_NAME: str = "model.pt"
TRAINED_MODEL_DIR: str = "trained_model"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
