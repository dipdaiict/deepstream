import torch
from typing import List
from datetime import datetime

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants
ARTIFACT_DIR: str = "artifacts"
BUCKET_NAME: str = "aerial-image-data"
S3_DATA_FOLDER: str = "data"
S3_FILE_NAME = "archieve.zip"

# data trasnforamtion

#model trainer constants
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "model.pt"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP_SIZE: int = 6
GAMMA: int = 0.5
EPOCH: int = 1
BENTOML_MODEL_NAME: str = "xray_model"
BENTOML_SERVICE_NAME: str = "xray_service"
BENTOML_ECR_URI: str = "xray_bento_image"

