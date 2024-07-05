from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader

@dataclass
class DataIngestionArtifact:
     zip_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader
    transformed_test_object: DataLoader
    train_transform_file_path: str

@dataclass
class ModelTrainerArtifact:
     trained_model_path: str
     
@dataclass
class ModelEvaluationArtifact:
    model_accuracy: float