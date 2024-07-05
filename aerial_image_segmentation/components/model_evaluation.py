import sys
import torch
from typing import Tuple
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Module
from aerial_image_segmentation.logger import logging
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.entity.config_entity import ModelEvaluationConfig
from aerial_image_segmentation.entity.artifact_entity import (DataTransformationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact)

class ModelEvaluation:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact,):
                self.data_transformation_artifact = data_transformation_artifact
                self.model_evaluation_config = model_evaluation_config
                self.model_trainer_artifact = model_trainer_artifact

