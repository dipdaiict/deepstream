import sys
import time
import torch
import numpy as np
from typing import Tuple
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from . . ml.models.arch import get_model
from . . ml.models.losses import cross_entropy_loss, pixel_accuracy, mean_iou
from aerial_image_segmentation.logger import logging
from . . ml.models.optimizer_setup import create_optimizer
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.entity.artifact_entity import (DataIngestionArtifact,
                                                              DataTransformationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact)
from aerial_image_segmentation.entity.config_entity import ExternalModelConfig, ModelTrainerConfig

# model_external_config = ExternalModelConfig()
# model_trainer_config = ModelTrainerConfig()
class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, 
                 model_trainer_config: ModelTrainerConfig, 
                 model_external_config: ExternalModelConfig):
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config
        self.model_external_config: ExternalModelConfig = model_external_config
        self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        
        try:
            self.model = get_model(self.model_external_config)
            self.device = torch.device(self.model_external_config.device)
            self.model = self.model.to(self.device) 
            logging.info("Model initialized with pre-trained weights.")
        except Exception as e:
            logging.error(f"Error loading model state: {e}")
            raise DataException(f"Error loading model state: {e}")
        self.optimizer = create_optimizer(self.model, **self,model_trainer_config.optimizer_params)  # Initialize optimizer
        self.train_loader = self.data_transformation_artifact.transformed_train_object
        self.val_loader = self.data_transformation_artifact.transformed_val_object

    def get_lr(self, optimizer: Optimizer) -> float:
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def train(self) -> dict:
        # criterion = CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.model_trainer_config.optimizer_params['lr'], epochs=self.model_trainer_config.epochs, steps_per_epoch=len(train_loader))
        optimizer = SGD(self.model.parameters(), **self.model_trainer_config.optimizer_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        losses_train, losses_test = [], []
        train_iou, train_acc, val_iou, val_acc = [], [], []
        lrs = []
        min_loss = np.inf  # Infinity Loss For Comparison
        decreases = 1
        num_of_times_loss_not_improving = 0

        torch.cuda.empty_cache()
        fit_time = time.time()

        for epoch in range(self.model_trainer_config.epochs):
            start_time = time.time()
            running_loss, iou_score, accuracy = 0.0, 0.0, 0.0

            # Training Loop:
            self.model.train()
            for i, data in enumerate(self.train_loader):
                image_tiles, mask_tiles = data
                image, mask = image_tiles.to(self.device), mask_tiles.to(self.device)

                self.optimizer.zero_grad()
                predicted_image = self.model(image)
                loss = cross_entropy_loss(predicted_image, mask)

                # Metrics
                iou_score += mean_iou(predicted_image, mask)
                accuracy += pixel_accuracy(predicted_image, mask)

                loss.backward()
                optimizer.step()

                lrs.append(self.get_lr(optimizer))
                scheduler.step()

                running_loss += loss.item()

            # Validation Loop
            self.model.eval()
            test_loss, test_accuracy, val_iou_score = 0.0, 0.0, 0.0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data
                    image, mask = image_tiles.to(self.device), mask_tiles.to(self.device)

                    predicted_image = self.model(image)
                    loss = criterion(predicted_image, mask)
                    test_loss += loss.item()

                    val_iou_score += mean_iou(predicted_image, mask)
                    test_accuracy += pixel_accuracy(predicted_image, mask)

            # Append metrics
            losses_train.append(running_loss / len(train_loader))
            losses_test.append(test_loss / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_iou.append(val_iou_score / len(val_loader))
            val_acc.append(test_accuracy / len(val_loader))

            # Check for minimum validation loss
            if min_loss > (test_loss / len(val_loader)):
                logging.info(f"Loss Decreasing... {min_loss:.3f} >> {(test_loss / len(val_loader)):.3f}")
                min_loss = test_loss / len(val_loader)
                decreases += 1
                if decreases % 5 == 0:
                    model_save_path = os.path.join(self.model_trainer_config.trained_model_dir,
                                                   f"{self.model_trainer_config.trained_model_name}_epoch_{epoch + 1}.pt")
                    torch.save(self.model.state_dict(), model_save_path)
                    logging.info(f"Model saved as loss is decreasing: {model_save_path}")
            else:
                num_of_times_loss_not_improving += 1
                logging.info(f"Loss Not Decreasing for {num_of_times_loss_not_improving} time(s)")
                if num_of_times_loss_not_improving == 6:
                    logging.info("Loss not decreasing for 6 times, hence stopping training")
                    break

            # Logging epoch results
            logging.info(
                f"Epoch {epoch + 1}/{self.model_trainer_config.epochs}.. "
                f"Train Loss: {running_loss / len(train_loader):.3f}.. "
                f"Validation Loss: {test_loss / len(val_loader):.3f}.. "
                f"Train mean_iou: {iou_score / len(train_loader):.3f}.. "
                f"Validation mean_iou: {val_iou_score / len(val_loader):.3f}.. "
                f"Train Acc: {accuracy / len(train_loader):.3f}.. "
                f"Val Acc: {test_accuracy / len(val_loader):.3f}.. "
                f"Time: {(time.time() - start_time) / 60:.2f}m"
            )

        total_time = (time.time() - fit_time) / 60
        logging.info(f"Total training time: {total_time:.2f}m")

        history = {
            "train_loss": losses_train,
            "val_loss": losses_test,
            "train_miou": train_iou,
            "val_miou": val_iou,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lrs": lrs
        }

        return history