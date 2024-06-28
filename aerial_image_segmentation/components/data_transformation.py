import os
import re
import sys
import joblib
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from aerial_image_segmentation.logger import logging
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.entity.artifact_entity import (DataIngestionArtifact,
                                                              DataTransformationArtifact)
from aerial_image_segmentation.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def unzip_artifact(self, artifact_path: str) -> str:
        try:
            logging.info(f"Unzipping artifact {artifact_path} to {os.path.dirname(artifact_path)}.")
            with zipfile.ZipFile(artifact_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(artifact_path))
            logging.info("Artifact unzipped successfully.")

            unzipped_folder = os.path.splitext(artifact_path)[0]  # Assuming artifact is a zip file
            return unzipped_folder

        except Exception as e:
            logging.error(f"Error occurred while unzipping artifact: {e}")
            raise DataException(e, sys)
        
    def return_specific_folders(self, artifact_path: str) -> Tuple[str, str]:
        try:
            unzipped_folder = self.unzip_artifact(artifact_path)
            label_images_semantic_path = None
            original_image_path = None

            for root, dirs, files in os.walk(unzipped_folder):
                if 'label_images_semantic' in dirs:
                    label_images_semantic_path = os.path.join(root, 'label_images_semantic')
                if 'original_image' in dirs:
                    original_image_path = os.path.join(root, 'original_image')
            if not (label_images_semantic_path and original_image_path):
                raise DataException("Required directories not found in the artifact.")
            return label_images_semantic_path, original_image_path
        except Exception as e:
            logging.error(f"Error occurred while searching for specific folders: {e}")
            raise DataException(e, sys)
        
    def get_image_info_df(self, original_image_path: str, label_images_semantic_path: str) -> pd.DataFrame:
        try:
            image_files = os.listdir(original_image_path)
            mask_files = os.listdir(label_images_semantic_path)
            full_image_paths = sorted([os.path.join(original_image_path, img) for img in image_files])
            full_mask_paths = sorted([os.path.join(label_images_semantic_path, mask) for mask in mask_files])
            ids = sorted([os.path.splitext(img)[0] for img in image_files])
            df = pd.DataFrame({
                "ids": ids,
                "image_paths": full_image_paths,
                "mask_paths": full_mask_paths
            })
            return df
        except Exception as e:
            logging.error(f"Error occurred while creating image info DataFrame: {e}")
            raise DataException(e, sys)
        
    def get_versioned_filename(self, base_path: str, base_name: str) -> str:
        version = 1
        existing_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        pattern = re.compile(rf"{base_name}_(\d+).csv")

        for file in existing_files:
            match = pattern.match(file)
            if match:
                version = max(version, int(match.group(1)) + 1)

        return os.path.join(base_path, f"{base_name}_{version}.csv")

    def calculate_band_stats_and_save_csv(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        band_means, band_stds = [], []
        image_paths = df['image_paths']

        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                img_norm = img_array / 255.0
                band_means.append(np.mean(img_norm, axis=(0, 1)))
                band_stds.append(np.std(img_norm, axis=(0, 1)))
            except Exception as e:
                raise DataException(f"Error processing image {img_path}: {e}")

        overall_mean = np.mean(band_means, axis=0)
        overall_std = np.mean(band_stds, axis=0)

        try:
            stats_dir = os.path.join(os.getcwd(), "stats")
            os.makedirs(stats_dir, exist_ok=True)
            output_csv_path = self.get_versioned_filename(stats_dir, "version_name_stats")

            band_stats_df = pd.DataFrame({
                "Band": np.arange(1, overall_mean.shape[0] + 1),
                "Mean": overall_mean,
                "Standard Deviation": overall_std
            })

            band_stats_df.to_csv(output_csv_path, index=False)
            print(f"Band-wise mean and standard deviation saved to {output_csv_path}")

            return band_means, band_stds

        except Exception as e:
            raise DataException(f"Error saving band-wise statistics to CSV: {e}")
        
    def initiate_data_transformation(self, artifact_path: str, extraction_path: str) -> DataTransformationArtifact:
        try:
            logging.info("Initiating data transformation process.")

            unzipped_folder = self.unzip_artifact(artifact_path)

            label_images_semantic_path, original_image_path = self.return_specific_folders(unzipped_folder)

            data_df = self.get_image_info_df(label_images_semantic_path, original_image_path)

            band_means, band_stds = self.calculate_band_stats_and_save_csv(data_df)

            train_transform: transforms.Compose = self.transforming_training_data()
            test_transform: transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file)

            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file)

            logging.info("Transformation files saved.")

            train_loader, test_loader = self.data_loader(
                train_transform=train_transform, test_transform=test_transform
            )

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
            )

            logging.info("Data transformation process completed successfully.")

            return data_transformation_artifact

        except Exception as e:
            logging.error(f"Error occurred during data transformation process: {e}")
            raise DataException(e, sys)
