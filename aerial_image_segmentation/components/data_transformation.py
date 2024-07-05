import os
import re
import sys
import cv2
import torch
import joblib
import tempfile
import shutil
import zipfile
import time
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from typing import Tuple, List
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from aerial_image_segmentation.logger import logging
from aerial_image_segmentation.exceptions import DataException
from aerial_image_segmentation.entity.artifact_entity import (DataIngestionArtifact,
                                                              DataTransformationArtifact)
from aerial_image_segmentation.entity.config_entity import DataTransformationConfig

class DataGen(Dataset):
    def __init__(self, df, mean, std, transform=None, patch=False):
        self.df = df
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        try:
            image_path = self.df.iloc[idx]['image_paths']
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            mask_path = self.df.iloc[idx]['mask_paths']
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = Image.fromarray(augmented["image"])
                mask = augmented["mask"]
            
            if self.transform is None:
                img = Image.fromarray(img)
            
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)
            mask = torch.from_numpy(mask).long()
            
            if self.patches:
                img, mask = self.get_img_patches(img, mask)
            
            return img, mask
        
        except Exception as e:
            logging.error(f"Error occurred while loading data at index {idx}: {e}")
            raise DataException(e, sys)
        
    def get_img_patches(self, img, mask):
        # Implement your patch extraction logic here
        # For demonstration purposes, this function just returns the input as is
        return img, mask
    
class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def unzip_artifact(self, artifact_path: str) -> str:
        try:
            base_dir = os.path.dirname(artifact_path)
            zip_file_name = os.path.basename(artifact_path)
            target_dir = os.path.join(base_dir, os.path.splitext(zip_file_name)[0])
            logging.info(f"Unzipping artifact {artifact_path} to {target_dir}.")

            # Create the target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)

            # Create a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(artifact_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                # Move contents from temp directory to target directory
                for item in os.listdir(tmpdir):
                    s = os.path.join(tmpdir, item)
                    d = os.path.join(target_dir, item)
                    if os.path.exists(d):
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    shutil.move(s, d)
            
            # Change permissions if needed
            self.change_permissions(target_dir)

            logging.info("Artifact unzipped successfully.")
            return target_dir

        except Exception as e:
            logging.error(f"Error occurred while unzipping artifact: {e}")
            raise DataException(str(e))
        
    def change_permissions(self, directory_path):
        try:
            # Check current permissions
            current_permissions = oct(os.stat(directory_path).st_mode & 0o777)
            logging.info(f"Current permissions: {current_permissions} of Artifact: {directory_path}")

            # Change permissions to read/write/execute for owner, group, and others
            if current_permissions != '0o777':
                os.chmod(directory_path, 0o777)
                logging.info(f"Permissions changed to: {oct(os.stat(directory_path).st_mode & 0o777)} of Artifact: {directory_path}")
            else:
                logging.info(f"Permissions are already set to 777of Artifact: {directory_path}")

        except PermissionError as e:
            logging.error(f"Permission denied: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    def return_specific_folders(self, artifact_path: str) -> Tuple[str, str]:
        try:
            unzipped_folder = self.unzip_artifact(artifact_path)
            if not unzipped_folder:
                raise DataException("Failed to unzip artifact or empty unzipped folder.")

            label_images_semantic_path = None
            original_image_path = None

            for root, dirs, files in os.walk(unzipped_folder):
                if 'label_images_semantic' in dirs:
                    label_images_semantic_path = os.path.join(root, 'label_images_semantic')
                    print(f"Found label_images_semantic at: {label_images_semantic_path}")

                if 'original_images' in dirs:
                    original_image_path = os.path.join(root, 'original_images')
                    print(f"Found original_images at: {original_image_path}")

            if not (label_images_semantic_path and original_image_path):
                raise DataException("Required directories not found in the artifact.")

            return label_images_semantic_path, original_image_path
        
        except DataException as de:
            logging.error(f"Data exception: {de}")
            raise
        except Exception as e:
            logging.error(f"Error occurred while searching for specific folders: {e}")
            raise DataException(str(e))
    
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
            print(df.head(2))
            return df
        except Exception as e:
            logging.error(f"Error occurred while creating image info DataFrame: {e}")
            raise DataException(e)
    
    def manage_stats_files(self, directory: str) -> str:
        """Manages stats files in the directory by creating a new file with the next number in sequence."""
        
        def extract_number_from_filename(filename):
            """Extracts the number from a filename in the format 'stats_x.csv'."""
            base_name = os.path.splitext(filename)[0]
            number = int(base_name.split('_')[1])
            return number

        # Find the maximum number from all filenames in the directory
        max_number = 0
        
        for filename in os.listdir(directory):
            if filename.startswith("stats_") and filename.endswith(".csv"):
                number = extract_number_from_filename(filename)
                if number > max_number:
                    max_number = number
        
        # Define the new number and filename
        new_number = max_number + 1
        new_filename = f"stats_{new_number}.csv"
        new_file_path = os.path.join(directory, new_filename)
        return new_file_path

    def calculate_band_stats_and_save_csv(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        band_means, band_stds = [], []
        image_paths = df['image_paths']
        print(f"Image paths: {image_paths[10]}")

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
            output_csv_path = self.manage_stats_files(stats_dir)
            print(f"Writing stats: {output_csv_path}")
            print(f"Mean: {overall_mean} | std: {overall_std}")

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
        
    def train_test_validation_split(self, data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            train_ratio: float = self.data_transformation_config.train_ratio
            test_ratio: float = self.data_transformation_config.test_ratio
            validation_ratio: float = self.data_transformation_config.validation_ratio

            total_ratio = float(train_ratio) + float(test_ratio) + float(validation_ratio)
            tolerance = 1e-8
            assert abs(total_ratio - 1.0) < tolerance, f"Ratios should sum up to 1. Current sum: {total_ratio}"
            # assert total_ratio == 1.0, f"Ratios should sum up to 1. Current sum: {total_ratio}"

            train_df, test_validation_df = train_test_split(data_df, test_size=(test_ratio + validation_ratio))
            test_df, validation_df = train_test_split(test_validation_df, test_size=(validation_ratio / (test_ratio + validation_ratio)))
            total_size, train_size, test_size, validation_size  = len(data_df), len(train_df), len(test_df), len(validation_df)
            
            logging.info(f"Data split into Train: {train_size} ({train_size/total_size:.2%}), Test: {test_size} ({test_size/total_size:.2%}), Validation: {validation_size} ({validation_size/total_size:.2%})")
            return train_df, test_df, validation_df       
        
        except AssertionError as ae:
            logging.error(f"Invalid ratio configuration: {ae}. Train: {train_ratio}, Test: {test_ratio}, Validation: {validation_ratio}")
            raise DataException(ae)
        
        except Exception as e:
            logging.error(f"Error occurred during train-test-validation split: {e}")
            raise DataException(e)
        
    def create_transformations(self) -> Tuple[A.Compose, A.Compose]:
        t_train_transforms = []
        t_val_transforms = []

        train_transforms = self.data_transformation_config.transform_config.get('TRAIN_TRANSFORMS', {})
        val_transforms = self.data_transformation_config.transform_config.get('VAL_TRANSFORMS', {})

        # Define a mapping between transformation names and Albumentations functions
        transform_mapping = {
            'RESIZE': A.Resize,
            'HORIZONTAL_FLIP': A.HorizontalFlip,
            'GRID_DISTORTION': A.GridDistortion,
            # Add more transformations as needed
        }

        # Process train transforms
        for transform_name, transform_params in train_transforms.items():
            if transform_name in transform_mapping:
                transform_fn = transform_mapping[transform_name]
                if transform_name == 'RESIZE':
                    t_train_transforms.append(transform_fn(height=transform_params["HEIGHT"], width=transform_params["WIDTH"], interpolation=cv2.INTER_NEAREST))
                elif isinstance(transform_params, bool) and transform_params:
                    t_train_transforms.append(transform_fn())
                else:
                    t_train_transforms.append(transform_fn(**transform_params))

        # Process val transforms
        for transform_name, transform_params in val_transforms.items():
            if transform_name in transform_mapping:
                transform_fn = transform_mapping[transform_name]
                if transform_name == 'RESIZE':
                    t_val_transforms.append(transform_fn(height=transform_params["HEIGHT"], width=transform_params["WIDTH"], interpolation=cv2.INTER_NEAREST))
                elif isinstance(transform_params, bool) and transform_params:
                    t_val_transforms.append(transform_fn())
                else:
                    t_val_transforms.append(transform_fn(**transform_params))

        t_train = A.Compose(t_train_transforms)
        t_val = A.Compose(t_val_transforms)
        return t_train, t_val
        
    def data_generator(self, train_data_df: pd.DataFrame, mean: List, std: List, val_data_df: pd.DataFrame, transform_train=None, transform_val=None, patch=False) -> Tuple[DataLoader, DataLoader]:
        try:
            train_dataset = DataGen(df=train_data_df, mean=mean, std=std, transform=transform_train, patch=patch)
            val_dataset = DataGen(df=val_data_df, mean=mean, std=std, transform=transform_val, patch=patch)
            # test_dataset = DataGen()
            train_dataloader = DataLoader(train_dataset, batch_size=self.data_transformation_config.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.data_transformation_config.batch_size, shuffle=True)
            return train_dataloader, val_dataloader
        except Exception as e:
            logging.error(f"Error occurred while creating data generator: {e}")
            raise DataException(e)

    def initiate_data_transformation(self, artifact_path: str) -> DataTransformationArtifact:
        try:
            logging.info("Initiating data transformation process...")

            label_images_semantic_path, original_image_path = self.return_specific_folders(artifact_path=artifact_path)

            data_df = self.get_image_info_df(label_images_semantic_path=label_images_semantic_path,
                                              original_image_path=original_image_path)

            band_means, band_stds = self.calculate_band_stats_and_save_csv(data_df)

            train_df, test_df, validation_df = self.train_test_validation_split(data_df)

            t_train, t_val = self.create_transformations()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(t_train, self.data_transformation_config.train_transforms_file)
            logging.info("Transformation files saved.")

            train_dataloader, val_dataloader = self.data_generator(train_data_df=train_df, mean=band_means, std=band_stds, 
                                                                   val_data_df=validation_df, transform_train=t_train, transform_val=t_val)

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                                                                    transformed_train_object=train_dataloader,
                                                                    transformed_test_object=val_dataloader,
                                                                    train_transform_file_path=self.data_transformation_config.train_transforms_file)
            logging.info("Data transformation process completed successfully.")

            return data_transformation_artifact

        except Exception as e:
            logging.error(f"Error occurred during data transformation process: {e}")
            raise DataException(e)
