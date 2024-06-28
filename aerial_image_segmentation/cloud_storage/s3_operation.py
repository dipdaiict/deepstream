import os
import sys
from aerial_image_segmentation.exceptions import DataException


class S3Operation:
    # def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
    #     """
    #     Syncs a local folder to an S3 bucket.

    #     Args:
    #     - folder (str): Local folder path to sync.
    #     - bucket_name (str): Name of the S3 bucket.
    #     - bucket_folder_name (str): Folder path within the S3 bucket.

    #     Raises:
    #     - DataException: If syncing fails.
    #     """
    #     try:
    #         command = f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/"
    #         os.system(command)

    #     except Exception as e:
    #         raise DataException(f"Error syncing folder to S3: {e}", sys)

    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_file: str) -> None:
        """
        Synchronizes the contents of a specified folder from an S3 bucket to a local folder.

        Args:
            folder (str): Path to the local folder where downloaded files will be stored.
            bucket_name (str): Name of the S3 bucket containing the folder to download.
            bucket_folder_name (str): Path to the folder within the S3 bucket to download.

        Raises:
            DataException: If an error occurs during the download process.
        """
        try:
            command: str = f"aws s3 cp s3://{bucket_name}/{bucket_folder_file} {folder}"
            os.system(command)

        except Exception as e:
            raise DataException(f"Error syncing folder from S3: {e}", sys)