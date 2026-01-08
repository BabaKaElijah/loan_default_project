import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

from src.logger.logger import get_logger
from src.exception.exception import CustomException

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting data ingestion")

            raw_path = self.config["data"]["raw_data_path"]
            train_path = self.config["data"]["train_data_path"]
            test_path = self.config["data"]["test_data_path"]

            df = pd.read_csv(raw_path)
            logger.info(f"Raw data loaded with shape {df.shape}")

            train_df, test_df = train_test_split(
                df,
                test_size=self.config["data"]["test_size"],
                random_state=self.config["data"]["random_state"]
            )

            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("Data ingestion completed")

            return train_path, test_path

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
