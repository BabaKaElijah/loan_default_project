import pandas as pd
import os
import sys
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger.logger import get_logger
from src.exception.exception import CustomException

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = self.config["preprocessing"]["target_column"]
            drop_cols = self.config["preprocessing"]["drop_columns"]
            num_features = self.config["preprocessing"]["numerical_features"]
            cat_features = self.config["preprocessing"]["categorical_features"]

            # Drop identifier columns
            train_df = train_df.drop(columns=drop_cols)
            test_df = test_df.drop(columns=drop_cols)

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
                ]
            )

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            preprocessor_dir = self.config["model"]["preprocessor_dir"]
            os.makedirs(preprocessor_dir, exist_ok=True)
            preprocessor_path = os.path.join(preprocessor_dir, "preprocessor.pkl")
            joblib.dump(preprocessor, preprocessor_path)

            logger.info("Data transformation completed successfully")

            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_path

        except Exception as e:
            logger.error("Error during data transformation")
            raise CustomException(e, sys)
