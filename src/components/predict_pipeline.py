import os
import sys
import joblib
import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import CustomException

logger = get_logger(__name__)

class PredictPipeline:
    def __init__(self, model_dir="artifacts/model"):
        try:
            model_files = [
                f for f in os.listdir(model_dir)
                if f.endswith("_pipeline.pkl")
            ]

            if len(model_files) == 0:
                raise FileNotFoundError("No model pipeline found")

            if len(model_files) > 1:
                logger.warning("Multiple model pipelines found. Using the most recent one.")

            model_candidates = []
            for filename in model_files:
                path = os.path.join(model_dir, filename)
                model_candidates.append((os.path.getmtime(path), filename, path))

            model_candidates.sort()
            _, _, model_path = model_candidates[-1]
            logger.info(f"Loading model pipeline from {model_path}")

            self.model = joblib.load(model_path)

        except Exception as e:
            logger.error("Failed to load model pipeline")
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            logger.info("Starting prediction")

            df = pd.DataFrame([input_data])

            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]

            logger.info("Prediction completed")

            return {
                "prediction": int(prediction),
                "probability": float(probability)
            }

        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)
