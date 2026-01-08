import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from src.logger.logger import get_logger
from src.exception.exception import CustomException

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        rs = self.config["data"]["random_state"]

        self.models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=rs
            ),
            "DecisionTree": DecisionTreeClassifier(
                random_state=rs,
                class_weight="balanced",
                min_samples_leaf=50
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                random_state=rs,
                class_weight="balanced",
                min_samples_leaf=50
            )
        }

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

    def train_and_select_best(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting model training and evaluation")

            model_report = {}
            trained_models = {}

            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}")
                model.fit(X_train, y_train)

                metrics = self.evaluate_model(model, X_test, y_test)
                model_report[model_name] = metrics
                trained_models[model_name] = model

                logger.info(f"{model_name} metrics {metrics}")

            best_model_name = max(
                model_report,
                key=lambda m: model_report[m]["f1"]
            )
            best_f1 = model_report[best_model_name]["f1"]

            log_f1 = model_report["LogisticRegression"]["f1"]
            if log_f1 >= best_f1 - 0.05:
                best_model_name = "LogisticRegression"
                best_f1 = log_f1

            best_model = trained_models[best_model_name]

            logger.info(f"Selected model {best_model_name}")
            logger.info(f"Final F1 score {best_f1}")

            preprocessor_path = os.path.join(
                self.config["model"]["preprocessor_dir"],
                "preprocessor.pkl"
            )

            preprocessor = joblib.load(preprocessor_path)

            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", best_model)
            ])

            model_dir = self.config["model"]["model_dir"]
            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(
                model_dir,
                f"{best_model_name}_pipeline.pkl"
            )

            joblib.dump(full_pipeline, model_path)

            logger.info(f"Saved pipeline at {model_path}")

            return model_path, best_model_name, model_report

        except Exception as e:
            logger.error("Error during model training")
            raise CustomException(e, sys)
