from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.utils import read_yaml

if __name__ == "__main__":
    # Load config
    config = read_yaml("config/config.yaml")

    # Data ingestion
    ingestion = DataIngestion(config)
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Data transformation
    transformation = DataTransformation(config)
    X_train, X_test, y_train, y_test, preprocessor_path = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    # Model training
    trainer = ModelTrainer(config)  # <-- pass config here
    model_path, best_model_name, model_report = trainer.train_and_select_best(
        X_train, y_train, X_test, y_test
    )
