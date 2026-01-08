import yaml
import sys
from src.exception.exception import CustomException

def read_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)
