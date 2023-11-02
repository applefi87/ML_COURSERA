import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_dir, 'logs')
# Set up logging
import logging
from src.utils.config_loader import ENVIRONMENT
if ENVIRONMENT == "cloud_training":
    logging.basicConfig(filename=os.path.join(log_dir, 'cloud_training_log.txt'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
else:  # Local training environment
    logging.basicConfig(filename=os.path.join(log_dir, 'training_log.txt'), level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("****Starting final model training on full dataset with hyperparameters:****")

import json
config_path = os.path.join(project_dir, 'config.json')
with open(config_path, "r") as f:
    config = json.load(f)
SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]
POLY_DEGREE = config.get("POLY_DEGREE", 1)  # get from config or default to 1

from src.preprocessing_pipeline import create_pipeline
def load_processed_data(raw_data_path,columns_to_filter=[SIZE_COLUMN, WEIGHT_COLUMN], exclude_columns=[], y_column_name=WEIGHT_COLUMN, degree=POLY_DEGREE):
    try:
        raw_data = pd.read_csv(raw_data_path)
        pipeline = create_pipeline(columns_to_filter, exclude_columns, y_column_name,degree )
        processed_data = pipeline.transform(raw_data)
        return processed_data
    except Exception as e:
        logging.error(f"Error loading and processing data: {str(e)}")
        raise  