##The code copy from initial_eda.ipynb, remove print & plt.
# import pandas as pd
# import os
# current_data_dir = os.getcwd()
# src_dir = os.path.dirname(current_data_dir)
# project_dir = os.path.dirname(src_dir)
# data_path = os.path.join(project_dir, 'data', 'raw', 'initial_dataset.csv')
# data = pd.read_csv(data_path)
# data_dropped_na = data.dropna()
# data_positive_weight = data_dropped_na[data_dropped_na['weight'] > 0]
# dt = data_positive_weight
# Q1 = data.quantile(0.25)
# Q3 = data.quantile(0.75)
# IQR = Q3 - Q1
# outliers_count = ((dt < (Q1 - 1.5 * IQR)) | (dt > (Q3 + 1.5 * IQR))).sum()
# data_no_outliers = dt[~((dt < (Q1 - 1.5 * IQR)) | (dt > (Q3 + 1.5 * IQR))).any(axis=1)]

## Build into function:
import pandas as pd
from src.utils.data_utils import convert_to_float
import os
import logging
from src.utils.config_loader import ENVIRONMENT
from sklearn.base import BaseEstimator, TransformerMixin
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_dir, 'logs')
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
if ENVIRONMENT == "production":
    logging.basicConfig(filename=os.path.join(log_dir, 'production_log.txt'), level=logging.ERROR, format=LOGGING_FORMAT)
else:
    logging.basicConfig(filename=os.path.join(log_dir, 'development_log.txt'), level=logging.DEBUG, format=LOGGING_FORMAT)


# This way if later the csv columns name change,you just need to change config.json. Or you can directly declare here.
# If directly write size into function like filter_positive(data,column_name="size"), would hard for modification in future and for test.
import json
config_path = os.path.join(project_dir, 'config.json')
with open(config_path, "r") as f:
    config = json.load(f)
SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]


#
def drop_na(data):
    """Drop rows with missing values during training. Fill NA with mean during prediction."""
    if ENVIRONMENT == 'production':
        return data.fillna(data.mean())
    else:
        return data.dropna()
def filter_positive(data, column_names):
    """Filter rows where values in specified columns are positive."""
    for column in column_names:
        data = data[data[column] > 0]
    return data
def remove_outliers(data):
  """Remove rows considered as outliers based on IQR."""
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1
  return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

def clean_data(data, columns_to_filter=[],exclude_columns=[]):
    """Clean data."""
    data = convert_to_float(drop_na(data),exclude_columns)
    data = filter_positive(data, columns_to_filter,)
    data = remove_outliers(data)
    data = data.reset_index(drop=True)
    return data

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_filter=[], exclude_columns=[]):
        self.columns_to_filter = columns_to_filter
        self.exclude_columns = exclude_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return clean_data(X, self.columns_to_filter,self.exclude_columns)

def process_data(input_path, output_path, columns_to_filter=[SIZE_COLUMN, WEIGHT_COLUMN], exclude_columns=[]):
    """Load, clean, and save data."""
    data = pd.read_csv(input_path)
    cleaned_data = clean_data(data,columns_to_filter,exclude_columns)
    cleaned_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = os.path.join(project_dir, 'data', 'raw', 'initial_dataset.csv')
    output_path = os.path.join(project_dir, 'data', 'interim', 'cleaned_data.csv')
    process_data(input_path, output_path)