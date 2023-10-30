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
import os
# This way if later the csv columns name change,you just need to change config.json. Or you can directly declare here.
# If directly write size into function like filter_positive(data,column_name="size"), would hard for modification in future and for test.
import json
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(project_dir, 'config.json')
with open(config_path, "r") as f:
    config = json.load(f)
SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]
#

def convert_types(data,column_names,type='int32'):
    """Convert data types for certain columns."""
    for column in column_names:
        data[column] = data[column].astype(type)
    return data
  
def drop_na(data):
  """Drop rows with missing values."""
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

def clean_data(data, columns_to_filter=[],columns_change_types=[]):
    """Clean data."""
    data = drop_na(data)
    data = filter_positive(data, columns_to_filter,)
    data = remove_outliers(data)
    data = convert_types(data,columns_change_types) 
    return data

def process_data(input_path, output_path, columns_to_filter=[SIZE_COLUMN, WEIGHT_COLUMN], columns_change_types=[]):
    """Load, clean, and save data."""
    data = pd.read_csv(input_path)
    cleaned_data = clean_data(data,columns_to_filter,columns_change_types)
    cleaned_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = os.path.join(project_dir, 'data', 'raw', 'initial_dataset.csv')
    output_path = os.path.join(project_dir, 'data', 'interim', 'cleaned_data.csv')
    
    process_data(input_path, output_path)