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

def drop_na(data):
  """Drop rows with missing values."""
  return data.dropna()
def filter_positive(data,column_name):
  """Filter rows where weight is positive."""
  return data[data[column_name] > 0]
def remove_outliers(data):
  """Remove rows considered as outliers based on IQR."""
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1
  return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

def clean_data(input_path,output_path,column_name="weight"):
  """Main function to clean data."""
  data = pd.read_csv(input_path)
  data =drop_na(data)
  data=filter_positive(data,column_name)
  data=remove_outliers(data)
  data.to_csv(output_path,index=False)
  return data

#The if __name__ == "__main__": block allows you to run the script as a standalone file, 
# but you can also import and use the functions elsewhere without executing the whole script
if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    input_path = os.path.join(project_dir, 'data', 'raw', 'initial_dataset.csv')
    output_path = os.path.join(project_dir, 'data', 'interim', 'cleaned_data.csv')
    
    clean_data(input_path, output_path)