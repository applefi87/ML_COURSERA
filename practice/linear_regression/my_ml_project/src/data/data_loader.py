import os
import pandas as pd

def load_dataset(folder_name='raw', file_name='initial_dataset.csv'):
    """
    Load dataset from the specified folder and file name.
    
    Parameters:
    - folder_name (str): The name of the folder (e.g., 'raw', 'processed').
    - file_name (str): The name of the file to load.
    
    Returns:
    - DataFrame: Loaded dataset.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_directory, '..', '..','data', folder_name, file_name)
    
    return pd.read_csv(data_path)