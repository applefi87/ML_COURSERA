import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main(input_path, train_output_path, test_output_path, test_size=0.2, random_state=None):
    """Load cleaned data, split it, and save training and testing sets."""
    data = pd.read_csv(input_path)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    input_path = os.path.join(project_dir, 'data', 'processed', 'features_engineered.csv')
    train_output_path = os.path.join(project_dir, 'data', 'processed', 'train_data.csv')
    test_output_path = os.path.join(project_dir, 'data', 'processed', 'test_data.csv')
    
    main(input_path, train_output_path, test_output_path)