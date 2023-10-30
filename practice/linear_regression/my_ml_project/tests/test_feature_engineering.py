import os
import pandas as pd
from src.features.feature_engineering import engineer_features
import json

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_dir, 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)

SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]

def test_engineer_features():
    # Sample cleaned data
    cleaned_data = pd.DataFrame({
        SIZE_COLUMN: [1, 2, 3],
        WEIGHT_COLUMN: [10, 20, 30]
    }, dtype='float64')
    
    # Assuming engineer_features adds a 'x_squared' feature as an example
    engineered = engineer_features(cleaned_data,WEIGHT_COLUMN,2)
    
    # Expected engineered data
    expected = pd.DataFrame({
        SIZE_COLUMN: [1, 2, 3],
        SIZE_COLUMN+'^2': [1, 4, 9],
        WEIGHT_COLUMN: [10, 20, 30]
    }, dtype='float64')

    pd.testing.assert_frame_equal(engineered, expected)