import os
import pandas as pd
from src.data.data_cleaning import clean_data
import json
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_dir, 'config.json')
with open(config_path, "r") as f:
    config = json.load(f)
SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]

def test_clean_data():
    # Sample raw data with known issues
    raw_data = pd.DataFrame({
        "x": [1, 2, 3, None, 5],
        WEIGHT_COLUMN: [10, 20, None, 40, 50]
    })
    cleaned = clean_data(raw_data,[],["x",WEIGHT_COLUMN])
    cleaned = cleaned.reset_index(drop=True)
    # Expected cleaned data
    expected = pd.DataFrame({
        "x": [1, 2, 5],
        WEIGHT_COLUMN: [10, 20, 50]
    }, dtype='float64')

    pd.testing.assert_frame_equal(cleaned, expected)