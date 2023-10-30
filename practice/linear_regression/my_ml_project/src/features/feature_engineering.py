import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import json

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(project_dir, 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)

SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]

def engineer_features(data, y_column_name, degree):
    """Feature engineering function."""
    x_data = data.drop(y_column_name, axis=1)  # Drop the y_column_name
    
    poly = PolynomialFeatures(degree, include_bias=False)
    poly_data = poly.fit_transform(x_data)
    poly_df = pd.DataFrame(poly_data, columns=poly.get_feature_names_out(x_data.columns))
    
    # Concatenate the polynomial features, original y column, and potentially other columns
    data_poly = pd.concat([ poly_df,data[[y_column_name]]], axis=1)
    return data_poly


def process_feature_engineering(input_path, output_path, y_column_name=WEIGHT_COLUMN, degree=1):
    """Load data, engineer features, and save the engineered data."""
    data = pd.read_csv(input_path)
    engineered_data = engineer_features(data, y_column_name, degree)
    engineered_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    DEGREE = config.get("POLY_DEGREE", 1)  # get from config or default to 1
    input_path = os.path.join(project_dir, 'data', 'interim', 'cleaned_data.csv')
    output_path = os.path.join(project_dir, 'data', 'processed', 'features_engineered.csv')
    process_feature_engineering(input_path, output_path, WEIGHT_COLUMN, DEGREE)