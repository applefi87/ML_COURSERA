import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import json
from sklearn.base import BaseEstimator, TransformerMixin

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(project_dir, 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)

SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]
POLY_DEGREE = config.get("POLY_DEGREE", 1)  # get from config or default to 1
    
def engineer_features(data, y_column_name=None, degree=1):
    """Feature engineering function."""
    x_data = data.copy()
    if y_column_name and y_column_name in data.columns:
        x_data = data.drop(y_column_name, axis=1)  # Drop the y_column_name if it exists

    poly = PolynomialFeatures(degree, include_bias=False)
    poly_data = poly.fit_transform(x_data)
    poly_df = pd.DataFrame(poly_data, columns=poly.get_feature_names_out(x_data.columns))
    
    # Concatenate the polynomial features, original y column (if it exists), and potentially other columns
    if y_column_name and y_column_name in data.columns:
        data_poly = pd.concat([poly_df, data[[y_column_name]]], axis=1)
    else:
        data_poly = poly_df
    
    return data_poly

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, y_column_name, degree=1):
        self.y_column_name = y_column_name
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return engineer_features(X, self.y_column_name, self.degree)

def process_feature_engineering(input_path, output_path, y_column_name=WEIGHT_COLUMN, degree=1):
    """Load data, engineer features, and save the engineered data."""
    data = pd.read_csv(input_path)
    engineered_data = engineer_features(data, y_column_name, degree)
    engineered_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = os.path.join(project_dir, 'data', 'interim', 'cleaned_data.csv')
    output_path = os.path.join(project_dir, 'data', 'processed', 'features_engineered.csv')
    process_feature_engineering(input_path, output_path, WEIGHT_COLUMN, POLY_DEGREE)