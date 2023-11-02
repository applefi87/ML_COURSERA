from sklearn.pipeline import Pipeline
from src.data.data_cleaning import DataCleaner
from src.features.feature_engineering import FeatureEngineer

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Print the shape or other info
        print(f"After step '{self.name}', data shape: {X.shape}")
        print(X)
        return X
    
def create_pipeline(columns_to_filter, exclude_columns, y_column_name, degree):
    steps = [
        # ('debug_after_cleaning', DebugTransformer(name="start")),
        ('clean_data', DataCleaner(columns_to_filter, exclude_columns)),
        # ('debug_after_cleaning', DebugTransformer(name="after_cleaning")),
        
        ('feature_engineering', FeatureEngineer(y_column_name, degree)),
        # ('debug_after_fe', DebugTransformer(name="after_feature_engineering"))
    ]
    return Pipeline(steps)
