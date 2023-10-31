import os
import json
import pandas as pd
from src.models.train import train_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

YOUR_THRESHOLD = 30

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_path = os.path.join(project_dir,"data","processed","train_data.csv")
config_path = os.path.join(project_dir,"config.json")

with open(config_path,"r") as f:
  config = json.load(f)

SIZE_COLUMN = config["SIZE_COLUMN"]
WEIGHT_COLUMN = config["WEIGHT_COLUMN"]


def test_train_model():
    train_data = pd.DataFrame({
        SIZE_COLUMN: [1, 2, 3, 4, 5],
        WEIGHT_COLUMN: [2, 4, 6, 8, 10]  # perfect linear relationship y = 2x
    })
    
    X_train = train_data[[SIZE_COLUMN]]
    y_train = train_data[WEIGHT_COLUMN]

    model = train_model(X_train, y_train)

    # Basic checks
    assert isinstance(model, LinearRegression)  # Check if the model is of the correct type
    assert model.coef_[0] == 2  # Since y = 2x, the coefficient for x should be close to 2 (might vary slightly due to precision)
    
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    
    # Since the data is perfectly linear, the MSE should be very close to 0
    assert mse < 1e-10
    
def test_train_model_on_actual_data():
    # Load a subset of actual data for testing
    train_data = pd.read_csv(train_data_path).sample(frac=0.1, random_state=42)
    
    X_train = train_data.drop(WEIGHT_COLUMN, axis=1)
    y_train = train_data[WEIGHT_COLUMN]

    model = train_model(X_train, y_train)
    
    # Basic functionality check
    assert model is not None
    
    # Model evaluation (this threshold should be set based on domain knowledge or prior benchmarks)
    predictions = model.predict(X_train)
    mae = mean_absolute_error(y_train, predictions)
    assert mae < YOUR_THRESHOLD  

