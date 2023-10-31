import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
import logging
from src.utils.config_loader import ENVIRONMENT

if ENVIRONMENT == "production":
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data():
    from src.data.data_loader import load_dataset
    return load_dataset("processed", "train_data.csv")

def train_model(X_train, y_train):
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    logging.info(f"Mean Squared Error on Validation Set: {mse}")
    logging.info(f"Mean Absolute Error on Validation Set: {mae}")
    logging.info(f"R^2 Score: {r2}")

def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    logging.info(f"5-fold cross-validated MSE: {mse_scores}")
    logging.info(f"Mean MSE: {mse_scores.mean()}")
    logging.info(f"Standard Deviation of MSE: {mse_scores.std()}")

def save_model(model, filename='train_linear_regression_model.pkl'):
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    output_path = os.path.join(project_dir, 'models', filename)
    joblib.dump(model, output_path)
    logging.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    ##### 5. Model Selection & Training #####
    data = load_data()
    # Splitting the data
    X = data.drop('weight', axis=1)
    y = data['weight']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    ##### 7. Model Tuning using Cross-Validation ############
    cross_validate_model(model, X_train, y_train)
    ##### 6. Model Evaluation ############
    evaluate_model(model, X_val, y_val)
    # Save the trained model
    save_model(model)