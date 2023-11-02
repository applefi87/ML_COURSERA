import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_dir, 'data', 'raw', 'initial_dataset.csv')
log_dir = os.path.join(project_dir, 'logs')
# Set up logging
import logging
from src.utils.config_loader import ENVIRONMENT

if ENVIRONMENT == "cloud_training":
    logging.basicConfig(filename=os.path.join(log_dir, 'cloud_training_log.txt'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
else:  # Local training environment
    logging.basicConfig(filename=os.path.join(log_dir, 'training_log.txt'), level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("****Starting experimental model training with hyperparameters:") 

from src.data.data_utils import load_processed_data
def train_model(X_train, y_train):
    try:
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise  

def evaluate_model(model, X_val, y_val):
    try:
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logging.info(f"Best model performance on validation set: ")
        logging.info(f"Mean Squared Error on Validation Set: {mse}")
        logging.info(f"Mean Absolute Error on Validation Set: {mae}")
        logging.info(f"R^2 Score: {r2}")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise  

def cross_validate_model(model, X_train, y_train):
    try:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -scores
        logging.info(f"5-fold cross-validated MSE: {mse_scores}")
        logging.info(f"Mean MSE: {mse_scores.mean()}")
        logging.info(f"Standard Deviation of MSE: {mse_scores.std()}")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise  

def save_model(model, filename='train_linear_regression_model.pkl'):
    try:
        output_path = os.path.join(project_dir, 'models', filename)
        joblib.dump(model, output_path)
        logging.info(f"Model saved to {output_path}")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise  
if __name__ == "__main__":
    try:
        ##### 5. Model Selection & Training #####
        data = load_processed_data(data_path)
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
        logging.info(f"****Saved model. End****")
    except Exception as e:
        # This will capture any unexpected error that wasn't caught by the function-specific handlers
        logging.error(f"Unexpected error during training: {str(e)}")