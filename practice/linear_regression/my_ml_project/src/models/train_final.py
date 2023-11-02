import os
from sklearn.linear_model import LinearRegression
import joblib

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

logging.info("****Starting final model training on full dataset with hyperparameters:****")

##### 8. Final Model Training & Testing #####
from src.data.data_utils import load_processed_data



try:
  # Train the model on the full dataset (combining training and validation sets)
  data = load_processed_data(data_path)
  X = data.drop('weight', axis=1)
  y = data['weight']

  # Train the final model on the entire dataset
  final_model = LinearRegression()
  final_model.fit(X, y)

  # If you have a separate test set, you can load it and evaluate the final model
  # test_data = load_dataset("processed", "test_dataset.csv")
  # X_test = test_data.drop('weight', axis=1)
  # y_test = test_data['weight']
  # y_test_pred = final_model.predict(X_test)

  # mse_test = mean_squared_error(y_test, y_test_pred)
  # mae_test = mean_absolute_error(y_test, y_test_pred)
  # r2_test = r2_score(y_test, y_test_pred)
  # print(f"Mean Squared Error on Test Set: {mse_test}")
  # print(f"Mean Absolute Error on Test Set: {mae_test}")
  # print(f"R^2 Score on Test Set: {r2_test}")

  # Save the final model
  output_path_final = os.path.join(project_dir, 'models', 'linear_regression_model.pkl')
  joblib.dump(final_model, output_path_final)
  logging.info("****Saving final model to models/linear_regression_model.pkl. End.***")
except Exception as e:
  logging.error(f"Error loading data: {str(e)}")
  raise  


