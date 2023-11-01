import os
import argparse
import pandas as pd
import joblib

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_dir, 'logs')

# Set up logging
import logging
from src.utils.config_loader import ENVIRONMENT

if ENVIRONMENT == "production":
    logging.basicConfig(filename=os.path.join(log_dir, 'prediction_log.txt'), level=logging.WARNING)
else:
    logging.basicConfig(filename=os.path.join(log_dir, 'training_log.txt'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def make_predictions(model, data_path):
    try:
        data = pd.read_csv(data_path)
        predictions = model.predict(data)
        logging.info(f"Predictions made for data from {data_path}")
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions for data from {data_path}: {str(e)}")
        raise

# Mocking command-line arguments for the notebook
import sys
model_path = os.path.join(project_dir, 'models',"linear_regression_model.pkl")
data_path = os.path.join(project_dir, 'data',"new_data.csv")
output_path = os.path.join(project_dir, "outputs","predictions.csv")
sys.argv = ['script_name', model_path, data_path, '--output_path', output_path]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("data_path", type=str, help="Path to the new data on which predictions should be made.")
    parser.add_argument("--output_path", type=str, default="outputs/predictions.csv", help="Path where predictions will be saved.")
    args = parser.parse_args()
    
    try:
        # Load the trained model
        model = load_model(args.model_path)
    
        # Make predictions on the new data
        predictions = make_predictions(model, args.data_path)
    
        # Check if the file already exists
        if os.path.exists(args.output_path):
            logging.warning(f"{args.output_path} already exists!")
            print(f"Warning: {args.output_path} already exists!")
            should_proceed = input("Do you want to overwrite? (yes/no): ").lower().strip()
            if should_proceed != 'yes':
                logging.info("User aborted due to existing output file.")
                print("Aborting.")
                exit()

        # Save the predictions to disk
        pd.DataFrame(predictions, columns=["Prediction"]).to_csv(args.output_path, index=False)
        logging.info(f"Predictions saved to {args.output_path}")
        print(f"Predictions saved to {args.output_path}")
        
    except Exception as e:
        logging.critical(f"Critical error: {str(e)}")