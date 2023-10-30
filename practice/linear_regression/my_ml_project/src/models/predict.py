import os
import argparse
import pandas as pd
import joblib


def load_model(model_path):
    model = joblib.load(model_path)
    return model

def make_predictions(model, data_path):
    """
    Use the trained model to make predictions on new data.
    """
    data = pd.read_csv(data_path)
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("--model_path", type=str, default="models/final_linear_regression_model.pkl", help="Path to the trained model file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the new data on which predictions should be made.")
    parser.add_argument("--output_path", type=str, default="outputs/predictions.csv", help="Path where predictions will be saved.")
    args = parser.parse_args()
    
    
    # Load the trained model
    model = load_model(args.model_path)
    
    # Make predictions on the new data
    predictions = make_predictions(model, args.data_path)
    
    # Check if the file already exists
    if os.path.exists(args.output_path):
        print(f"Warning: {args.output_path} already exists!")
        should_proceed = input("Do you want to overwrite? (yes/no): ").lower().strip()
        if should_proceed != 'yes':
            print("Aborting.")
            exit()

    # Save the predictions to disk
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")