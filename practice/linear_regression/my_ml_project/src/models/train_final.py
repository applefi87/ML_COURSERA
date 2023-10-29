import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

##### 8. Final Model Training & Testing #####
from src.data.data_loader import load_dataset

# Train the model on the full dataset (combining training and validation sets)
data = load_dataset("processed", "features_engineered.csv")
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
project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
output_path_final = os.path.join(project_dir, 'models', 'final_linear_regression_model.pkl')
joblib.dump(final_model, output_path_final)


