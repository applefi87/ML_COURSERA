import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

##### 5. Model Selection & Training #####
from src.data.data_loader import load_dataset
data = load_dataset("processed","features_engineered.csv")

# Splitting the data
X = data.drop('weight', axis=1)
y = data['weight']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

###### 6. Model Evaluation ############
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Mean Squared Error on Validation Set: {mse}")
print(f"Mean Absolute Error on Validation Set: {mae}")
print(f"R^2 Score: {r2}")

# Save the model
project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
output_path = os.path.join(project_dir, 'models', 'linear_regression_model.pkl')

joblib.dump(model, output_path)