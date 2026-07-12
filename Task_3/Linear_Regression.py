# ============================================================
# TASK 3 : LINEAR REGRESSION
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Load Dataset
# ============================================================

# Replace with your dataset filename
df = pd.read_csv("Housing.csv")

print("First 5 Rows")
print(df.head())

# ============================================================
# Dataset Information
# ============================================================

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# Convert Categorical Variables
# ============================================================

# Convert yes/no columns into numeric

binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in binary_columns:
    df[col] = df[col].map({"yes":1, "no":0})

# Furnishing status

df = pd.get_dummies(df,
                    columns=["furnishingstatus"],
                    drop_first=True)

# ============================================================
# Define Features and Target
# ============================================================

X = df.drop("price", axis=1)

y = df["price"]

# ============================================================
# Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ============================================================
# Train Linear Regression Model
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation Metrics
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===============================")
print("Model Evaluation")
print("===============================")

print("Mean Absolute Error :", mae)

print("Mean Squared Error :", mse)

print("Root Mean Squared Error :", rmse)

print("R² Score :", r2)

# ============================================================
# Model Coefficients
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients")

print(coefficients)

# ============================================================
# Intercept
# ============================================================

print("\nIntercept")

print(model.intercept_)

# ============================================================
# Scatter Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.show()

# ============================================================
# Residual Plot
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color='red')

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()

# ============================================================
# Save Predictions
# ============================================================

results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

results.to_csv("Predictions.csv", index=False)

print("\nPredictions saved successfully!")

print("\nLinear Regression Completed Successfully!")