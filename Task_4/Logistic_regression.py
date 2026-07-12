# ==========================================================
# TASK 4 : LOGISTIC REGRESSION CLASSIFICATION
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# ==========================================================
# Load Dataset
# ==========================================================

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

print("First 5 Rows")
print(df.head())

# ==========================================================
# Dataset Information
# ==========================================================

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nSummary Statistics")
print(df.describe())

# ==========================================================
# Features and Target
# ==========================================================

X = df.drop("target", axis=1)
y = df["target"]

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Standardize Features
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Train Logistic Regression Model
# ==========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================================
# Predictions
# ==========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# ==========================================================
# Evaluation Metrics
# ==========================================================

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("Precision :", precision_score(y_test, y_pred))

print("Recall :", recall_score(y_test, y_pred))

print("ROC-AUC :", roc_auc_score(y_test, y_prob))

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label="ROC Curve")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nFeature Coefficients")

print(coefficients.sort_values(by="Coefficient", ascending=False))

# ==========================================================
# Save Predictions
# ==========================================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("LogisticRegressionPredictions.csv", index=False)

print("\nPredictions saved successfully!")

print("\nTask Completed Successfully!")