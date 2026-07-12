# ==========================================================
# TASK 5 : DECISION TREE & RANDOM FOREST (Heart Disease)
# ==========================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("heart.csv")

print("First 5 Rows")
print(df.head())

# ==========================================================
# Dataset Information
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
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
# Decision Tree Model
# ==========================================================

dt_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# ==========================================================
# Decision Tree Evaluation
# ==========================================================

print("\n==============================")
print("Decision Tree Results")
print("==============================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_dt))

print("\nClassification Report")

print(classification_report(y_test, y_pred_dt))

# ==========================================================
# Decision Tree Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================================
# Visualize Decision Tree
# ==========================================================

plt.figure(figsize=(20,10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Disease","Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree")

plt.show()

# ==========================================================
# Random Forest Model
# ==========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# ==========================================================
# Random Forest Evaluation
# ==========================================================

print("\n==============================")
print("Random Forest Results")
print("==============================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("\nClassification Report")

print(classification_report(y_test, y_pred_rf))

# ==========================================================
# Accuracy Comparison
# ==========================================================

dt_accuracy = accuracy_score(y_test, y_pred_dt)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n==============================")
print("Model Comparison")
print("==============================")

print("Decision Tree Accuracy :", dt_accuracy)
print("Random Forest Accuracy :", rf_accuracy)

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features")

print(importance)

# ==========================================================
# Feature Importance Plot
# ==========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.show()

# ==========================================================
# Cross Validation
# ==========================================================

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")

print(scores)

print("\nAverage Accuracy:", scores.mean())

# ==========================================================
# Save Predictions
# ==========================================================

results = pd.DataFrame({
    "Actual": y_test,
    "Decision Tree Prediction": y_pred_dt,
    "Random Forest Prediction": y_pred_rf
})

results.to_csv(
    "HeartDisease_Predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")

print("\nTask Completed Successfully!")