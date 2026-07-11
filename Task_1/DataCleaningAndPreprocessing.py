# ============================================
# TASK 1: Data Cleaning & Preprocessing
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------
# Load Dataset
# --------------------------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# --------------------------------------------
# Basic Information
# --------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nShape of Dataset:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# --------------------------------------------
# Handle Missing Values
# --------------------------------------------

# Fill Age with Median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because of many missing values
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------------------------
# Encode Categorical Variables
# --------------------------------------------

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# --------------------------------------------
# Drop Unnecessary Columns
# --------------------------------------------

df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# --------------------------------------------
# Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# --------------------------------------------
# Detect Outliers
# --------------------------------------------

plt.figure(figsize=(10,5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot Before Removing Outliers")
plt.show()

# --------------------------------------------
# Remove Outliers using IQR
# --------------------------------------------

numerical_columns = ['Age', 'Fare']

for col in numerical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# --------------------------------------------
# Boxplot After Removing Outliers
# --------------------------------------------

plt.figure(figsize=(10,5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot After Removing Outliers")
plt.show()

# --------------------------------------------
# Final Dataset Information
# --------------------------------------------

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows of Cleaned Dataset:")
print(df.head())

# --------------------------------------------
# Save Cleaned Dataset
# --------------------------------------------

df.to_csv("cleaned_titanic.csv", index=False)

print("\nData Cleaning and Preprocessing Completed Successfully!")
print("Cleaned dataset saved as 'cleaned_titanic.csv'")