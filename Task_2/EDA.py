# ======================================================
# TASK 2 : Exploratory Data Analysis (EDA)
# ======================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional (Interactive Graphs)
import plotly.express as px

# Set plot style
plt.style.use('ggplot')

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("First 5 Rows")
print(df.head())

# ======================================================
# Basic Information
# ======================================================

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ======================================================
# Summary Statistics
# ======================================================

print("\nSummary Statistics")
print(df.describe())

print("\nMedian")
print(df.median(numeric_only=True))

print("\nMode")
print(df.mode().iloc[0])

# ======================================================
# Histograms
# ======================================================

df.hist(figsize=(12,10), bins=20)

plt.suptitle("Histograms of Numerical Features")
plt.show()

# ======================================================
# Boxplots
# ======================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(y=df["Age"])
plt.title("Boxplot of Age")

plt.subplot(1,2,2)
sns.boxplot(y=df["Fare"])
plt.title("Boxplot of Fare")

plt.show()

# ======================================================
# Countplots
# ======================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")

plt.subplot(1,2,2)
sns.countplot(x="Sex", data=df)
plt.title("Gender Count")

plt.show()

# ======================================================
# Survival by Gender
# ======================================================

plt.figure(figsize=(6,5))

sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")

plt.show()

# ======================================================
# Age Distribution
# ======================================================

plt.figure(figsize=(8,5))

sns.histplot(df["Age"], bins=25, kde=True)

plt.title("Age Distribution")

plt.show()

# ======================================================
# Fare Distribution
# ======================================================

plt.figure(figsize=(8,5))

sns.histplot(df["Fare"], bins=30, kde=True)

plt.title("Fare Distribution")

plt.show()

# ======================================================
# Pairplot
# ======================================================

sns.pairplot(
    df[['Survived','Age','Fare','Pclass']],
    hue='Survived'
)

plt.show()

# ======================================================
# Correlation Matrix
# ======================================================

temp = df.copy()

# Convert categorical values into numbers

temp["Sex"] = temp["Sex"].map({
    "male":0,
    "female":1
})

temp["Embarked"] = temp["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
})

correlation = temp.corr(numeric_only=True)

plt.figure(figsize=(10,8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix")

plt.show()

# ======================================================
# Scatter Plot
# ======================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Age",
    y="Fare",
    hue="Survived",
    data=df
)

plt.title("Age vs Fare")

plt.show()

# ======================================================
# Plotly Visualization (Optional)
# ======================================================

fig = px.histogram(
    df,
    x="Age",
    color="Survived",
    title="Age Distribution by Survival"
)

fig.show()

# ======================================================
# Basic Insights
# ======================================================

print("\n==============================")
print("Basic Insights")
print("==============================")

print("Average Age :", round(df["Age"].mean(),2))

print("Median Age :", df["Age"].median())

print("Average Fare :", round(df["Fare"].mean(),2))

print("Maximum Fare :", df["Fare"].max())

print("Minimum Fare :", df["Fare"].min())

print("\nPassengers Survived")

print(df["Survived"].value_counts())

print("\nPassenger Class")

print(df["Pclass"].value_counts())

print("\nGender Distribution")

print(df["Sex"].value_counts())

# ======================================================
# Save Summary Statistics
# ======================================================

summary = df.describe()

summary.to_csv("summary_statistics.csv")

print("\nSummary statistics saved successfully.")

print("\nEDA Completed Successfully!")