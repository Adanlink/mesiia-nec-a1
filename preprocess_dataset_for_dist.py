import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("data/dielectron.csv")

# Drop the columns "Run" and "Event"
data = data.drop(columns=["Run", "Event"])

# 1. Check for Missing Values
# Impute missing numerical values with the mean
num_imputer = SimpleImputer(strategy='mean')
numerical_columns = data.select_dtypes(include=[np.number]).columns
data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])

# 2. Handle Categorical Variables
# In this dataset, Q1 and Q2 are categorical but already numerical, so no transformation is needed.

# 3. Detect and Handle Outliers
# Using Z-score to identify outliers in numerical columns
z_scores = np.abs((data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std())
outliers = (z_scores > 3).any(axis=1)

# Remove rows with outliers (aggressive, but we have lots of patterns/rows)
non_outliers = data[~outliers]
print("Amount of outliers rows removed: ", len(data) - len(non_outliers))
data = non_outliers

# 4. Normalize/Scale the Data
# Normalize numerical features using Min-Max Scaling
# scaler = MinMaxScaler()
# data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Save the preprocessed data to a new CSV file
data.to_csv("data/preprocessed_data.csv", index=False)