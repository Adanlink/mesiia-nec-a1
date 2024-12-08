import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("data/dielectron.csv")
data.columns = data.columns.str.strip()

# Drop the columns "Run" and "Event"
data = data.drop(columns=["Run", "Event"])

print(f"Dataset size: {len(data)} rows")

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
## Standardize columns "px1", "py1", "pz1", "px2", "py2", "pz2" (Gaussian-like distributions)
standardize_columns = ["px1", "py1", "pz1", "px2", "py2", "pz2"]
scaler_standard = StandardScaler()
data[standardize_columns] = scaler_standard.fit_transform(data[standardize_columns])

## Apply Yeo-Johnson power transformation to columns "E1" and "E2"
power_transformer_yeo_johnson = PowerTransformer(method="yeo-johnson")
power_transform_columns_1 = ["E1", "E2"]
data[power_transform_columns_1] = power_transformer_yeo_johnson.fit_transform(data[power_transform_columns_1])

## Apply Yeo-Johnson power transformation to columns "pt1", "pt2" and "M"
# Note: This may not be the proper way to preprocess these features, but it is the closest thing.
power_transform_columns_2 = ["pt1", "pt2", "M"]
data[power_transform_columns_2] = power_transformer_yeo_johnson.fit_transform(data[power_transform_columns_2])

## Min-Max scale columns "eta1", "phi1", "eta2", and "phi2" to range [-1, 1]
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
min_max_scale_columns = ["eta1", "phi1", "eta2", "phi2"]
data[min_max_scale_columns] = min_max_scaler.fit_transform(data[min_max_scale_columns])

# Convert all numerical columns to float32 for lower precision
data[numerical_columns] = data[numerical_columns].astype(np.float32)

# 5. Randomly shuffle and split the dataset into training (80%) and testing (20%)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)

print(f"Training set size: {len(train_data)} rows")
print(f"Testing set size: {len(test_data)} rows")