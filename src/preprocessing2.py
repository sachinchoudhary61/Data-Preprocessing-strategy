# ------------------------- 1. IMPORT LIBRARIES ------------------------- #
# These are core Python libraries for data cleaning, encoding, and scaling

import numpy as np  # Used for numerical operations and arrays
import pandas as pd  # Powerful data manipulation library
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
)  # For encoding and scaling
from sklearn.impute import SimpleImputer  # For handling missing values

# ------------------------- 2. LOAD DATA ------------------------- #
# Load the dataset into a Pandas DataFrame
df = pd.read_csv("./data/AB_NYC_2019.csv")
print(
    "Initial shape of data:", df.shape
)  # Shows the number of rows and columns in the dataset

# ------------------------- 3. EXPLORE AND UNDERSTAND ------------------------- #
# Understand the dataset's structure to guide preprocessing decisions

print(
    "\n Column Data Types:\n", df.dtypes
)  # See which columns are numeric, object, etc.
print("\n Missing Values Per Column:\n", df.isnull().sum())  # Find where nulls exist

# ------------------------- 4. DROP NON-ESSENTIAL COLUMNS ------------------------- #
# Some columns aren't helpful or are identifiers not useful for ML models:
# - 'id' and 'host_id': Just unique IDs
# - 'name' and 'host_name': Text-heavy, unstructured, and mostly irrelevant here
# - 'last_review': Could be useful, but not used in this workflow

df.drop(["id", "name", "host_id", "host_name", "last_review"], axis=1, inplace=True)

# ------------------------- 5. HANDLE MISSING VALUES ------------------------- #
# We still have missing values in 'reviews_per_month'
# We'll fill them using the most frequently occurring value (simple and safe)

imputer = SimpleImputer(strategy="most_frequent")
df[["reviews_per_month"]] = imputer.fit_transform(df[["reviews_per_month"]])

# ------------------------- 6. IDENTIFY CATEGORICAL FEATURES ------------------------- #
# Categorical features are stored as object (text/string)
# We'll identify them for encoding

cat_columns = [col for col in df.columns if df[col].dtype == "object"]

print("\n Categorical Features Identified:")
for col in cat_columns:
    print(f"• {col}: {df[col].nunique()} unique values")

# Knowing unique values helps determine:
# - Ordinal encoding (if few categories)
# - One-Hot encoding (if many categories or no inherent order)

# ------------------------- 7. ENCODING CATEGORICAL FEATURES ------------------------- #

# --- 7A: Ordinal Encoding for 'room_type' ---
# This column has a few meaningful categories (e.g. 'Private room', 'Entire home/apt')
# We convert these to 0, 1, 2 — giving ML models numeric data to work with

ordinal_encoder = OrdinalEncoder()
df["room_type"] = ordinal_encoder.fit_transform(df[["room_type"]])
print("\n 'room_type' encoded as:", ordinal_encoder.categories_)

# --- 7B: One-Hot Encoding for 'neighbourhood' ---
# This column has 221 unique values — typical for one-hot encoding
# It turns one column into multiple binary columns (1 if present, 0 otherwise)

onehot_neigh = OneHotEncoder(sparse_output=False)
neigh_encoded = onehot_neigh.fit_transform(df[["neighbourhood"]])
neigh_df = pd.DataFrame(
    neigh_encoded, columns=onehot_neigh.get_feature_names_out(["neighbourhood"])
)
print(f" Created {neigh_df.shape[1]} one-hot columns from 'neighbourhood'")

# --- 7C: One-Hot Encoding for 'neighbourhood_group' ---
# This one has just 5 boroughs — easy to one-hot encode too

onehot_ng = OneHotEncoder(sparse_output=False)
ng_encoded = onehot_ng.fit_transform(df[["neighbourhood_group"]])
ng_df = pd.DataFrame(
    ng_encoded, columns=onehot_ng.get_feature_names_out(["neighbourhood_group"])
)
print(f" Created {ng_df.shape[1]} one-hot columns from 'neighbourhood_group'")

# Add the encoded columns to the main DataFrame
df = pd.concat([df, neigh_df, ng_df], axis=1)

# Drop original string columns as they are now represented by the encoded versions
df.drop(["neighbourhood", "neighbourhood_group"], axis=1, inplace=True)

# ------------------------- 8. SCALE NUMERIC DATA ------------------------- #
# Different features have different ranges (e.g. price vs reviews)
# Min-Max Scaling ensures all features range from 0 to 1 — fair input for ML models

columns_to_scale = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
print("\n Scaled features:", columns_to_scale)

# ------------------------- 9. REVIEW & EXPORT ------------------------- #
# Check how the final dataset looks and save for future use

print("\n Final dataset shape:", df.shape)
print(" Preview of processed data:\n", df.head())

# Export the cleaned, encoded, and scaled dataset to a CSV file
df.to_csv("Final_Cleaned_AB_NYC_2019.csv", index=False)
print("\n Dataset saved as 'Final_Cleaned_AB_NYC_2019.csv'")
