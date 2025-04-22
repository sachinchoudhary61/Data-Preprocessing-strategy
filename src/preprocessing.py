import pandas as pd

# ------------------------ STEP 1: LOAD DATA ------------------------ #

# Load the dataset into a pandas DataFrame
# This allows us to work with rows/columns efficiently for cleaning and analysis
df = pd.read_csv("./data/Employee.csv")

# ------------------------ STEP 2: INITIAL INSPECTION ------------------------ #

# Print the names of all columns to get a quick overview of dataset structure
print(" Columns in the dataset:\n", df.columns.tolist())

# Print the datatype of each column (int, float, object, etc.)
# Useful to identify which columns are numerical or categorical
print("\n Data Types:\n", df.dtypes)

# Count missing (null) values in each column
# This tells us where we need to apply imputation or drop records
print("\n Missing Values:\n", df.isnull().sum())

# Display basic statistics like mean, min, max, count, etc.
# Helps identify data distribution and spot unusual values
print("\n Descriptive Statistics:\n", df.describe(include="all"))

# ------------------------ STEP 3: DROP USELESS COLUMNS ------------------------ #

# Identify columns with only 1 unique value — they don't add any useful information
useless_cols = [col for col in df.columns if df[col].nunique() <= 1]

# If such columns exist, drop them to keep dataset clean
if useless_cols:
    print(f"\n Dropping useless columns: {useless_cols}")
    df.drop(columns=useless_cols, inplace=True)

# ------------------------ STEP 4: CLEANING TEXT FIELDS ------------------------ #

# For every text (object-type) column:
# - Convert values to string
# - Strip whitespace (e.g., " TCS " → "TCS")
# - Capitalize properly (e.g., "infosys" → "Infosys")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.title()

# ------------------------ STEP 5: HANDLE MISSING VALUES ------------------------ #

# Fill missing values in 'Company' with the most frequent value (mode)
# Assumes the most common company name is a reasonable guess
if df["Company"].isnull().sum() > 0:
    company_mode = df["Company"].mode()[0]
    df["Company"] = df["Company"].fillna(company_mode)


# Define a function to fill missing 'Place' values
# Uses the most common place for each company
# Falls back to "Unknown" if no mode exists for that group
def fill_place(x):
    if not x.mode().empty:
        return x.fillna(x.mode()[0])
    else:
        return x.fillna("Unknown")


# Apply the function group-wise per company
df["Place"] = df.groupby("Company")["Place"].transform(fill_place)

# Fill missing 'Age' using the median of each company group
# Median is preferred because it is robust to outliers
df["Age"] = df.groupby("Company")["Age"].transform(lambda x: x.fillna(x.median()))

# Fill missing 'Salary' using the mean of each company group
# Mean makes sense here as it reflects average pay per company
df["Salary"] = df.groupby("Company")["Salary"].transform(lambda x: x.fillna(x.mean()))

# ------------------------ STEP 6: CONVERT ENCODINGS ------------------------ #

# Convert numerical gender codes into readable strings
# Assumes 0 = Male and 1 = Female
if df["Gender"].dtype in [int, float]:
    df["Gender"] = df["Gender"].map({0: "Male", 1: "Female"})

# ------------------------ STEP 7: VALIDATION & FILTERING ------------------------ #

# Remove exact duplicate rows to avoid skewing data analysis
df.drop_duplicates(inplace=True)

# Remove rows with illogical age values (outside realistic work age range)
df = df[(df["Age"] >= 18) & (df["Age"] <= 65)]

# Remove records with suspiciously low salary (< ₹1000)
# These may be data entry errors or unpaid interns
df = df[df["Salary"] >= 1000]

# Drop rows that still have null in critical fields after imputation
# These fields are essential for analysis
df.dropna(subset=["Company", "Place", "Gender"], inplace=True)

# ------------------------ STEP 8: OUTLIER REMOVAL ------------------------ #


# Define function to remove outliers using IQR method
# Keeps only values within 1.5 * IQR of Q1–Q3 range
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)  # 25th percentile
    Q3 = data[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile Range
    lower = Q1 - 1.5 * IQR  # Lower bound
    upper = Q3 + 1.5 * IQR  # Upper bound
    return data[(data[column] >= lower) & (data[column] <= upper)]


# Remove outliers in 'Age' and 'Salary' columns
df = remove_outliers_iqr(df, "Age")
df = remove_outliers_iqr(df, "Salary")

# ------------------------ STEP 9: FEATURE ENGINEERING ------------------------ #

# Create a new column 'Seniority' based on age brackets
# This turns a numeric feature into a categorical one (good for grouping or ML)
df["Seniority"] = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 50, 65],
    labels=["Junior", "Mid", "Senior", "Executive"],
)

# ------------------------ STEP 10: EXPORT CLEANED DATA ------------------------ #

# Save the cleaned and processed dataset to a new CSV file
df.to_csv("Cleaned_Employee.csv", index=False)

# Display a sample of the cleaned data for confirmation
print("\n Data cleaning complete. Sample cleaned data:")
print(df.head())
