# data_cleaning_utils.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ----------------------------------------
# 🔍 Missing Values Handling
# ----------------------------------------

def missing_value_summary(df):
    """Returns a DataFrame summarizing missing values."""
    missing = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    return pd.DataFrame({'Missing Values': missing, '% of Total': percent}).sort_values('% of Total', ascending=False)

def fill_missing_with_mode(df, columns):
    """Fills missing values in specified columns with mode."""
    for col in columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def fill_missing_with_mean(df, columns):
    """Fills missing values in specified columns with mean."""
    for col in columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def drop_missing_threshold(df, threshold=0.5):
    """Drops columns with missing value ratio above threshold."""
    return df.loc[:, df.isnull().mean() < threshold]

# ----------------------------------------
# 🧹 Duplicate & Outlier Handling
# ----------------------------------------

def drop_duplicates(df):
    """Drops duplicate rows."""
    return df.drop_duplicates()

def remove_outliers_iqr(df, columns):
    """Removes outliers using IQR method on specified columns."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# ----------------------------------------
# 🧠 Encoding Categorical Features
# ----------------------------------------

def encode_categorical(df, columns, drop_first=True):
    """One-hot encodes specified categorical columns."""
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)

def label_encode(df, columns):
    """Label encodes specified categorical columns."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

# ----------------------------------------
# 🧪 Data Type Conversion
# ----------------------------------------

def convert_to_datetime(df, columns):
    """Converts specified columns to datetime."""
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def convert_to_category(df, columns):
    """Converts specified columns to category dtype."""
    for col in columns:
        df[col] = df[col].astype('category')
    return df

# ----------------------------------------
# 🧾 Text Cleaning
# ----------------------------------------

def clean_text_column(df, column):
    """Lowercases and strips whitespace in text column."""
    df[column] = df[column].str.lower().str.strip()
    return df

def clean_multiple_text_columns(df, columns):
    """Cleans multiple text columns (lowercase + strip)."""
    for col in columns:
        df = clean_text_column(df, col)
    return df

# ----------------------------------------
# 📊 General Utilities
# ----------------------------------------

def standardize_column_names(df):
    """Standardizes column names to snake_case."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace(r'[^\w\s]', '', regex=True)
    )
    return df

def drop_unused_columns(df, columns):
    """Drops specified columns."""
    return df.drop(columns=columns, errors='ignore')

def summarize_dataset(df):
    """Prints basic info and head of dataset."""
    print("🔎 Data Types:\n", df.dtypes)
    print("\n📏 Shape:", df.shape)
    print("\n🧾 First 5 rows:\n", df.head())

# ----------------------------------------
# 🧪 Example Usage (if run as standalone)
# ----------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("your_dataset.csv")
    summarize_dataset(df)
    df = standardize_column_names(df)
    df = drop_duplicates(df)
    df = fill_missing_with_mean(df, columns=["age", "income"])
    df = remove_outliers_iqr(df, columns=["income"])
    df = encode_categorical(df, columns=["gender", "city"])
    print("\n✅ Cleaned dataset ready!")
