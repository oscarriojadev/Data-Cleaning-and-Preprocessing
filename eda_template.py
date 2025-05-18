# eda_template.py

"""
📊 EDA Template Script
Description: A reusable script for exploratory data analysis (EDA) using pandas, seaborn, and matplotlib.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Optional: uncomment for pretty notebooks
# %matplotlib inline  

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# =============================
# 🔧 CONFIGURATION
# =============================

DATA_PATH = "sample_data.csv"  # Replace with dataset path
MAX_ROWS = 5000  # Limit to speed up big data analysis

# =============================
# 🧰 UTILITY FUNCTIONS
# =============================

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path=DATA_PATH, nrows=MAX_ROWS)
        print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

def quick_summary(df: pd.DataFrame):
    print("\n📌 Data Types:\n", df.dtypes)
    print("\n📌 Missing Values:\n", df.isnull().sum())
    print("\n📌 Descriptive Statistics:\n", df.describe(include="all").T)

def missing_value_plot(df: pd.DataFrame):
    nulls = df.isnull().mean().sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("🎉 No missing values to display.")
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=nulls.values, y=nulls.index, palette="coolwarm")
    plt.title("🔍 Missing Value Percentage by Feature")
    plt.xlabel("Percentage")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        print("⚠️ Not enough numerical features for correlation matrix.")
        return
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", mask=np.triu(np.ones_like(corr, dtype=bool)))
    plt.title("📈 Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_distributions(df: pd.DataFrame, columns=None):
    numeric_cols = df.select_dtypes(include=np.number).columns if columns is None else columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"📊 Distribution: {col}")
        plt.tight_layout()
        plt.show()

def categorical_value_counts(df: pd.DataFrame, threshold=20):
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals <= threshold:
            print(f"\n🗂️ Value Counts for '{col}':")
            print(df[col].value_counts(dropna=False))
        else:
            print(f"\n'{col}' skipped (too many categories: {unique_vals})")

# =============================
# 🚀 MAIN WORKFLOW
# =============================

def run_eda(path=DATA_PATH):
    df = load_data(path)

    print("\n🔎 --- QUICK SUMMARY ---")
    quick_summary(df)

    print("\n📉 --- MISSING VALUES ---")
    missing_value_plot(df)

    print("\n📊 --- DISTRIBUTIONS ---")
    plot_distributions(df)

    print("\n🧩 --- CATEGORICAL VARIABLES ---")
    categorical_value_counts(df)

    print("\n📐 --- CORRELATION HEATMAP ---")
    correlation_heatmap(df)

# =============================
# 📌 EXECUTION
# =============================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_eda(sys.argv[1])
    else:
        run_eda()
