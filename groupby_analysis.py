"""
groupby_analysis.py

Purpose:
    Perform advanced groupby operations using pandas for data summarization,
    aggregation, transformation, and pivoting.

Sections:
    - Basic Aggregation
    - Multiple Aggregations
    - Custom Aggregations
    - Filtering Groups
    - Group-wise Transformation
    - Pivot Tables
    - Time-based Grouping
"""

import pandas as pd
import numpy as np

#  =============================
# 📊 Sample Data Creation
#  =============================
def load_sample_data():
    data = {
        'Region': ['East', 'West', 'East', 'North', 'West', 'East', 'North', 'South'],
        'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
        'Sales': [250, 200, 340, 560, 450, 230, 670, 120],
        'Profit': [30, 20, 45, 70, 60, 25, 90, 10],
        'Date': pd.date_range(start='2023-01-01', periods=8, freq='M')
    }
    return pd.DataFrame(data)

df = load_sample_data()
print("📋 Original Data:")
print(df)

#  =============================
# 🧮 Basic Aggregation
#  =============================
def basic_groupby(df):
    print("\n📌 Total Sales by Region:")
    return df.groupby('Region')['Sales'].sum()

#  =============================
# 📊 Multiple Aggregations
#  =============================
def multiple_aggregations(df):
    print("\n📌 Sales & Profit Aggregations by Category:")
    return df.groupby('Category')[['Sales', 'Profit']].agg(['sum', 'mean', 'max'])

#  =============================
# 🧠 Custom Aggregation Function
#  =============================
def custom_aggregation(df):
    print("\n📌 Custom Aggregation (Range of Sales by Region):")
    return df.groupby('Region')['Sales'].agg(lambda x: x.max() - x.min())

#  =============================
# 🧹 Filtering Groups
#  =============================
def filter_high_sales(df, threshold=400):
    print(f"\n📌 Groups with Total Sales > {threshold}:")
    return df.groupby('Region').filter(lambda x: x['Sales'].sum() > threshold)

#  =============================
# 🔁 Group-wise Transformation (Normalization)
#  =============================
def normalize_sales(df):
    print("\n📌 Normalized Sales within each Region:")
    df['Sales_Normalized'] = df.groupby('Region')['Sales'].transform(lambda x: (x - x.mean()) / x.std())
    return df

#  =============================
# 🔄 Pivot Table
#  =============================
def sales_pivot_table(df):
    print("\n📌 Pivot Table: Total Sales by Region and Category:")
    return pd.pivot_table(df, values='Sales', index='Region', columns='Category', aggfunc='sum', fill_value=0)

#  =============================
# ⏱️ Time-based Grouping
#  =============================
def monthly_sales_summary(df):
    df['Month'] = df['Date'].dt.to_period('M')
    print("\n📌 Monthly Sales Summary:")
    return df.groupby('Month')['Sales'].sum()

# ✅ Run all analyses
if __name__ == "__main__":
    print(basic_groupby(df))
    print(multiple_aggregations(df))
    print(custom_aggregation(df))
    print(filter_high_sales(df))
    print(normalize_sales(df))
    print(sales_pivot_table(df))
    print(monthly_sales_summary(df))
