"""
pandas_tips_and_tricks.py

🔥 Handy Pandas Tips & Tricks for Efficient Data Analysis
"""

import pandas as pd
import numpy as np

# Sample DataFrame for demo
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Age": [25, np.nan, 35, 45, 29],
    "Gender": ["F", "M", "M", "M", "F"],
    "Salary": [50000, 60000, np.nan, 80000, 70000]
})

#  =============================
# 🧼 1. Chain operations
#  =============================
cleaned_df = (
    df.dropna(subset=["Age"])  # remove rows with missing age
      .assign(Salary_USD=lambda x: x["Salary"] * 1.0)  # add new column
      .sort_values(by="Salary_USD", ascending=False)  # sort
)

print("📌 Cleaned DataFrame (chained operations):\n", cleaned_df)

#  =============================
# 🔍 2. Fast filtering with query()
#  =============================
high_earners = df.query("Salary > 60000")
print("\n💸 High Earners (query method):\n", high_earners)

#  =============================
# 🧠 3. Use map and apply smartly
#  =============================
df["Gender_Flag"] = df["Gender"].map({"F": 1, "M": 0})
df["Age_Group"] = df["Age"].apply(lambda x: "Youth" if x < 30 else "Adult")
print("\n🧬 Gender Flag & Age Group:\n", df[["Name", "Gender_Flag", "Age_Group"]])

#  =============================
# 🧊 4. Memory optimization
#  =============================
def optimize_dtypes(df):
    for col in df.select_dtypes(include="float"):
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="int"):
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

df_optimized = optimize_dtypes(df.copy())
print("\n⚙️ Optimized dtypes:\n", df_optimized.dtypes)

#  =============================
# 📐 5. Reshape with melt & pivot
#  =============================
pivoted = df.pivot_table(index="Gender", values="Salary", aggfunc="mean")
print("\n📊 Pivot Table:\n", pivoted)

#  =============================
# 🌀 6. Melt for wide to long format
#  =============================
long_df = df.melt(id_vars=["Name"], value_vars=["Age", "Salary"])
print("\n🔄 Melted DataFrame:\n", long_df.head())

#  =============================
# 🧼 7. Fill missing values cleverly
#  =============================
df["Salary_filled"] = df["Salary"].fillna(df["Salary"].median())
print("\n🧩 Salary Filled with Median:\n", df[["Name", "Salary", "Salary_filled"]])

#  =============================
# 🧾 8. Conditional column creation with np.select
#  =============================
conditions = [
    df["Age"] < 30,
    df["Age"] >= 30
]
choices = ["<30", "30+"]
df["Age_Bucket"] = np.select(conditions, choices, default="Unknown")
print("\n📚 Age Bucket:\n", df[["Name", "Age", "Age_Bucket"]])

#  =============================
# 📎 9. Efficient groupby tricks
#  =============================
grouped = df.groupby("Gender")["Salary"].agg(["mean", "median", "count"]).reset_index()
print("\n🧮 Salary Stats by Gender:\n", grouped)

#  =============================
# 🔂 10. Use explode() to flatten lists in columns
#  =============================
df["Skills"] = [["Python", "SQL"], ["Excel"], ["Python", "Tableau"], [], ["SQL"]]
exploded = df.explode("Skills")
print("\n🎇 Exploded Skills Column:\n", exploded[["Name", "Skills"]])

#  =============================
# 🧪 11. Quickly detect duplicates
#  =============================
duplicates = df[df.duplicated(subset=["Name"], keep=False)]
print("\n♻️ Duplicates Detected:\n", duplicates)

#  =============================
# 📌 12. Vectorized string ops
#  =============================
df["Name_Upper"] = df["Name"].str.upper()
df["Gender_Initial"] = df["Gender"].str[0]
print("\n🔠 String Operations:\n", df[["Name", "Name_Upper", "Gender_Initial"]])

#  =============================
# 🕵️‍♂️ 13. Detect and filter outliers with IQR
#  =============================
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

df_no_outliers = remove_outliers_iqr(df, "Salary_filled")
print("\n🚫 DataFrame without Salary Outliers:\n", df_no_outliers)

#  =============================
# ✅ 14. Apply column renaming best practice
#  =============================
df_renamed = df.rename(columns=str.lower).rename(columns=lambda x: x.replace(" ", "_"))
print("\n🔧 Renamed Columns:\n", df_renamed.columns)

#  =============================
# 🧹 15. Drop columns with high % of missing values
#  =============================
def drop_missing_columns(df, threshold=0.5):
    missing_ratio = df.isnull().mean()
    return df.loc[:, missing_ratio < threshold]

df_dropped = drop_missing_columns(df)
print("\n🚿 Columns after dropping high-missing:\n", df_dropped.columns)

# 🧵 Bonus: Style your DataFrame (great for Jupyter Notebooks)
def highlight_max(s):
    return ['background-color: yellow' if v == s.max() else '' for v in s]

styled = df.style.apply(highlight_max, subset=["Salary_filled"])
# styled  # Uncomment in Jupyter Notebooks only

