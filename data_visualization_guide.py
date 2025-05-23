"""
data_visualization_guide.py

Purpose:
    Showcase essential data visualization types using seaborn, matplotlib, and plotly.
    Includes guidance on when to use each type.

Sections:
    - Distribution Plots
    - Categorical Plots
    - Correlation & Heatmaps
    - Time Series Plots
    - Comparison Plots
    - Interactive Visuals with Plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#  =============================
# ✅ Load or simulate data:
# -- Generate Sample Data
#  =============================

def generate_sample_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C'], size=100),
        'Subcategory': np.random.choice(['X', 'Y'], size=100),
        'Value': np.random.normal(50, 15, 100),
        'Score': np.random.uniform(0, 1, 100),
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D')
    })
    return df

df = generate_sample_data()

#  =============================
# 🎯 Distribution Plots (Univariate Analysis):
# -- Distribution Plots
# -- Categorical Plots
# -- Correlation Heatmap
# -- Timeseries Plot
# -- Interactive Plotly
# -- Print Plot Guide
#  =============================
def distribution_plots(df):
    print("🎯 Distribution Plots: Use to understand variable distribution and skew.")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['Value'], bins=20, kde=True)
    plt.title('Seaborn Histogram + KDE')

    plt.subplot(1, 2, 2)
    plt.boxplot(df['Value'])
    plt.title('Matplotlib Boxplot')

    plt.tight_layout()
    plt.show()

#  =============================
# 📊 Categorical Plots (Group Comparison)
#  =============================
def categorical_plots(df):
    print("📊 Categorical Plots: Use to compare distributions across categories.")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Category', y='Value')
    plt.title('Seaborn Barplot')

    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x='Category', y='Value')
    plt.title('Seaborn Violin Plot')

    plt.tight_layout()
    plt.show()

#  =============================
# 🔗 Correlation Heatmap
#  =============================
def correlation_heatmap(df):
    print("🔗 Heatmaps: Use to show correlation or relationship matrix.")
    corr = df[['Value', 'Score']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Seaborn Heatmap')
    plt.show()

#  =============================
# 📈 Time Series Line Plot
#  =============================
def time_series_plot(df):
    print("📈 Time Series Plot: Use for trend analysis over time.")
    df_ts = df.groupby('Date')['Value'].mean().reset_index()
    plt.plot(df_ts['Date'], df_ts['Value'])
    plt.title('Matplotlib Time Series')
    plt.xlabel('Date')
    plt.ylabel('Average Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#  =============================
# 🚀 Interactive Plotly Visuals
#  =============================
def interactive_plotly(df):
    print("🚀 Plotly Interactive Plot: Use for dashboards and exploration.")
    fig = px.scatter(
        df, x='Value', y='Score',
        color='Category', size='Score',
        title='Plotly Bubble Chart (Value vs Score)'
    )
    fig.show()

#  =============================
# 🧠 When to Use What (Quick Reference)
# Print Plot Guide
#  =============================
def print_plot_guide():
    guide = {
        'Histogram': '➡️ Use to show distribution of a numeric variable.',
        'Boxplot': '➡️ Great for detecting outliers and comparing distributions.',
        'Barplot': '➡️ Compare values across categories.',
        'Violinplot': '➡️ Distribution + density per category.',
        'Heatmap': '➡️ Correlation or matrix values.',
        'Lineplot': '➡️ Best for trends over time.',
        'Scatter': '➡️ Reveal relationships between two variables.',
        'Bubble Chart': '➡️ Like scatter but encodes 3rd variable as size.',
    }
    print("🧠 Plot Selection Guide:")
    for k, v in guide.items():
        print(f"{k}: {v}")

# ✅ Run all
if __name__ == "__main__":
    print_plot_guide()
    distribution_plots(df)
    categorical_plots(df)
    correlation_heatmap(df)
    time_series_plot(df)
    interactive_plotly(df)
