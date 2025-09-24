import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_credible_intervals(df):
    """Plot credible intervals for Bayesian predictions"""
    sns.set_theme(style="whitegrid", font_scale=1.3)
    df_clean = df.dropna(subset=['Melting_Temperature'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Predicted vs Actual with CIs
    ax = axes[0]
    ax.scatter(df_clean['Melting_Temperature'], df_clean['predicted_mean'],
               alpha=0.7, s=60, edgecolor='k', label="Predicted")
    min_temp, max_temp = df_clean['Melting_Temperature'].min(), df_clean['Melting_Temperature'].max()
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', label="Ideal")
    # 68% and 95% CI ribbons
    ax.fill_between(df_clean['Melting_Temperature'], df_clean['ci_68%_lower'], df_clean['ci_68%_upper'],
                    color='blue', alpha=0.2, label="68% CI")
    ax.fill_between(df_clean['Melting_Temperature'], df_clean['ci_95%_lower'], df_clean['ci_95%_upper'],
                    color='lightblue', alpha=0.2, label="95% CI")
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("Predicted vs Actual with Credible Intervals")
    ax.legend()

    # 2. CI Coverage vs Sample Index
    ax = axes[1]
    df_sorted = df_clean.sort_values('predicted_mean')
    x = range(len(df_sorted))
    ax.fill_between(x, df_sorted['ci_95%_lower'], df_sorted['ci_95%_upper'], color='lightblue', alpha=0.4, label="95% CI")
    ax.fill_between(x, df_sorted['ci_68%_lower'], df_sorted['ci_68%_upper'], color='blue', alpha=0.6, label="68% CI")
    ax.scatter(x, df_sorted['Melting_Temperature'], color='red', s=25, label="Actual")
    ax.plot(x, df_sorted['predicted_mean'], color='black', linewidth=1, label="Predicted")
    ax.set_xlabel("Sample Index (sorted)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Credible Intervals vs Actual Values")
    ax.legend()

    plt.tight_layout()
    plt.show()


# Usage example
if __name__ == "__main__":
    filepath = "bayesian_test_results.csv"  # Replace with your CSV file
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Clean column names
    plot_credible_intervals(df)
