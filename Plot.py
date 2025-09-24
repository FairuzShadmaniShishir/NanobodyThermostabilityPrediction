import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


# Load the data
def load_data(filepath):
    """Load and clean the Bayesian test results data"""
    df = pd.read_csv(filepath)

    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()

    # Convert melting temperature to numeric, handling any non-numeric values
    df['Melting_Temperature'] = pd.to_numeric(df['Melting_Temperature'], errors='coerce')

    return df


# Basic data exploration
def explore_data(df):
    """Perform basic data exploration"""
    print("=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n=== BASIC STATISTICS ===")
    numeric_cols = ['predicted_mean', 'predicted_std', 'epistemic_uncertainty',
                    'aleatoric_uncertainty', 'Melting_Temperature']
    print(df[numeric_cols].describe())

    print("\n=== UNIQUE VALUES IN CATEGORICAL COLUMNS ===")
    categorical_cols = ['Technique', 'Solvent', 'Expression', 'Source']
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].value_counts()
            print(f"\n{col}:")
            print(unique_vals)


# Model performance analysis
def analyze_model_performance(df):
    """Analyze Bayesian model performance"""
    # Remove rows with missing actual temperatures
    df_clean = df.dropna(subset=['Melting_Temperature'])

    if len(df_clean) == 0:
        print("No valid temperature data for performance analysis")
        return None

    actual = df_clean['Melting_Temperature']
    predicted = df_clean['predicted_mean']

    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    correlation = stats.pearsonr(actual, predicted)[0]

    print("=== MODEL PERFORMANCE METRICS ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}°C")
    print(f"R² Score: {r2:.3f}")
    print(f"Pearson Correlation: {correlation:.3f}")

    # Calculate prediction intervals coverage
    ci_68_coverage = np.mean((actual >= df_clean['ci_68%_lower']) &
                             (actual <= df_clean['ci_68%_upper']))
    ci_95_coverage = np.mean((actual >= df_clean['ci_95%_lower']) &
                             (actual <= df_clean['ci_95%_upper']))

    print(f"\n=== CONFIDENCE INTERVAL COVERAGE ===")
    print(f"68% CI Coverage: {ci_68_coverage:.1%} (expected: 68%)")
    print(f"95% CI Coverage: {ci_95_coverage:.1%} (expected: 95%)")

    return df_clean


# Uncertainty analysis
def analyze_uncertainty(df):
    """Analyze prediction uncertainties"""
    print("\n=== UNCERTAINTY ANALYSIS ===")

    # Basic uncertainty statistics
    print("Predicted Standard Deviation:")
    print(f"  Mean: {df['predicted_std'].mean():.2f}")
    print(f"  Median: {df['predicted_std'].median():.2f}")
    print(f"  Range: {df['predicted_std'].min():.2f} - {df['predicted_std'].max():.2f}")

    # Confidence interval widths
    ci_68_width = df['ci_68%_upper'] - df['ci_68%_lower']
    ci_95_width = df['ci_95%_upper'] - df['ci_95%_lower']

    print(f"\n68% CI Width:")
    print(f"  Mean: {ci_68_width.mean():.2f}°C")
    print(f"  Median: {ci_68_width.median():.2f}°C")

    print(f"\n95% CI Width:")
    print(f"  Mean: {ci_95_width.mean():.2f}°C")
    print(f"  Median: {ci_95_width.median():.2f}°C")

    # Identify high uncertainty predictions
    high_uncertainty_threshold = df['predicted_std'].quantile(0.9)
    high_uncertainty_samples = df[df['predicted_std'] > high_uncertainty_threshold]

    print(f"\nHigh Uncertainty Samples (top 10%):")
    print(f"Threshold: >{high_uncertainty_threshold:.2f}")
    print(f"Count: {len(high_uncertainty_samples)}")
    if len(high_uncertainty_samples) > 0:
        print("Sample IDs:", high_uncertainty_samples['ID'].tolist())


# Technique comparison
def compare_techniques(df):
    """Compare performance across different experimental techniques"""
    if 'Technique' not in df.columns or df['Technique'].isnull().all():
        print("No technique data available for comparison")
        return

    print("\n=== TECHNIQUE COMPARISON ===")

    # Remove rows with missing data for this analysis
    df_tech = df.dropna(subset=['Technique', 'Melting_Temperature'])

    if len(df_tech) == 0:
        print("No complete data for technique comparison")
        return

    techniques = df_tech['Technique'].value_counts()
    print("Sample counts by technique:")
    print(techniques)

    # Performance by technique
    print("\nPerformance by technique:")
    for technique in techniques.index:
        subset = df_tech[df_tech['Technique'] == technique]
        if len(subset) > 1:
            mae = mean_absolute_error(subset['Melting_Temperature'],
                                      subset['predicted_mean'])
            rmse = np.sqrt(mean_squared_error(subset['Melting_Temperature'],
                                              subset['predicted_mean']))
            print(f"  {technique}: MAE={mae:.2f}°C, RMSE={rmse:.2f}°C (n={len(subset)})")


# Visualization functions
def create_visualizations(df):
    """Create comprehensive visualizations"""
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # 1. Predicted vs Actual scatter plot
    ax1 = plt.subplot(3, 4, 1)
    df_clean = df.dropna(subset=['Melting_Temperature'])
    if len(df_clean) > 0:
        plt.scatter(df_clean['Melting_Temperature'], df_clean['predicted_mean'],
                    alpha=0.6, s=50)
        plt.plot([40, 90], [40, 90], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Temperature (°C)')
        plt.ylabel('Predicted Temperature (°C)')
        plt.title('Predicted vs Actual Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 2. Residuals plot
    ax2 = plt.subplot(3, 4, 2)
    if len(df_clean) > 0:
        residuals = df_clean['predicted_mean'] - df_clean['Melting_Temperature']
        plt.scatter(df_clean['predicted_mean'], residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Temperature (°C)')
        plt.ylabel('Residuals (°C)')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)

    # 3. Prediction uncertainty distribution
    ax3 = plt.subplot(3, 4, 3)
    plt.hist(df['predicted_std'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainty')
    plt.grid(True, alpha=0.3)

    # 4. Confidence interval coverage plot
    ax4 = plt.subplot(3, 4, 4)
    if len(df_clean) > 0:
        # Sort by predicted mean for better visualization
        df_sorted = df_clean.sort_values('predicted_mean')
        x = range(len(df_sorted))

        plt.fill_between(x, df_sorted['ci_95%_lower'], df_sorted['ci_95%_upper'],
                         alpha=0.3, label='95% CI', color='lightblue')
        plt.fill_between(x, df_sorted['ci_68%_lower'], df_sorted['ci_68%_upper'],
                         alpha=0.5, label='68% CI', color='blue')
        plt.scatter(x, df_sorted['Melting_Temperature'], color='red', s=20,
                    label='Actual', alpha=0.7)
        plt.scatter(x, df_sorted['predicted_mean'], color='black', s=20,
                    label='Predicted', alpha=0.7)

        plt.xlabel('Sample Index (sorted by prediction)')
        plt.ylabel('Temperature (°C)')
        plt.title('Confidence Intervals vs Actual Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 5. Temperature distribution comparison
    ax5 = plt.subplot(3, 4, 5)
    if len(df_clean) > 0:
        plt.hist(df_clean['Melting_Temperature'], bins=15, alpha=0.7,
                 label='Actual', color='red', edgecolor='black')
        plt.hist(df_clean['predicted_mean'], bins=15, alpha=0.7,
                 label='Predicted', color='blue', edgecolor='black')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Frequency')
        plt.title('Temperature Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. Uncertainty vs Prediction scatter
    ax6 = plt.subplot(3, 4, 6)
    plt.scatter(df['predicted_mean'], df['predicted_std'], alpha=0.6, s=50)
    plt.xlabel('Predicted Temperature (°C)')
    plt.ylabel('Prediction Uncertainty (std)')
    plt.title('Prediction vs Uncertainty')
    plt.grid(True, alpha=0.3)

    # 7. Box plot by technique (if available)
    ax7 = plt.subplot(3, 4, 7)
    if 'Technique' in df.columns and not df['Technique'].isnull().all():
        df_tech = df.dropna(subset=['Technique'])
        techniques = df_tech['Technique'].value_counts()
        if len(techniques) > 1:
            sns.boxplot(data=df_tech, x='Technique', y='predicted_std', ax=ax7)
            plt.xticks(rotation=45)
            plt.title('Uncertainty by Technique')
        else:
            plt.text(0.5, 0.5, 'Insufficient technique data',
                     ha='center', va='center', transform=ax7.transAxes)
            plt.title('Uncertainty by Technique')
    else:
        plt.text(0.5, 0.5, 'No technique data available',
                 ha='center', va='center', transform=ax7.transAxes)
        plt.title('Uncertainty by Technique')

    # 8. Error vs Uncertainty correlation
    ax8 = plt.subplot(3, 4, 8)
    if len(df_clean) > 0:
        errors = np.abs(df_clean['predicted_mean'] - df_clean['Melting_Temperature'])
        plt.scatter(df_clean['predicted_std'], errors, alpha=0.6, s=50)
        plt.xlabel('Prediction Uncertainty (std)')
        plt.ylabel('Absolute Error (°C)')
        plt.title('Uncertainty vs Absolute Error')
        plt.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(errors) > 1:
            corr = stats.pearsonr(df_clean['predicted_std'], errors)[0]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax8.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 9. Sequence length analysis (if sequence data available)
    ax9 = plt.subplot(3, 4, 9)
    if 'Sequence' in df.columns and not df['Sequence'].isnull().all():
        df['seq_length'] = df['Sequence'].str.len()
        plt.scatter(df['seq_length'], df['predicted_mean'], alpha=0.6, s=50)
        plt.xlabel('Sequence Length')
        plt.ylabel('Predicted Temperature (°C)')
        plt.title('Prediction vs Sequence Length')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No sequence data available',
                 ha='center', va='center', transform=ax9.transAxes)
        plt.title('Prediction vs Sequence Length')

    # 10. Calibration plot
    ax10 = plt.subplot(3, 4, 10)
    if len(df_clean) > 0:
        # Create calibration plot for 68% CI
        n_bins = 10
        df_sorted = df_clean.sort_values('predicted_std')
        bin_size = len(df_sorted) // n_bins

        expected_coverage = []
        observed_coverage = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(df_sorted))
            bin_data = df_sorted.iloc[start_idx:end_idx]

            if len(bin_data) > 0:
                coverage = np.mean((bin_data['Melting_Temperature'] >= bin_data['ci_68%_lower']) &
                                   (bin_data['Melting_Temperature'] <= bin_data['ci_68%_upper']))
                observed_coverage.append(coverage)
                expected_coverage.append(0.68)

        plt.plot(expected_coverage, observed_coverage, 'bo-', label='68% CI')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Expected Coverage')
        plt.ylabel('Observed Coverage')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 11. CI width distribution
    ax11 = plt.subplot(3, 4, 11)
    ci_68_width = df['ci_68%_upper'] - df['ci_68%_lower']
    ci_95_width = df['ci_95%_upper'] - df['ci_95%_lower']

    plt.hist(ci_68_width, bins=15, alpha=0.7, label='68% CI Width',
             color='blue', edgecolor='black')
    plt.hist(ci_95_width, bins=15, alpha=0.7, label='95% CI Width',
             color='red', edgecolor='black')
    plt.xlabel('Confidence Interval Width (°C)')
    plt.ylabel('Frequency')
    plt.title('CI Width Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 12. Performance summary text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    if len(df_clean) > 0:
        mae = mean_absolute_error(df_clean['Melting_Temperature'], df_clean['predicted_mean'])
        rmse = np.sqrt(mean_squared_error(df_clean['Melting_Temperature'], df_clean['predicted_mean']))
        r2 = r2_score(df_clean['Melting_Temperature'], df_clean['predicted_mean'])

        ci_68_coverage = np.mean((df_clean['Melting_Temperature'] >= df_clean['ci_68%_lower']) &
                                 (df_clean['Melting_Temperature'] <= df_clean['ci_68%_upper']))
        ci_95_coverage = np.mean((df_clean['Melting_Temperature'] >= df_clean['ci_95%_lower']) &
                                 (df_clean['Melting_Temperature'] <= df_clean['ci_95%_upper']))

        summary_text = f"""Performance Summary:

MAE: {mae:.2f}°C
RMSE: {rmse:.2f}°C
R²: {r2:.3f}

CI Coverage:
68% CI: {ci_68_coverage:.1%}
95% CI: {ci_95_coverage:.1%}

Data Points: {len(df_clean)}
Mean Uncertainty: {df['predicted_std'].mean():.2f}°C"""

        plt.text(0.1, 0.5, summary_text, transform=ax12.transAxes,
                 fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()


# Main analysis function
def main_analysis(filepath):
    """Run complete analysis of Bayesian test results"""
    # Load data
    df = load_data(filepath)

    # Basic exploration
    explore_data(df)

    # Model performance analysis
    df_clean = analyze_model_performance(df)

    # Uncertainty analysis
    analyze_uncertainty(df)

    # Technique comparison
    compare_techniques(df)

    # Create visualizations
    create_visualizations(df)

    return df


# Usage example:
if __name__ == "__main__":
    # Replace with your actual file path
    filepath = "bayesian_test_results.csv"
    df = main_analysis(filepath)

    # Additional custom analysis can be added here
    print("\n=== CUSTOM ANALYSIS SUGGESTIONS ===")
    print("1. Sequence-based analysis: Analyze amino acid composition effects")
    print("2. Outlier detection: Identify samples with unusual predictions")
    print("3. Feature importance: If you have sequence features, analyze their impact")
    print("4. Cross-validation analysis: Evaluate model stability")
    print("5. Ensemble analysis: Compare multiple models if available")