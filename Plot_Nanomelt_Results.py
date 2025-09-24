import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('NanoMelt_Reuslts_83.csv', index_col=0)

# Basic dataset information
print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\n=== DATA TYPES AND MISSING VALUES ===")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Basic statistics for melting temperature
print("\n=== MELTING TEMPERATURE STATISTICS ===")
print(f"Count: {df['Melting_Temperature'].count()}")
print(f"Mean: {df['Melting_Temperature'].mean():.2f}°C")
print(f"Median: {df['Melting_Temperature'].median():.2f}°C")
print(f"Std Dev: {df['Melting_Temperature'].std():.2f}°C")
print(f"Min: {df['Melting_Temperature'].min():.2f}°C")
print(f"Max: {df['Melting_Temperature'].max():.2f}°C")

# Distribution of experimental techniques
print("\n=== EXPERIMENTAL TECHNIQUES ===")
technique_counts = df['Technique'].value_counts()
print(technique_counts)

# Distribution of expression systems
print("\n=== EXPRESSION SYSTEMS ===")
expression_counts = df['Expression'].value_counts(dropna=False)
print(expression_counts)

# pH analysis
print("\n=== pH CONDITIONS ===")
ph_stats = df['pH'].describe()
print(ph_stats)

# Sequence length analysis
print("\n=== SEQUENCE LENGTH ANALYSIS ===")
df['Sequence_Length'] = df['Sequence'].str.len()
print(f"Mean sequence length: {df['Sequence_Length'].mean():.1f}")
print(f"Min sequence length: {df['Sequence_Length'].min()}")
print(f"Max sequence length: {df['Sequence_Length'].max()}")

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Single-Domain Antibody Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Melting temperature distribution
axes[0, 0].hist(df['Melting_Temperature'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Melting Temperature (°C)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Melting Temperatures')
axes[0, 0].axvline(df['Melting_Temperature'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["Melting_Temperature"].mean():.1f}°C')
axes[0, 0].legend()

# 2. Melting temperature by technique
technique_data = [df[df['Technique'] == tech]['Melting_Temperature'].dropna()
                  for tech in df['Technique'].unique() if pd.notna(tech)]
technique_labels = [tech for tech in df['Technique'].unique() if pd.notna(tech)]
bp = axes[0, 1].boxplot(technique_data, labels=technique_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
axes[0, 1].set_xlabel('Technique')
axes[0, 1].set_ylabel('Melting Temperature (°C)')
axes[0, 1].set_title('Melting Temperature by Experimental Technique')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Sequence length vs melting temperature
valid_data = df.dropna(subset=['Melting_Temperature', 'Sequence_Length'])
axes[0, 2].scatter(valid_data['Sequence_Length'], valid_data['Melting_Temperature'],
                   alpha=0.6, color='green')
axes[0, 2].set_xlabel('Sequence Length')
axes[0, 2].set_ylabel('Melting Temperature (°C)')
axes[0, 2].set_title('Sequence Length vs Melting Temperature')

# Calculate correlation
corr_coef = stats.pearsonr(valid_data['Sequence_Length'], valid_data['Melting_Temperature'])[0]
axes[0, 2].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[0, 2].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. pH vs melting temperature (where pH data exists)
ph_data = df.dropna(subset=['pH', 'Melting_Temperature'])
if len(ph_data) > 1:
    axes[1, 0].scatter(ph_data['pH'], ph_data['Melting_Temperature'], alpha=0.6, color='purple')
    axes[1, 0].set_xlabel('pH')
    axes[1, 0].set_ylabel('Melting Temperature (°C)')
    axes[1, 0].set_title('pH vs Melting Temperature')

    # Add trend line if enough data points
    if len(ph_data) > 2:
        z = np.polyfit(ph_data['pH'], ph_data['Melting_Temperature'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(ph_data['pH'], p(ph_data['pH']), "r--", alpha=0.8)
else:
    axes[1, 0].text(0.5, 0.5, 'Insufficient pH data', ha='center', va='center',
                    transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('pH vs Melting Temperature (No Data)')

# 5. Experimental vs Predicted melting temperatures
pred_data = df.dropna(subset=['Melting_Temperature', 'Prediction'])
if len(pred_data) > 0:
    axes[1, 1].scatter(pred_data['Prediction'], pred_data['Melting_Temperature'],
                       alpha=0.6, color='orange')
    axes[1, 1].set_xlabel('Predicted Temperature (°C)')
    axes[1, 1].set_ylabel('Experimental Temperature (°C)')
    axes[1, 1].set_title('Experimental vs Predicted Melting Temperature')

    # Add diagonal line for perfect prediction
    min_temp = min(pred_data['Prediction'].min(), pred_data['Melting_Temperature'].min())
    max_temp = max(pred_data['Prediction'].max(), pred_data['Melting_Temperature'].max())
    axes[1, 1].plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, label='Perfect Prediction')

    # Calculate R²
    r2 = r2_score(pred_data['Melting_Temperature'], pred_data['Prediction'])
    mae = mean_absolute_error(pred_data['Melting_Temperature'], pred_data['Prediction'])
    axes[1, 1].text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}°C',
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1, 1].legend()
else:
    axes[1, 1].text(0.5, 0.5, 'No prediction data', ha='center', va='center',
                    transform=axes[1, 1].transAxes)

# 6. Expression system analysis
expr_data = df.dropna(subset=['Expression', 'Melting_Temperature'])
if len(expr_data) > 0:
    expr_grouped = expr_data.groupby('Expression')['Melting_Temperature'].agg(['mean', 'std', 'count'])
    expr_grouped = expr_grouped.sort_values('mean', ascending=False)

    axes[1, 2].bar(range(len(expr_grouped)), expr_grouped['mean'],
                   yerr=expr_grouped['std'], capsize=5, alpha=0.7, color='teal')
    axes[1, 2].set_xticks(range(len(expr_grouped)))
    axes[1, 2].set_xticklabels(expr_grouped.index, rotation=45)
    axes[1, 2].set_ylabel('Mean Melting Temperature (°C)')
    axes[1, 2].set_title('Mean Melting Temperature by Expression System')

    # Add count annotations
    for i, (count, mean_temp) in enumerate(zip(expr_grouped['count'], expr_grouped['mean'])):
        axes[1, 2].text(i, mean_temp + 1, f'n={count}', ha='center', va='bottom', fontsize=8)
else:
    axes[1, 2].text(0.5, 0.5, 'No expression data', ha='center', va='center',
                    transform=axes[1, 2].transAxes)

plt.tight_layout()
plt.show()

# Statistical analysis
print("\n=== STATISTICAL ANALYSIS ===")

# ANOVA for techniques
technique_groups = [group['Melting_Temperature'].dropna() for name, group in df.groupby('Technique')]
technique_groups = [group for group in technique_groups if len(group) > 1]

if len(technique_groups) > 1:
    f_stat, p_value = stats.f_oneway(*technique_groups)
    print(f"ANOVA for techniques: F-statistic = {f_stat:.3f}, p-value = {p_value:.3f}")
    if p_value < 0.05:
        print("Significant difference between techniques (p < 0.05)")
    else:
        print("No significant difference between techniques (p >= 0.05)")

# Amino acid composition analysis
print("\n=== AMINO ACID COMPOSITION ANALYSIS ===")
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Calculate amino acid frequencies
aa_freq = pd.DataFrame()
for aa in amino_acids:
    aa_freq[aa] = df['Sequence'].str.count(aa) / df['Sequence'].str.len()

# Correlations with melting temperature
print("Amino acid correlations with melting temperature:")
correlations = []
for aa in amino_acids:
    valid_indices = df['Melting_Temperature'].notna()
    if valid_indices.sum() > 10:  # Need at least 10 valid data points
        corr = stats.pearsonr(aa_freq.loc[valid_indices, aa],
                              df.loc[valid_indices, 'Melting_Temperature'])[0]
        correlations.append((aa, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for aa, corr in correlations[:10]:  # Top 10 correlations
    print(f"{aa}: {corr:.3f}")

# Summary statistics by source
print("\n=== ANALYSIS BY DATA SOURCE ===")
source_summary = df.groupby('Source')['Melting_Temperature'].agg(['count', 'mean', 'std']).round(2)
source_summary = source_summary.sort_values('count', ascending=False)
print("Top sources by number of entries:")
print(source_summary.head(10))

# Final summary
print("\n=== DATASET SUMMARY ===")
print(f"Total entries: {len(df)}")
print(f"Entries with melting temperature: {df['Melting_Temperature'].notna().sum()}")
print(f"Unique proteins: {df['ID'].nunique()}")
print(f"Temperature range: {df['Melting_Temperature'].min():.1f}°C - {df['Melting_Temperature'].max():.1f}°C")
print(f"Most common technique: {df['Technique'].mode().iloc[0]} ({df['Technique'].value_counts().iloc[0]} entries)")

# Data quality assessment
print(f"\nData completeness:")
for col in ['pH', 'Concentration', 'Expression', 'Solvent']:
    completeness = (df[col].notna().sum() / len(df)) * 100
    print(f"{col}: {completeness:.1f}% complete")