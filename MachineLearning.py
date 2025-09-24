

#######################################
#####      environment tf-gpu     ###########

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_validate, KFold # For train/test splits
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.pipeline import Pipeline # For setting up pipeline
# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.decomposition import PCA # For PCA
from sklearn.model_selection import GridSearchCV # For optimization

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
df = pd.read_csv('/home/f087s426/Research/Nanobody_Thermo_Prediction/processed_protein_sequences.csv')
df_nano_ab = np.load('/home/f087s426/Research/Nanobody_Thermo_Prediction/Nanobody_ablang_embedding.npy')
df_nano_esm = np.load('/home/f087s426/Research/Nanobody_Thermo_Prediction/Nanobody_esm_embedding.npy')
df_nano_protbert = np.load('/home/f087s426/Research/Nanobody_Thermo_Prediction/Nanobody_protbert_embedding.npy')
df_nano_esmfold = np.load('/home/f087s426/Research/Nanobody_Thermo_Prediction/Nanobody_esmfold_embedding.npy')

# Creating new df that will contain melting temperatures of protein sequences and take embeddings as input
df_new_esm = pd.DataFrame(df_nano_esm)
df_new_ab = pd.DataFrame(df_nano_ab)
df_new_protbert = pd.DataFrame(df_nano_protbert)
df_new_esmfold = pd.DataFrame(df_nano_esmfold)

# Adding Melting Temperature columns
df_new_esm['Melting_Temperature'] = df['Melting_Temperature']
df_new_ab['Melting_Temperature'] = df['Melting_Temperature']
df_new_protbert['Melting_Temperature'] = df['Melting_Temperature']
df_new_esmfold['Melting_Temperature'] = df['Melting_Temperature']


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# ✅ Parent output directory
# -----------------------------
parent_output_dir = '/home/f087s426/Research/Nanobody_Thermo_Prediction/Trained_Models'
os.makedirs(parent_output_dir, exist_ok=True)

# -----------------------------
# ✅ Assign target column
# -----------------------------
df_new_esm['Melting_Temperature'] = df['Melting_Temperature']
df_new_ab['Melting_Temperature'] = df['Melting_Temperature']
df_new_protbert['Melting_Temperature'] = df['Melting_Temperature']
df_new_esmfold['Melting_Temperature'] = df['Melting_Temperature']

all_dfs = {
    'esm': df_new_esm,
    'ab': df_new_ab,
    'protbert': df_new_protbert,
    'esmfold': df_new_esmfold
}

# -----------------------------
# ✅ Helper functions
# -----------------------------
def create_features_and_target_split(df):
    X = df.drop(columns=['Melting_Temperature'])
    y = df['Melting_Temperature']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('pca', PCA(n_components=0.95)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', model)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Kernel Ridge': KernelRidge(kernel="rbf", gamma=0.1),
    'Support Vector': SVR(kernel="rbf", gamma=0.1),
    'Stochastic Regressor': SGDRegressor(max_iter=1000, tol=1e-3),
    #'Random Forest Regressor': RandomForestRegressor(),
    'XGBoost Regressor': XGBRegressor()
    #'LightGBM Regressor': LGBMRegressor()
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
global_results = []

# -----------------------------
# ✅ Main loop
# -----------------------------
for df_name, df in all_dfs.items():
    print(f"\n🔍 Processing dataset: {df_name}")
    dataset_results = []

    # Dataset folder
    dataset_dir = os.path.join(parent_output_dir, df_name)
    os.makedirs(dataset_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = create_features_and_target_split(df)

    for model_name, model in models.items():
        print(f"  → Model: {model_name}")

        model_dir = os.path.join(dataset_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)

        pipeline = create_pipeline(model)

        # Cross-validation
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'),
            return_train_score=False
        )

        # Train/test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Metrics
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        result = {
            'Dataset': df_name,
            'Model': model_name,
            'CV R2 Mean': np.mean(cv_results['test_r2']),
            'CV MSE Mean': -np.mean(cv_results['test_neg_mean_squared_error']),
            'CV MAE Mean': -np.mean(cv_results['test_neg_mean_absolute_error']),
            'Test R2': test_r2,
            'Test MSE': test_mse,
            'Test MAE': test_mae
        }

        dataset_results.append(result)
        global_results.append(result)

        # Save individual result as .txt
        with open(os.path.join(model_dir, 'results.txt'), 'w') as f:
            for key, value in result.items():
                f.write(f"{key}: {value}\n")

        # Plot: Actual vs Predicted
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Actual vs Predicted ({df_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'actual_vs_predicted.png'))
        plt.close()

        # Plot: Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.title(f'{model_name}: Residuals Distribution ({df_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'residuals.png'))
        plt.close()

    # Save dataset-level CSV
    dataset_results_df = pd.DataFrame(dataset_results)
    dataset_csv_path = os.path.join(dataset_dir, 'regression_model_comparison.csv')
    dataset_results_df.to_csv(dataset_csv_path, index=False)
    print(f"  ✅ Saved dataset results: {dataset_csv_path}")

# Save global comparison CSV
global_results_df = pd.DataFrame(global_results)
global_csv_path = os.path.join(parent_output_dir, 'all_regression_model_comparisons.csv')
global_results_df.to_csv(global_csv_path, index=False)
print(f"\n📄 Global results saved to: {global_csv_path}")
