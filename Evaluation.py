import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, kl_divergence
from esm import pretrained
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import joblib

warnings.filterwarnings('ignore')


# ---------------------------
# 1. Model Architecture (Same as training)
# ---------------------------
class EnhancedBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        if use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.prior_std = prior_std

    def forward(self, x):
        # Reparameterization trick
        weight_std = F.softplus(self.weight_rho) + 1e-6
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * weight_eps

        if self.use_bias:
            bias_std = F.softplus(self.bias_rho) + 1e-6
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_std * bias_eps
        else:
            bias = None

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_std = F.softplus(self.weight_rho) + 1e-6
        weight_prior = Normal(0, self.prior_std)
        weight_posterior = Normal(self.weight_mu, weight_std)
        weight_kl = kl_divergence(weight_posterior, weight_prior).sum()

        if self.use_bias:
            bias_std = F.softplus(self.bias_rho) + 1e-6
            bias_prior = Normal(0, self.prior_std)
            bias_posterior = Normal(self.bias_mu, bias_std)
            bias_kl = kl_divergence(bias_posterior, bias_prior).sum()
            return weight_kl + bias_kl

        return weight_kl


class EnhancedBayesianESMFusionModel(nn.Module):
    def __init__(self, esm_dim=1280, physchem_dim=17, fusion_hidden=512, prior_std=0.5):
        super().__init__()

        # ESM feature processing
        self.esm_processor = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

        # Enhanced physicochemical branch
        self.physchem_branch = nn.Sequential(
            nn.LayerNorm(physchem_dim),
            EnhancedBayesianLinear(physchem_dim, 64, prior_std=prior_std),
            nn.ReLU(),
            nn.Dropout(0.1),
            EnhancedBayesianLinear(64, 128, prior_std=prior_std),
            nn.ReLU(),
            nn.Dropout(0.1),
            EnhancedBayesianLinear(128, 128, prior_std=prior_std)
        )

        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )

        # Enhanced fusion network
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(256 + 128),
            EnhancedBayesianLinear(256 + 128, fusion_hidden, prior_std=prior_std),
            nn.ReLU(),
            nn.Dropout(0.2),
            EnhancedBayesianLinear(fusion_hidden, 256, prior_std=prior_std),
            nn.ReLU(),
            nn.Dropout(0.1),
            EnhancedBayesianLinear(256, 128, prior_std=prior_std),
            nn.ReLU(),
            EnhancedBayesianLinear(128, 2, prior_std=prior_std)  # mean and log_var
        )

    def forward(self, esm_vec, physchem_vec):
        # Process ESM features
        esm_processed = self.esm_processor(esm_vec)

        # Process physicochemical features
        physchem_embed = self.physchem_branch(physchem_vec)

        # Apply attention to ESM features
        esm_attended, _ = self.attention(
            esm_processed.unsqueeze(1),
            esm_processed.unsqueeze(1),
            esm_processed.unsqueeze(1)
        )
        esm_attended = esm_attended.squeeze(1)

        # Fusion
        fused = torch.cat((esm_attended, physchem_embed), dim=1)
        out = self.fusion_mlp(fused)

        mean = out[:, :1]
        log_var = out[:, 1:]

        return mean, log_var


# ---------------------------
# 2. Feature Extraction
# ---------------------------
def extract_enhanced_physchem_features(seq):
    """Extract comprehensive physicochemical features"""
    analysed = ProteinAnalysis(seq)
    helix, turn, sheet = analysed.secondary_structure_fraction()

    # Basic features
    basic_feats = [
        analysed.molecular_weight(),
        analysed.isoelectric_point(),
        analysed.aromaticity(),
        analysed.instability_index(),
        analysed.gravy(),
        np.mean(analysed.flexibility()) if analysed.flexibility() else 0.0,
        helix, turn, sheet
    ]

    # Enhanced features
    aa_counts = analysed.amino_acids_percent
    hydrophobic_aa = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    polar_aa = ['S', 'T', 'N', 'Q']
    charged_aa = ['R', 'K', 'D', 'E']
    aromatic_aa = ['F', 'Y', 'W']

    enhanced_feats = [
        sum(aa_counts.get(aa, 0) for aa in hydrophobic_aa),
        sum(aa_counts.get(aa, 0) for aa in polar_aa),
        sum(aa_counts.get(aa, 0) for aa in charged_aa),
        sum(aa_counts.get(aa, 0) for aa in aromatic_aa),
        len(seq),
        seq.count('C') / len(seq) if len(seq) > 0 else 0,
        seq.count('P') / len(seq) if len(seq) > 0 else 0,
        seq.count('G') / len(seq) if len(seq) > 0 else 0,
    ]

    return basic_feats + enhanced_feats


# ---------------------------
# 3. Test Dataset Class
# ---------------------------
class TestProteinDataset(Dataset):
    def __init__(self, sequences, labels, alphabet, scaler):
        self.sequences = sequences
        self.labels = labels if labels is not None else [0] * len(sequences)  # Dummy labels if None
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        # Extract and scale features
        self.feats = [extract_enhanced_physchem_features(seq) for seq in sequences]
        self.feats = scaler.transform(self.feats)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.feats[idx]

    def collate_fn(self, batch):
        data = [(str(i), seq) for i, (seq, _, _) in enumerate(batch)]
        labels = torch.tensor([label for _, label, _ in batch], dtype=torch.float32).unsqueeze(1)
        feats_array = np.array([feat for _, _, feat in batch])
        feats = torch.tensor(feats_array, dtype=torch.float32)
        _, _, tokens = self.batch_converter(data)
        return tokens, feats, labels


# ---------------------------
# 4. Bayesian Prediction with Uncertainty
# ---------------------------
def bayesian_predict_with_uncertainty(model, esm_model, tokens, feats, alphabet,
                                      n_samples=100, confidence_levels=[0.68, 0.95]):
    """
    Make predictions with comprehensive uncertainty quantification

    Args:
        model: Trained Bayesian model
        esm_model: ESM model for embeddings
        tokens: Tokenized sequences
        feats: Physicochemical features
        alphabet: ESM alphabet
        n_samples: Number of Monte Carlo samples
        confidence_levels: List of confidence levels (e.g., [0.68, 0.95] for 1σ, 2σ)

    Returns:
        Dictionary with predictions and uncertainties
    """
    esm_model.eval()

    # Get ESM representations
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[33])
        reps = out["representations"][33]
        mask = (tokens != alphabet.padding_idx)
        esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    # Monte Carlo sampling
    model.train()  # Keep in training mode for stochastic behavior
    all_means, all_vars = [], []

    for _ in range(n_samples):
        with torch.no_grad():
            mean, log_var = model(esm_vec, feats)
            all_means.append(mean)
            all_vars.append(torch.exp(log_var))

    # Stack predictions
    means = torch.stack(all_means)  # [n_samples, batch_size, 1]
    vars_aleatoric = torch.stack(all_vars)  # [n_samples, batch_size, 1]

    # Calculate uncertainties
    epistemic = means.var(dim=0)  # Model uncertainty
    aleatoric = vars_aleatoric.mean(dim=0)  # Data uncertainty
    total_uncertainty = epistemic + aleatoric

    # Predictive statistics
    predictive_mean = means.mean(dim=0)
    predictive_std = torch.sqrt(total_uncertainty)

    # Calculate confidence intervals
    confidence_intervals = {}
    for conf_level in confidence_levels:
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        margin = z_score * predictive_std
        confidence_intervals[f'{int(conf_level * 100)}%'] = {
            'lower': predictive_mean - margin,
            'upper': predictive_mean + margin
        }

    return {
        'mean': predictive_mean,
        'std': predictive_std,
        'epistemic_uncertainty': torch.sqrt(epistemic),
        'aleatoric_uncertainty': torch.sqrt(aleatoric),
        'total_uncertainty': predictive_std,
        'confidence_intervals': confidence_intervals,
        'all_samples': means  # For further analysis if needed
    }


# ---------------------------
# 5. Model Ensemble Predictions
# ---------------------------
def ensemble_predict(model_paths, esm_model, tokens, feats, alphabet, device,
                     n_samples=50, confidence_levels=[0.68, 0.95]):
    """
    Make ensemble predictions using multiple fold models
    """
    all_predictions = []

    # Load model architecture
    model = EnhancedBayesianESMFusionModel(
        esm_dim=1280, physchem_dim=17, fusion_hidden=512, prior_std=0.5
    ).to(device)

    for model_path in model_paths:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

        pred_results = bayesian_predict_with_uncertainty(
            model, esm_model, tokens, feats, alphabet, n_samples, confidence_levels
        )
        all_predictions.append(pred_results['mean'])

    # Ensemble statistics
    ensemble_preds = torch.stack(all_predictions)
    ensemble_mean = ensemble_preds.mean(dim=0)
    ensemble_std = ensemble_preds.std(dim=0)

    # Calculate ensemble confidence intervals
    confidence_intervals = {}
    for conf_level in confidence_levels:
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        margin = z_score * ensemble_std
        confidence_intervals[f'{int(conf_level * 100)}%'] = {
            'lower': ensemble_mean - margin,
            'upper': ensemble_mean + margin
        }

    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'confidence_intervals': confidence_intervals,
        'individual_predictions': all_predictions
    }


# ---------------------------
# 6. Comprehensive Testing Function
# ---------------------------
def test_bayesian_model(model_paths, test_data_path, scaler_path=None,
                        output_path="NB_Bench_test_results.csv", plot_results=True,
                        n_samples=100, confidence_levels=[0.68, 0.95]):
    """
    Comprehensive testing function for Bayesian models

    Args:
        model_paths: List of paths to saved model files (from different folds)
        test_data_path: Path to CSV file with test data
        scaler_path: Path to saved scaler (if None, will fit new scaler)
        output_path: Path to save results
        plot_results: Whether to create plots
        n_samples: Number of Monte Carlo samples
        confidence_levels: Confidence levels for intervals
    """

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ESM model
    print("Loading ESM model...")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    print(test_df.columns)

    # Check required columns
    if 'Sequence' not in test_df.columns:
        raise ValueError("Test data must contain 'Sequence' column")

    sequences = test_df['Sequence'].tolist()

    # Check if we have true labels for evaluation
    has_labels = 'Melting_Temperature' in test_df.columns
    labels = test_df['Melting_Temperature'].tolist() if has_labels else None

    # Prepare scaler
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded existing scaler")
    else:
        print("Fitting new scaler on test data...")
        all_feats = np.array([extract_enhanced_physchem_features(seq) for seq in sequences])
        scaler = RobustScaler()
        scaler.fit(all_feats)
        if scaler_path:
            joblib.dump(scaler, scaler_path)

    # Create test dataset
    test_dataset = TestProteinDataset(sequences, labels, alphabet, scaler)
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        collate_fn=test_dataset.collate_fn, num_workers=2
    )

    # Make predictions
    print(f"Making predictions with {len(model_paths)} models...")
    all_results = []

    for batch_idx, (tokens, feats, true_labels) in enumerate(test_loader):
        tokens, feats = tokens.to(device), feats.to(device)

        if len(model_paths) > 1:
            # Ensemble prediction
            batch_results = ensemble_predict(
                model_paths, esm_model, tokens, feats, alphabet, device,
                n_samples, confidence_levels
            )
        else:
            # Single model prediction
            model = EnhancedBayesianESMFusionModel(
                esm_dim=1280, physchem_dim=17, fusion_hidden=512, prior_std=0.5
            ).to(device)

            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_paths[0]))
            else:
                model.load_state_dict(torch.load(model_paths[0], map_location='cpu'))

            batch_results = bayesian_predict_with_uncertainty(
                model, esm_model, tokens, feats, alphabet, n_samples, confidence_levels
            )

        # Store results
        batch_size = tokens.size(0)
        for i in range(batch_size):
            result = {
                'sequence_idx': batch_idx * 8 + i,
                'predicted_mean': batch_results['mean'][i].item(),
                'predicted_std': batch_results['std'][i].item(),
                'epistemic_uncertainty': batch_results.get('epistemic_uncertainty', [0])[
                    i].item() if 'epistemic_uncertainty' in batch_results else 0,
                'aleatoric_uncertainty': batch_results.get('aleatoric_uncertainty', [0])[
                    i].item() if 'aleatoric_uncertainty' in batch_results else 0,
            }

            # Add confidence intervals
            for conf_level in confidence_levels:
                conf_key = f'{int(conf_level * 100)}%'
                result[f'ci_{conf_key}_lower'] = batch_results['confidence_intervals'][conf_key]['lower'][i].item()
                result[f'ci_{conf_key}_upper'] = batch_results['confidence_intervals'][conf_key]['upper'][i].item()

            # Add true label if available
            if has_labels:
                result['true_value'] = true_labels[i].item()

            all_results.append(result)

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * 8} sequences...")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    print(results_df)

    # Add original data
    for col in test_df.columns:
        if col not in ['Melting_Temperature']:  # Avoid duplicate
            results_df[col] = test_df[col].values[:len(results_df)]

    # Calculate evaluation metrics if we have true labels
    if has_labels:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from scipy.stats import spearmanr
        import math

        # Compute metrics
        mae = mean_absolute_error(test_df['Melting_Temperature'], results_df['predicted_mean'])
        rmse = math.sqrt(mean_squared_error(test_df['Melting_Temperature'], results_df['predicted_mean']))
        r2 = r2_score(test_df['Melting_Temperature'], results_df['predicted_mean'])

        # Spearman correlation
        spearman_corr, spearman_pval = spearmanr(test_df['Melting_Temperature'], results_df['predicted_mean'])

        print(f"\nEvaluation Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f} (p={spearman_pval:.4e})")


        # Check calibration (how well uncertainty estimates match actual errors)
        errors = np.abs(results_df['labels'] - results_df['predicted_mean'])
        predicted_std = results_df['predicted_std']

        # Calibration: percentage of points within predicted intervals
        for conf_level in confidence_levels:
            conf_key = f'{int(conf_level * 100)}%'
            within_interval = (
                    (results_df['labels'] >= results_df[f'ci_{conf_key}_lower']) &
                    (results_df['labels'] <= results_df[f'ci_{conf_key}_upper'])
            ).mean()
            print(f"Calibration ({conf_key}): {within_interval:.3f} (Expected: {conf_level:.3f})")

    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Create plots if requested
    if plot_results and has_labels:
        create_prediction_plots(results_df, confidence_levels)

    return results_df


# ---------------------------
# 7. Visualization Functions
# ---------------------------
def create_prediction_plots(results_df, confidence_levels=[0.68, 0.95]):
    """Create comprehensive prediction plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Prediction vs True scatter plot
    ax1 = axes[0, 0]
    x = results_df['Melting_Temperature']
    y = results_df['predicted_mean']
    yerr = results_df['predicted_std']

    ax1.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.6, capsize=2)

    # Perfect prediction line
    min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')

    ax1.set_xlabel('True Melting Temperature')
    ax1.set_ylabel('Predicted Melting Temperature')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Predictions
    ax2 = axes[0, 1]
    residuals = results_df['Melting_Temperature'] - results_df['predicted_mean']
    ax2.scatter(results_df['predicted_mean'], residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Uncertainty vs Error
    ax3 = axes[1, 0]
    errors = np.abs(residuals)
    ax3.scatter(results_df['predicted_std'], errors, alpha=0.6)
    ax3.plot([0, results_df['predicted_std'].max()], [0, results_df['predicted_std'].max()],
             'r--', alpha=0.8, label='Perfect Calibration')
    ax3.set_xlabel('Predicted Uncertainty (σ)')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Uncertainty Calibration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Confidence Interval Coverage
    ax4 = axes[1, 1]
    coverages = []
    conf_labels = []

    for conf_level in confidence_levels:
        conf_key = f'{int(conf_level * 100)}%'
        within_interval = (
                (results_df['true_value'] >= results_df[f'ci_{conf_key}_lower']) &
                (results_df['true_value'] <= results_df[f'ci_{conf_key}_upper'])
        ).mean()
        coverages.append(within_interval)
        conf_labels.append(f'{int(conf_level * 100)}%')

    x_pos = np.arange(len(conf_labels))
    ax4.bar(x_pos, coverages, alpha=0.7, label='Observed')
    ax4.bar(x_pos, confidence_levels, alpha=0.5, label='Expected')
    ax4.set_xlabel('Confidence Level')
    ax4.set_ylabel('Coverage')
    ax4.set_title('Confidence Interval Coverage')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(conf_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------
# 8. Example Usage
# ---------------------------
if __name__ == "__main__":
    # Example usage
    model_paths = [
        "enhanced_bayesian_model_fold1.pt",
        "enhanced_bayesian_model_fold2.pt",
        "enhanced_bayesian_model_fold3.pt",
        "enhanced_bayesian_model_fold4.pt",
        "enhanced_bayesian_model_fold5.pt"
    ]

    # Test the models
    results = test_bayesian_model(
        model_paths=model_paths,
        test_data_path="NB_bench_test_dataset.csv",  # Replace with your test data path
        output_path="NB_Bench_bayesian_test_results.csv",
        plot_results=True,
        n_samples=83,
        confidence_levels=[0.68, 0.95]  # 68% and 95% confidence intervals
    )

    print("Testing completed!")
    print(f"Results shape: {results.shape}")
    print("\nFirst few predictions:")
    print(results[['Sequence', 'predicted_mean', 'predicted_std',
                   'ci_68%_lower', 'ci_68%_upper', 'ci_95%_lower', 'ci_95%_upper']].head())