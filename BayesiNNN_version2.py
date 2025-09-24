import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, kl_divergence
from esm import pretrained
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


# ---------------------------
# 1. Enhanced Physicochemical Feature Extraction
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

    # Enhanced features - use amino_acids_percent attribute instead of deprecated method
    aa_counts = analysed.amino_acids_percent
    hydrophobic_aa = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    polar_aa = ['S', 'T', 'N', 'Q']
    charged_aa = ['R', 'K', 'D', 'E']
    aromatic_aa = ['F', 'Y', 'W']

    enhanced_feats = [
        sum(aa_counts.get(aa, 0) for aa in hydrophobic_aa),  # Hydrophobic content
        sum(aa_counts.get(aa, 0) for aa in polar_aa),  # Polar content
        sum(aa_counts.get(aa, 0) for aa in charged_aa),  # Charged content
        sum(aa_counts.get(aa, 0) for aa in aromatic_aa),  # Aromatic content
        len(seq),  # Sequence length
        seq.count('C') / len(seq) if len(seq) > 0 else 0,  # Cysteine content
        seq.count('P') / len(seq) if len(seq) > 0 else 0,  # Proline content
        seq.count('G') / len(seq) if len(seq) > 0 else 0,  # Glycine content
    ]

    return basic_feats + enhanced_feats


# ---------------------------
# 2. Enhanced Bayesian Linear Layer
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
        self.reset_parameters()

    def reset_parameters(self):
        # Better initialization
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -4.0)  # Start with lower variance

        if self.use_bias:
            nn.init.constant_(self.bias_mu, 0.0)
            nn.init.constant_(self.bias_rho, -4.0)

    def forward(self, x):
        # Reparameterization trick
        weight_std = F.softplus(self.weight_rho) + 1e-6  # Add small epsilon for stability
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


# ---------------------------
# 3. Enhanced Dataset
# ---------------------------
class EnhancedFusionProteinDataset(Dataset):
    def __init__(self, sequences, labels, alphabet, scaler=None, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.augment = augment

        # Extract enhanced features
        self.feats = [extract_enhanced_physchem_features(seq) for seq in sequences]

        if scaler:
            self.feats = scaler.transform(self.feats)
        else:
            self.feats = np.array(self.feats)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        feat = self.feats[idx]

        # Simple data augmentation for training
        if self.augment and np.random.random() < 0.1:
            # Add small noise to features
            feat = feat + np.random.normal(0, 0.01, feat.shape)

        return seq, label, feat

    def collate_fn(self, batch):
        data = [(str(i), seq) for i, (seq, _, _) in enumerate(batch)]
        labels = torch.tensor([label for _, label, _ in batch], dtype=torch.float32).unsqueeze(1)
        # Convert to numpy array first to avoid warning
        feats_array = np.array([feat for _, _, feat in batch])
        feats = torch.tensor(feats_array, dtype=torch.float32)
        _, _, tokens = self.batch_converter(data)
        return tokens, feats, labels


# ---------------------------
# 4. Enhanced Model Architecture
# ---------------------------
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

    def kl_divergence(self):
        kl_div = 0
        for module in self.modules():
            if isinstance(module, EnhancedBayesianLinear):
                kl_div += module.kl_divergence()
        return kl_div


# ---------------------------
# 5. Enhanced Loss Function
# ---------------------------
def enhanced_bayesian_loss(mean, log_var, target, kl_div, num_batches,
                           kl_weight=1.0, beta_schedule=None, epoch=None):
    """Enhanced loss with adaptive KL weighting"""

    # Adaptive KL weight scheduling
    if beta_schedule == 'cyclical' and epoch is not None:
        cycle_length = 10
        kl_weight = kl_weight * (0.5 * (1 + np.cos(np.pi * (epoch % cycle_length) / cycle_length)))
    elif beta_schedule == 'linear' and epoch is not None:
        kl_weight = min(kl_weight, kl_weight * epoch / 20)  # Gradually increase

    # Negative log-likelihood with heteroscedastic uncertainty
    precision = torch.exp(-log_var)
    nll = torch.mean(0.5 * (precision * (target - mean) ** 2 + log_var))

    # KL divergence term
    kl_term = kl_weight * kl_div / num_batches

    # Total loss
    total_loss = nll + kl_term

    return total_loss, nll, kl_term


# ---------------------------
# 6. Enhanced Prediction with Monte Carlo Dropout
# ---------------------------
def enhanced_bayesian_predict_with_uncertainty(model, esm_model, tokens, feats, alphabet, n_samples=50):
    """Enhanced prediction with both epistemic and aleatoric uncertainty"""
    esm_model.eval()

    # Get ESM representations once
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[33])
        reps = out["representations"][33]
        mask = (tokens != alphabet.padding_idx)
        esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    # Monte Carlo sampling
    model.train()  # Keep in training mode for dropout and Bayesian sampling
    all_means, all_vars = [], []

    for _ in range(n_samples):
        with torch.no_grad():
            mean, log_var = model(esm_vec, feats)
            all_means.append(mean)
            all_vars.append(torch.exp(log_var))

    # Calculate uncertainties
    means = torch.stack(all_means)
    vars_aleatoric = torch.stack(all_vars)

    # Epistemic uncertainty (model uncertainty)
    epistemic = means.var(dim=0)

    # Aleatoric uncertainty (data uncertainty)
    aleatoric = vars_aleatoric.mean(dim=0)

    # Total uncertainty
    total_uncertainty = epistemic + aleatoric

    # Predictive mean
    predictive_mean = means.mean(dim=0)

    return predictive_mean, total_uncertainty, epistemic, aleatoric


# ---------------------------
# 7. Enhanced Training Loop
# ---------------------------
def train_and_eval_enhanced_bayesian(train_loader, val_loader, model, esm_model, optimizer,
                                     scheduler, device, alphabet, patience=5, max_epochs=100,
                                     kl_weight=1.0, beta_schedule='cyclical'):
    best_loss = float('inf')
    patience_counter = 0
    best_model = None
    best_metrics = {}
    num_batches = len(train_loader)

    # Training history for analysis
    train_history = {'loss': [], 'nll': [], 'kl': []}
    val_history = {'loss': [], 'mae': [], 'rmse': [], 'r2': []}

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        esm_model.eval()
        epoch_loss = epoch_nll = epoch_kl = 0

        for batch_idx, (tokens, feats, labels) in enumerate(train_loader):
            tokens, feats, labels = tokens.to(device), feats.to(device), labels.to(device)

            # Get ESM representations
            with torch.no_grad():
                out = esm_model(tokens, repr_layers=[33])
                reps = out["representations"][33]
                mask = (tokens != alphabet.padding_idx)
                esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

            # Forward pass
            mean, log_var = model(esm_vec, feats)
            kl_div = model.kl_divergence()

            # Compute loss
            loss, nll, kl_term = enhanced_bayesian_loss(
                mean, log_var, labels, kl_div, num_batches,
                kl_weight, beta_schedule, epoch
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kl += kl_term.item()

        # Validation phase
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        all_epistemic, all_aleatoric, all_total = [], [], []

        with torch.no_grad():
            for tokens, feats, labels in val_loader:
                tokens, feats, labels = tokens.to(device), feats.to(device), labels.to(device)

                pred_mean, total_unc, epistemic, aleatoric = enhanced_bayesian_predict_with_uncertainty(
                    model, esm_model, tokens, feats, alphabet, n_samples=30
                )

                loss = F.mse_loss(pred_mean, labels)
                val_loss += loss.item()

                all_preds.extend(pred_mean.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_epistemic.extend(epistemic.cpu().numpy())
                all_aleatoric.extend(aleatoric.cpu().numpy())
                all_total.extend(total_unc.cpu().numpy())

        # Calculate metrics
        val_loss /= len(val_loader)
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
        val_r2 = r2_score(all_labels, all_preds)
        avg_epistemic = np.mean(all_epistemic)
        avg_aleatoric = np.mean(all_aleatoric)
        avg_total = np.mean(all_total)

        # Store history
        train_history['loss'].append(epoch_loss / num_batches)
        train_history['nll'].append(epoch_nll / num_batches)
        train_history['kl'].append(epoch_kl / num_batches)
        val_history['loss'].append(val_loss)
        val_history['mae'].append(val_mae)
        val_history['rmse'].append(val_rmse)
        val_history['r2'].append(val_r2)

        # Update learning rate scheduler
        if scheduler:
            scheduler.step(val_r2)  # Pass the metric for ReduceLROnPlateau

        # Print progress
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch + 1}/{max_epochs}:")
            print(
                f"  Train - Loss: {epoch_loss / num_batches:.4f}, NLL: {epoch_nll / num_batches:.4f}, KL: {epoch_kl / num_batches:.4f}")
            print(f"  Val - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
            print(
                f"  Uncertainty - Epistemic: {avg_epistemic:.4f}, Aleatoric: {avg_aleatoric:.4f}, Total: {avg_total:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping based on validation R² (higher is better)
        if val_r2 > best_metrics.get('r2', -float('inf')):
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
            best_metrics = {
                "val_loss": val_loss,
                "mae": val_mae,
                "rmse": val_rmse,
                "r2": val_r2,
                "epistemic": avg_epistemic,
                "aleatoric": avg_aleatoric,
                "total_uncertainty": avg_total
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_model)
    return model, best_metrics, train_history, val_history


# ---------------------------
# 8. Enhanced Main Function
# ---------------------------
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("/home/f087s426/Research/Nanobody_Thermo_Prediction/processed_protein_sequences.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ESM model
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)

    sequences = df.Sequence.tolist()
    labels = df.Melting_Temperature.tolist()

    # Prepare cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Feature scaling
    all_feats = np.array([extract_enhanced_physchem_features(seq) for seq in sequences])
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    scaler.fit(all_feats)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"\n{'=' * 60}\nFOLD {fold + 1}\n{'=' * 60}")

        # Create datasets
        train_dataset = EnhancedFusionProteinDataset(
            [sequences[i] for i in train_idx],
            [labels[i] for i in train_idx],
            alphabet, scaler, augment=True
        )
        val_dataset = EnhancedFusionProteinDataset(
            [sequences[i] for i in val_idx],
            [labels[i] for i in val_idx],
            alphabet, scaler, augment=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=True,
            collate_fn=train_dataset.collate_fn, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=4, shuffle=False,
            collate_fn=val_dataset.collate_fn, num_workers=2
        )

        # Initialize model
        model = EnhancedBayesianESMFusionModel(
            esm_dim=1280, physchem_dim=17, fusion_hidden=512, prior_std=0.5
        ).to(device)

        # Optimizer with different learning rates for different parts
        optimizer = optim.AdamW([
            {'params': model.esm_processor.parameters(), 'lr': 1e-5},
            {'params': model.physchem_branch.parameters(), 'lr': 1e-4},
            {'params': model.attention.parameters(), 'lr': 5e-5},
            {'params': model.fusion_mlp.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-5)

        # Learning rate scheduler - remove verbose to avoid deprecation warning
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        # Train model
        trained_model, best_metrics, train_hist, val_hist = train_and_eval_enhanced_bayesian(
            train_loader, val_loader, model, esm_model, optimizer, scheduler,
            device, alphabet, patience=8, max_epochs=80, kl_weight=0.1, beta_schedule='cyclical'
        )

        # Save results
        torch.save(trained_model.state_dict(), f"enhanced_bayesian_model_fold{fold + 1}.pt")

        with open(f"enhanced_bayesian_metrics_fold{fold + 1}.txt", "w") as f:
            for k, v in best_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        fold_results.append(best_metrics)
        print(f"\nFold {fold + 1} Results:")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Summary across all folds
    print(f"\n{'=' * 60}\nSUMMARY ACROSS ALL FOLDS\n{'=' * 60}")
    metrics_names = ['mae', 'rmse', 'r2', 'epistemic', 'aleatoric', 'total_uncertainty']

    for metric in metrics_names:
        values = [fold[metric] for fold in fold_results]
        print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"\n{'=' * 60}\nENHANCED BAYESIAN TRAINING COMPLETED\n{'=' * 60}")