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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# 1. Physicochemical Feature Extraction
# ---------------------------
def extract_physchem_features(seq):
    analysed = ProteinAnalysis(seq)
    helix, turn, sheet = analysed.secondary_structure_fraction()
    feats = [
        analysed.molecular_weight(),
        analysed.isoelectric_point(),
        analysed.aromaticity(),
        analysed.instability_index(),
        analysed.gravy(),
        np.mean(analysed.flexibility()),
        helix,
        turn,
        sheet
    ]
    return feats

# ---------------------------
# 2. Bayesian Linear Layer
# ---------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.prior_std = prior_std
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        weight_prior = Normal(0, self.prior_std)
        bias_prior = Normal(0, self.prior_std)
        weight_posterior = Normal(self.weight_mu, weight_std)
        bias_posterior = Normal(self.bias_mu, bias_std)
        weight_kl = kl_divergence(weight_posterior, weight_prior).sum()
        bias_kl = kl_divergence(bias_posterior, bias_prior).sum()
        return weight_kl + bias_kl

# ---------------------------
# 3. Dataset
# ---------------------------
class FusionProteinDataset(Dataset):
    def __init__(self, sequences, labels, alphabet, scaler=None):
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = alphabet.get_batch_converter()
        self.scaler = scaler
        self.feats = [extract_physchem_features(seq) for seq in sequences]
        if scaler:
            self.feats = scaler.transform(self.feats)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.feats[idx]

    def collate_fn(self, batch):
        data = [(str(i), seq) for i, (seq, _, _) in enumerate(batch)]
        labels = torch.tensor([label for _, label, _ in batch], dtype=torch.float32).unsqueeze(1)
        feats = torch.tensor([feat for _, _, feat in batch], dtype=torch.float32)
        _, _, tokens = self.batch_converter(data)
        return tokens, feats, labels

# ---------------------------
# 4. Model
# ---------------------------
class BayesianESMFusionModel(nn.Module):
    def __init__(self, esm_dim=1280, physchem_dim=9, fusion_hidden=512, prior_std=1.0):
        super().__init__()
        self.physchem_branch = nn.Sequential(
            BayesianLinear(physchem_dim, 64, prior_std=prior_std),
            nn.ReLU(),
            BayesianLinear(64, 128, prior_std=prior_std),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            BayesianLinear(esm_dim + 128, fusion_hidden, prior_std=prior_std),
            nn.ReLU(),
            nn.Dropout(0.2),
            BayesianLinear(fusion_hidden, 2, prior_std=prior_std)
        )

    def forward(self, esm_vec, physchem_vec):
        physchem_embed = self.physchem_branch(physchem_vec)
        x = torch.cat((esm_vec, physchem_embed), dim=1)
        out = self.fusion_mlp(x)
        mean = out[:, :1]
        log_var = out[:, 1:]
        return mean, log_var

    def kl_divergence(self):
        return sum(m.kl_divergence() for m in self.modules() if isinstance(m, BayesianLinear))

# ---------------------------
# 5. Loss
# ---------------------------
def bayesian_loss(mean, log_var, target, kl_div, num_batches, kl_weight=1.0):
    precision = torch.exp(-log_var)
    nll = torch.mean(precision * (target - mean) ** 2 + log_var)
    kl_term = kl_weight * kl_div / num_batches
    return nll + kl_term, nll, kl_term

# ---------------------------
# 6. Prediction
# ---------------------------
def bayesian_predict_with_uncertainty(model, esm_model, tokens, feats, alphabet, n_samples=20):
    model.train()
    esm_model.eval()
    all_means, all_vars = [], []
    for _ in range(n_samples):
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[33])
            reps = out["representations"][33]
            mask = (tokens != alphabet.padding_idx)
            esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            mean, log_var = model(esm_vec, feats)
            all_means.append(mean)
            all_vars.append(torch.exp(log_var))
    means = torch.stack(all_means)
    vars_aleatoric = torch.stack(all_vars)
    epistemic = means.var(dim=0)
    aleatoric = vars_aleatoric.mean(dim=0)
    total_uncertainty = epistemic + aleatoric
    predictive_mean = means.mean(dim=0)
    return predictive_mean, total_uncertainty, epistemic, aleatoric

# ---------------------------
# 7. Training Loop
# ---------------------------
def train_and_eval_bayesian(train_loader, val_loader, model, esm_model, optimizer,
                            device, alphabet, patience=3, max_epochs=50, kl_weight=1.0):
    best_loss = float('inf')
    patience_counter = 0
    best_model = None
    best_metrics = {}
    num_batches = len(train_loader)

    for epoch in range(max_epochs):
        model.train()
        esm_model.eval()
        epoch_loss = 0
        epoch_nll = 0
        epoch_kl = 0

        for tokens, feats, labels in train_loader:
            tokens, feats, labels = tokens.to(device), feats.to(device), labels.to(device)
            with torch.no_grad():
                out = esm_model(tokens, repr_layers=[33])
                reps = out["representations"][33]
                mask = (tokens != alphabet.padding_idx)
                esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            mean, log_var = model(esm_vec, feats)
            kl_div = model.kl_divergence()
            loss, nll, kl_term = bayesian_loss(mean, log_var, labels, kl_div, num_batches, kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kl += kl_term.item()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        all_epistemic, all_aleatoric, all_total = [], [], []
        for tokens, feats, labels in val_loader:
            tokens, feats, labels = tokens.to(device), feats.to(device), labels.to(device)
            pred_mean, total_unc, epistemic, aleatoric = bayesian_predict_with_uncertainty(
                model, esm_model, tokens, feats, alphabet, n_samples=10
            )
            loss = F.mse_loss(pred_mean, labels)
            val_loss += loss.item()
            all_preds.extend(pred_mean.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_epistemic.extend(epistemic.cpu().numpy())
            all_aleatoric.extend(aleatoric.cpu().numpy())
            all_total.extend(total_unc.cpu().numpy())

        val_loss /= len(val_loader)
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
        val_r2 = r2_score(all_labels, all_preds)
        avg_epistemic = np.mean(all_epistemic)
        avg_aleatoric = np.mean(all_aleatoric)
        avg_total = np.mean(all_total)

        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {epoch_loss/num_batches:.4f}, NLL: {epoch_nll/num_batches:.4f}, KL: {epoch_kl/num_batches:.4f}")
        print(f"  Val - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
        print(f"  Uncertainty - Epistemic: {avg_epistemic:.4f}, Aleatoric: {avg_aleatoric:.4f}, Total: {avg_total:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
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
                print("Early stopping.")
                break

    model.load_state_dict(best_model)
    return model, best_metrics

# ---------------------------
# 8. Main
# ---------------------------
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("/home/f087s426/Research/Nanobody_Thermo_Prediction/processed_protein_sequences.csv")  # Replace with your data path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    sequences = df.Sequence.tolist()
    labels = df.Melting_Temperature.tolist()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    feats = np.array([extract_physchem_features(seq) for seq in sequences])
    scaler = StandardScaler().fit(feats)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"\n{'='*50}\nFOLD {fold+1}\n{'='*50}")
        train_dataset = FusionProteinDataset([sequences[i] for i in train_idx], [labels[i] for i in train_idx], alphabet, scaler)
        val_dataset = FusionProteinDataset([sequences[i] for i in val_idx], [labels[i] for i in val_idx], alphabet, scaler)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=val_dataset.collate_fn)

        model = BayesianESMFusionModel(prior_std=1.0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        trained_model, best_metrics = train_and_eval_bayesian(train_loader, val_loader, model, esm_model, optimizer, device, alphabet)

        torch.save(trained_model.state_dict(), f"bayesian_model_fold{fold+1}.pt")
        with open(f"bayesian_metrics_fold{fold+1}.txt", "w") as f:
            for k, v in best_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        print(f"Saved Bayesian model and metrics for fold {fold+1}")

    print("\n" + "="*50)
    print("BAYESIAN TRAINING COMPLETED")
    print("="*50)
