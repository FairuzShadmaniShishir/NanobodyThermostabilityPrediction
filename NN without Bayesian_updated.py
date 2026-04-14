import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from esm import pretrained
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


# ---------------------------
# 1. Physicochemical Feature Extraction
# ---------------------------
def extract_enhanced_physchem_features(seq):
    """Extract comprehensive physicochemical features"""
    analysed = ProteinAnalysis(seq)
    helix, turn, sheet = analysed.secondary_structure_fraction()

    basic_feats = [
        analysed.molecular_weight(),
        analysed.isoelectric_point(),
        analysed.aromaticity(),
        analysed.instability_index(),
        analysed.gravy(),
        np.mean(analysed.flexibility()) if analysed.flexibility() else 0.0,
        helix, turn, sheet
    ]

    aa_counts    = analysed.amino_acids_percent
    hydrophobic  = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    polar        = ['S', 'T', 'N', 'Q']
    charged      = ['R', 'K', 'D', 'E']
    aromatic     = ['F', 'Y', 'W']

    enhanced_feats = [
        sum(aa_counts.get(aa, 0) for aa in hydrophobic),
        sum(aa_counts.get(aa, 0) for aa in polar),
        sum(aa_counts.get(aa, 0) for aa in charged),
        sum(aa_counts.get(aa, 0) for aa in aromatic),
        len(seq),
        seq.count('C') / len(seq) if len(seq) > 0 else 0.0,
        seq.count('P') / len(seq) if len(seq) > 0 else 0.0,
        seq.count('G') / len(seq) if len(seq) > 0 else 0.0,
    ]

    return basic_feats + enhanced_feats  # 17 features total


# ---------------------------
# 2. Pre-compute ESM Embeddings (run ONCE before training)
# ---------------------------
def precompute_esm_embeddings(sequences, esm_model, alphabet, device, batch_size=8):
    """
    Compute mean-pooled ESM-2 representations for all sequences.
    Returns a CPU tensor of shape [N, 1280].
    """
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    all_embeddings = []

    print(f"Pre-computing ESM embeddings for {len(sequences)} sequences...")
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out    = esm_model(tokens, repr_layers=[33])
            reps   = out["representations"][33]
            mask   = (tokens != alphabet.padding_idx)
            esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

        all_embeddings.append(esm_vec.cpu())

        done = min(i + batch_size, len(sequences))
        if done % 50 == 0 or done == len(sequences):
            print(f"  {done}/{len(sequences)} sequences embedded")

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, 1280]
    print(f"Embedding complete. Shape: {embeddings.shape}\n")
    return embeddings


# ---------------------------
# 3. Dataset (uses pre-computed embeddings — no ESM in loop)
# ---------------------------
class CachedProteinDataset(Dataset):
    def __init__(self, esm_embeddings, physchem_feats, labels, augment=False):
        """
        Args:
            esm_embeddings : torch.Tensor [N, 1280]
            physchem_feats : np.ndarray  [N, 17]  (already scaled)
            labels         : list of floats, length N
            augment        : whether to add small noise to physchem features
        """
        self.embeddings = esm_embeddings
        self.feats      = physchem_feats
        self.labels     = labels
        self.augment    = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        esm_vec = self.embeddings[idx]                         # [1280]
        feat    = self.feats[idx].copy()                       # [17]
        label   = self.labels[idx]

        if self.augment and np.random.random() < 0.1:
            feat = feat + np.random.normal(0, 0.01, feat.shape)

        return (
            esm_vec,
            torch.tensor(feat,    dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
        )


# ---------------------------
# 4. Model Architecture
# ---------------------------
class ESMFusionModel(nn.Module):
    def __init__(self, esm_dim=1280, physchem_dim=17, fusion_hidden=512):
        super().__init__()

        # --- ESM feature processor ---
        self.esm_processor = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

        # --- Physicochemical branch (standard Linear, was Bayesian) ---
        self.physchem_branch = nn.Sequential(
            nn.LayerNorm(physchem_dim),
            nn.Linear(physchem_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
        )

        # --- Multi-head self-attention on ESM features ---
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )

        # --- Fusion MLP (standard Linear, was Bayesian) ---
        # Input: 256 (ESM attended) + 128 (physchem) = 384
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(256 + 128),
            nn.Linear(256 + 128, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),   # single output: predicted melting temperature
        )

    def forward(self, esm_vec, physchem_vec):
        # Process ESM
        esm_processed = self.esm_processor(esm_vec)           # [B, 256]

        # Self-attention over ESM features
        esm_attended, _ = self.attention(
            esm_processed.unsqueeze(1),
            esm_processed.unsqueeze(1),
            esm_processed.unsqueeze(1),
        )
        esm_attended = esm_attended.squeeze(1)                 # [B, 256]

        # Process physicochemical features
        physchem_embed = self.physchem_branch(physchem_vec)    # [B, 128]

        # Concatenate and predict
        fused = torch.cat((esm_attended, physchem_embed), dim=1)  # [B, 384]
        out   = self.fusion_mlp(fused)                            # [B, 1]
        return out


# ---------------------------
# 5. Parameter Printing Utility
# ---------------------------
def print_model_parameters(model):
    print("\n" + "=" * 70)
    print(f"{'MODULE':<50} {'PARAMETERS':>12}  {'TRAINABLE':>10}")
    print("=" * 70)

    total_params     = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:          # leaf modules only
            num_params    = sum(p.numel() for p in module.parameters())
            num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if num_params > 0:
                print(f"  {name:<48} {num_params:>12,}  {str(num_trainable > 0):>10}")
            total_params     += num_params
            trainable_params += num_trainable

    # Per-block summary
    blocks = {
        "esm_processor":   model.esm_processor,
        "physchem_branch": model.physchem_branch,
        "attention":       model.attention,
        "fusion_mlp":      model.fusion_mlp,
    }
    print(f"\n{'BLOCK SUMMARY':<50} {'PARAMETERS':>12}  {'TRAINABLE':>10}")
    print("-" * 70)
    for block_name, block in blocks.items():
        bp = sum(p.numel() for p in block.parameters())
        bt = sum(p.numel() for p in block.parameters() if p.requires_grad)
        print(f"  {block_name:<48} {bp:>12,}  {str(bt > 0):>10}")

    print("=" * 70)
    print(f"  {'TOTAL PARAMETERS':<48} {total_params:>12,}")
    print(f"  {'TRAINABLE PARAMETERS':<48} {trainable_params:>12,}")
    print(f"  {'NON-TRAINABLE PARAMETERS':<48} {total_params - trainable_params:>12,}")
    print("=" * 70 + "\n")


# ---------------------------
# 6. Training & Evaluation Loop
# ---------------------------
def train_and_eval(train_loader, val_loader, model, optimizer,
                   scheduler, device, patience=8, max_epochs=80):
    best_r2          = -float('inf')
    patience_counter = 0
    best_model_state = None
    best_metrics     = {}
    num_batches      = len(train_loader)

    for epoch in range(max_epochs):

        # ---- Training ----
        model.train()
        epoch_loss = 0.0

        for esm_vec, feats, labels in train_loader:
            esm_vec = esm_vec.to(device)
            feats   = feats.to(device)
            labels  = labels.to(device)

            pred = model(esm_vec, feats)
            loss = F.mse_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for esm_vec, feats, labels in val_loader:
                esm_vec = esm_vec.to(device)
                feats   = feats.to(device)
                labels  = labels.to(device)

                pred      = model(esm_vec, feats)
                val_loss += F.mse_loss(pred, labels).item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_mae   = mean_absolute_error(all_labels, all_preds)
        val_rmse  = math.sqrt(mean_squared_error(all_labels, all_preds))
        val_r2    = r2_score(all_labels, all_preds)

        if scheduler:
            scheduler.step(val_r2)

        if epoch % 5 == 0 or epoch < 10:
            print(
                f"Epoch {epoch+1:>3}/{max_epochs} | "
                f"Train Loss: {epoch_loss/num_batches:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE: {val_mae:.4f} | "
                f"RMSE: {val_rmse:.4f} | "
                f"R²: {val_r2:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # ---- Early stopping on R² ----
        if val_r2 > best_r2:
            best_r2          = val_r2
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_metrics     = {
                "val_loss": val_loss,
                "mae":      val_mae,
                "rmse":     val_rmse,
                "r2":       val_r2,
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model, best_metrics


# ---------------------------
# 7. Main
# ---------------------------
if __name__ == "__main__":

    # ---- Config ----
    CSV_PATH    = "/home/f087s426/Research/Nanobody_Thermo_Prediction/processed_protein_sequences.csv"
    OUTPUT_DIR  = "Standard_Model_630_Sequences"
    BATCH_SIZE  = 16       # larger batch → better GPU utilisation
    N_SPLITS    = 2
    MAX_EPOCHS  = 80
    PATIENCE    = 8
    ESM_BATCH   = 8        # batch size for ESM embedding pre-computation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load data ----
    df        = pd.read_csv(CSV_PATH)
    sequences = df.Sequence.tolist()
    labels    = df.Melting_Temperature.tolist()
    print(f"Dataset size: {len(sequences)} sequences")

    # ---- Load ESM model ----
    print("Loading ESM-2 (650M)...")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)

    # ---- Pre-compute ALL embeddings ONCE ----
    all_embeddings = precompute_esm_embeddings(
        sequences, esm_model, alphabet, device, batch_size=ESM_BATCH
    )

    # Free ESM from GPU — no longer needed
    esm_model.cpu()
    torch.cuda.empty_cache()
    print("ESM model moved to CPU. GPU memory freed.\n")

    # ---- Physicochemical features + scaling ----
    print("Extracting physicochemical features...")
    all_feats = np.array([extract_enhanced_physchem_features(seq) for seq in sequences])
    scaler    = RobustScaler()
    all_feats_scaled = scaler.fit_transform(all_feats)
    print(f"Feature matrix shape: {all_feats_scaled.shape}\n")

    # ---- Print model parameters (once, before folds) ----
    sample_model = ESMFusionModel(esm_dim=1280, physchem_dim=17, fusion_hidden=512)
    print_model_parameters(sample_model)
    del sample_model

    # ---- Cross-validation ----
    kf           = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"\n{'='*60}\nFOLD {fold+1}/{N_SPLITS}\n{'='*60}")

        train_dataset = CachedProteinDataset(
            all_embeddings[train_idx],
            all_feats_scaled[train_idx],
            [labels[i] for i in train_idx],
            augment=True,
        )
        val_dataset = CachedProteinDataset(
            all_embeddings[val_idx],
            all_feats_scaled[val_idx],
            [labels[i] for i in val_idx],
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # ---- Model ----
        model = ESMFusionModel(esm_dim=1280, physchem_dim=17, fusion_hidden=512).to(device)

        # ---- Optimizer (separate LRs per block) ----
        optimizer = optim.AdamW([
            {'params': model.esm_processor.parameters(),   'lr': 1e-4},
            {'params': model.physchem_branch.parameters(), 'lr': 1e-4},
            {'params': model.attention.parameters(),        'lr': 5e-5},
            {'params': model.fusion_mlp.parameters(),       'lr': 1e-4},
        ], weight_decay=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        # ---- Train ----
        trained_model, best_metrics = train_and_eval(
            train_loader, val_loader, model, optimizer, scheduler,
            device, patience=PATIENCE, max_epochs=MAX_EPOCHS
        )

        # ---- Save ----
        torch.save(
            trained_model.state_dict(),
            os.path.join(OUTPUT_DIR, f"model_fold{fold+1}.pt")
        )
        with open(os.path.join(OUTPUT_DIR, f"metrics_fold{fold+1}.txt"), "w") as f:
            for k, v in best_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        fold_results.append(best_metrics)
        print(f"\nFold {fold+1} best metrics:")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")

    # ---- Summary ----
    print(f"\n{'='*60}\nSUMMARY ACROSS ALL FOLDS\n{'='*60}")
    for metric in ['mae', 'rmse', 'r2']:
        values = [f[metric] for f in fold_results]
        print(f"  {metric.upper():<8} {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")