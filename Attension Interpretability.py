"""
model_interpretability.py
=========================
Interpretability additions for EnhancedBayesianESMFusionModel.

Two complementary analyses:
  1. Attention-weight analysis  – extracts per-head weights from the
     MultiheadAttention layer that fuses ESM embeddings, then summarises
     which ESM embedding dimensions drive each attention head.
  2. SHAP feature attribution  – uses KernelExplainer (model-agnostic) on
     the physicochemical branch to rank the 17 hand-crafted features by
     their signed contribution to the predicted melting temperature.

Both helpers are self-contained and can be dropped straight into the
existing test_bayesian_model() pipeline.

Requires: shap>=0.42, matplotlib, seaborn  (all standard in the training env)
"""

import math
import warnings
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

warnings.filterwarnings("ignore")

# ── Canonical feature names (must match extract_enhanced_physchem_features) ──
PHYSCHEM_FEATURE_NAMES = [
    "Molecular Weight",
    "Isoelectric Point",
    "Aromaticity",
    "Instability Index",
    "GRAVY",
    "Mean Flexibility",
    "Helix Fraction",
    "Turn Fraction",
    "Sheet Fraction",
    "Hydrophobic AA Frac.",
    "Polar AA Frac.",
    "Charged AA Frac.",
    "Aromatic AA Frac.",
    "Sequence Length",
    "Cys Frac.",
    "Pro Frac.",
    "Gly Frac.",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ATTENTION-WEIGHT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class AttentionRecorder(nn.Module):
    """
    Thin wrapper around nn.MultiheadAttention that caches the attention
    weight tensor produced during the last forward pass.

    Swap the model's self.attention with an AttentionRecorder(model.attention)
    to enable recording without touching the trained weights.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.last_attn_weights: Optional[torch.Tensor] = None  # [B, H, T, T]

    def forward(self, query, key, value, **kwargs):
        # need_weights=True and average_attn_weights=False → per-head weights
        out, attn_weights = self.mha(
            query, key, value,
            need_weights=True,
            average_attn_weights=False,  # → [B, num_heads, T, T]
            **{k: v for k, v in kwargs.items()
               if k not in ("need_weights", "average_attn_weights")}
        )
        self.last_attn_weights = attn_weights.detach().cpu()
        return out, attn_weights


def attach_attention_recorder(model) -> AttentionRecorder:
    """
    Replace model.attention in-place with an AttentionRecorder and return
    the recorder so the caller can read .last_attn_weights after each forward.
    """
    recorder = AttentionRecorder(model.attention)
    model.attention = recorder
    return recorder


@torch.no_grad()
def extract_attention_weights(
    model,
    esm_model,
    tokens: torch.Tensor,
    feats: torch.Tensor,
    alphabet,
    n_samples: int = 30,
) -> Dict:
    """
    Run n_samples stochastic forward passes and return:
      • mean_attn  [B, num_heads]  – per-sample, per-head mean self-attention
                                      score (averaged over T×T positions)
      • head_importance [num_heads] – variance across heads (which heads
                                      disagree most → most informative split)
    """
    recorder = attach_attention_recorder(model)

    # ESM representations (deterministic)
    esm_model.eval()
    out = esm_model(tokens, repr_layers=[33])
    reps = out["representations"][33]
    mask = (tokens != alphabet.padding_idx)
    esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    model.train()  # keep Bayesian dropout/sampling active
    head_weights_samples = []

    for _ in range(n_samples):
        esm_proc = model.esm_processor(esm_vec)
        _ = model.attention(
            esm_proc.unsqueeze(1),
            esm_proc.unsqueeze(1),
            esm_proc.unsqueeze(1),
        )
        # recorder.last_attn_weights: [B, num_heads, 1, 1]  (T=1 after squeeze)
        w = recorder.last_attn_weights.squeeze(-1).squeeze(-1)  # [B, num_heads]
        head_weights_samples.append(w)

    head_weights = torch.stack(head_weights_samples)        # [S, B, H]
    mean_attn    = head_weights.mean(dim=0)                  # [B, H]
    head_std     = head_weights.std(dim=0).mean(dim=0)       # [H]  – variability

    return {
        "mean_attn_per_sample_per_head": mean_attn.numpy(),  # [B, H]
        "head_variability":              head_std.numpy(),   # [H]
        "n_heads":                       mean_attn.shape[1],
        "n_samples":                     n_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHAP FEATURE ATTRIBUTION (physicochemical branch)
# ─────────────────────────────────────────────────────────────────────────────

class PhyschemPredictor:
    """
    Callable wrapper that feeds ONLY the physicochemical features through
    the model and returns the predicted mean, holding the ESM embedding fixed.

    This lets KernelExplainer attribute the output change to the 17
    physicochemical features while the ESM contribution is treated as a
    constant baseline.
    """

    def __init__(self, model, fixed_esm_vec: torch.Tensor, device):
        self.model     = model
        self.esm_vec   = fixed_esm_vec.to(device)  # [B, esm_dim]
        self.device    = device
        self.model.eval()

    def __call__(self, physchem_np: np.ndarray) -> np.ndarray:
        """physchem_np: [n, 17]  →  predicted mean [n]"""
        feats = torch.tensor(physchem_np, dtype=torch.float32).to(self.device)

        # Broadcast ESM vec to match batch size if needed
        b = feats.shape[0]
        # esm = self.esm_vec[:1].expand(b, -1) if self.esm_vec.shape[0] == 1 \
        #       else self.esm_vec[:b]
        esm = self.esm_vec[0:1].expand(b, -1).contiguous()

        self.model.train()   # stochastic; averaged over a single sample here
        with torch.no_grad():
            mean, _ = self.model(esm, feats)
        return mean.squeeze(-1).cpu().numpy()


def compute_shap_values(
    model,
    esm_model,
    tokens: torch.Tensor,
    feats: torch.Tensor,
    alphabet,
    device,
    background_size: int = 50,
    n_explain: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compute SHAP values for physicochemical features.

    Args:
        background_size: Number of background samples for KernelExplainer.
        n_explain:       Number of test samples to explain.

    Returns dict with:
        shap_values  [n_explain, 17]
        base_value   scalar
        feature_names list[str]
        feats_explained [n_explain, 17]
    """
    np.random.seed(seed)

    # 1. Get fixed ESM embedding (deterministic)
    esm_model.eval()
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[33])
        reps = out["representations"][33]
        mask = (tokens != alphabet.padding_idx)
        esm_vec = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    # 2. Build callable predictor
    predictor = PhyschemPredictor(model, esm_vec, device)

    # 3. Build background dataset
    feats_np = feats.cpu().numpy()                       # [B, 17]
    bg_idx   = np.random.choice(
        len(feats_np),
        size=min(background_size, len(feats_np)),
        replace=False
    )
    background = feats_np[bg_idx]                        # [bg, 17]

    # 4. KernelExplainer
    explainer  = shap.KernelExplainer(predictor, background)
    n_explain  = min(n_explain, len(feats_np))
    feats_exp  = feats_np[:n_explain]

    shap_vals  = explainer.shap_values(feats_exp, nsamples=100, silent=True)
    # shap_vals: [n_explain, 17]

    return {
        "shap_values":       shap_vals,
        "base_value":        explainer.expected_value,
        "feature_names":     PHYSCHEM_FEATURE_NAMES,
        "feats_explained":   feats_exp,
        "n_explained":       n_explain,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

# def plot_interpretability(
#     attn_results:  Dict,
#     shap_results:  Dict,
#     sample_labels: Optional[List[str]] = None,
#     save_path:     str = "interpretability_panel.png",
#     predicted_means: Optional[np.ndarray] = None,
# ):
#     """
#     Produce a two-panel figure:
#       Left  – Attention head activity heatmap (samples × heads)
#       Right – SHAP beeswarm / bar plot for physicochemical features
#     """
#     # ── Aesthetics ──────────────────────────────────────────────────────────
#     BG       = "#0d1117"
#     PANEL    = "#161b22"
#     ACCENT1  = "#58a6ff"   # cool blue
#     ACCENT2  = "#f78166"   # warm coral
#     TEXT     = "#e6edf3"
#     MUTED    = "#8b949e"
#     GRID     = "#21262d"
#
#     plt.rcParams.update({
#         "figure.facecolor":  BG,
#         "axes.facecolor":    PANEL,
#         "axes.edgecolor":    GRID,
#         "axes.labelcolor":   TEXT,
#         "xtick.color":       MUTED,
#         "ytick.color":       MUTED,
#         "text.color":        TEXT,
#         "grid.color":        GRID,
#         "font.family":       "monospace",
#     })
#
#     fig = plt.figure(figsize=(18, 9), facecolor=BG)
#     gs  = gridspec.GridSpec(
#         1, 2, figure=fig,
#         left=0.07, right=0.97, top=0.88, bottom=0.12,
#         wspace=0.38
#     )
#
#     # ── Panel A: Attention heatmap ───────────────────────────────────────────
#     ax_attn = fig.add_subplot(gs[0])
#
#     mean_attn = attn_results["mean_attn_per_sample_per_head"]  # [B, H]
#     n_heads   = attn_results["n_heads"]
#     B         = mean_attn.shape[0]
#
#     if sample_labels is None:
#         sample_labels = [f"Sample {i+1}" for i in range(B)]
#
#     # Normalise per-row so colours reflect *relative* head preference
#     row_max  = mean_attn.max(axis=1, keepdims=True) + 1e-9
#     norm_attn = mean_attn / row_max
#
#     sns.heatmap(
#         norm_attn,
#         ax=ax_attn,
#         cmap=sns.diverging_palette(220, 20, as_cmap=True),
#         vmin=0, vmax=1,
#         linewidths=0.4,
#         linecolor=BG,
#         xticklabels=[f"H{i+1}" for i in range(n_heads)],
#         yticklabels=[lab[:22] for lab in sample_labels],
#         cbar_kws={"shrink": 0.7, "label": "Rel. attention (norm.)"},
#         annot=True, fmt=".2f", annot_kws={"size": 7, "color": TEXT},
#     )
#     ax_attn.set_title(
#         "A  ·  ESM Self-Attention Head Activity",
#         color=TEXT, fontsize=13, fontweight="bold", pad=12, loc="left"
#     )
#     ax_attn.set_xlabel("Attention Head", fontsize=10)
#     ax_attn.set_ylabel("Input Sample", fontsize=10)
#
#     # Annotate predicted means on the right y-axis if supplied
#     if predicted_means is not None:
#         ax_r = ax_attn.twinx()
#         ax_r.set_ylim(ax_attn.get_ylim())
#         ax_r.set_yticks(np.arange(B) + 0.5)
#         ax_r.set_yticklabels(
#             [f"{v:.1f} °C" for v in predicted_means[:B]],
#             color=ACCENT1, fontsize=8
#         )
#         ax_r.set_ylabel("Predicted Tₘ", color=ACCENT1, fontsize=9)
#         ax_r.tick_params(axis="y", colors=ACCENT1)
#         ax_r.spines["right"].set_edgecolor(ACCENT1)
#
#     # Head variability bar below heatmap
#     ax_var = ax_attn.inset_axes([0, -0.18, 1, 0.12])
#     head_var = attn_results["head_variability"]
#     bars = ax_var.bar(
#         range(n_heads), head_var,
#         color=[ACCENT1 if v > head_var.mean() else MUTED for v in head_var],
#         edgecolor=BG, linewidth=0.5
#     )
#     ax_var.set_xlim(-0.5, n_heads - 0.5)
#     ax_var.set_xticks([])
#     ax_var.set_ylabel("σ (variability)", color=MUTED, fontsize=7)
#     ax_var.set_facecolor(PANEL)
#     for spine in ax_var.spines.values():
#         spine.set_edgecolor(GRID)
#     ax_var.tick_params(colors=MUTED, labelsize=7)
#     ax_var.set_title("Head variability across MC samples",
#                      color=MUTED, fontsize=7, pad=3, loc="left")
#
#     # ── Panel B: SHAP bar chart ──────────────────────────────────────────────
#     ax_shap = fig.add_subplot(gs[1])
#
#     shap_vals    = shap_results["shap_values"]        # [n, 17]
#     feat_names   = shap_results["feature_names"]
#     mean_abs_shap = np.abs(shap_vals).mean(axis=0)    # [17]
#     mean_shap     = shap_vals.mean(axis=0)            # signed mean
#
#     # Sort by importance
#     order     = np.argsort(mean_abs_shap)
#     names_ord = [feat_names[i] for i in order]
#     abs_ord   = mean_abs_shap[order]
#     sign_ord  = mean_shap[order]
#
#     colors = [ACCENT1 if s >= 0 else ACCENT2 for s in sign_ord]
#
#     ax_shap.barh(
#         range(len(order)), abs_ord,
#         color=colors, edgecolor=BG, linewidth=0.5, height=0.7
#     )
#     ax_shap.set_yticks(range(len(order)))
#     ax_shap.set_yticklabels(names_ord, fontsize=9)
#     ax_shap.set_xlabel("Mean |SHAP value|  (°C)", fontsize=10)
#     ax_shap.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
#     ax_shap.set_title(
#         "B  ·  Physicochemical Feature Importance (SHAP)",
#         color=TEXT, fontsize=13, fontweight="bold", pad=12, loc="left"
#     )
#     ax_shap.grid(axis="x", color=GRID, linewidth=0.5)
#
#     # Legend for sign
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor=ACCENT1, label="Positive contribution (↑ Tₘ)"),
#         Patch(facecolor=ACCENT2, label="Negative contribution (↓ Tₘ)"),
#     ]
#     ax_shap.legend(
#         handles=legend_elements, loc="lower right",
#         fontsize=8, facecolor=PANEL, edgecolor=GRID,
#         labelcolor=TEXT
#     )
#
#     # Annotate top-3 with signed value
#     top3 = np.argsort(abs_ord)[-3:]
#     for rank in top3:
#         val = sign_ord[rank]
#         ax_shap.text(
#             abs_ord[rank] + abs_ord.max() * 0.01, rank,
#             f"{val:+.3f}", va="center", fontsize=8,
#             color=ACCENT1 if val >= 0 else ACCENT2
#         )
#
#     # ── Global title ─────────────────────────────────────────────────────────
#     fig.suptitle(
#         "Model Interpretability  ·  Bayesian ESM-Fusion  ·  Nanobody Tₘ Prediction",
#         color=TEXT, fontsize=14, fontweight="bold", y=0.97
#     )
#
#     plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG)
#     plt.close()
#     print(f"[interpretability] Figure saved → {save_path}")
def plot_shap_summary(shap_results: Dict, save_path: str = "shap_summary.png"):
    shap_vals  = shap_results["shap_values"]       # [n, 17]
    feat_names = shap_results["feature_names"]
    feats_exp  = shap_results["feats_explained"]

    # Bar plot – mean absolute SHAP
    shap.summary_plot(
        shap_vals, feats_exp,
        feature_names=feat_names,
        plot_type="bar",
        max_display=min(20, len(feat_names)),
        show=False,
    )
    plt.title("Physicochemical Feature Importance (SHAP)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path.replace('.png', '_bar.png')}")

    # Beeswarm – shows direction and spread
    shap.summary_plot(
        shap_vals, feats_exp,
        feature_names=feat_names,
        plot_type="dot",
        max_display=min(20, len(feat_names)),
        show=False,
    )
    plt.title("SHAP Feature Impact on Predicted Tₘ", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path.replace('.png', '_beeswarm.png')}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONVENIENCE WRAPPER  (drop-in call inside test_bayesian_model)
# ─────────────────────────────────────────────────────────────────────────────

def run_interpretability_analysis(
    model,
    esm_model,
    tokens: torch.Tensor,
    feats: torch.Tensor,
    alphabet,
    device,
    sample_labels:   Optional[List[str]] = None,
    predicted_means: Optional[np.ndarray] = None,
    n_attn_samples:  int  = 50,
    n_shap_explain:  int  = 5,
    background_size: int  = 50,
    save_path:       str  = "interpretability_panel.png",
) -> Dict:
    """
    End-to-end interpretability pipeline.

    Call this once per batch (or for a representative subset) after your
    existing bayesian_predict_with_uncertainty() call:

    Example
    -------
    interp = run_interpretability_analysis(
     model, esm_model, tokens, feats, alphabet, device, sample_labels=sequences[:5],predicted_means=results['mean'].squeeze().cpu().numpy(),)
    """
    print("[interpretability] Extracting attention weights …")
    attn_results = extract_attention_weights(
        model, esm_model, tokens, feats, alphabet,
        n_samples=n_attn_samples,
    )

    print("[interpretability] Computing SHAP values (this may take ~30 s) …")
    shap_results = compute_shap_values(
        model, esm_model, tokens, feats, alphabet, device,
        background_size=background_size,
        n_explain=n_shap_explain,
    )

    print("[interpretability] Rendering figure …")
    # plot_interpretability(
    #     attn_results, shap_results,
    #     sample_labels=sample_labels,
    #     save_path=save_path,
    #     predicted_means=predicted_means,
    # )
    plot_shap_summary(shap_results, save_path=os.path.join(PROJECT_DIR, "shap_summary.png"))

    # Print concise textual summary
    _print_summary(attn_results, shap_results)

    return {"attention": attn_results, "shap": shap_results}


def _print_summary(attn_results: Dict, shap_results: Dict):
    feat_names   = shap_results["feature_names"]
    shap_vals    = shap_results["shap_values"]
    mean_abs     = np.abs(shap_vals).mean(axis=0)
    top5_idx     = np.argsort(mean_abs)[::-1][:5]

    print("\n── Physicochemical Feature Importance (SHAP) ─────────────────")
    for rank, idx in enumerate(top5_idx, 1):
        signed = shap_vals[:, idx].mean()
        print(f"  #{rank:1d}  {feat_names[idx]:<28s}  "
              f"|SHAP|={mean_abs[idx]:.4f}  "
              f"mean_signed={signed:+.4f}")

    hv   = attn_results["head_variability"]
    tops = np.argsort(hv)[::-1][:3]
    print("\n── Most Variable Attention Heads (ESM branch) ────────────────")
    for h in tops:
        print(f"  Head {h+1:2d}  σ={hv[h]:.4f}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INTEGRATION PATCH for test_bayesian_model()
# ─────────────────────────────────────────────────────────────────────────────
#
#  Add the following block inside the per-batch loop in test_bayesian_model(),
#  right after you have `batch_results` and before `all_results.append(...)`:
#
#  ─── paste start ────────────────────────────────────────────────────────────
#
#  if batch_idx == 0:          # run once on the first batch as a representative
#      from model_interpretability import run_interpretability_analysis
#      run_interpretability_analysis(
#          model          = model,          # the loaded model for this fold
#          esm_model      = esm_model,
#          tokens         = tokens,
#          feats          = feats,
#          alphabet       = alphabet,
#          device         = device,
#          sample_labels  = sequences[: tokens.size(0)],
#          predicted_means= batch_results['mean'].squeeze().cpu().numpy(),
#          n_attn_samples = 50,
#          n_shap_explain = min(5, tokens.size(0)),
#          background_size= 50,
#          save_path      = output_path.replace('.csv', '_interpretability.png'),
#      )
#
#  ─── paste end ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import joblib
    from esm import pretrained
    from sklearn.preprocessing import RobustScaler

    # ── Config ────────────────────────────────────────────────────────────
    PROJECT_DIR = "/home/f087s426/PycharmProjects/Nanobody Thermostability Prediction"
    MODEL_PATH  = os.path.join(PROJECT_DIR, "enhanced_bayesian_model_fold1.pt")
    DATA_PATH   = os.path.join(PROJECT_DIR, "NB_Bench_bayesian_20testset.csv")
    SCALER_PATH = os.path.join(PROJECT_DIR, "scaler.pkl")
    SAVE_PATH   = os.path.join(PROJECT_DIR, "interpretability_panel.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load ESM ──────────────────────────────────────────────────────────
    print("Loading ESM model...")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)

    # ── Load test data ────────────────────────────────────────────────────
    import pandas as pd
    from torch.utils.data import DataLoader

    #import these from your main script
    from Nb_bence_20smapel_evaluation import (
        EnhancedBayesianESMFusionModel,
        TestProteinDataset,
        extract_enhanced_physchem_features,
    )

    df        = pd.read_csv(DATA_PATH)
    sequences = df["Sequence"].tolist()
    labels    = df["Melting_Temperature"].tolist() if "Melting_Temperature" in df.columns else None

    # ── Scaler ────────────────────────────────────────────────────────────
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        all_feats = np.array([extract_enhanced_physchem_features(s) for s in sequences])
        scaler    = RobustScaler().fit(all_feats)
        joblib.dump(scaler, SCALER_PATH)

    # ── DataLoader (just first batch) ─────────────────────────────────────
    dataset    = TestProteinDataset(sequences, labels, alphabet, scaler)
    loader     = DataLoader(dataset, batch_size=8, shuffle=False,
                            collate_fn=dataset.collate_fn)
    tokens, feats, _ = next(iter(loader))
    tokens, feats    = tokens.to(device), feats.to(device)

    # ── Load model ────────────────────────────────────────────────────────
    model = EnhancedBayesianESMFusionModel(
        esm_dim=1280, physchem_dim=17, fusion_hidden=512, prior_std=0.5
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded.")

    # ── Run interpretability ──────────────────────────────────────────────
    run_interpretability_analysis(
        model           = model,
        esm_model       = esm_model,
        tokens          = tokens,
        feats           = feats,
        alphabet        = alphabet,
        device          = device,
        sample_labels   = sequences[:tokens.size(0)],
        n_attn_samples  = 50,
        n_shap_explain  = min(5, tokens.size(0)),
        background_size = min(50, tokens.size(0)),
        save_path       = SAVE_PATH,
    )
    print(f"\nDone. Figure at: {SAVE_PATH}")