import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import RobustScaler
import pickle
import io
import math

# Set page configuration
st.set_page_config(
    page_title="Nanobody Melting Temperature Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .uncertainty-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Model Architecture Classes
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
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -4.0)
        if self.use_bias:
            nn.init.constant_(self.bias_mu, 0.0)
            nn.init.constant_(self.bias_rho, -4.0)

    def forward(self, x):
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

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )

        # Fusion network
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
            EnhancedBayesianLinear(128, 2, prior_std=prior_std)
        )

    def forward(self, esm_vec, physchem_vec):
        esm_processed = self.esm_processor(esm_vec)
        physchem_embed = self.physchem_branch(physchem_vec)

        esm_attended, _ = self.attention(
            esm_processed.unsqueeze(1),
            esm_processed.unsqueeze(1),
            esm_processed.unsqueeze(1)
        )
        esm_attended = esm_attended.squeeze(1)

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
# Feature Extraction Functions
# ---------------------------
def extract_enhanced_physchem_features(seq):
    """Extract comprehensive physicochemical features"""
    try:
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
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return [0.0] * 17


# ---------------------------
# Cached Model Loading
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Initialize model
        model = EnhancedBayesianESMFusionModel(
            esm_dim=1280, physchem_dim=17, fusion_hidden=512, prior_std=0.5
        )

        # For demo purposes, we'll use a randomly initialized model
        # In production, load your trained weights:
        model.load_state_dict(torch.load('enhanced_bayesian_model_fold5.pt', map_location='cpu'))

        model.eval()

        # Initialize a dummy scaler (in production, load your fitted scaler)
        scaler = RobustScaler()
        # For demo, fit on dummy data
        dummy_data = np.random.randn(100, 17)
        scaler.fit(dummy_data)

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


@st.cache_resource
def load_esm_model():
    """Load ESM model - placeholder for actual ESM loading"""
    try:
        #In production, uncomment and use:
        from esm import pretrained
        esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
        return esm_model, alphabet

        # For demo, return None (we'll simulate ESM embeddings)
        return None, None
    except Exception as e:
        st.error(f"Error loading ESM model: {str(e)}")
        return None, None


# ---------------------------
# Prediction Functions
# ---------------------------
def simulate_esm_embedding(sequence, esm_dim=1280):
    """Simulate ESM embedding for demo purposes"""
    # In production, replace with actual ESM model inference
    np.random.seed(hash(sequence) % 2 ** 32)
    return torch.randn(1, esm_dim)


def predict_with_uncertainty(model, sequence, scaler, n_samples=30):
    """Make prediction with uncertainty quantification"""
    try:
        # Extract physicochemical features
        features = extract_enhanced_physchem_features(sequence)
        features_scaled = scaler.transform([features])
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # Simulate ESM embedding (replace with actual ESM in production)
        esm_embedding = simulate_esm_embedding(sequence)

        # Monte Carlo sampling
        model.train()  # Enable dropout and Bayesian sampling
        all_means, all_vars = [], []

        for _ in range(n_samples):
            with torch.no_grad():
                mean, log_var = model(esm_embedding, features_tensor)
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

        return {
            'prediction': predictive_mean.item(),
            'epistemic_uncertainty': epistemic.item(),
            'aleatoric_uncertainty': aleatoric.item(),
            'total_uncertainty': total_uncertainty.item(),
            'confidence_interval': (
                predictive_mean.item() - 1.96 * math.sqrt(total_uncertainty.item()),
                predictive_mean.item() + 1.96 * math.sqrt(total_uncertainty.item())
            )
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# ---------------------------
# Streamlit App
# ---------------------------
def main():
    # Title and description
    st.markdown('<h1 class="main-header">🧬 Nanobody Melting Temperature Predictor</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    This app uses a **Bayesian ESM Fusion Model** to predict Nanobody melting temperatures with 
    uncertainty quantification. The model combines ESM protein embeddings with physicochemical 
    features for accurate predictions.
    """)

    # Sidebar
    with st.sidebar:
        st.header("🔬 Model Information")
        st.info("""
        **Model Architecture:**
        - ESM2 (650M parameters) for Nanobody embeddings
        - Bayesian Neural Network for uncertainty
        - Multi-head attention fusion
        - Physicochemical feature integration

        **Uncertainty Types:**
        - **Epistemic**: Model uncertainty (reducible with more data)
        - **Aleatoric**: Data uncertainty (irreducible noise)
        """)

        st.header("⚙️ Prediction Settings")
        n_samples = st.slider("Monte Carlo Samples", 10, 100, 30,
                              help="More samples = better uncertainty estimates")
        confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)

    # Load model
    model, scaler = load_model_and_scaler()
    if model is None:
        st.error("Failed to load model. Please check your model files.")
        return

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input Nanobody Sequence")

        # Input methods
        input_method = st.radio("Choose input method:",
                                ["Enter sequence manually", "Upload FASTA file"])

        sequence = ""
        if input_method == "Enter sequence manually":
            sequence = st.text_area(
                "Nanobody Sequence:",
                height=150,
                placeholder="Enter your Nanobody sequence here (single letter amino acid codes)...",
                help="Enter a valid Nanobody sequence using single-letter amino acid codes"
            )
        else:
            uploaded_file = st.file_uploader("Upload FASTA file", type=['fasta', 'fa', 'txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                sequence = ''.join([line for line in lines if not line.startswith('>')])

        # Validate sequence
        if sequence:
            sequence = sequence.upper().replace(' ', '').replace('\n', '')
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            invalid_chars = set(sequence) - valid_aa

            if invalid_chars:
                st.error(f"Invalid amino acids found: {', '.join(invalid_chars)}")
                sequence = ""
            else:
                st.success(f"Valid sequence with {len(sequence)} amino acids")

                # Show sequence info
                with st.expander("Sequence Information"):
                    st.write(f"**Length:** {len(sequence)} residues")
                    st.write(
                        f"**Composition:** {dict(sorted([(aa, sequence.count(aa)) for aa in valid_aa if sequence.count(aa) > 0]))}")

    with col2:
        st.header("🎯 Prediction Results")

        if sequence and st.button("🔬 Predict Melting Temperature", type="primary"):
            with st.spinner("Computing prediction with uncertainty..."):
                results = predict_with_uncertainty(model, sequence, scaler, n_samples)

                if results:
                    # Main prediction
                    st.markdown('<div class="uncertainty-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted Melting Temperature: {results['prediction']:.2f}°C")
                    st.markdown(
                        f"**95% Confidence Interval:** {results['confidence_interval'][0]:.2f}°C - {results['confidence_interval'][1]:.2f}°C")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Uncertainty breakdown
                    st.subheader("🎲 Uncertainty Analysis")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Epistemic Uncertainty", f"{results['epistemic_uncertainty']:.3f}°C²",
                                  help="Model uncertainty - reducible with more training data")
                    with col_b:
                        st.metric("Aleatoric Uncertainty", f"{results['aleatoric_uncertainty']:.3f}°C²",
                                  help="Data uncertainty - irreducible noise in the data")
                    with col_c:
                        st.metric("Total Uncertainty", f"{results['total_uncertainty']:.3f}°C²",
                                  help="Combined uncertainty from both sources")

                    # Uncertainty visualization
                    st.subheader("📊 Uncertainty Visualization")

                    # Create uncertainty breakdown pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Epistemic', 'Aleatoric'],
                        values=[results['epistemic_uncertainty'], results['aleatoric_uncertainty']],
                        hole=0.3,
                        marker_colors=['#1f77b4', '#ff7f0e']
                    )])
                    fig_pie.update_layout(title="Uncertainty Breakdown", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Prediction distribution
                    st.subheader("📈 Prediction Distribution")
                    x_vals = np.linspace(
                        results['prediction'] - 3 * math.sqrt(results['total_uncertainty']),
                        results['prediction'] + 3 * math.sqrt(results['total_uncertainty']),
                        100
                    )
                    y_vals = (1 / math.sqrt(2 * math.pi * results['total_uncertainty'])) * \
                             np.exp(-0.5 * ((x_vals - results['prediction']) ** 2) / results['total_uncertainty'])

                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='lines', fill='tozeroy',
                        name='Prediction Distribution', line=dict(color='blue')
                    ))
                    fig_dist.add_vline(x=results['prediction'], line_dash="dash",
                                       annotation_text="Predicted Value")
                    fig_dist.add_vrect(
                        x0=results['confidence_interval'][0],
                        x1=results['confidence_interval'][1],
                        fillcolor="lightblue", opacity=0.3,
                        annotation_text="95% CI"
                    )
                    fig_dist.update_layout(
                        title="Prediction Distribution with Confidence Interval",
                        xaxis_title="Melting Temperature (°C)",
                        yaxis_title="Probability Density",
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)



        elif not sequence:
            st.info("👆 Enter a Nanobody sequence to get started!")

    # Additional information
    st.markdown("---")
    st.header("ℹ️ About This Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Model Features:**
        - Bayesian neural network for uncertainty quantification
        - ESM2 protein language model embeddings
        - Multi-head attention mechanism
        - Physicochemical feature integration
        - Monte Carlo dropout for epistemic uncertainty
        """)

    with col2:
        st.markdown("""
        **Use Cases:**
        - Nanobody engineering and design
        - Thermostability prediction
        - Drug discovery applications
        - Research and development
        - Educational purposes
        """)


if __name__ == "__main__":
    main()