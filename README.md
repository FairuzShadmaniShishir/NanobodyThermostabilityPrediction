# Nanobody Thermostability Prediction

This repository contains code and data for predicting the thermal stability (melting temperature, **T<sub>m</sub>**) of nanobodies using a **Bayesian neural network (BNN)** that integrates:
- **Protein language model embeddings** (e.g., [ESM-2](https://github.com/facebookresearch/esm, AbLang)  
- **Physicochemical features** (e.g., hydrophobicity, charge, cysteine frequency)  

The framework achieves accurate T<sub>m</sub> predictions while providing **uncertainty quantification**, making it a practical tool for guiding experimental prioritization in nanobody engineering.

---

## 🚀 Key Features
- **Multimodal fusion** of PLM embeddings and physicochemical descriptors  
- **Bayesian neural network** for robust prediction and well-calibrated uncertainty  
- **Epistemic vs. aleatoric uncertainty decomposition**  
- **Performance**: MAE = 1.89 °C, R² = 0.67 (5-fold cross-validation)  
- Outperforms unimodal and deterministic baselines  

---


---

## ⚙️ Installation

```bash
git clone https://github.com/FairuzShadmaniShishir
cd NanobodyThermostabilityPrediction

pip install -r requirements.txt


🧪 Reproducing Results
Train the model
python BayesianNN_version2.py


'''''
To test the model
run python Nb_bence_20smapel_evaluation.py # test_data_path will be the testing your data

@article{nanobody_bnn,
  title={NbBayesLM: Bayesian Prediction of Nanobody
Thermostability Using Protein Language Model},
  author={Fairuz Shadmani Shishir, Rokunuzjahan Rudro, Bishnu Sarker, Cuncong Zhong, and Sumaiya Shomaji},
  journal={...},
  year={2026}
}
