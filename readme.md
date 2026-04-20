<div align="center">

# 🪐 TESS Exoplanet Detection
** Multi-Modal Fusion Strategies for Exoplanet Transit Detection**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/W&B-Experiment_Tracking-FFBE00.svg?style=flat-square&logo=weightsandbiases)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> An end-to-end deep learning pipeline for identifying exoplanet transits in NASA TESS photometry, featuring a comprehensive ablation study on injecting physical stellar metadata into Transformer sequences.

</div>

---

## 🚀 Overview

Distinguishing true planetary transits from stellar-mass companions (e.g., grazing Eclipsing Binaries) is a fundamental challenge in astrophysics. Traditional models treat physical metadata as an afterthought. This repository presents a **Dual-View CNN-Transformer** that solves this by evaluating four distinct methods of physical data fusion. 

**Key Contributions:**
* **Dual-View Processing:** Simultaneously processes a global phase-folded view (baseline noise) and a local zoomed view (transit morphology).
* **The "Meta Token" Strategy:** Proves that projecting physical stellar metadata (Temperature, Radius, Gravity) as a native sequence token inside a Transformer is vastly superior to traditional Late Fusion (MLP).
* **Imbalance Handling:** Addresses severe class imbalance dynamically in-memory using a Weighted Random Sampler and Focal Loss, eliminating the need for static oversampling.
* **Explainability:** Provides mathematical proof of physical reasoning via attention weight extraction, showing exactly *when* the model relies on astrophysical parameters to make decisions.

---

## 🧠 Architecture

```text
Global view (2048 pts) ──► CNN Encoder ──► [Global Token] \
Local view  (201 pts)  ──► CNN Encoder ──► [Local Token]  ──► Cross-Attention Transformer ──► p(planet)
Stellar Physics        ──► Linear Proj ──► [Meta Token]   /
(Temp, Radius, Grav)
Unlike standard classifiers that concatenate data at the bottleneck, the Meta Token architecture allows the Local View to directly query the host star's physics via Self-Attention before classification.📂 Repository StructurePlaintexttess-exoplanet-hybrid/
├── configs/
│   └── config.yaml             # Hyperparameters, paths, and W&B config
├── src/
│   ├── data_pipeline.py        # AWS S3 cloud-accelerated ingestion & phase-folding
│   ├── model.py                # Dual-View CNN-Transformer with toggleable fusion
│   ├── train.py                # Main training loop with Focal Loss & SMOTE balancing
│   ├── evaluate.py             # Calculates metrics and optimal PR thresholds
│   ├── plot_roc.py             # Generates comparative ROC curves for Ablation Study
│   ├── visualise_attention.py  # Extracts 3x3 Transformer attention heatmaps
│   ├── error_analysis.py       # Plots False Positive/Negative boundary cases
│   └── utils.py                # Helpers for metrics and GPU allocation
├── tests/
│   └── test_model.py           # Unit tests for tensor dimension validation
├── requirements.txt            
└── README.md
⚡ Quickstart1. Clone and InstallBashgit clone [https://github.com/GoodAsur2023/tess-exoplanet-hybrid.git](https://github.com/GoodAsur2023/tess-exoplanet-hybrid.git)
cd tess-exoplanet-hybrid
pip install -r requirements.txt
2. Authenticate Tracking (Optional but Recommended)Bashwandb login
3. Run the PipelineBash# Verify architecture tensor dimensions
python tests/test_model.py

# Build the dataset from NASA MAST
python src/data_pipeline.py --config configs/config.yaml

# Train the Champion Model (Meta Token)
python src/train.py --config configs/config.yaml --fusion_type meta_token

# Evaluate against the Test Set
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/meta_token/best_model.pt
📊 Ablation Study & ResultsWe systematically tested four methods of physical metadata fusion to identify the most effective architecture. Evaluated on an unseen 15% test split.Fusion MethodTest AUCKey FindingMeta Token (Champion)0.9832Native sequence attention on physics is the superior fusion method.FiLM Modulation0.9602Early feature-wise scaling improves convergence over the baseline.Late Fusion (MLP)0.9542Standard concatenation is effective but lacks early physical context.The Science Way0.6558Manual physical scaling ($R^2$) destroys normalization and hinders feature extraction.Final Champion Metrics (Threshold = 0.6015)Recall: 97.39% (Successfully identifies nearly all actual exoplanets)Precision: 88.69% (High reliability; low false-alarm rate)F1-Score: 0.9283👁️ Interpretability & Error AnalysisAttention Heatmaps: By extracting weights from the final Transformer layer, we confirmed that the model exhibits physics-dependent gating. When querying the background noise (Macro View), the model ignores the star's physics (3.8% attention). However, when querying the transit dip (Local View), the model allocates 14.7% of its attention to the Stellar Meta Token to validate the physical probability of the planet.Error Analysis:Qualitative analysis of boundary cases shows:False Negatives: Primarily SNR-limited (missed planets are visually indistinguishable from background photon noise).False Positives: High-confidence errors are predominantly grazing Eclipsing Binaries, representing the fundamental photometric limits of the TESS instrument.<div align="center"><i>Research conducted at the Department of Computer Science and Engineering, PES University.Data provided by the NASA Mikulski Archive for Space Telescopes (MAST).</i></div>
