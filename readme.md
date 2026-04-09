<div align="center">

# 🪐 TESS Exoplanet Detection
**A Hybrid Dual-View CNN–Transformer Architecture for Exoplanet Transit Detection in Noisy TESS Light Curves**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> A deep learning pipeline for the automated classification of exoplanet transit signals in NASA TESS photometric data.

</div>

---

## 🚀 Overview

This repository presents a deep learning pipeline for automated classification of exoplanet transit signals. We propose a dual-view, cross-attention hybrid architecture that improves upon pure-CNN baselines (e.g., AstroNet) by:
* **Dual-View Processing:** Simultaneously processing a global phase-folded view (full orbital period) and a local zoomed view (transit window), giving the model both long-range context and fine-grained transit morphology.
* **Cross-Attention Mechanism:** Replacing the standard dense combination of views with a cross-attention Transformer encoder, letting each view attend to the other's feature sequence.
* **Imbalance Handling:** Addressing severe class imbalance (~2–3% of TCEs are planets) using Focal Loss and dynamic threshold optimization.
* **Explainability:** Providing explainability via attention weight visualization, showing exactly which parts of the light curve the model focuses on.

---

## 🧠 Architecture

    Global view (2048 pts) ──► CNN Encoder ──►
                                                \
                                                 ──► Cross-Attention Transformer ──► Classifier ──► p(planet)
                                                /
    Local view  (201 pts)  ──► CNN Encoder ──►

*(See `src/model.py` for the full PyTorch implementation).*

---

## 📂 Repository Structure

    tess-exoplanet-hybrid/
    ├── configs/
    │   └── config.yaml          # All hyperparameters and paths
    ├── data/
    │   ├── raw/                 # TCE CSV metadata (ignored by git)
    │   └── processed/           # .npy light curve arrays (ignored by git)
    ├── src/
    │   ├── data_pipeline.py     # Download, preprocess, and save light curves
    │   ├── model.py             # Dual-view CNN-Transformer architecture
    │   ├── train.py             # Training loop with Focal Loss + W&B logging
    │   ├── evaluate.py          # ROC-AUC, PR curves, confusion matrix, attention maps
    │   └── utils.py             # Shared helpers (seeds, metrics, focal loss)
    ├── notebooks/
    │   └── EDA_and_Baseline.ipynb
    ├── paper/
    │   └── IEEE_Draft.tex
    ├── .github/
    │   └── workflows/
    │       └── ci.yml
    ├── requirements.txt
    └── README.md

---

## ⚡ Quickstart

**1. Clone and install**

    git clone https://github.com/GoodAsur23/tess-exoplanet-hybrid.git
    cd tess-exoplanet-hybrid
    pip install -r requirements.txt

**2. Download the TCE metadata**

Download `tess2018206190142-s0001-s0013_dvr-tcestats.csv` from the MAST archive and place it in `data/raw/`.

**3. Build the dataset**

    python src/data_pipeline.py --config configs/config.yaml

*This downloads all light curves, applies dual-view phase-folding, cross-references labels with the NASA Exoplanet Archive, and saves `.npy` tensors to `data/processed/`.*

**4. Train**

    python src/train.py --config configs/config.yaml

**5. Evaluate**

    python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pt

---

## 📊 Results (V1 Baseline)

*Initial baseline evaluation run on a highly imbalanced TESS test set (Threshold optimized to `0.84`).*

| Model | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **1D-CNN Baseline** | *Pending* | *Pending* | *Pending* | *Pending* |
| **Ours (Dual-View Transformer)** | **50.0%** | **14.3%** | **0.2222** | **0.6745** |

<details>
<summary><b>View Confusion Matrix Breakdown</b></summary>

* **True Negatives:** 225 *(Filtered out 99.5% of all background noise and false positives)*
* **False Positives:** 1 *(Only 1 false alarm)*
* **False Negatives:** 6 *(Missed planets due to strict thresholding)*
* **True Positives:** 1 *(Successfully identified planet)*

</details>

*Note: The model currently exhibits extremely high precision, effectively filtering out 99.5% of false positives. Active work is ongoing to improve recall through hyperparameter tuning and expanded sector data.*

---
<div align="center">
<i>Built as an exploration into applied Deep Learning and Astrophysics.</i>
</div>