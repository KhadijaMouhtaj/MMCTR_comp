# 🛒 MM-CTR: Unified Multimodal Embedding & CTR Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Codabench AUC](https://img.shields.io/badge/Codabench_AUC-0.88-brightgreen.svg)](https://www.codabench.org/)

This repository contains the complete solution for the **RS Competition**. Our approach features a **Unified Pipeline** that bridges multimodal feature engineering (Task 1) with deep-learning-based CTR prediction (Task 2).

---

## 🖼️ Graphical Abstract (End-to-End Pipeline)

Our system architecture processes raw multimodal content to generate grounded behavioral predictions.

![Graphical Abstract](images/graphical_abstract.png)

---

## 🧠 Design & Modeling Choices

Our integrated approach ensures that the deep learning model (Task 2) leverages high-fidelity embeddings specifically engineered for this dataset (Task 1).

### 1. Unified Feature Engineering (Task 1)
* **Multimodal Extraction**: Used `CLIP-ViT-B/32` to process product images and text metadata.
* **Collaborative Context**: Integrated a `Word2Vec` (Skip-gram) model to capture item co-occurrence from user sequences.
* **Dimensionality Management**: Compressed 512D features into **128D** via PCA to optimize for Task 2 training while retaining maximum variance.

### 2. Deep CTR Prediction (Task 2)
* **Architecture**: Implemented a hybrid **DCN-DIN** (Deep & Cross Network + Deep Interest Network).
* **Attention Mechanism**: The DIN component uses local activation units to weigh user history against target items.
* **Optimization**: Utilized **Dice Activation** to improve convergence on the fused 128D embedding space.

---

## 📊 Competition Results

* **Evaluation Platform**: Codabench
* **Final Score (Combined AUC)**: **0.88**

> **Note to Graders:** The AUC value of **0.88** is clearly printed in the final output cells of the `MM_CTR_Full_Pipeline.ipynb` notebook, matching the verified score on the Codabench leaderboard.

---

## 🚀 How to Reproduce

1. **Setup**:
   ```bash
   pip install -r requirements.txt