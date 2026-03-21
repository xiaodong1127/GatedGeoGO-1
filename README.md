# GatedGeoGO: Multi-Modal Geometry-Aware Network with Gated Fusion and GO Semantic Attention for Protein Function Prediction

# Overview

**GatedGeoGO** is a multimodal deep learning framework for protein function prediction, integrating:

- 🧬 **Sequence information** (via ESM pretrained model)
- 🧱 **Structural information** 
- 🧠 **GO semantic embeddings**
- 🔗 **Graph-based learning**

The model is designed to improve **prediction accuracy**, **robustness**, and **generalization ability** for Gene Ontology (GO) annotation tasks.

---

# Key Features

- ✅ Multimodal fusion (Sequence + Structure + GO)
- ✅ Pretrained protein language model (ESM)
- ✅ Geometric Vector Perceptron for 3D structure modeling
- ✅ Support for **case study analysis**
- ✅ Flexible architecture for backbone replacement
- ✅ Designed for CAFA-style protein function prediction

---
```bash
## 📂 Project Structure
├── esm/ # ESM sequence encoder
│ ├── axial_attention.py
│ ├── constants.py
│ ├── data.py
│ ├── extract.py # Sequence feature extraction
│ ├── model.py
│ ├── modules.py
│ ├── multhead_attention.py
│ ├── pretrained.py
│ └── version.py
│
├── gvp
│ ├── go_embedding_utils.py
│
├── predgo
│ ├── data.py # Dataset processing
│ ├── model.py # Model definition
│ └── modules.py # Model components
│
├── tools/ # Utility scripts
│
├── train_PredGOModel_cafa3.py # Training entry
└── README.md
