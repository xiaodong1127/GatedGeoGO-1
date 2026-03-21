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

```
---

## ⚙️ Requirements

### 🔧 Environment

- Python >= 3.8
- PyTorch >= 1.10
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn

### 📦 Installation

```bash
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas scikit-learn

```
📊 Data Preparation [点击下载数据集](https://pan.baidu.com/s/1valwROkws_IyUOlPYxIMkQ?pwd=p2kh)

Input data should be in .tsv format:

```bash
 sequences    protein_id    annotation_all    annotation_mf    annotation_bp    annotation_cc    orgs
```
🧾 Field Description
Column	Description
sequences	Protein amino acid sequence
protein_id	Unique protein identifier
annotation_*	GO labels
orgs	Organism
