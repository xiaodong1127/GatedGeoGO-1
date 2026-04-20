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

## ⚙️ Requirements

### 🔧 Environment

- Python >= 3.8
- PyTorch >= 1.10
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn

### 📦 Installation

We provide a `requirements.txt` file for convenient environment setup.

#### Option 1 (Recommended)

```bash
pip install -r requirements.txt
```

#### Option 2 (Manual Installation)

```bash
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas scikit-learn
```


📊 Data Preparation [点击下载数据集](https://pan.baidu.com/s/1valwROkws_IyUOlPYxIMkQ?pwd=p2kh)
After downloading, place all files under the `data/CAFA3/` directory.
### 📁 Required Dataset Files

```bash
data/CAFA3/
├── train.tsv
├── validation.tsv
├── test.tsv
├── train_seqs.fasta
├── validation_seqs.fasta
├── test_seqs.fasta
├── ppi_seqs.fasta
├── ppi_score.tsv
├── terms-50.tsv
├── go.obo
├── afdb_dir/              # AlphaFold predicted structures
├── esm_dir/               # Extracted ESM embeddings
└── PredGODataset/         # Cached processed graph data
```

### 📄 File Description

| File | Description |
|------|-------------|
| train.tsv | Training set annotations |
| validation.tsv | Validation set annotations |
| test.tsv | Test set annotations |
| *_seqs.fasta | Protein sequences in FASTA format |
| ppi_seqs.fasta | PPI neighbor protein sequences |
| ppi_score.tsv | STRING interaction scores |
| terms-50.tsv | Selected GO term labels |
| go.obo | Gene Ontology hierarchy file |
| afdb_dir/ | AlphaFold structure files |
| esm_dir/ | Precomputed ESM sequence embeddings |
| PredGODataset/ | Intermediate processed graph data |

### 🧾 TSV Format

```bash
sequences    protein_id    annotation_all    annotation_mf    annotation_bp    annotation_cc    orgs
```

| Column | Description |
|--------|-------------|
| sequences | Protein amino acid sequence |
| protein_id | Unique protein identifier |
| annotation_* | GO labels |
| orgs | Organism |


## 🚀 Training and Testing

After completing the environment setup and data preparation, run the following command to start training and evaluation:

```bash
python train_PredGOModel_cafa3.py
```

Before model training begins, the pipeline will automatically perform:

- ESM-based protein sequence feature extraction
- Structural information preprocessing and graph construction
- Dataset loading and feature preparation

The script will then automatically execute:

- Model training
- Validation during training
- Final testing on the benchmark dataset
- Prediction result generation
