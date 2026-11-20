"""
TerraSpectra: Day 4 - Notebook Generator
Generates notebooks/04_deep_learning_spatial_spectral.ipynb with complete documentation,
mathematical formulations, PyTorch model definitions, convergence curves, and thematic prediction maps.
"""

import os
import nbformat as nbf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# =============================================================================
# Cell 1: Title & Overview
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🌿 TerraSpectra: Intelligent Crop Disease Forecasting
## Day 4 — Spatial-Spectral Deep Learning Architectures (1D-CNN vs 2D-CNN vs 3D-CNN & HybridSN)

**Module:** ML Engineering & Hyperspectral Remote Sensing Pipeline  
**Author:** Rohit (TerraSpectra ML Team)  
**Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 4 — Spatial-Spectral Deep Learning Architectures, Receptive Field Analysis & Overcoming the Generalization Cliff  
**Date:** November 21, 2026  

---

### 1. Research Objectives for Day 4
1. **Architectural Paradigm Benchmarking:** Implement and rigorously compare four distinct convolutional deep learning paradigms for hyperspectral remote sensing:
   - **Spectral 1D-CNN:** Pixel-wise baseline learning sequential 1D spectral absorption dynamics without spatial awareness ($1 \\times 1$ spatial receptive field).
   - **Spatial 2D-CNN:** Standard planar spatial architecture ingesting dimension-reduced spatial patches ($k \\times k \\times 30$).
   - **Spatial-Spectral 3D-CNN:** Volumetric convolutional network simultaneously preserving inter-band continuum and intra-band spatial texture.
   - **HybridSN (3D-2D CNN):** Hybrid architecture (Roy et al., IEEE GRSL 2019) combining 3D convolutions for low-level joint representation with 2D convolutions for higher-level spatial feature abstraction.
2. **Empirical Audit of the Deep Learning "Generalization Cliff":** Evaluate all four neural architectures across both:
   - **Regime A (Naive Random Split 80/20):** Characterized by high spatial autocorrelation and overlapping patch leakage.
   - **Regime B (Spatial Disjoint Parcel Split):** Independent agricultural parcels isolated via connected components (zero spatial patch leakage).
3. **Full-Scene Thematic Spatial Inference:** Reconstruct complete $145 \\times 145$ pixel thematic prediction rasters for Indian Pines, visually demonstrating spatial parcel cohesion, field row continuity, and pixel-level noise suppression across architectures.
4. **Spatial Receptive Field Ablation:** Analyze sensitivity to spatial patch context window ($k \\in \\{5, 9, 13\\}$), measuring the trade-off between parcel contextual smoothing and boundary edge blur.
5. **Computational Profiling:** Quantify parameter efficiency, FLOPs complexity, and per-sample inference latency for real-time agricultural remote sensing."""))

# =============================================================================
# Cell 2: Imports & Environment Configuration
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Environment & Library Setup
import os
import sys
import json
import time
import tempfile
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# Ensure clean multiprocessing and non-interactive matplotlib caching
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# Plotting style configuration
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

print("Environment initialized. PyTorch Version:", torch.__version__, "| CUDA Available:", torch.cuda.is_available())"""))

# =============================================================================
# Cell 3: Mathematical Formulations
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 2. Mathematical Formulation of Hyperspectral Convolutions

Hyperspectral data cubes are naturally 3D tensors $\\mathcal{X} \\in \\mathbb{R}^{H \\times W \\times B}$, where $(H, W)$ denote spatial coordinates and $B$ denotes continuous spectral channels.

#### 2.1. 1D Spectral Convolutions
In 1D-CNN, the spatial coordinates are collapsed to a single pixel vector $\\mathbf{x} \\in \\mathbb{R}^B$. The activation value $v_{i, j}^l$ at position $j$ in the $i$-th feature map of layer $l$ is:
$$v_{i, j}^l = f\\left(\\sum_{m} \\sum_{p=0}^{P_l-1} w_{i, m, p}^l v_{m, j+p}^{l-1} + b_i^l\\right)$$
Where $P_l$ is the spectral kernel width. Pure 1D convolutions isolate absorption slope and inflection dynamics, but possess zero spatial context ($1 \\times 1$ spatial receptive field).

#### 2.2. 2D Spatial Convolutions
In 2D-CNN, spatial patches $\\mathbf{P} \\in \\mathbb{R}^{k \\times k \\times B}$ are convolved spatially after PCA projection:
$$v_{i, (x, y)}^l = f\\left(\\sum_{m} \\sum_{p=-P/2}^{P/2} \\sum_{q=-Q/2}^{Q/2} w_{i, m, p, q}^l v_{m, (x+p, y+q)}^{l-1} + b_i^l\\right)$$
While capturing spatial parcel homogeneity, standard 2D convolutions collapse the spectral dimension through channel summation, losing subtle inter-band continuum.

#### 2.3. 3D Volumetric Convolutions
In 3D-CNN, 3D kernel filters $w \\in \\mathbb{R}^{k_s \\times k_s \\times k_\\lambda}$ simultaneously sweep across space and spectral depth:
$$v_{i, (x, y, z)}^l = f\\left(\\sum_{m} \\sum_{p} \\sum_{q} \\sum_{r} w_{i, m, p, q, r}^l v_{m, (x+p, y+q, z+r)}^{l-1} + b_i^l\\right)$$
Simultaneously preserves joint spatial-spectral dynamics.

#### 2.4. HybridSN (3D-2D Hierarchical Coupling)
HybridSN cascades 3D convolutions with a 2D spatial convolution:
1. Stacked 3D Conv ($7\\times 3\\times 3$, $5\\times 3\\times 3$, $3\\times 3\\times 3$) preserves low-level inter-band spectral continuity.
2. The remaining spectral depth dimension is reshaped into planar channels: $\\mathbf{F} \\in \\mathbb{R}^{B \\times (C \\cdot D) \\times H \\times W}$.
3. Standard 2D Conv ($3\\times 3$) learns higher-level spatial texture without 3D parameter explosion."""))

# =============================================================================
# Cell 4: Data Ingestion & PCA Dimensionality Reduction
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 1. Load Calibrated Indian Pines Dataset & Ground Truth
mat_cube = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat"))
mat_gt = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat"))

cube = mat_cube["indian_pines_corrected"]
gt = mat_gt["indian_pines_gt"]
H, W, B = cube.shape

# Calibrated reflectance normalization
normalized_cube = cube.astype(np.float32) / 10000.0

# 2. PCA Dimensionality Reduction to 30 Components
flat_pixels = normalized_cube.reshape(-1, B)
pca = PCA(n_components=30, random_state=42)
flat_pca = pca.fit_transform(flat_pixels)
cube_pca = flat_pca.reshape(H, W, 30)

explained_var = float(np.sum(pca.explained_variance_ratio_))
print(f"Cube Shape: {H}x{W}x{B} | Labeled Pixels: {np.sum(gt > 0):,}")
print(f"PCA-30 Explained Variance: {explained_var * 100.0:.3f}%")"""))

# =============================================================================
# Cell 5: Patch Extraction & Dataset Splitting
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 3. Patch Extraction Engine & Parcel Partitioning
patch_size = 9
m = patch_size // 2
padded_pca = np.pad(cube_pca, ((m, m), (m, m), (0, 0)), mode="reflect")

labeled_mask = (gt > 0)
coords = np.argwhere(labeled_mask)
labels = gt[labeled_mask] - 1  # 0-indexed [0..15]
n_samples = len(labels)

# 1D Spectral Vectors (full 200 calibrated channels)
X_1d = normalized_cube[labeled_mask]

# 3D Spatial-Spectral Patches (9x9x30)
X_patches = np.zeros((n_samples, patch_size, patch_size, 30), dtype=np.float32)
for i, (r, c) in enumerate(coords):
    X_patches[i] = padded_pca[r:r + patch_size, c:c + patch_size, :]

# Split 1: Naive Random Split (80/20)
idx_rnd_tr, idx_rnd_te = train_test_split(np.arange(n_samples), test_size=0.20, random_state=42, stratify=labels)

# Split 2: Spatial Disjoint Parcel Split (Connected Components)
parcel_labels, num_parcels = ndi.label(labeled_mask)
parcel_records = []
for p_id in range(1, num_parcels + 1):
    p_mask = (parcel_labels == p_id)
    p_gt_vals = gt[p_mask] - 1
    dom_cls = int(np.bincount(p_gt_vals).argmax())
    parcel_records.append((p_id, dom_cls, int(np.sum(p_mask))))

np.random.seed(42)
test_parcels = []
for cls_id in np.unique(labels):
    cls_p = [p[0] for p in parcel_records if p[1] == cls_id]
    if len(cls_p) > 1:
        n_test = max(1, int(len(cls_p) * 0.20))
        test_parcels.extend(list(np.random.choice(cls_p, n_test, replace=False)))

test_parcel_set = set(test_parcels)
parcel_test_mask = np.zeros((H, W), dtype=bool)
for p_id in test_parcel_set:
    parcel_test_mask[parcel_labels == p_id] = True

idx_prc_te = np.array([i for i, (r, c) in enumerate(coords) if parcel_test_mask[r, c]])
idx_prc_tr = np.array([i for i, (r, c) in enumerate(coords) if not parcel_test_mask[r, c]])

print(f"Total Labeled Parcels: {num_parcels} | Disjoint Test Parcels: {len(test_parcel_set)}")
print(f"Regime A (Random Split):   {len(idx_rnd_tr)} train / {len(idx_rnd_te)} test")
print(f"Regime B (Parcel Split):   {len(idx_prc_tr)} train / {len(idx_prc_te)} test")"""))

# =============================================================================
# Cell 6: Model Definitions
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 4. Neural Network Architectures Suite
class Spectral1DCNN(nn.Module):
    def __init__(self, in_bands=200, num_classes=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.features(x)
        return self.classifier(feat.view(feat.size(0), -1))

class Spatial2DCNN(nn.Module):
    def __init__(self, in_channels=30, num_classes=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 4 and x.shape[3] == 30 and x.shape[1] != 30:
            x = x.permute(0, 3, 1, 2)
        feat = self.features(x)
        return self.classifier(feat.view(feat.size(0), -1))

class SpatialSpectral3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        pooled = self.pool(f3)
        return self.classifier(pooled.view(pooled.size(0), -1))

class HybridSN(nn.Module):
    def __init__(self, in_channels=1, num_classes=16):
        super().__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * 18, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        f1 = self.conv3d_1(x)
        f2 = self.conv3d_2(f1)
        f3 = self.conv3d_3(f2)
        b, c, d, h, w = f3.shape
        f4 = self.conv2d(f3.view(b, c * d, h, w))
        return self.classifier(f4.view(b, -1))

# Parameter counts
print("Model Parameter Summary:")
for name, m in [("1D-CNN", Spectral1DCNN(200, 16)),
                ("2D-CNN", Spatial2DCNN(30, 16)),
                ("3D-CNN", SpatialSpectral3DCNN(1, 16)),
                ("HybridSN", HybridSN(1, 16))]:
    params = sum(p.numel() for p in m.parameters())
    print(f"  -> {name:<10}: {params:>8,} parameters")"""))

# =============================================================================
# Cell 7: Benchmark Results & Generalization Cliff Table
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 5. Load Precomputed Benchmark Results & Generalization Cliff
bench_file = os.path.join(PROC_DATA_DIR, "day04_dl_benchmark_results.json")
with open(bench_file, "r") as f:
    bench_data = json.load(f)

# Display comparative summary table
table_rows = []
for name, cliff in bench_data["generalization_cliff"].items():
    table_rows.append({
        "Architecture": name,
        "Random OA (%)": f"{cliff['random_split_oa'] * 100.0:.2f}%",
        "Parcel OA (%)": f"{cliff['parcel_split_oa'] * 100.0:.2f}%",
        "Absolute Cliff Drop": f"-{cliff['absolute_accuracy_drop'] * 100.0:.2f}%",
        "Relative Drop (%)": f"-{cliff['relative_accuracy_drop_pct']:.1f}%",
        "Random Kappa (κ)": f"{cliff['random_split_kappa']:.4f}",
        "Parcel Kappa (κ)": f"{cliff['parcel_split_kappa']:.4f}",
        "Parameters": f"{cliff['parameter_count']:,}"
    })

summary_df = pd.DataFrame(table_rows)
summary_df"""))

# =============================================================================
# Cell 8: Figure 1 - Training Convergence Curves
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 6. Display Figure 1: Deep Learning Training Dynamics
fig1_path = os.path.join(FIG_DIR, "day04_dl_training_curves.png")
if os.path.exists(fig1_path):
    img = plt.imread(fig1_path)
    plt.figure(figsize=(16, 12), dpi=150)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
else:
    print("Figure 1 not found. Run scripts/day04_deep_learning_spatial_spectral.py first.")"""))

# =============================================================================
# Cell 9: Figure 2 - The Generalization Cliff
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 7. Display Figure 2: The Deep Learning Generalization Cliff
fig2_path = os.path.join(FIG_DIR, "day04_deep_learning_benchmark_cliff.png")
if os.path.exists(fig2_path):
    img = plt.imread(fig2_path)
    plt.figure(figsize=(16, 7), dpi=150)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
else:
    print("Figure 2 not found.")"""))

# =============================================================================
# Cell 10: Figure 3 - Thematic Spatial Prediction Maps
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 8. Display Figure 3: Full Scene Thematic Spatial Prediction Maps
fig3_path = os.path.join(FIG_DIR, "day04_full_scene_thematic_maps.png")
if os.path.exists(fig3_path):
    img = plt.imread(fig3_path)
    plt.figure(figsize=(18, 12), dpi=150)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
else:
    print("Figure 3 not found.")"""))

# =============================================================================
# Cell 11: Figure 4 - Patch Size Receptive Field Ablation
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 9. Display Figure 4: Spatial Context Window Ablation & Efficiency
fig4_path = os.path.join(FIG_DIR, "day04_patch_size_ablation_and_confusion.png")
if os.path.exists(fig4_path):
    img = plt.imread(fig4_path)
    plt.figure(figsize=(16, 6), dpi=150)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
else:
    print("Figure 4 not found.")"""))

# =============================================================================
# Cell 12: Class-Wise Performance Table
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# 10. Class-by-Class Generalization Profiling on Disjoint Parcels
class_csv = os.path.join(PROC_DATA_DIR, "day04_dl_per_class_performance.csv")
if os.path.exists(class_csv):
    class_df = pd.read_csv(class_csv)
    print("Per-Class Recall & F1 Scores under Disjoint Parcel Evaluation:")
    display(class_df)"""))

# =============================================================================
# Cell 13: Conclusions & Day 5 Roadmap
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 3. Key Findings & Strategic Conclusions (Day 4)

1. **Confirmation of the Deep Learning Generalization Cliff:**
   - On **Regime A (Naive Random Split)**, spatial deep models achieve near-perfect metrics ($>98-99\\% \\text{ OA}$), driven by spatial autocorrelation and overlapping patch leakage.
   - On **Regime B (Spatial Disjoint Parcel Split)**, all models experience a substantial performance drop (the Generalization Cliff), proving that naive random splitting severely distorts model reliability.
2. **Superiority of Hierarchical Spatial-Spectral Modeling (HybridSN):**
   - **1D-CNN (33.87% OA)** lacks spatial context, matching classical pixel-level baselines (SVM 32.89%).
   - **3D-CNN (26.67% OA)** suffers from over-parameterized volumetric kernels that overfit to parcel-specific training characteristics.
   - **2D-CNN (35.44% OA)** captures spatial parcel regularity but collapses the inter-band spectral continuum.
   - **HybridSN (38.99% OA, \\kappa = 0.3032)** outperforms all classical and deep baselines on disjoint parcels, demonstrating the power of cascading 3D spectral feature extraction with 2D spatial texture abstraction.
3. **Full-Scene Thematic Cohesion:**
   - Visual inspection of the $145 \\times 145$ prediction rasters demonstrates that spatial-spectral networks drastically suppress the "salt-and-pepper" classification noise characteristic of pixel-wise classifiers, producing sharp parcel boundaries and homogeneous agricultural fields.

---

### 4. Day 5 Research Roadmap
To systematically close the remaining Generalization Cliff on disjoint agricultural fields:
- **Self-Supervised Masked Autoencoders (MAE) for Hyperspectral Cubes:** Pretraining on uncurated scenes to learn invariant spectral-spatial priors before downstream supervised fine-tuning.
- **Cross-Sensor Domain Adaptation & Transfer Learning:** Evaluating model transferability between AVIRIS (Indian Pines, Salinas) and satellite sensors (EnMAP, PRISMA)."""))

nb.cells = cells

# Save notebook
notebook_path = os.path.join(NOTEBOOK_DIR, "04_deep_learning_spatial_spectral.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated Day 4 Notebook at {notebook_path}")
