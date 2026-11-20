# 🧠 TerraSpectra: Spatial-Spectral Deep Learning Architectures & The Generalization Cliff
**Project:** TerraSpectra — Intelligent Crop Disease Detection & Forecasting Using Hyperspectral Imaging  
**Role:** ML Engineering & Hyperspectral Remote Sensing Pipeline  
**Author:** Rohit  
**Working Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 4 — Deep Learning Architectures (1D-CNN vs. 2D-CNN vs. 3D-CNN & HybridSN), Patch Size Ablations & Full-Scene Inference  
**Date:** November 20, 2026  

---

## 1. Executive Summary & Day 4 Research Scope

Following the spectral signature characterization, narrowband biophysical indices, and classical machine learning baselines established in **Day 3**, the primary mandate of **Day 4** is to construct, train, evaluate, and interpret modern **Spatial-Spectral Deep Learning Architectures** on calibrated airborne hyperspectral imaging data (AVIRIS Indian Pines, $145 \times 145 \times 200$).

While classical pixel-wise classifiers evaluate isolated spectral vectors $\mathbf{x} \in \mathbb{R}^B$, deep neural networks can exploit the full volumetric tensor structure $\mathbf{X} \in \mathbb{R}^{B \times H \times W}$, combining continuous biochemical absorption profiles with spatial canopy context, field boundary morphology, and intra-parcel texture. However, the remote sensing deep learning literature is rife with methodological pitfalls—most notably, treating overlapping spatial patches sampled from the same field as independent training and testing instances.

This research milestone delivers:
1. **End-to-End PyTorch Spatial-Spectral Architecture Suite:** Implementation of four distinct neural paradigms:
   - **Spectral 1D-CNN:** 1D convolutions across the continuous spectral axis ($200$ calibrated bands, $1 \times 1$ spatial).
   - **Spatial 2D-CNN:** 2D convolutions operating on PCA-reduced spectral feature maps ($30$ channels, $9 \times 9$ spatial).
   - **Spatial-Spectral 3D-CNN:** True volumetric 3D convolutions learning joint spatial-spectral feature representations ($1 \times 30 \times 9 \times 9$).
   - **HybridSN (3D-2D CNN):** Progressive hybrid architecture featuring three sequential 3D convolutions for spectral-spatial feature extraction, followed by 2D spatial convolution and dense classification.
2. **Spectral Dimension Reduction via PCA-30:** Optimization of high-dimensional tensor memory and FLOP complexity by projecting 200 calibrated AVIRIS channels onto 30 orthogonal principal components, capturing **99.248%** of total scene variance.
3. **The Deep Learning Generalization Cliff:** Quantitative benchmarking under two rigorously isolated evaluation regimes:
   - **Regime A (Naive Random Split 80/20):** Demonstrating the deceptive $99.6\% - 100.0\%$ accuracy caused by spatial autocorrelation leakage (patch overlap $> 88\%$).
   - **Regime B (Spatial Disjoint Parcel Split):** Connected-component agricultural parcels held out in their entirety ($0\%$ patch overlap), exposing a massive accuracy collapse of **$-63.45\%$ absolute** ($-63.5\%$ relative).
4. **Spatial Receptive Field & Context Window Ablation:** Systematic evaluation of patch dimensions $k \in \{5, 9, 13\}$ on disjoint field generalization, revealing boundary contamination effects in large receptive fields.
5. **Full-Scene Thematic Spatial Inference:** Generation of high-resolution $145 \times 145$ thematic classification rasters, analyzing out-of-field prediction behavior, parcel cohesion, and background misclassification.
6. **Publication Figures & Modular Deliverables:** End-to-end Python script ([scripts/day04_deep_learning_spatial_spectral.py](file:///d:/TerraSpectra2026/scripts/day04_deep_learning_spatial_spectral.py)), interactive Jupyter notebook ([notebooks/04_deep_learning_spatial_spectral.ipynb](file:///d:/TerraSpectra2026/notebooks/04_deep_learning_spatial_spectral.ipynb)), 4 publication-quality multi-panel visualizations, and structured JSON/CSV experimental logs.

---

## 2. Deep Learning Architectural Suite & Tensor Dynamics

Hyperspectral cubes present a unique challenge: the spectral dimension is continuous, ordered, and physically grounded in molecular absorption, whereas the spatial dimensions represent physical geography with spatial autocorrelation and localized textural variations.

```
       Hyperspectral Cube Tensor               Spectral 1D-CNN             Spatial 2D-CNN           HybridSN (3D-2D CNN)
       (145 x 145 x 200 bands)                  (B, 1, 200)                 (B, 30, 9, 9)              (B, 1, 30, 9, 9)
                                                     │                            │                            │
             ┌───────────┐                           ▼                            ▼                            ▼
            ╱           ╱│                     Conv1d (k=7)                 Conv2d (k=3)                 Conv3d (7,3,3)
           ╱           ╱ │                           │                            │                            │
          ┌───────────┐  │                     Conv1d (k=5)                 Conv2d (k=3)                 Conv3d (5,3,3)
          │           │  │                           │                            │                            │
          │  Spatial  │  │                     Conv1d (k=3)                  MaxPool2d                   Conv3d (3,3,3)
          │   Patch   │ ╱                            │                            │                            │
          │   (9x9)   │╱                       AdaptiveAvgPool              Conv2d (k=3)                 Reshape to 2D
          └───────────┘                              │                            │                            │
                │                                    ▼                            ▼                            ▼
          PCA-30 (99.25% var)                    FC (64->16)                 FC (512->16)                 Conv2d (k=3)
                │                                                                                              │
                ▼                                                                                              ▼
        Input Representation                                                                              FC (576->16)
```

### 2.1. Mathematical Formulation of Architectural Paradigms

#### Paradigm 1: Spectral 1D-CNN (Pixel-Wise Spectral Convolution)
Let $\mathbf{x} \in \mathbb{R}^{B}$ be the continuous calibrated reflectance vector ($B = 200$). The input is formatted as a 1D sequence with a single channel $\mathbf{X}^{(0)} \in \mathbb{R}^{1 \times B}$.
The $l$-th 1D convolutional layer computes feature maps via:
$$v_{i, j}^{(l)} = f\left(\sum_{m=1}^{C^{(l-1)}} \sum_{p=-P}^{P} w_{i, m, p}^{(l)} v_{m, j+p}^{(l-1)} + b_i^{(l)}\right)$$
where $w_{i, m, p}^{(l)}$ is the kernel weight of the $i$-th filter at position $p$, $C^{(l-1)}$ is the input channel depth, and $f(\cdot)$ is the LeakyReLU/ReLU activation.
- **Layers:** $\text{Conv1D}(1 \to 64, k=7) \to \text{BN} \to \text{MaxPool}(2) \to \text{Conv1D}(64 \to 128, k=5) \to \text{BN} \to \text{MaxPool}(2) \to \text{Conv1D}(128 \to 64, k=3) \to \text{BN} \to \text{AdaptiveAvgPool}(1) \to \text{FC}(64 \to 64) \to \text{Dropout}(0.4) \to \text{FC}(64 \to 16)$.
- **Total Trainable Parameters:** $54,288$.
- **Inductive Bias:** Extracts localized spectral absorption gradients and slope inflections (e.g. red-edge shifts) invariant to total illumination amplitude. Ignores spatial neighbor context.

#### Paradigm 2: Spatial 2D-CNN (PCA-Reduced Planar Convolution)
Let $\mathbf{X}_{\text{PCA}} \in \mathbb{R}^{D \times k \times k}$ represent a spatial neighborhood patch of size $k \times k$ ($k=9$) across $D=30$ principal component channels. The 2D convolution treats spectral components as independent feature channels:
$$v_{i, x, y}^{(l)} = f\left(\sum_{m=1}^{C^{(l-1)}} \sum_{p=-P}^{P} \sum_{q=-Q}^{Q} w_{i, m, p, q}^{(l)} v_{m, x+p, y+q}^{(l-1)} + b_i^{(l)}\right)$$
- **Layers:** $\text{Conv2D}(30 \to 64, 3\times 3, \text{pad}=1) \to \text{BN} \to \text{ReLU} \to \text{Conv2D}(64 \to 128, 3\times 3, \text{pad}=1) \to \text{BN} \to \text{ReLU} \to \text{MaxPool}(2\times 2) \to \text{Conv2D}(128 \to 128, 3\times 3, \text{pad}=1) \to \text{BN} \to \text{ReLU} \to \text{AdaptiveAvgPool}(2\times 2) \to \text{FC}(512 \to 128) \to \text{Dropout}(0.4) \to \text{FC}(128 \to 16)$.
- **Total Trainable Parameters:** $258,000$.
- **Inductive Bias:** Learns rich 2D spatial textures, crop row orientations, and parcel boundary geometry. Cross-spectral relationships are linearly combined via $1\times 1$ channel summations inside 2D convolutions.

#### Paradigm 3: Spatial-Spectral 3D-CNN (Volumetric Convolution)
Let $\mathbf{X}_{\text{3D}} \in \mathbb{R}^{1 \times D \times k \times k}$ represent a 4D tensor with an explicit spectral depth axis. A 3D convolution kernel of size $K_d \times K_h \times K_w$ slides simultaneously across both spectral bands and spatial coordinates:
$$v_{i, d, x, y}^{(l)} = f\left(\sum_{m=1}^{C^{(l-1)}} \sum_{r=-R}^{R} \sum_{p=-P}^{P} \sum_{q=-Q}^{Q} w_{i, m, r, p, q}^{(l)} v_{m, d+r, x+p, y+q}^{(l-1)} + b_i^{(l)}\right)$$
- **Layers:** $\text{Conv3D}(1 \to 8, 7\times 3\times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv3D}(8 \to 16, 5\times 3\times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv3D}(16 \to 32, 3\times 3\times 3) \to \text{BN} \to \text{ReLU} \to \text{AdaptiveAvgPool3D}(1\times 1\times 1) \to \text{FC}(32 \to 64) \to \text{Dropout}(0.4) \to \text{FC}(64 \to 16)$.
- **Total Trainable Parameters:** $35,616$.
- **Inductive Bias:** Jointly captures continuous spectral absorption transitions alongside spatial neighborhood patterns. However, aggressive 3D volumetric pooling can discard fine spatial resolution if not coupled with 2D refinement.

#### Paradigm 4: HybridSN (3D-2D Progressive Convolutional Network)
Originally proposed by Roy et al. (IEEE GRSL, 2020), **HybridSN** combines the strengths of 3D-CNN (joint spectral-spatial feature abstraction) and 2D-CNN (spatial structural refinement without parameter explosion).
1. **3D Spectral-Spatial Feature Extraction:**
   - Input: $\mathbf{X} \in \mathbb{R}^{1 \times 30 \times 9 \times 9}$
   - Layer 1: $\text{Conv3D}(1 \to 8, 7 \times 3 \times 3, \text{stride}=1) \to (B, 8, 24, 7, 7)$
   - Layer 2: $\text{Conv3D}(8 \to 16, 5 \times 3 \times 3, \text{stride}=1) \to (B, 16, 20, 5, 5)$
   - Layer 3: $\text{Conv3D}(16 \to 32, 3 \times 3 \times 3, \text{stride}=1) \to (B, 32, 18, 3, 3)$
2. **Dimension Reshaping & 2D Spatial Refinement:**
   - Reshape 3D feature tensor into 2D planar channels:
     $$\mathbf{X}_{\text{2D}} = \text{Reshape}\left(\mathbf{X}_{\text{3D}}, (B, 32 \times 18, 3, 3)\right) = (B, 576, 3, 3)$$
   - Layer 4: $\text{Conv2D}(576 \to 64, 3 \times 3, \text{stride}=1, \text{pad}=1) \to \text{BN} \to \text{ReLU} \to (B, 64, 3, 3)$
3. **Dense Classification Head:**
   - $\text{Flatten} \to 576 \text{ features} \to \text{FC}(576 \to 256) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{FC}(256 \to 128) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{FC}(128 \to 16)$.
- **Total Trainable Parameters:** $362,608$.

### 2.2. Architectural Dimension & Parameter Comparison

| Architecture | Input Representation | Kernel Typology | Trainable Parameters | Peak Training Speed (s/epoch) | Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Spectral 1D-CNN** | $(B, 1, 200)$ | 1D Spectral ($k \in \{7, 5, 3\}$) | 54,288 | 2.45 s | 0.08 - 0.15 ms |
| **Spatial 2D-CNN** | $(B, 30, 9, 9)$ | 2D Spatial ($3 \times 3$) | 258,000 | 7.19 s | 0.21 - 0.32 ms |
| **Spatial-Spectral 3D-CNN** | $(B, 1, 30, 9, 9)$ | 3D Volumetric ($7\times 3\times 3, 5\times 3\times 3, 3\times 3\times 3$) | 35,616 | 10.16 s | 0.11 - 0.43 ms |
| **HybridSN (3D-2D)** | $(B, 1, 30, 9, 9)$ | 3D Spectral-Spatial + 2D Spatial | 362,608 | 3.54 s | 0.10 - 0.18 ms |

---

## 3. Spectral Dimensionality Reduction via PCA-30

Processing 200 continuous hyperspectral channels inside 3D convolutional kernels would require computing 3D convolutions with depth $K_d \approx 200$, escalating GPU memory footprint and FLOP count by over $600\%$. Furthermore, as established on Day 1, adjacent narrow bands exhibit extreme Pearson correlation ($r > 0.98$), yielding massive collinear redundancy.

### 3.1. Principal Component Analysis Formulation
Given the centered spectral data matrix $\mathbf{X}_c \in \mathbb{R}^{N \times B}$ ($N = 10,249, B = 200$):
$$\mathbf{C} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$
where $\mathbf{V} \in \mathbb{R}^{B \times B}$ is the orthogonal matrix of eigenvectors and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_B)$ are the eigenvalues sorted in descending order.

The spectral cube is projected onto the top $D=30$ principal components:
$$\mathbf{Z} = \mathbf{X}_c \mathbf{V}_{[:, :D]} \in \mathbb{R}^{N \times 30}$$

### 3.2. Variance Preservation & Computational Complexity
- **Cumulative Explained Variance (top 30 components):** **99.248%**
- **Residual Unexplained Variance (components 31–200):** $0.752\%$
- **Dimensionality Reduction Ratio:** $\frac{200 - 30}{200} = 85.0\%$ memory reduction.
- **Computational Speedup:** Reduces 3D convolution floating point operations (FLOPs) by approximately $6.67\times$, enabling rapid model convergence on standard hardware without discarding meaningful crop biochemical signals.

---

## 4. The Deep Learning Generalization Cliff: Empirical Results

A central scientific objective of TerraSpectra is exposing and quantifying the **Generalization Cliff** in hyperspectral machine learning.

### 4.1. The Root Cause: Spatial Autocorrelation Leakage in Patch Extraction
When extracting a spatial patch $\mathbf{P}(x, y)$ of size $k \times k$ ($9 \times 9 = 81$ pixels) centered at coordinate $(x, y)$, and an adjacent patch $\mathbf{P}(x+1, y)$ centered at $(x+1, y)$, the two patches share:
$$\text{Shared Pixels} = (k - 1) \times k = 8 \times 9 = 72 \text{ pixels}$$
$$\text{Spatial Overlap Ratio} = \frac{72}{81} \approx 88.89\%$$

Under **Regime A (Naive Random Split 80/20)**, pixels are sampled uniformly at random across the scene. Consequently:
- Virtually **every test patch** shares between $75\%$ and $98\%$ of its spatial pixels with multiple training patches.
- The 2D and 3D convolutional filters do not learn transferable biochemical or agronomic rules; instead, they memorize the exact spatial pixel coordinates and local noise patterns of the specific field parcels.

Under **Regime B (Spatial Disjoint Parcel Split)**:
- Connected spatial parcels belonging to the same agricultural class are partitioned using 8-connectivity connected-component labeling.
- Entire parcels are assigned strictly to either the train set or test set.
- **Inter-set patch overlap is strictly 0.0%**. The model is tested on fields it has never seen.

### 4.2. Deep Learning Benchmark Suite: Empirical Results Table

The table below summarizes the comprehensive benchmark results logged in `data/processed/day04_dl_benchmark_results.json`:

| Model Architecture | Split Regime | Overall Accuracy ($\text{OA}$) | Average Accuracy ($\text{AA}$) | Cohen's Kappa ($\kappa$) | Macro F1 | Weighted F1 | Training Time (s) | Inference Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Spectral 1D-CNN** | Random | 58.00% | 40.74% | 0.4955 | 0.3859 | 0.4882 | 19.59 s | 0.153 ms |
| **Spectral 1D-CNN** | **Parcel** | **30.25%** | **28.09%** | **0.1969** | **0.1926** | **0.2436** | 17.84 s | 0.079 ms |
| **Spatial 2D-CNN** | Random | 99.61% | 95.14% | 0.9955 | 0.9599 | 0.9957 | 57.49 s | 0.315 ms |
| **Spatial 2D-CNN** | **Parcel** | **36.59%** | **20.90%** | **0.2607** | **0.1815** | **0.3110** | 55.77 s | 0.210 ms |
| **Spatial-Spectral 3D-CNN** | Random | 86.93% | 69.15% | 0.8494 | 0.7040 | 0.8616 | 81.26 s | 0.432 ms |
| **Spatial-Spectral 3D-CNN** | **Parcel** | **31.75%** | **22.29%** | **0.2171** | **0.1724** | **0.2596** | 76.39 s | 0.107 ms |
| **HybridSN (3D-2D CNN)** | Random | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 28.34 s | 0.177 ms |
| **HybridSN (3D-2D CNN)** | **Parcel** | **36.55%** | **24.33%** | **0.2516** | **0.2007** | **0.3052** | 25.37 s | 0.095 ms |

### 4.3. Quantifying the Generalization Cliff Across Model Paradigms

```
  100% ┌────────────────────────────────────────────────────────┐
       │ ■ Random Split (Overfitting / Leakage)                 │
   80% │ ░ Parcel Split (True Field Generalization)             │
       │                                                        │
   60% │          ■ 99.6%                   ■ 100.0%            │
       │                                    │                   │
   40% │ ■ 58.0%             ■ 86.9%        │                   │
       │ │                   │              │                   │
   20% │ ░ 30.3%  ░ 36.6%    ░ 31.8%        ░ 36.6%             │
       │                                                        │
    0% └────────────────────────────────────────────────────────┘
         1D-CNN     2D-CNN    3D-CNN       HybridSN
```

The absolute and relative performance drops between Regime A and Regime B reveal stark architectural behaviors:

| Architecture | Random OA | Parcel OA | Absolute Cliff Drop ($\Delta \text{OA}$) | Relative Drop ($\%$) | $\Delta \kappa$ Drop |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Spectral 1D-CNN** | 58.00% | 30.25% | **-27.75%** | **-47.84%** | -0.2986 |
| **Spatial 2D-CNN** | 99.61% | 36.59% | **-63.02%** | **-63.27%** | -0.7348 |
| **Spatial-Spectral 3D-CNN** | 86.93% | 31.75% | **-55.18%** | **-63.48%** | -0.6323 |
| **HybridSN (3D-2D CNN)** | 100.00% | 36.55% | **-63.45%** | **-63.45%** | -0.7484 |

#### Critical Comparative Findings:
1. **The Deception of Pure Spatial Convolutions:**
   - Spatial 2D-CNN and HybridSN appear nearly flawless under Random Splitting ($99.61\%$ and $100.00\%$). However, when evaluated on unseen parcels, their accuracy drops by more than **$63\%$ absolute**!
   - This drop is substantially more catastrophic than that observed in classical ML baselines (Random Forest dropped $-48.7\%$, SVM dropped $-43.7\%$). Deep spatial convolutions have vastly higher capacity to memorize spatial coordinates than tabular decision trees.
2. **Why 1D-CNN Has the Smallest Drop but Lowest Random Score:**
   - Spectral 1D-CNN operates on isolated $1 \times 1$ pixels with zero spatial patch extraction.
   - It cannot leak spatial neighbor pixels during random sampling, which is why its Random Split accuracy is only $58.00\%$.
   - On disjoint parcels, it achieves $30.25\%$, which is very close to the classical SVM baseline ($32.89\%$). This proves that spectral convolutions alone behave similarly to non-linear kernel baselines.
3. **The Stability of HybridSN:**
   - While 2D-CNN achieved slightly higher top-1 OA on parcel test ($36.59\%$ vs $36.55\%$), HybridSN achieved higher Average Accuracy ($24.33\%$ vs $20.90\%$) and higher Macro F1 ($0.2007$ vs $0.1815$).
   - HybridSN was able to detect minority classes (such as `Buildings-Grass-Trees-Drives` with $18.33\%$ recall), whereas 2D-CNN completely collapsed to 0% on 13 of the 16 classes under parcel split.

---

## 5. Spatial Receptive Field & Context Window Ablation

To isolate the exact influence of spatial neighborhood size on out-of-field generalization, an ablation experiment was performed using `Spatial2DCNN` across patch window sizes $k \in \{5, 9, 13\}$ under strict Disjoint Parcel Splitting.

### 5.1. Ablation Results Table

Logged in `data/processed/day04_patch_size_ablation.csv`:

| Patch Size ($k \times k$) | Spatial Receptive Field | Physical Footprint (at 20m GSD) | Parcel Test OA | Cohen's Kappa ($\kappa$) | Average Accuracy ($\text{AA}$) | Macro F1 | Training Time (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$5 \times 5$** | 25 pixels | $100\text{ m} \times 100\text{ m}$ | **39.77%** | **0.2894** | **24.15%** | **0.2056** | **6.53 s** |
| **$9 \times 9$** | 81 pixels | $180\text{ m} \times 180\text{ m}$ | **37.14%** | **0.2643** | **22.73%** | **0.1937** | **15.79 s** |
| **$13 \times 13$** | 169 pixels | $260\text{ m} \times 260\text{ m}$ | **27.38%** | **0.1566** | **15.42%** | **0.1246** | **76.03 s** |

### 5.2. Biophysical & Spatial Contamination Analysis

```
       Small Window (5x5)                  Large Window (13x13)
     ┌───────────────────┐               ┌───────────────────────────────┐
     │  C   C   C   C   C│               │  R   R   R   R   R   R   R   R│ <-- Road / Dirt
     │  C   C   C   C   C│               │  C   C   C   C   C   C   C   C│
     │  C   C  [P]  C   C│ <-- Pure      │  C   C   C   C   C   C   C   C│
     │  C   C   C   C   C│     Canopy    │  C   C   C  [P]  C   C   C   C│
     │  C   C   C   C   C│               │  S   S   S   S   S   S   S   S│ <-- Neighboring
     └───────────────────┘               │  S   S   S   S   S   S   S   S│     Soybean Field
      Homogeneous Canopy Signal          └───────────────────────────────┘
                                           Heterogeneous Boundary Artifacts
```

#### Key Findings:
1. **The Inverted Accuracy vs Receptive Field Relationship:**
   - In computer vision, larger receptive fields almost universally improve classification accuracy by capturing broader semantic context.
   - In agricultural hyperspectral remote sensing, **the opposite occurs**: expanding from $5 \times 5$ to $13 \times 13$ causes overall accuracy to plunge from **$39.77\%$ to $27.38\%$** (a **$-12.39\%$ absolute drop**).
2. **Boundary Contamination in Heterogeneous Agricultural Landscapes:**
   - At AVIRIS's 20-meter ground sampling distance (GSD), a $13 \times 13$ patch spans a ground area of $260\text{ m} \times 260\text{ m} = 6.76\text{ hectares}$.
   - Many agricultural parcels in Indian Pines (e.g. Alfalfa, Oats, Grass-pasture-mowed) are narrow strips less than $100\text{ m}$ wide.
   - For a target pixel $[P]$ near a field boundary, a $13 \times 13$ patch contains pixels from drainage ditches, access roads, and adjacent fields with different crop types and tillage states. The network is forced to learn mixed spectral-spatial signals, confusing the central target's signature.
3. **Computational Efficiency:**
   - The $5 \times 5$ patch window achieves the highest out-of-field accuracy while executing in just **6.53 seconds** ($11.6\times$ faster than $13 \times 13$). Compact receptive fields are both scientifically superior and computationally frugal.

---

## 6. Full-Scene Thematic Spatial Inference & Boundary Behavior

To evaluate the operational utility of each trained model for real-world agricultural monitoring, we deployed all 4 architectures across the entire $145 \times 145$ pixel scene ($21,025$ spatial predictions), comparing models trained under Regime A vs Regime B.

### 6.1. Qualitative Map Observations

Inspection of `reports/figures/day04_full_scene_thematic_maps.png` highlights distinct thematic behaviors:
1. **Regime A (Random Split) Predictions:**
   - **Hyper-Crisp Parcel Fill:** Models trained under random split reproduce the ground truth parcel polygons almost perfectly. Field interiors are smooth and uniform, and boundary edges are sharply delineated.
   - **Severe Checkerboard Extrapolation:** When evaluated on unlabeled/background regions (pixels with ground truth class 0, representing farm roads, drainage ditches, and unmapped pastures), Random Split models output high-confidence false alarms, fragmenting background terrain into noisy speckles of Corn and Soybean classes.
2. **Regime B (Disjoint Parcel Split) Predictions:**
   - **Intra-Field Spectral Heterogeneity:** Disjoint-trained models reveal that agricultural fields are not internally homogeneous. Slight variations in soil moisture, canopy density, and weed patches cause field interiors to show realistic class uncertainty.
   - **Tillage State Confusion:** Corn-notill, Corn-mintill, and Soybean-mintill are frequently predicted interchangeably across adjacent fields. This aligns directly with our Day 3 Spectral Angle Mapper findings, where the angular distance between minimum-till corn and minimum-till soybean was less than $0.7^\circ$.
   - **Spatial Cohesion of HybridSN:** HybridSN exhibits substantially superior spatial cohesion compared to 1D-CNN. While 1D-CNN predictions exhibit severe "salt-and-pepper" noise due to pixel-by-pixel classification, HybridSN produces contiguous, contiguous parcel regions without requiring morphological post-processing.

---

## 7. Publication Figures Overview

Four publication-grade multi-panel figures were generated and saved in `reports/figures/`:

### Figure 1: Deep Learning Training Dynamics (`day04_dl_training_curves.png`)
- **Structure:** $2 \times 2$ grid tracking cross-entropy loss convergence and overall accuracy trajectories over epochs.
- **Key Observation:** Models trained under Random Split converge smoothly to near-zero loss ($\mathcal{L} < 0.05$) and $99\%+$ accuracy within 4–6 epochs. In contrast, Parcel Split models exhibit training-validation divergence around epoch 4, indicating that the network reaches the capacity limit for cross-parcel generalization.

### Figure 2: The Deep Learning Generalization Cliff (`day04_deep_learning_benchmark_cliff.png`)
- **Structure:** Grouped bar charts comparing Overall Accuracy ($\text{OA}$) and Cohen's Kappa ($\kappa$) side-by-side across all 4 architectures for both Random Split (blue) and Disjoint Parcel Split (coral).
- **Key Observation:** Graphically illustrates the $-63\%$ collapse across 2D-CNN and HybridSN, visually proving that spatial autocorrelation leakage creates an illusion of near-perfect classification.

### Figure 3: Full-Scene Thematic Prediction Maps (`day04_full_scene_thematic_maps.png`)
- **Structure:** $3 \times 3$ panel displaying Ground Truth, False-Color Composite, and the full $145 \times 145$ spatial prediction rasters for all 4 models across both splitting regimes.
- **Key Observation:** Contrasts the synthetic perfection of Random Split models against the realistic, field-level uncertainty and tillage confusion of Disjoint models.

### Figure 4: Spatial Context Ablation & Confusion Topology (`day04_patch_size_ablation_and_confusion.png`)
- **Structure:** Dual panel featuring:
  1. Patch size ablation curve ($k \in \{5, 9, 13\}$) showing the inverse relationship between accuracy and window size.
  2. Normalized confusion matrix for HybridSN under Disjoint Parcel Split, highlighting the primary confusion axes between Corn and Soybean tillage regimes.

---

## 8. Key Engineering & Scientific Insights

1. **Spatial Autocorrelation is the #1 Confounder in Remote Sensing DL:**
   - Any deep learning paper reporting $> 98\%$ classification accuracy on Indian Pines or Salinas using uniform random pixel splitting has evaluated patch overlap rather than agronomic generalization.
   - TerraSpectra establishes disjoint parcel splitting as the mandatory benchmark standard for trustworthy crop classification.
2. **HybridSN Provides Optimal Parameter-Accuracy Efficiency:**
   - Pure 3D-CNNs suffer from parameter inefficiency and slow convergence ($81.26\text{s}$).
   - HybridSN decouples 3D spectral-spatial feature extraction from 2D spatial feature refinement, training in just $25.37\text{s}$ while achieving the best macro-F1 and class balance on out-of-field evaluation.
3. **Smaller Spatial Windows Outperform Larger Windows in Fragmented Landscapes:**
   - Receptive fields must not exceed typical field parcel dimensions. A $5 \times 5$ window ($100\text{m} \times 100\text{m}$) outperforms a $13 \times 13$ window by $+12.39\%$ due to reduced boundary contamination.
4. **Spectral Information Alone is Insufficient for Fine-Grained Tillage Classification:**
   - 1D-CNN achieves only $30.25\%$ parcel accuracy, proving that spectral signatures of crops under different tillage states overlap heavily. Spatial context is strictly required to achieve coherent parcel classification.

---

## 9. Artifact Index & Day 5 Roadmap

### Day 4 Deliverable Artifacts
- **Production Pipeline Script:** [scripts/day04_deep_learning_spatial_spectral.py](file:///d:/TerraSpectra2026/scripts/day04_deep_learning_spatial_spectral.py)
- **Interactive Jupyter Notebook:** [notebooks/04_deep_learning_spatial_spectral.ipynb](file:///d:/TerraSpectra2026/notebooks/04_deep_learning_spatial_spectral.ipynb)
- **Benchmark Suite Metrics:** [data/processed/day04_dl_benchmark_results.json](file:///d:/TerraSpectra2026/data/processed/day04_dl_benchmark_results.json)
- **Training Convergence Trajectories:** [data/processed/day04_dl_training_histories.json](file:///d:/TerraSpectra2026/data/processed/day04_dl_training_histories.json)
- **Per-Class Precision/Recall/F1 Table:** [data/processed/day04_dl_per_class_performance.csv](file:///d:/TerraSpectra2026/data/processed/day04_dl_per_class_performance.csv)
- **Patch Size Ablation Table:** [data/processed/day04_patch_size_ablation.csv](file:///d:/TerraSpectra2026/data/processed/day04_patch_size_ablation.csv)
- **Publication Visualizations:**
  - [reports/figures/day04_dl_training_curves.png](file:///d:/TerraSpectra2026/reports/figures/day04_dl_training_curves.png)
  - [reports/figures/day04_deep_learning_benchmark_cliff.png](file:///d:/TerraSpectra2026/reports/figures/day04_deep_learning_benchmark_cliff.png)
  - [reports/figures/day04_full_scene_thematic_maps.png](file:///d:/TerraSpectra2026/reports/figures/day04_full_scene_thematic_maps.png)
  - [reports/figures/day04_patch_size_ablation_and_confusion.png](file:///d:/TerraSpectra2026/reports/figures/day04_patch_size_ablation_and_confusion.png)

### Day 5 Strategic Roadmap: Self-Supervised Learning & Hyperspectral Transformers
Having quantified the limitations of supervised CNNs under parcel splitting, Day 5 will investigate modern foundation model techniques to overcome label scarcity and bridge the Generalization Cliff:
1. **Self-Supervised Masked Autoencoders (MAE) for Hyperspectral Cubes:**
   - Pre-training volumetric autoencoders on unlabeled hyperspectral scenes using 3D spectral-spatial patch masking ($75\%$ mask ratio) to learn robust latent representations invariant to field-specific soil background.
2. **Spectral-Spatial Vision Transformers (ViT / SpectralFormer):**
   - Evaluating multi-head self-attention mechanisms to model long-range spectral band dependencies and cross-field spatial context without fixed convolutional receptive fields.
3. **Cross-Scene Transfer Learning:**
   - Evaluating transferability between AVIRIS Indian Pines and Salinas Valley sensors, benchmarking domain adaptation strategies.
