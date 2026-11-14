# 🌿 TerraSpectra: Structural Tensor Inspection, Spectrometer Calibration & Spatial Patch Slicing
**Project:** TerraSpectra — Intelligent Crop Disease Detection & Forecasting Using Hyperspectral Imaging  
**Role:** ML Engineering & Data Pipeline (Hyperspectral Preprocessing & Spatial Representation)  
**Author:** Rohit  
**Working Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 2 — Structural Inspection, Channel Calibrations, Spatial Patch Slicing & Spatial Autocorrelation Audit  
**Date:** November 14, 2026  

---

## 1. Executive Summary & Day 2 Research Scope

Following the dataset acquisition and initial exploratory profiling in **Day 1**, the primary objective of **Day 2** is to establish a mathematically rigorous data ingestion, calibration, and spatial-spectral patch representation pipeline. Hyperspectral cubes differ fundamentally from standard RGB imagery; each sample is a continuous 3D volume $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$ containing hundreds of contiguous spectral channels across the Visible (VIS), Near-Infrared (NIR), and Shortwave-Infrared (SWIR) domains.

This document details:
1. **Deep Structural Tensor Inspection:** Memory footprints, array contiguity, bit depths, and radiometric normalization across Indian Pines and Salinas Valley.
2. **Spectrometer Subsystem Mapping & Atmospheric Calibration:** Physical mapping of the 4 AVIRIS optical grating spectrometers (Spectrometers A, B, C, D) and mathematical verification of atmospheric water vapor absorption bands (104–108, 150–163, and 220) via variance collapse.
3. **Spatial-Spectral 3D Patch Extraction Engine:** Design and benchmarking of the modular `HyperspectralPatchExtractor` supporting symmetric boundary reflection padding and receptive fields from $5 \times 5$ to $17 \times 17$ for downstream 3D-CNN and Vision Transformer (ViT) architectures.
4. **Spatial Autocorrelation Audit & Data Leakage Proof:** Empirical quantification of Tobler's First Law of Geography in agricultural fields, proof of catastrophic spatial leakage under naive random pixel splitting, and the formal implementation of **Disjoint Parcel Splitting**.

---

## 2. Mathematical & Structural Tensor Representation

### 2.1. Tensor Formalism & Memory Layout
Let a calibrated hyperspectral scene be represented as a 3rd-order tensor:
$$\mathcal{X} \in \mathbb{R}^{H \times W \times B}$$
where:
- $H$ represents the along-track spatial lines (flight direction).
- $W$ represents the cross-track spatial samples (swath width).
- $B$ represents the calibrated spectral channels ($B = 200$ for Indian Pines, $B = 204$ for Salinas).

Each spatial coordinate $(i, j)$ corresponds to a continuous spectral signature vector $\mathbf{x}_{i, j} \in \mathbb{R}^B$. The semantic ground truth is represented as a 2D integer matrix $Y \in \{0, 1, \dots, C\}^{H \times W}$, where $0$ denotes unlabeled background and $1 \le c \le C$ indexes verified agricultural classes.

### 2.2. Structural Audit & Memory Profiling
Inspection results from `data/processed/day02_structural_inspection_summary.json`:

| Tensor Name | Spatial Grid ($H \times W$) | Spectral Bands ($B$) | Total Voxels | Data Type | Memory (MB) | Contiguity Layout | Dynamic Range (DN) | Null / NaN Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Indian Pines Raw** | $145 \times 145$ | 220 | 4,625,500 | `uint16` (2 bytes) | 8.82 MB | C-Contiguous (Row-Major) | [955, 9,604] | 0 |
| **Indian Pines Calibrated** | $145 \times 145$ | 200 | 4,205,000 | `uint16` (2 bytes) | 8.02 MB | C-Contiguous (Row-Major) | [955, 9,604] | 0 |
| **Indian Pines GT** | $145 \times 145$ | 1 | 21,025 | `uint8` (1 byte) | 0.02 MB | C-Contiguous (Row-Major) | [0, 16] | 0 |
| **Salinas Calibrated** | $512 \times 217$ | 204 | 22,665,216 | `int16` (2 bytes) | 43.23 MB | C-Contiguous (Row-Major) | [-11, 9,207] | 0 |
| **Salinas GT** | $512 \times 217$ | 1 | 111,104 | `uint8` (1 byte) | 0.11 MB | C-Contiguous (Row-Major) | [0, 16] | 0 |

### 2.3. Radiometric Scaling & Precision Normalization
AVIRIS radiometric calibration scales surface reflectance by a constant multiplier of $10^4$:
$$\rho_{i, j, b} = \mathrm{clip}\left(\frac{\mathcal{X}_{i, j, b}}{10000.0}, 0.0, 1.0\right)$$
- In Salinas, residual atmospheric calibration offsets produced a negligible fraction of negative values ($-11 \le DN < 0$, $< 0.001\%$ of voxels), which are strictly non-physical in surface reflectance. Applying $\mathrm{clip}(0.0, 1.0)$ guarantees valid physical reflectance $\rho \in [0.0, 1.0]$.
- C-contiguous ordering guarantees optimal stride access along spatial lines and spectral depth during convolutional kernel slicing.

---

## 3. AVIRIS Spectrometer Subsystems & Wavelength Calibration

### 3.1. Optical Subsystem Decomposition
The AVIRIS instrument disperses incident solar photons across four independent grating optical spectrometers, each optimized for specific optical coatings and detector arrays:

```
+-------------------------------------------------------------------------------------------------------+
|                                    AVIRIS OPTICAL SUBSYSTEMS                                          |
+------------------------------------+-----------------------------------+------------------------------+
| Spectrometer Subsystem             | Calibrated Spectral Domain        | Diagnostic Agricultural Role |
+------------------------------------+-----------------------------------+------------------------------+
| Spectrometer A: VIS (Bands 1–32)   | 400 nm - 700 nm                   | Chlorophyll absorption,      |
|                                    | (Blue ~450nm, Green ~550nm,       | carotenoids, anthocyanins    |
|                                    |  Red ~670nm)                      |                              |
+------------------------------------+-----------------------------------+------------------------------+
| Spectrometer B: NIR (Bands 33–96)  | 700 nm - 1300 nm                  | Mesophyll cellular structure,|
|                                    | (Red-edge 700-740nm, NIR plateau) | vegetative biomass, LAI      |
+------------------------------------+-----------------------------------+------------------------------+
| Spectrometer C: SWIR-1 (97–160)    | 1300 nm - 1900 nm                 | Canopy liquid water content, |
|                                    | (Atmospheric water dip ~1400nm)   | leaf turgor pressure         |
+------------------------------------+-----------------------------------+------------------------------+
| Spectrometer D: SWIR-2 (161–220)   | 1900 nm - 2450 nm                 | Cellulose, lignin, starch,   |
|                                    | (Atmospheric water dip ~1900nm)   | dry matter & crop residue    |
+------------------------------------+-----------------------------------+------------------------------+
```

### 3.2. Verification of Atmospheric Water Absorption Windows
Earth's atmospheric column contains strong vibrational absorption bands of triatomic water molecules ($H_2O$), making the atmosphere opaque at specific narrow spectral intervals:
1. **$1.4\,\mu\text{m}$ Resonance Dip ($\lambda \in [1360, 1420\text{ nm}]$):** Raw Bands 104, 105, 106, 107, 108.
2. **$1.9\,\mu\text{m}$ Resonance Dip ($\lambda \in [1800, 1950\text{ nm}]$):** Raw Bands 150 through 163 (14 contiguous bands).
3. **Detector Overlap / Instrument Edge ($\lambda \sim 2450\text{ nm}$):** Raw Band 220.

**Mathematical Proof via Radiometric Variance Profile:**  
As seen in `data/processed/day02_spectral_calibration_details.csv` and plotted in Figure 2:
- Within clean agricultural channels, the spatial variance of radiance spans $1.5 \times 10^5 \text{ to } 3.8 \times 10^6\, DN^2$, reflecting healthy field heterogeneity.
- In raw bands 104–108 and 150–163, atmospheric attenuation causes the received ground radiance to collapse, while sensor readout noise fluctuates erratically.
- Retaining these 20 noisy bands would force machine learning models to correlate random atmospheric detector noise with crop labels. The calibrated 200-band cube removes all 20 atmospheric bands, preserving an uncorrupted signal.

---

## 4. 3D Orthogonal Cube Slicing

To analyze how spectral information couples with spatial morphology, orthogonal slicing was performed across three coordinate planes (Figure 1: `reports/figures/day02_spectral_cube_slicing.png`):

1. **Axial Spatial Slices ($X-Y$ Plane):**
   - **Band 15 ($\lambda \approx 550\text{ nm}$, Green Peak):** Shows moderate reflectance across photosynthetically active crops; low contrast between tilled soil and young vegetation.
   - **Band 35 ($\lambda \approx 720\text{ nm}$, Red-Edge):** Highlights the boundary of chlorophyll reflectance.
   - **Band 55 ($\lambda \approx 850\text{ nm}$, NIR Peak):** High dynamic range; healthy crop canopies reflect strongly ($DN > 5,000$), whereas soil and unplanted areas appear dark ($DN < 2,000$).
   - **Band 140 ($\lambda \approx 1650\text{ nm}$, SWIR-1):** Inverted contrast where dense, moisture-rich canopies absorb strongly, while bare dry soil reflects solar irradiance.
2. **Coronal Spectral Cross-Section ($X-\lambda$ Plane at $Y = 72$):**
   - Crosses multiple contiguous agricultural fields. Abrupt vertical boundaries in the $(X, \lambda)$ slice indicate sharp field edges where the crop cultivar and tillage regime change abruptly.
3. **Sagittal Spectral Cross-Section ($Y-\lambda$ Plane at $X = 72$):**
   - Tracks along the sensor flight line, confirming spectral stability across uniform parcels and continuity of the Red-Edge transition across adjacent scanlines.

---

## 5. Spatial-Spectral 3D Patch Extraction Engine

### 5.1. Formulation & Receptive Field Selection
Conventional pixel-wise 1D spectral classifiers evaluate a single vector $\mathbf{x} \in \mathbb{R}^B$, discarding spatial context. However, agricultural fields exhibit characteristic spatial textures, tillage rows, soil background variations, and canopy density gradients.

For a spatial kernel size $k$ (where $k$ is an odd integer), the patch extractor creates a 3D tensor:
$$\mathcal{P}_{i, j} \in \mathbb{R}^{k \times k \times B}$$
centered at pixel $(i, j)$ for all labeled coordinates where $Y_{i, j} > 0$.

### 5.2. Boundary Condition & Symmetric Reflection Padding
To extract patches for pixels located within distance $m = \lfloor k/2 \rfloor$ of the image boundaries, padding is required. Three strategies were analyzed:
- **Zero Padding (`constant`):** Introduces severe artificial step gradients at field margins, causing convolution kernels to learn border artifacts.
- **Replicate Padding (`edge`):** Duplicates border pixels, flattening spatial gradient information.
- **Symmetric Reflection Padding (`reflect` — Implemented):**
  $$\mathcal{X}_{\text{padded}}[-r, c, b] = \mathcal{X}[r, c, b]$$
  Reflects natural field texture across the boundary without introducing discontinuity or step artifacts.

### 5.3. Computational Footprint & Precision Benchmarks
For the 10,249 labeled pixels of Indian Pines ($B = 200$ bands):

| Patch Size ($k \times k$) | Half-Margin ($m$) | Spatial Ground Footprint ($m \times m$) | Voxels per Patch | FP32 Memory (MB) | FP16 Memory (MB) | Recommended Architecture |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$5 \times 5$** | 2 | $100\text{ m} \times 100\text{ m}$ | 5,000 | 195.5 MB | 97.7 MB | Lightweight 2D-CNN |
| **$7 \times 7$** | 3 | $140\text{ m} \times 140\text{ m}$ | 9,800 | 383.1 MB | 191.6 MB | 3D-CNN Baseline |
| **$9 \times 9$ (Standard)** | **4** | **$180\text{ m} \times 180\text{ m}$** | **16,200** | **633.2 MB** | **316.6 MB** | **Hybrid 3D-CNN / ViT (Optimal)** |
| **$11 \times 11$** | 5 | $220\text{ m} \times 220\text{ m}$ | 24,200 | 946.0 MB | 473.0 MB | Vision Transformer |
| **$13 \times 13$** | 6 | $260\text{ m} \times 260\text{ m}$ | 33,800 | 1,321.3 MB | 660.6 MB | Large ViT / High VRAM |
| **$15 \times 15$** | 7 | $300\text{ m} \times 300\text{ m}$ | 45,000 | 1,758.8 MB | 879.4 MB | Spatial Context Boundary Model |

*Strategic Decision:* The **$9 \times 9$ patch** provides an ideal balance: an $180\text{ m} \times 180\text{ m}$ ground footprint capturing field homogeneity and canopy texture, while requiring only $316.6\text{ MB}$ in `float16` precision, enabling mini-batch GPU training with batch sizes of 64 or 128.

---

## 6. Spatial Autocorrelation & The Random Split Fallacy

### 6.1. Empirical Proof of Tobler's First Law
In agricultural remote sensing, pixels belonging to the same crop parcel are cultivated under uniform soil, irrigation, fertilizer, and weather conditions. Consequently, adjacent pixels exhibit extreme spectral autocorrelation.

We measured the spectral cosine similarity between pixel pairs as a function of Euclidean spatial distance $d$:
$$S_{\cos}(\mathbf{x}_1, \mathbf{x}_2) = \frac{\mathbf{x}_1 \cdot \mathbf{x}_2}{\|\mathbf{x}_1\|_2 \|\mathbf{x}_2\|_2}$$

As shown in Figure 4 (`reports/figures/day02_spatial_autocorrelation_leakage.png`):
- For pixel pairs within the same parcel, cosine similarity remains **$> 0.985$** across distances up to 30 pixels (600 meters).
- This extreme spatial correlation proves that spatial proximity dominates spectral variation within individual agricultural parcels.

### 6.2. The Random Split Fallacy & Data Leakage
Many published papers in hyperspectral classification report test accuracies of $98\% - 99\%$ achieved by standard random pixel partitioning ($80\%$ train / $20\%$ test). Our Day 2 audit reveals why this metric is illusory:

1. **Voxel Overlap Audit:** In a naive $80/20$ random pixel split, when a $9 \times 9$ patch is centered on a test pixel, **$62.51\%$ of the voxels in that test patch belong to training samples**.
2. **Patch Contamination:** **$99.8\%$ of test patches contain training voxels**.
3. **Consequence:** The deep network does not learn invariant biochemical crop signatures; instead, it memorizes the spatial coordinates of adjacent pixels that it has already seen during training. When deployed on a new, unseen field parcel, performance drops significantly.

```
NAIVE RANDOM SPLIT (Severe Leakage)       DISJOINT PARCEL SPLIT (Zero Leakage)
+------------------------------------+    +------------------------------------+
|  [Train]  [Test]   [Train] [Train] |    |  [==============================]  |
|  [Train]  [Train]  [Test]  [Train] |    |  [      PARCEL A: ALL TRAIN     ]  |
|  [Test]   [Train]  [Train] [Test]  |    |  [==============================]  |
|  (Test patch contains Train voxels)|    |  [------------------------------]  |
|  --> 62.5% Voxel Contamination     |    |  [      PARCEL B: ALL TEST      ]  |
|  --> Deceptively high ~99% accuracy|    |  [------------------------------]  |
+------------------------------------+    +------------------------------------+
```

### 6.3. The Solution: Disjoint Parcel Splitting
To establish a valid evaluation framework:
1. We apply connected component labeling (`scipy.ndimage.label`) to segment the 10,249 labeled pixels into **43 distinct field parcels**.
2. Parcels are partitioned into training and testing sets as indivisible units.
3. This guarantees **zero test-patch contamination**, measuring true generalization to unvisited agricultural fields—an essential requirement for disease forecasting in real farms.

---

## 7. Generated Project Artifacts (Day 2)

All code, data, figures, and notebooks for Day 2 have been produced and verified:

### 7.1. Executable Scripts
- `scripts/day02_structural_inspection_patching.py`: Production pipeline executing structural audits, spectrometer subsystem mapping, patch engine instantiation, leakage proofs, and figure generation.
- `scripts/generate_day02_notebook.py`: Automated generator for the reproducible Jupyter Notebook.

### 7.2. Processed Data & Metadata (`data/processed/`)
- `data/processed/day02_structural_inspection_summary.json`: Complete structural audit, memory layouts, radiometric stats, and null checks for all 5 tensors.
- `data/processed/day02_spectral_calibration_details.csv`: Band-by-band calibration table (220 rows) including center wavelengths, spectrometer subsystems (A, B, C, D), atmospheric flags, and variance profiles.
- `data/processed/day02_patch_extraction_metadata.json`: Receptive field parameters, reflection padding metadata, sample batch verification, and memory benchmarks across patch sizes $k \in [5, 17]$.

### 7.3. Publication Figures (`reports/figures/`)
- `day02_spectral_cube_slicing.png`: 3D orthogonal slicing across Axial ($X-Y$), Coronal ($X-\lambda$), and Sagittal ($Y-\lambda$) planes.
- `day02_spectral_channel_calibration.png`: Spectrometer subsystems, atmospheric water vapor variance profiles, and mean crop reflectance curves.
- `day02_spatial_patch_extraction.png`: Boundary reflection padding diagram, receptive field trade-offs, and sample CIR $9 \times 9$ patches for 6 crop classes.
- `day02_spatial_autocorrelation_leakage.png`: Spatial autocorrelation decay curve, quantitative leakage box, and spatial maps comparing Naive Random Split vs Disjoint Parcel Split.

### 7.4. Interactive Jupyter Notebook
- `notebooks/02_structural_inspection_and_patching.ipynb`: 17-cell reproducible analysis notebook with interactive visualizations and math formulations.

---

## 8. Strategic Direction & Day 3 Roadmap

With structural integrity, spectrometer calibrations, and patch extraction validated, the foundation for model training is complete:

- **Day 3:** Spectral Signature Characterization & Classical ML Baseline:
  - Computation of narrowband vegetation indices: **NDVI** (Normalized Difference Vegetation Index), **NDRE** (Normalized Difference Red-Edge), **PRI** (Photochemical Reflectance Index), and **WBI** (Water Band Index).
  - Training classical machine learning baselines (Random Forest, SVM, LightGBM) under both Random Split and Disjoint Parcel Split to quantify the real-world performance delta.
  - Spectral Angle Mapper (SAM) analysis across crop health categories.