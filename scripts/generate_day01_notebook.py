"""
Create notebooks/01_dataset_exploration.ipynb with full documentation, markdown, and runnable cells.
"""

import os
import nbformat as nbf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🌿 TerraSpectra: Intelligent Crop Disease Forecasting
## Day 1 — Hyperspectral Dataset Exploration, Scientific Sensor Justification & Ground-Truth Verification

**Module:** ML Engineering & Data Pipeline (EDA)  
**Author:** Rohit (ML Engineering & Hyperspectral EDA Lead)  
**Branch:** `Rohit-EDA-Analysis`  
**Focus:** Dataset selection, sensor comparison, spectral wavelength profiling, and ground-truth crop verification.

---

### 1. Research Objectives for Day 1
1. **Scientific Dataset Evaluation:** Compare leading spaceborne (NASA EO-1 Hyperion, ASI PRISMA, DLR EnMAP) and benchmark airborne hyperspectral systems (AVIRIS Indian Pines, AVIRIS Salinas).
2. **Dataset Acquisition & Verification:** Verify the acquired cubes in `data/raw/` (`Indian_pines_corrected.mat`, `Indian_pines.mat`, `Salinas_corrected.mat`).
3. **Spectral Dimension & Metadata Audit:** Extract spatial dimensions $(H \\times W)$, spectral depth $(C = 200+ \\text{ bands})$, radiometric ranges, and bit depths.
4. **Wavelength Calibration & Atmospheric Water Absorption:** Map spectral channels from $400\\text{ nm}$ to $2500\\text{ nm}$ across VIS, NIR, and SWIR, identifying bands contaminated by atmospheric water vapor.
5. **Crop Health & Ground-Truth Distribution:** Analyze class distributions, crop types, tillage practices, phenological growth stages, and evaluate class imbalance.
6. **False-Color Infrared (CIR) Visualization:** Synthesize CIR composite imagery to distinguish vegetative vigor from background soil and non-vegetation."""))

# Imports
cells.append(nbf.v4.new_code_cell("""# Environment & Library Setup
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

# Configure matplotlib formatting
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

print("Base Working Directory:", BASE_DIR)
print("Raw Data Directory:    ", DATA_RAW)
print("Processed Directory:   ", DATA_PROC)"""))

# Section 1: Sensor Comparison
cells.append(nbf.v4.new_markdown_cell("""---
### 2. Hyperspectral Sensor Benchmark Analysis: Spaceborne vs. Airborne

In precision agriculture and early crop disease detection, selecting the appropriate sensor involves balancing:
- **Spectral Resolution:** Contiguous narrow bands ($\\le 10\\text{ nm}$) to detect subtle chlorophyll-a/b breakdown and carotenoid/anthocyanin shifts prior to visible chlorosis.
- **Signal-to-Noise Ratio (SNR):** Airborne sensors (AVIRIS) achieve SNR $> 1000:1$ in VIS/NIR and $> 500:1$ in SWIR, whereas satellite platforms suffer greater atmospheric path attenuation.
- **Ground Sample Distance (GSD):** High spatial resolution ($3.7\\text{ m} - 20\\text{ m}$) enables individual field and canopy resolution, whereas satellite pixels ($30\\text{ m}$) produce mixed pixels (*spectral unmixing* challenges)."""))

cells.append(nbf.v4.new_code_cell("""# Compare sensor specifications
sensor_comparison = pd.DataFrame([
    {
        "Sensor": "NASA EO-1 Hyperion",
        "Platform": "Spaceborne Satellite",
        "Total Spectral Bands": 242,
        "Usable Bands": 198,
        "Spectral Range (nm)": "355 - 2577",
        "Spectral Bandwidth": "10 nm",
        "Spatial GSD": "30 m",
        "Agricultural Role": "Regional scale canopy monitoring"
    },
    {
        "Sensor": "ASI PRISMA",
        "Platform": "Spaceborne Satellite",
        "Total Spectral Bands": 239,
        "Usable Bands": 230,
        "Spectral Range (nm)": "400 - 2505",
        "Spectral Bandwidth": "9 - 12 nm",
        "Spatial GSD": "30 m",
        "Agricultural Role": "Precision agriculture & canopy chemistry"
    },
    {
        "Sensor": "DLR EnMAP",
        "Platform": "Spaceborne Satellite",
        "Total Spectral Bands": 242,
        "Usable Bands": 224,
        "Spectral Range (nm)": "420 - 2450",
        "Spectral Bandwidth": "6.5 - 10 nm",
        "Spatial GSD": "30 m",
        "Agricultural Role": "Global agricultural ecosystem assessment"
    },
    {
        "Sensor": "AVIRIS (Indian Pines)",
        "Platform": "Airborne (NASA ER-2)",
        "Total Spectral Bands": 220,
        "Usable Bands": 200,
        "Spectral Range (nm)": "400 - 2450",
        "Spectral Bandwidth": "10 nm",
        "Spatial GSD": "20 m",
        "Agricultural Role": "Benchmark crop classification & tillage analysis"
    },
    {
        "Sensor": "AVIRIS (Salinas Valley)",
        "Platform": "Airborne (NASA ER-2)",
        "Total Spectral Bands": 224,
        "Usable Bands": 204,
        "Spectral Range (nm)": "400 - 2500",
        "Spectral Bandwidth": "10 nm",
        "Spatial GSD": "3.7 m",
        "Agricultural Role": "High-res crop phenology, weeds, & senescence"
    }
])

display(sensor_comparison)"""))

# Section 2: Load Data
cells.append(nbf.v4.new_markdown_cell("""---
### 3. Loading and Inspecting Hyperspectral Cubes

We load both the **AVIRIS Indian Pines** agricultural scene ($145 \\times 145 \\times 200$) and the **AVIRIS Salinas Valley** scene ($512 \\times 217 \\times 204$)."""))

cells.append(nbf.v4.new_code_cell("""# Load Indian Pines
ip_corr_path = os.path.join(DATA_RAW, "Indian_pines_corrected.mat")
ip_raw_path = os.path.join(DATA_RAW, "Indian_pines.mat")
ip_gt_path = os.path.join(DATA_RAW, "Indian_pines_gt.mat")

ip_data = sio.loadmat(ip_corr_path)["indian_pines_corrected"]
ip_raw = sio.loadmat(ip_raw_path)["indian_pines"]
ip_gt = sio.loadmat(ip_gt_path)["indian_pines_gt"]

print("=== AVIRIS Indian Pines ===")
print(f"Corrected Cube Shape: {ip_data.shape}  | Data Type: {ip_data.dtype}")
print(f"Raw Cube Shape:       {ip_raw.shape}  | Data Type: {ip_raw.dtype}")
print(f"Ground Truth Shape:   {ip_gt.shape}    | Data Type: {ip_gt.dtype}")
print(f"Radiometric Range:    Min={ip_data.min()}, Max={ip_data.max()}, Mean={ip_data.mean():.2f}")

# Load Salinas Valley
sal_corr_path = os.path.join(DATA_RAW, "Salinas_corrected.mat")
sal_gt_path = os.path.join(DATA_RAW, "Salinas_gt.mat")

sal_data = sio.loadmat(sal_corr_path)["salinas_corrected"]
sal_gt = sio.loadmat(sal_gt_path)["salinas_gt"]

print("\\n=== AVIRIS Salinas Valley ===")
print(f"Corrected Cube Shape: {sal_data.shape} | Data Type: {sal_data.dtype}")
print(f"Ground Truth Shape:   {sal_gt.shape}   | Data Type: {sal_gt.dtype}")
print(f"Radiometric Range:    Min={sal_data.min()}, Max={sal_data.max()}, Mean={sal_data.mean():.2f}")"""))

# Section 3: Wavelength Analysis
cells.append(nbf.v4.new_markdown_cell("""---
### 4. Wavelength Profiling & Atmospheric Absorption Windows

The solar electromagnetic spectrum captured by AVIRIS spans $400\\text{ nm}$ to $2450\\text{ nm}$.
Atmospheric constituents (principally water vapor $\\text{H}_2\\text{O}$ and carbon dioxide $\\text{CO}_2$) cause total transmission loss in two major bands:
1. **$1350\\text{ nm} - 1420\\text{ nm}$:** Bands 104–108 in Indian Pines.
2. **$1800\\text{ nm} - 1950\\text{ nm}$:** Bands 150–163 in Indian Pines.
3. **Band 220 ($\sim 2450\\text{ nm}$):** Excessive instrument noise at the detector boundary.

Filtering these out yields the 200 calibrated bands utilized by deep learning models."""))

cells.append(nbf.v4.new_code_cell("""# Inspect wavelength lookup table
wl_df = pd.read_csv(os.path.join(DATA_PROC, "indian_pines_wavelengths.csv"))
print(f"Total mapped bands: {len(wl_df)}")
print("\\nSample bands across spectral regions:")
display(wl_df.iloc[[0, 20, 50, 80, 105, 120, 155, 180, 219]])

# Regional band breakdown
region_summary = wl_df.groupby(["spectral_region", "status"]).size().unstack(fill_value=0)
display(region_summary)"""))

# Section 4: Ground Truth Distribution
cells.append(nbf.v4.new_markdown_cell("""---
### 5. Ground-Truth Agricultural Classes & Distribution Analysis

We analyze class representation for both datasets to assess class balance, minimum sample representation, and agricultural relevance."""))

cells.append(nbf.v4.new_code_cell("""# Display Indian Pines class distribution
ip_dist = pd.read_csv(os.path.join(DATA_PROC, "indian_pines_class_distribution.csv"))
print("=== Indian Pines Ground Truth Summary ===")
display(ip_dist)

# Plot class distribution
fig, ax = plt.subplots(figsize=(12, 6))
ip_labeled = ip_dist[ip_dist["class_id"] > 0].sort_values(by="pixel_count", ascending=True)
ax.barh(ip_labeled["class_name"], ip_labeled["pixel_count"], color="#388e3c", edgecolor="#1b5e20")
ax.set_xlabel("Pixel Count")
ax.set_title("AVIRIS Indian Pines: Class Distribution (Sample Count per Agricultural Class)", fontweight="bold")
for idx, (val, name) in enumerate(zip(ip_labeled["pixel_count"], ip_labeled["class_name"])):
    ax.text(val + 30, idx, f"{val}", va="center", fontsize=9)
plt.tight_layout()
plt.show()"""))

# Section 5: Salinas Distribution
cells.append(nbf.v4.new_markdown_cell("""---
### 6. Salinas Valley Ground-Truth: Phenological & Senescence Stages

The Salinas scene provides high spatial resolution ($3.7\\text{ m}$) and targets crop stress and growth stages:
- **Phenology Tracking:** `Lettuce_romaine_4wk`, `5wk`, `6wk`, `7wk` demonstrate spectral shifts as canopy coverage expands.
- **Crop Senescence & Weeds:** `Corn_senesced_green_weeds` captures natural plant degradation and weed competition, mimicking disease-induced canopy deterioration."""))

cells.append(nbf.v4.new_code_cell("""sal_dist = pd.read_csv(os.path.join(DATA_PROC, "salinas_class_distribution.csv"))
display(sal_dist)

fig, ax = plt.subplots(figsize=(12, 6))
sal_labeled = sal_dist[sal_dist["class_id"] > 0].sort_values(by="pixel_count", ascending=True)
ax.barh(sal_labeled["class_name"], sal_labeled["pixel_count"], color="#1976d2", edgecolor="#0d47a1")
ax.set_xlabel("Pixel Count")
ax.set_title("AVIRIS Salinas Valley: Class Distribution (Crop & Phenological Stages)", fontweight="bold")
for idx, (val, name) in enumerate(zip(sal_labeled["pixel_count"], sal_labeled["class_name"])):
    ax.text(val + 100, idx, f"{val}", va="center", fontsize=9)
plt.tight_layout()
plt.show()"""))

# Section 6: False Color Composite & Ground Truth Maps
cells.append(nbf.v4.new_markdown_cell("""---
### 7. False-Color Infrared (CIR) Composite & Ground Truth Spatial Maps

In Color-Infrared (CIR) imagery:
- **Red Channel:** Near-Infrared ($\sim 800\\text{ nm}$) $\\rightarrow$ Vigorous vegetation reflects intensely due to spongy mesophyll cell scattering.
- **Green Channel:** Red ($\sim 650\\text{ nm}$) $\\rightarrow$ Absorbed by chlorophyll-a/b.
- **Blue Channel:** Green ($\sim 550\\text{ nm}$) $\\rightarrow$ Slight green reflection peak.

Healthy crops appear bright crimson/red; senesced, diseased, or bare soil areas appear dull pink, cyan, or brown."""))

cells.append(nbf.v4.new_code_cell("""def norm(band):
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    return np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)

# Generate CIR Composites
ip_cir = np.dstack([norm(ip_data[:, :, 40]), norm(ip_data[:, :, 25]), norm(ip_data[:, :, 15])])
sal_cir = np.dstack([norm(sal_data[:, :, 40]), norm(sal_data[:, :, 25]), norm(sal_data[:, :, 15])])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].imshow(ip_cir)
axes[0, 0].set_title("Indian Pines: CIR Composite (NIR / Red / Green)", fontweight="bold")
axes[0, 0].axis("off")

im0 = axes[0, 1].imshow(ip_gt, cmap="tab20", vmin=0, vmax=16)
axes[0, 1].set_title("Indian Pines: Ground Truth Class Map", fontweight="bold")
axes[0, 1].axis("off")

axes[1, 0].imshow(sal_cir)
axes[1, 0].set_title("Salinas Valley: CIR Composite (3.7m GSD)", fontweight="bold")
axes[1, 0].axis("off")

im1 = axes[1, 1].imshow(sal_gt, cmap="tab20", vmin=0, vmax=16)
axes[1, 1].set_title("Salinas Valley: Ground Truth Class Map", fontweight="bold")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()"""))

# Section 7: Key Conclusions
cells.append(nbf.v4.new_markdown_cell("""---
### 8. Key Findings & Strategic ML Insights (Day 1)

1. **Dimensionality Confirmation:**
   - **Indian Pines:** $145 \\times 145 \\times 200$ (clean bands, 220 raw). Calibrated reflectance values range from $955$ to $9604$.
   - **Salinas Valley:** $512 \\times 217 \\times 204$ (clean bands, 224 raw). High spatial detail ($3.7\\text{ m}$).
2. **Spectral Coverage:**
   - Continuous contiguous coverage from $400\\text{ nm}$ to $2450\\text{ nm}$ at $10\\text{ nm}$ FWHM.
   - 20 water-vapor contaminated bands confirmed and isolated (Bands 104–108, 150–163, 220).
3. **Class Imbalance Challenge:**
   - In Indian Pines, largest class (`Soybean-mintill`, 2,455 pixels) vs smallest class (`Oats`, 20 pixels) represents a $> 120:1$ imbalance ratio.
   - For downstream 3D-CNN / ViT model training, stratified spatial patch sampling or weighted loss formulations will be mandatory.
4. **Relevance to Disease Forecasting:**
   - While primary labels are crop categories and tillage, the presence of senescence (`Corn_senesced_green_weeds`), weed competition, and multi-week phenology (`Lettuce_romaine 4-7wk`) provides the exact biochemical spectral signatures (chlorophyll loss, cell wall breakdown) essential for modeling pre-visual crop stress.
5. **Next Day Objective:**
   - **Day 2:** In-depth loading and structural inspection of data dimensions, metadata header parsing, exact channel calibrations, and spatial patch slicing verification."""))

nb.cells = cells

notebook_path = os.path.join(BASE_DIR, "notebooks", "01_dataset_exploration.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {notebook_path} with {len(cells)} cells.")
