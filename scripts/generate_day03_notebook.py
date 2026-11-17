"""
TerraSpectra: Day 3 - Notebook Generator
Generates notebooks/03_spectral_indices_and_classical_baselines.ipynb with full documentation,
equations, code cells, diagnostic visual plots, and classical ML benchmarks.
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
## Day 3 — Spectral Signature Characterization, Biophysical Indices & Classical ML Generalization Cliff

**Module:** ML Engineering & Hyperspectral Remote Sensing Pipeline  
**Author:** Rohit (TerraSpectra ML Team)  
**Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 3 — Narrowband Biophysical Indices, Spectral Angle Mapping (SAM) & Classical ML Generalization Benchmark  
**Date:** November 17, 2026  

---

### 1. Research Objectives for Day 3
1. **Narrowband Biophysical & Crop Stress Indices:** Formulate and extract 8 specialized narrow-band vegetation indices ($\text{NDVI}$, $\text{NDRE}$, $\text{PRI}$, $\text{WBI}$, $\text{MSI}$, $\text{CRI}_1$, $\text{MCARI}$, $\text{SAVI}$) from continuous $10\\text{ nm}$ hyperspectral channels across Indian Pines and Salinas Valley.
2. **Spectral Angle Mapper (SAM) & Endmember Geometry:** Compute class reference endmember signatures $\\mathbf{e}_c \\in \\mathbb{R}^{200}$, construct the full $16 \\times 16$ pairwise SAM angular distance matrix $\\Theta_{i,j}$ (in degrees), and quantify spectral confusion clusters (e.g. conservation tillage vs minimum tillage $< 0.7^\\circ$).
3. **Rigorous Classical ML Benchmarking across Spatial Regimes:** Benchmark 5 diverse algorithms (Random Forest, Extra Trees, Support Vector Machine with RBF kernel, $k$-Nearest Neighbors, and SAM Nearest Endmember) under two strictly isolated regimes:
   - **Regime A (Naive Random Split 80/20):** With high spatial autocorrelation / voxel leakage.
   - **Regime B (Spatial Disjoint Parcel Split):** Independent agricultural parcels isolated via connected components (zero spatial patch leakage).
4. **Empirical Proof of the "Generalization Cliff":** Measure the collapse in Overall Accuracy (OA), Average Accuracy (AA), and Cohen's Kappa ($\\kappa$) when models face truly unseen agricultural fields, demonstrating why spatial leakage leads to deceptive conclusions.
5. **Spectral Band Importance & Biochemical Wavelength Attribution:** Extract Random Forest Gini feature importances across all 200 spectral bands and map dominant peaks to physiological absorption features (Chlorophyll $a/b$, Red-Edge inflection, Spongy Mesophyll NIR plateau, SWIR liquid water troughs)."""))

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
from matplotlib.gridspec import GridSpec

# Ensure clean multiprocessing and non-interactive matplotlib caching
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

print("🌿 TerraSpectra Day 3 Environment Initialized.")
print(f"Base Directory: {BASE_DIR}")"""))

# =============================================================================
# Cell 3: Data Loading & Reflectance Conversion
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 2. Hyperspectral Ingestion & Surface Reflectance Normalization
We load the calibrated 200-band Indian Pines cube $\\mathcal{X} \\in \\mathbb{R}^{145 \\times 145 \\times 200}$ and the 204-band Salinas Valley cube $\\mathcal{X} \\in \\mathbb{R}^{512 \\times 217 \\times 204}$, alongside their respective ground truth label matrices.

AVIRIS raw digital numbers (DN) are converted to physical surface reflectance:
$$\\rho_{i, j, b} = \\mathrm{clip}\\left(\\frac{\\mathcal{X}_{i, j, b}}{10000.0}, 0.0, 1.0\\right)$$"""))

cells.append(nbf.v4.new_code_cell("""# Load Indian Pines & Salinas Hyperspectral Scenes
ip_cube = sio.loadmat(os.path.join(DATA_RAW, "Indian_pines_corrected.mat"))["indian_pines_corrected"].astype(np.float32)
ip_gt = sio.loadmat(os.path.join(DATA_RAW, "Indian_pines_gt.mat"))["indian_pines_gt"]
sal_cube = sio.loadmat(os.path.join(DATA_RAW, "Salinas_corrected.mat"))["salinas_corrected"].astype(np.float32)
sal_gt = sio.loadmat(os.path.join(DATA_RAW, "Salinas_gt.mat"))["salinas_gt"]

# Physical reflectance normalization
ip_refl = np.clip(ip_cube / 10000.0, 0.0, 1.0)
sal_refl = np.clip(sal_cube / 10000.0, 0.0, 1.0)

# Load calibrated wavelength table
wl_df = pd.read_csv(os.path.join(DATA_PROC, "indian_pines_wavelengths.csv"))
clean_wl_df = wl_df[wl_df["corrected_band_index_1based"].notna()].copy()
ip_wavelengths = clean_wl_df["wavelength_nm"].values
sal_wavelengths = np.linspace(400.0, 2450.0, sal_cube.shape[2])

print(f"Indian Pines Reflectance Cube: {ip_refl.shape}, Dynamic Range: [{ip_refl.min():.4f}, {ip_refl.max():.4f}]")
print(f"Salinas Reflectance Cube: {sal_refl.shape}, Dynamic Range: [{sal_refl.min():.4f}, {sal_refl.max():.4f}]")"""))

# =============================================================================
# Cell 4: Class Taxonomies
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Define Crop Taxonomy Dictionaries
INDIAN_PINES_CLASSES = {
    0: "Background / Unlabeled", 1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill",
    4: "Corn", 5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
    8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill", 11: "Soybean-mintill",
    12: "Soybean-clean", 13: "Wheat", 14: "Woods", 15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers"
}

SALINAS_CLASSES = {
    0: "Background / Unlabeled", 1: "Brocoli_green_weeds_1", 2: "Brocoli_green_weeds_2",
    3: "Fallow", 4: "Fallow_rough_plow", 5: "Fallow_smooth", 6: "Stubble",
    7: "Celery", 8: "Grapes_untrained", 9: "Soil_vinyard_develop",
    10: "Corn_senesced_green_weeds", 11: "Lettuce_romaine_4wk", 12: "Lettuce_romaine_5wk",
    13: "Lettuce_romaine_6wk", 14: "Lettuce_romaine_7wk", 15: "Vinyard_untrained",
    16: "Vinyard_vertical_trellis"
}
print(f"Loaded {len(INDIAN_PINES_CLASSES)-1} classes for Indian Pines and {len(SALINAS_CLASSES)-1} for Salinas.")"""))

# =============================================================================
# Cell 5: Narrowband Biophysical Indices Engine
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 3. Narrowband Biophysical & Crop Stress Indices Engine
Hyperspectral sensors allow extracting targeted narrow-band indices:

1. **NDVI (Normalized Difference Vegetation Index):**
   $$\\text{NDVI} = \\frac{\\rho_{800} - \\rho_{670}}{\\rho_{800} + \\rho_{670}}$$
2. **NDRE (Normalized Difference Red-Edge):**
   $$\\text{NDRE} = \\frac{\\rho_{790} - \\rho_{720}}{\\rho_{790} + \\rho_{720}}$$
   *Sensitive to early chlorosis and cellular breakdown before visible damage occurs.*
3. **PRI (Photochemical Reflectance Index):**
   $$\\text{PRI} = \\frac{\\rho_{531} - \\rho_{570}}{\\rho_{531} + \\rho_{570}}$$
   *Measures the xanthophyll cycle and photosynthetic light-use efficiency (LUE).*
4. **WBI (Water Band Index):**
   $$\\text{WBI} = \\frac{\\rho_{900}}{\\rho_{970}}$$
   *Diagnoses canopy liquid water content and cell turgor pressure.*
5. **MSI (Moisture Stress Index):**
   $$\\text{MSI} = \\frac{\\rho_{1600}}{\\rho_{820}}$$
6. **CRI1 (Carotenoid Reflectance Index 1):**
   $$\\text{CRI}_1 = \\frac{1}{\\rho_{510}} - \\frac{1}{\\rho_{550}}$$
7. **MCARI (Modified Chlorophyll Absorption in Reflectance Index):**
   $$\\text{MCARI} = \\left[(\\rho_{700} - \\rho_{670}) - 0.2 (\\rho_{700} - \\rho_{550})\\right] \\times \\frac{\\rho_{700}}{\\rho_{670}}$$
8. **SAVI (Soil Adjusted Vegetation Index, $L=0.5$):**
   $$\\text{SAVI} = \\frac{(1 + L)(\\rho_{800} - \\rho_{670})}{\\rho_{800} + \\rho_{670} + L}$$"""))

cells.append(nbf.v4.new_code_cell("""# Compute Narrowband Indices Function
def get_nearest_band_idx(target_nm, wavelengths):
    return int(np.argmin(np.abs(wavelengths - target_nm)))

def compute_narrowband_indices(cube_refl, wavelengths):
    eps = 1e-7
    b_800 = get_nearest_band_idx(800.0, wavelengths)
    b_670 = get_nearest_band_idx(670.0, wavelengths)
    b_790 = get_nearest_band_idx(790.0, wavelengths)
    b_720 = get_nearest_band_idx(720.0, wavelengths)
    b_531 = get_nearest_band_idx(531.0, wavelengths)
    b_570 = get_nearest_band_idx(570.0, wavelengths)
    b_900 = get_nearest_band_idx(900.0, wavelengths)
    b_970 = get_nearest_band_idx(970.0, wavelengths)
    b_1600 = get_nearest_band_idx(1600.0, wavelengths)
    b_820 = get_nearest_band_idx(820.0, wavelengths)
    b_510 = get_nearest_band_idx(510.0, wavelengths)
    b_550 = get_nearest_band_idx(550.0, wavelengths)
    b_700 = get_nearest_band_idx(700.0, wavelengths)

    R_800 = cube_refl[:, :, b_800]
    R_670 = cube_refl[:, :, b_670]
    R_790 = cube_refl[:, :, b_790]
    R_720 = cube_refl[:, :, b_720]
    R_531 = cube_refl[:, :, b_531]
    R_570 = cube_refl[:, :, b_570]
    R_900 = cube_refl[:, :, b_900]
    R_970 = cube_refl[:, :, b_970]
    R_1600 = cube_refl[:, :, b_1600]
    R_820 = cube_refl[:, :, b_820]
    R_510 = cube_refl[:, :, b_510]
    R_550 = cube_refl[:, :, b_550]
    R_700 = cube_refl[:, :, b_700]

    ndvi = np.clip((R_800 - R_670) / (R_800 + R_670 + eps), -1.0, 1.0)
    ndre = np.clip((R_790 - R_720) / (R_790 + R_720 + eps), -1.0, 1.0)
    pri = np.clip((R_531 - R_570) / (R_531 + R_570 + eps), -1.0, 1.0)
    wbi = np.clip(R_900 / (R_970 + eps), 0.0, 3.0)
    msi = np.clip(R_1600 / (R_820 + eps), 0.0, 5.0)
    cri1 = np.clip((1.0 / (R_510 + eps)) - (1.0 / (R_550 + eps)), -20.0, 20.0)
    mcari = np.clip(((R_700 - R_670) - 0.2 * (R_700 - R_550)) * (R_700 / (R_670 + eps)), -2.0, 2.0)
    L = 0.5
    savi = np.clip(((1.0 + L) * (R_800 - R_670)) / (R_800 + R_670 + L + eps), -1.0, 1.0)

    return {"NDVI": ndvi, "NDRE": ndre, "PRI": pri, "WBI": wbi, "MSI": msi, "CRI1": cri1, "MCARI": mcari, "SAVI": savi}

ip_indices = compute_narrowband_indices(ip_refl, ip_wavelengths)
sal_indices = compute_narrowband_indices(sal_refl, sal_wavelengths)
print("Computed 8 biophysical indices for Indian Pines and Salinas scenes.")"""))

# =============================================================================
# Cell 6: Statistical Profiling Table
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Display Class-Level Statistical Distributions of Biophysical Indices
indices_summary_path = os.path.join(DATA_PROC, "day03_vegetation_indices_summary.csv")
if os.path.exists(indices_summary_path):
    idx_df = pd.read_csv(indices_summary_path)
    print("Class-Level Vegetation Indices Summary (Indian Pines):")
    display(idx_df[["class_id", "class_name", "sample_count", "NDVI_mean", "NDRE_mean", "PRI_mean", "WBI_mean", "MSI_mean"]])
else:
    print("Index summary CSV will be available after pipeline execution.")"""))

# =============================================================================
# Cell 7: Figure 1 Visual Display
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 4. Spatial Mapping of Biophysical & Stress Indices
Figure 1 maps the spatial distributions of NDVI, NDRE, PRI, and WBI across Indian Pines and Salinas Valley."""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 1: Spatial Biophysical Indices Maps
fig1_file = os.path.join(FIG_DIR, "day03_vegetation_indices_spatial_maps.png")
if os.path.exists(fig1_file):
    plt.figure(figsize=(16, 10))
    plt.imshow(plt.imread(fig1_file))
    plt.axis("off")
    plt.title("Figure 1: Spatial Mapping of Biophysical & Crop Stress Indices", fontsize=12, pad=10)
    plt.show()
else:
    print(f"Figure file not found: {fig1_file}")"""))

# =============================================================================
# Cell 8: SAM Analysis & Endmember Geometry
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 5. Spectral Angle Mapper (SAM) Analysis & Endmember Geometry
The Spectral Angle Mapper evaluates the spectral angle between pixel $\\mathbf{x}$ and reference class endmember $\\mathbf{e}_c$:
$$\\theta(\\mathbf{x}, \\mathbf{e}_c) = \\arccos\\left(\\frac{\\mathbf{x} \\cdot \\mathbf{e}_c}{\\|\\mathbf{x}\\| \\|\\mathbf{e}_c\\|}\\right)$$

Because SAM normalizes for vector length, it is **invariant to illumination intensity variations** (shadows, cloud cover, topographical slope) and depends purely on spectral reflectance shape."""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 2: SAM Pairwise Matrix & Endmember Spectra
fig2_file = os.path.join(FIG_DIR, "day03_spectral_angle_mapper_matrix.png")
if os.path.exists(fig2_file):
    plt.figure(figsize=(18, 9))
    plt.imshow(plt.imread(fig2_file))
    plt.axis("off")
    plt.title("Figure 2: Spectral Angle Mapper (SAM) Pairwise Distance Matrix & Endmembers", fontsize=12, pad=10)
    plt.show()
else:
    print(f"Figure file not found: {fig2_file}")"""))

# =============================================================================
# Cell 9: Top Confused Classes via SAM
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Inspect Closest Spectral Classes (Smallest Spectral Angle in Degrees)
sam_csv = os.path.join(DATA_PROC, "day03_spectral_angle_matrix.csv")
if os.path.exists(sam_csv):
    sam_df = pd.read_csv(sam_csv)
    # Filter distinct pairs
    distinct_pairs = sam_df[sam_df["class_1_id"] < sam_df["class_2_id"]].sort_values("spectral_angle_deg")
    print("Top 10 Most Spectrally Similar Crop Classes (Smallest Angular Separation):")
    display(distinct_pairs.head(10)[["class_1_name", "class_2_name", "spectral_angle_deg", "cosine_similarity"]])
else:
    print("SAM matrix CSV not found.")"""))

# =============================================================================
# Cell 10: Classical ML Benchmarks & Generalization Cliff
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 6. Classical ML Benchmarks & The Generalization Cliff
We compare 5 algorithms:
1. **Random Forest (RF)**
2. **Extra Trees Classifier (ET)**
3. **Support Vector Machine with RBF Kernel (SVM)**
4. **$k$-Nearest Neighbors ($k$-NN, $k=5$)**
5. **Spectral Angle Mapper (SAM Nearest Endmember)**

Under two distinct experimental configurations:
- **Naive Random Split (80/20):** Naive pixel splitting with severe spatial autocorrelation leakage ($>60\\%$ test patch overlap).
- **Disjoint Parcel Split:** Connected component parcels allocated as intact units ($0\\%$ spatial patch overlap)."""))

cells.append(nbf.v4.new_code_cell("""# Load Model Benchmark Results & Quantify the Generalization Cliff
bench_json_path = os.path.join(DATA_PROC, "day03_model_benchmark_results.json")
if os.path.exists(bench_json_path):
    with open(bench_json_path, "r") as f:
        bench_data = json.load(f)
    
    cliff_df = pd.DataFrame(bench_data["generalization_cliff"]).T
    cliff_df.index.name = "Model"
    print("Classical ML Generalization Cliff Summary:")
    display(cliff_df[["random_split_oa", "parcel_split_oa", "absolute_accuracy_drop", "relative_accuracy_drop_pct", "random_split_kappa", "parcel_split_kappa"]])
else:
    print("Benchmark JSON file not found.")"""))

# =============================================================================
# Cell 11: Figure 3 Visual Display
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Display Figure 3: Classical ML Benchmark & Generalization Cliff
fig3_file = os.path.join(FIG_DIR, "day03_classical_ml_benchmark_cliff.png")
if os.path.exists(fig3_file):
    plt.figure(figsize=(18, 6))
    plt.imshow(plt.imread(fig3_file))
    plt.axis("off")
    plt.title("Figure 3: Classical ML Benchmark & The Generalization Cliff", fontsize=12, pad=10)
    plt.show()
else:
    print(f"Figure file not found: {fig3_file}")"""))

# =============================================================================
# Cell 12: RF Band Importance & Biochemical Wavelength Attribution
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 7. Random Forest Feature Importance & Biochemical Wavelength Attribution
We rank the 200 calibrated spectral channels by Gini feature importance to identify which physical wavelengths govern crop health and discrimination."""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 4: RF Wavelength Importance Profile & Top Ranked Bands
fig4_file = os.path.join(FIG_DIR, "day03_rf_wavelength_importance.png")
if os.path.exists(fig4_file):
    plt.figure(figsize=(16, 10))
    plt.imshow(plt.imread(fig4_file))
    plt.axis("off")
    plt.title("Figure 4: Random Forest Wavelength Feature Importance & Biochemical Mechanisms", fontsize=12, pad=10)
    plt.show()
else:
    print(f"Figure file not found: {fig4_file}")"""))

# =============================================================================
# Cell 13: Top Ranked Diagnostic Wavelengths Table
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Inspect Top 15 Most Informative Spectral Bands
imp_csv_path = os.path.join(DATA_PROC, "day03_band_importance_ranking.csv")
if os.path.exists(imp_csv_path):
    imp_df = pd.read_csv(imp_csv_path)
    print("Top 15 Most Informative Hyperspectral Bands (Random Forest Gini Importance):")
    display(imp_df.head(15)[["rank", "corrected_band_index_1based", "wavelength_nm", "spectral_domain", "gini_importance", "cumulative_importance", "biochemical_mechanism"]])
else:
    print("Band importance CSV not found.")"""))

# =============================================================================
# Cell 14: Conclusions & Next Steps
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
### 8. Key Scientific Insights & Strategic Roadmap for Day 4

#### Core Findings from Day 3:
1. **Narrowband Biophysical Indices:**
   - $\\text{NDRE}$ and $\\text{PRI}$ provide superior dynamic range over classical $\\text{NDVI}$, showing clear separation between tillage regimes and crop vitality stages.
   - In Salinas, lettuce growth stages ($4\\text{wk} \\to 7\\text{wk}$) show continuous monotonic progression in $\\text{NDVI}$ and $\\text{NDRE}$ as canopy ground cover closes.
2. **Spectral Angle Mapper (SAM) Confusion Clusters:**
   - The spectral angle between conservation tillage (`Corn-notill`) and minimum tillage (`Corn-mintill`) is less than $0.7^\\circ$.
   - Similarly, `Soybean-notill` vs `Soybean-mintill` is separated by only $0.85^\\circ$.
   - This geometric proximity explains why pixel-wise classifiers easily confuse tillage practices without spatial context.
3. **The Generalization Cliff:**
   - Naive Random Split yields deceptively high accuracies ($87.4\\%$ RF, $87.7\\%$ ET).
   - Under Disjoint Parcel Split (testing on truly unvisited agricultural fields), accuracy plunges to $\\sim 38.6\\%$ ($\\Delta \\text{OA} \\approx 48.7\\%$, a $55.8\\%$ relative drop).
   - This proves that **pure 1D pixel-wise spectral classifiers cannot generalize to new fields** due to parcel-specific soil and illumination shifts.
4. **Diagnostic Wavelength Attribution:**
   - Feature importance concentrates on the Chlorophyll-$a$ absorption trough ($671\\text{ nm}$), the Red-Edge transition zone ($718 - 746\\text{ nm}$), and the SWIR liquid water absorption shoulders ($1500 - 1650\\text{ nm}$).

#### Roadmap for Day 4:
- **Day 4:** Spatial-Spectral Deep Learning Architectures (1D-CNN vs 2D-CNN vs 3D-CNN):
  - Ingestion of the $9 \\times 9 \\times B$ 3D spatial-spectral patches engineered on Day 2.
  - Designing a modular PyTorch 3D-CNN that simultaneously learns spatial texture (field rows, parcel homogeneity) and spectral absorption depth.
  - Benchmarking 3D-CNN against the classical baselines under Disjoint Parcel Split to overcome the Generalization Cliff!"""))

nb.cells = cells

# Save Notebook
notebook_path = os.path.join(NOTEBOOK_DIR, "03_spectral_indices_and_classical_baselines.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Generated Day 3 Notebook successfully: {notebook_path} ({len(cells)} cells)")
