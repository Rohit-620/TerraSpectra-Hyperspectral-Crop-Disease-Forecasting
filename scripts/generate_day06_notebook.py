"""
TerraSpectra: Day 6 - Notebook Generator
Generates notebooks/06_crop_disease_detection_and_forecasting.ipynb with complete documentation,
mathematical formulations, biophysical index equations, RX anomaly spotting, PyTorch models,
epidemiological simulation, and interactive visualizations.
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
cells.append(nbf.v4.new_markdown_cell(r"""# 🌿 TerraSpectra: Intelligent Crop Disease Forecasting
## Day 6 — Intelligent Crop Disease Identification, Biophysical Stress Profiling, Early Pre-Visual Symptom Detection, & Spatiotemporal Epidemic Risk Forecasting

**Module:** Agricultural Pathology & Epidemiological Spatiotemporal Forecasting  
**Author:** Rohit (TerraSpectra ML Team)  
**Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 6 — Disease Identification, Early Pre-Visual Detection, and Predictive Spatiotemporal Risk Forecasting  
**Date:** November 22, 2026  

---

### 1. Research Scope & Objectives for Day 6
Following the foundation models, self-supervised pretraining (HS-MAE), and domain adaptation established on **Day 5**, **Day 6** addresses the core mission of TerraSpectra: **Intelligent Crop Disease Detection, Early Pre-Visual Symptom Identification, and Spatiotemporal Epidemiological Risk Forecasting**.

1. **Biophysical Pathophysiological Profiling & Disease Severity Index (DSI):**
   - Formulate a continuous biophysical metric integrating chlorophyll degradation (NDRE), senescence/carotenoid accumulation (PSRI), photosystem light-use efficiency shutdown (PRI), and cell water desiccation (NDWI).
   - Stratify agricultural canopy into 4 clinical epidemiological disease stages:
     - **Stage 0:** Healthy / Asymptomatic (Canopy core, low stress)
     - **Stage 1:** Mild / Pre-Visual Latent Stress (Pre-symptomatic chloroplast disruption)
     - **Stage 2:** Moderate Chlorotic Infection (Active chlorosis & carotenoid buildup)
     - **Stage 3:** Severe Necrotic Blight (Cell wall lysis, tissue desiccation)

2. **Early Pre-Visual Symptom Detection Engine:**
   - **Lagrangian Red Edge Inflection Point (REIP):** Quantify the diagnostic "Blue Shift" ($\approx 10-22\text{ nm}$) where peak first-derivative $dR/d\lambda$ shifts toward shorter wavelengths prior to macroscopic RGB symptoms.
   - **Reed-Xiaoli (RX) Subspace Anomaly Detection:** Compute within-parcel Mahalanobis distance relative to background canopy covariance to pinpoint sub-pixel latent infection foci.
   - **ROC-AUC Sensitivity Benchmark:** Contrast early pre-visual detection sensitivity between RGB indices (GLI), broadband NDVI, Red Edge NDRE, and Hyperspectral RX anomaly detectors.

3. **Multi-Stage Disease Severity Classification Benchmark:**
   - Benchmark three distinct architectures under both **Regime A (Random Split)** and **Regime B (Disjoint Parcel Split)**:
     - **Physics-Guided Random Forest (RF-Phys)**
     - **Spatial-Spectral HybridSN (3D-2D CNN)**
     - **Vision Transformer (SpectralFormer-Disease)**

4. **Spatiotemporal Disease Progression & Environmental Risk Forecasting:**
   - Implement a spatial reaction-diffusion epidemiological model (SEIR-Grid) driven by microclimate disease favorability $\Psi_{\text{env}}(t)$ (temperature, relative humidity, leaf wetness duration).
   - Generate multi-horizon spatial forecasts ($t = 0, 3, 7, 14$ days) and translate predictions into actionable farm management prescription maps:
     - **Green (Normal):** Low risk; standard field routine.
     - **Yellow (Advisory):** Latent stress; targeted field scouting required.
     - **Orange (Warning):** Active infection; precision fungicide application.
     - **Red (Critical):** Rapid epidemic expansion; quarantine barrier spray."""))

# =============================================================================
# Cell 2: Imports & Environment Setup
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

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

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
cells.append(nbf.v4.new_markdown_cell(r"""### 2. Mathematical Formulations of Crop Disease Detection & Forecasting

#### 2.1. Biophysical Disease Severity Index (DSI)
Foliar plant pathology manifests through four sequential physiological breakdowns:
1. **Photochemical Inhibition (PRI):** Xanthophyll pigment conversion ($531\text{ nm}$ vs $570\text{ nm}$):
   $$\text{PRI} = \frac{R_{531} - R_{570}}{R_{531} + R_{570}}$$
2. **Red Edge Chlorophyll Degradation (NDRE):** Loss of mesophyll absorption ($790\text{ nm}$ vs $720\text{ nm}$):
   $$\text{NDRE} = \frac{R_{790} - R_{720}}{R_{790} + R_{720}}$$
3. **Senescence & Carotenoid Accumulation (PSRI):** Accelerated foliar aging:
   $$\text{PSRI} = \frac{R_{680} - R_{500}}{R_{750}}$$
4. **Cellular Water Desiccation (NDWI):** Mesophyll cell wall rupture and water loss:
   $$\text{NDWI} = \frac{R_{860} - R_{1240}}{R_{860} + R_{1240}}$$

The continuous biophysical metric synthesizes these normalized components:
$$\text{DSI}(\mathbf{x}) = 0.35 \cdot \widetilde{\text{PSRI}}(\mathbf{x}) + 0.30 \cdot (1 - \widetilde{\text{NDRE}}(\mathbf{x})) + 0.20 \cdot (1 - \widetilde{\text{PRI}}(\mathbf{x})) + 0.15 \cdot (1 - \widetilde{\text{NDWI}}(\mathbf{x}))$$

#### 2.2. Lagrangian Red Edge Inflection Point (REIP) & Spectral Derivative
The Red Edge inflection point marks the maximum first derivative of canopy reflectance ($dR/d\lambda$). Following Guyot & Baret:
$$\text{REIP} = 700 + 40 \left( \frac{0.5(R_{670} + R_{780}) - R_{700}}{R_{740} - R_{700}} \right)$$
Under fungal pathogen colonization, early destruction of chloroplasts induces a **Blue Shift** ($\Delta \lambda \approx 10-25\text{ nm}$) toward shorter wavelengths days before symptoms become visible in RGB bands.

#### 2.3. Reed-Xiaoli (RX) Subspace Anomaly Detection
Within each agricultural field parcel $\mathcal{P}_k$, the background mean $\boldsymbol{\mu}_k$ and covariance matrix $\boldsymbol{\Sigma}_k$ are estimated. Early sub-pixel infection foci are identified via Mahalanobis anomaly distance:
$$r_{\text{RX}}(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu}_k)^\top (\boldsymbol{\Sigma}_k + \gamma \mathbf{I})^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)$$

#### 2.4. Spatiotemporal Reaction-Diffusion Epidemiological Forecasting
Spatiotemporal disease propagation across agricultural fields is governed by the continuous PDE:
$$\frac{\partial D(\mathbf{s}, t)}{\partial t} = \mathcal{D}_{\text{spatial}} \nabla^2 D(\mathbf{s}, t) + r \cdot D(\mathbf{s}, t) \left(1 - \frac{D(\mathbf{s}, t)}{K(\mathbf{s})}\right) \cdot \Psi_{\text{env}}(t) - \mu D(\mathbf{s}, t)$$
where $\Psi_{\text{env}}(t) = f_T(T_t) \cdot f_{RH}(RH_t) \cdot f_{\text{wet}}(LWD_t)$ integrates the classical plant disease triangle (temperature response, relative humidity saturation, and leaf wetness duration)."""))

# =============================================================================
# Cell 4: Ingestion & Feature Engineering
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Ingest Hyperspectral Scene & Compute Biophysical Features
ip_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat"))
gt_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat"))
cube_raw = [ip_mat[k] for k in ip_mat if not k.startswith("__")][0]
gt = [gt_mat[k] for k in gt_mat if not k.startswith("__")][0]

df_wl = pd.read_csv(os.path.join(PROC_DATA_DIR, "indian_pines_wavelengths.csv"))
wavelengths = df_wl[df_wl["status"] == "Retained (Clean Signal)"]["wavelength_nm"].values
cube_refl = cube_raw.astype(np.float32) / 10000.0
H, W, B = cube_refl.shape

print(f"Scene ingested: {cube_refl.shape} with {len(wavelengths)} spectral channels.")"""))

# =============================================================================
# Cell 5: Biophysical Indices & Continuous DSI
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Compute Biophysical Indices & Red Edge Derivatives
eps = 1e-7
def get_idx(wl):
    return int(np.argmin(np.abs(wavelengths - wl)))

r450, r500, r531 = cube_refl[:, :, get_idx(450)], cube_refl[:, :, get_idx(500)], cube_refl[:, :, get_idx(531)]
r550, r570, r670 = cube_refl[:, :, get_idx(550)], cube_refl[:, :, get_idx(570)], cube_refl[:, :, get_idx(670)]
r680, r700, r720 = cube_refl[:, :, get_idx(680)], cube_refl[:, :, get_idx(700)], cube_refl[:, :, get_idx(720)]
r740, r750, r780 = cube_refl[:, :, get_idx(740)], cube_refl[:, :, get_idx(750)], cube_refl[:, :, get_idx(780)]
r790, r800, r860 = cube_refl[:, :, get_idx(790)], cube_refl[:, :, get_idx(800)], cube_refl[:, :, get_idx(860)]
r900, r970, r1240 = cube_refl[:, :, get_idx(900)], cube_refl[:, :, get_idx(970)], cube_refl[:, :, get_idx(1240)]

ndvi = (r800 - r670) / (r800 + r670 + eps)
ndre = (r790 - r720) / (r790 + r720 + eps)
pri = (r531 - r570) / (r531 + r570 + eps)
psri = (r680 - r500) / (r750 + eps)
wbi = r900 / (r970 + eps)
ndwi = (r860 - r1240) / (r860 + r1240 + eps)
gli = (2.0 * r550 - r670 - r450) / (2.0 * r550 + r670 + r450 + eps)
reip = 700.0 + 40.0 * np.clip((0.5 * (r670 + r780) - r700) / (r740 - r700 + eps), 0.0, 1.0)

# Filter to crop parcels (exclude background 0, buildings 15, towers 16)
veg_mask = (gt > 0) & (gt != 15) & (gt != 16)

def scale_veg(arr):
    vals = arr[veg_mask]
    p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
    return np.clip((arr - p5) / (p95 - p5 + 1e-7), 0.0, 1.0)

dsi_raw = (
    0.35 * scale_veg(psri) +
    0.30 * scale_veg(1.0 - ndre) +
    0.20 * scale_veg(1.0 - pri) +
    0.15 * scale_veg(1.0 - ndwi)
)

dv = dsi_raw[veg_mask]
p_min, p_max = float(np.min(dv)), float(np.max(dv))
dsi = np.zeros_like(dsi_raw)
dsi[veg_mask] = np.clip((dsi_raw[veg_mask] - p_min) / (p_max - p_min + 1e-7), 0.0, 1.0)

q25, q50, q75 = np.percentile(dv, [25, 50, 75])
stages = np.zeros(dsi.shape, dtype=np.int64)
stages[veg_mask & (dsi_raw <= q25)] = 0
stages[veg_mask & (dsi_raw > q25) & (dsi_raw <= q50)] = 1
stages[veg_mask & (dsi_raw > q50) & (dsi_raw <= q75)] = 2
stages[veg_mask & (dsi_raw > q75)] = 3

stage_names = ["Stage 0: Healthy", "Stage 1: Pre-Visual", "Stage 2: Moderate", "Stage 3: Severe"]
for i in range(4):
    cnt = np.sum(veg_mask & (stages == i))
    print(f"{stage_names[i]}: {cnt} pixels ({cnt / np.sum(veg_mask) * 100:.2f}%)")"""))

# =============================================================================
# Cell 6: Reed-Xiaoli Anomaly Detection & ROC Analysis
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Reed-Xiaoli (RX) Subspace Anomaly Scoring & Pre-Visual ROC-AUC
pca = PCA(n_components=30, random_state=42)
cube_pca = pca.fit_transform(cube_refl.reshape(-1, B)).reshape(H, W, 30)
parcel_labels, num_parcels = ndi.label(veg_mask)

rx_anomaly_map = np.zeros((H, W), dtype=np.float32)
for p_id in range(1, num_parcels + 1):
    p_mask = (parcel_labels == p_id)
    if np.sum(p_mask) < 35:
        continue
    pix = cube_pca[p_mask]
    diff = pix - np.mean(pix, axis=0)
    cov = (diff.T @ diff) / (len(pix) - 1 + 1e-7) + 1e-4 * np.eye(30)
    rx_anomaly_map[p_mask] = np.sum((diff @ np.linalg.pinv(cov)) * diff, axis=1)

valid_rx = rx_anomaly_map > 0
if np.sum(valid_rx) > 0:
    rx_anomaly_map = np.clip(rx_anomaly_map / (np.percentile(rx_anomaly_map[valid_rx], 98) + 1e-7), 0.0, 1.0)

# Evaluate Early Detection ROC on Stage 0 vs Stage 1
eval_mask = veg_mask & (stages <= 1)
y_eval = stages[eval_mask]

reip_scores = 740.0 - reip[eval_mask]
reip_norm = (reip_scores - np.min(reip_scores)) / (np.ptp(reip_scores) + 1e-7)
rx_norm = rx_anomaly_map[eval_mask]

detectors = {
    "RGB (GLI)": 1.0 - gli[eval_mask],
    "Broadband NDVI": 1.0 - ndvi[eval_mask],
    "Narrowband NDRE": 1.0 - ndre[eval_mask],
    "REIP Blue Shift": reip_scores,
    "Hyperspectral Composite (REIP+RX)": 0.70 * reip_norm + 0.30 * rx_norm
}

print("=== Early Pre-Visual Detection Sensitivity (ROC-AUC) ===")
for name, scores in detectors.items():
    fpr, tpr, _ = roc_curve(y_eval, scores)
    print(f"  {name:35s} ROC-AUC: {auc(fpr, tpr):.4f}")"""))

# =============================================================================
# Cell 7: Display Publication Visualizations
# =============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""### 3. Publication Figures & Analytical Results

The complete research suite generated 4 multi-panel publication figures in `reports/figures/`:
1. **Biophysical Disease Severity Profiling & Spectral Signatures** (`day06_biophysical_disease_severity_profiling.png`)
2. **Early Pre-Visual Detection & ROC-AUC Comparison** (`day06_early_previsual_detection_and_roc.png`)
3. **Disease Severity Classification Benchmark & Diagnostic Biomarkers** (`day06_disease_classification_benchmark.png`)
4. **Spatiotemporal Disease Progression & Agricultural Prescription Maps** (`day06_spatiotemporal_epidemiological_forecasting.png`)"""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 1: Biophysical Disease Severity Profiling
from IPython.display import Image, display
display(Image(filename=os.path.join(FIG_DIR, "day06_biophysical_disease_severity_profiling.png")))"""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 2: Early Pre-Visual Detection & ROC-AUC
display(Image(filename=os.path.join(FIG_DIR, "day06_early_previsual_detection_and_roc.png")))"""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 3: Disease Classification Benchmark
display(Image(filename=os.path.join(FIG_DIR, "day06_disease_classification_benchmark.png")))"""))

cells.append(nbf.v4.new_code_cell("""# Display Figure 4: Spatiotemporal Disease Progression & Alerts
display(Image(filename=os.path.join(FIG_DIR, "day06_spatiotemporal_epidemiological_forecasting.png")))"""))

# =============================================================================
# Cell 8: Quantitative Summary Table
# =============================================================================
cells.append(nbf.v4.new_code_cell("""# Display Benchmark Results Summary
df_metrics = pd.read_csv(os.path.join(PROC_DATA_DIR, "day06_per_stage_classification_metrics.csv"))
print("=== Model Performance on Multi-Stage Disease Severity Classification ===")
print(df_metrics.to_string(index=False))

with open(os.path.join(PROC_DATA_DIR, "day06_spatiotemporal_forecast_summary.json")) as f:
    fc_sum = json.load(f)

print("\n=== Acreage Risk Distribution Progression Over 14-Day Forecast ===")
for horizon, vals in fc_sum["acreage_alert_distribution"].items():
    print(f"  {horizon.upper()}: Mean DSI={vals['mean_dsi']:.3f} | Normal={vals['normal_pct']:.1f}% | "
          f"Advisory={vals['advisory_pct']:.1f}% | Warning={vals['warning_pct']:.1f}% | Critical={vals['critical_pct']:.1f}%")"""))

nb.cells = cells
output_path = os.path.join(NOTEBOOK_DIR, "06_crop_disease_detection_and_forecasting.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated interactive notebook: {output_path}")
