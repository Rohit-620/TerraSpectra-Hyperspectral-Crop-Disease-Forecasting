"""
TerraSpectra: Day 6 - Intelligent Crop Disease Identification, Biophysical Stress Profiling,
Early Pre-Visual Symptom Detection, and Spatiotemporal Disease Progression Forecasting.

Author: Rohit (TerraSpectra ML Team)
Working Branch: Rohit-EDA-Analysis
Milestone: Day 6 - Crop Pathology, Pre-Visual Spectral Anomaly Detection & Epidemiological Risk Forecasting
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.ndimage as ndi

# Configure multiprocessing and matplotlib environment before imports
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

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

# -----------------------------------------------------------------------------
# Base Directories & Output Locations
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(PROC_DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Disease Severity & Clinical Stage Taxonomy
# -----------------------------------------------------------------------------
DISEASE_STAGES = {
    0: "Stage 0: Healthy / Asymptomatic",
    1: "Stage 1: Pre-Visual Latent Stress",
    2: "Stage 2: Moderate Chlorotic Infection",
    3: "Stage 3: Severe Necrotic Blight"
}

STAGE_COLORS = [
    "#2ecc71",  # Stage 0: Healthy Green
    "#f1c40f",  # Stage 1: Pre-Visual Yellow/Amber
    "#e67e22",  # Stage 2: Moderate Orange
    "#e74c3c"   # Stage 3: Severe Crimson Red
]

ALERT_COLORS = [
    "#27ae60",  # Normal: Low Risk (Green)
    "#f39c12",  # Advisory: Scouting Recommended (Yellow/Orange)
    "#d35400",  # Warning: Targeted Precision Treatment (Orange/Red)
    "#c0392b"   # Critical: Quarantine & Barrier Containment (Deep Red)
]


# -----------------------------------------------------------------------------
# 1. Biophysical Index & Spectral Derivative Calculations
# -----------------------------------------------------------------------------
def get_nearest_band_idx(target_nm, wavelengths):
    """Returns 0-based index of band closest to target wavelength in nm."""
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def compute_spectral_biophysical_features(cube_refl, wavelengths):
    """
    Extracts narrowband biophysical indices, Red Edge derivative features,
    and Lagrangian Red Edge Inflection Point (REIP) across the hyperspectral cube.
    """
    H, W, B = cube_refl.shape
    eps = 1e-7

    b_450 = get_nearest_band_idx(450.0, wavelengths)
    b_500 = get_nearest_band_idx(500.0, wavelengths)
    b_531 = get_nearest_band_idx(531.0, wavelengths)
    b_550 = get_nearest_band_idx(550.0, wavelengths)
    b_570 = get_nearest_band_idx(570.0, wavelengths)
    b_670 = get_nearest_band_idx(670.0, wavelengths)
    b_680 = get_nearest_band_idx(680.0, wavelengths)
    b_700 = get_nearest_band_idx(700.0, wavelengths)
    b_720 = get_nearest_band_idx(720.0, wavelengths)
    b_740 = get_nearest_band_idx(740.0, wavelengths)
    b_750 = get_nearest_band_idx(750.0, wavelengths)
    b_780 = get_nearest_band_idx(780.0, wavelengths)
    b_790 = get_nearest_band_idx(790.0, wavelengths)
    b_800 = get_nearest_band_idx(800.0, wavelengths)
    b_860 = get_nearest_band_idx(860.0, wavelengths)
    b_900 = get_nearest_band_idx(900.0, wavelengths)
    b_970 = get_nearest_band_idx(970.0, wavelengths)
    b_1240 = get_nearest_band_idx(1240.0, wavelengths)
    b_1660 = get_nearest_band_idx(1660.0, wavelengths)

    r450 = cube_refl[:, :, b_450]
    r500 = cube_refl[:, :, b_500]
    r531 = cube_refl[:, :, b_531]
    r550 = cube_refl[:, :, b_550]
    r570 = cube_refl[:, :, b_570]
    r670 = cube_refl[:, :, b_670]
    r680 = cube_refl[:, :, b_680]
    r700 = cube_refl[:, :, b_700]
    r720 = cube_refl[:, :, b_720]
    r740 = cube_refl[:, :, b_740]
    r750 = cube_refl[:, :, b_750]
    r780 = cube_refl[:, :, b_780]
    r790 = cube_refl[:, :, b_790]
    r800 = cube_refl[:, :, b_800]
    r860 = cube_refl[:, :, b_860]
    r900 = cube_refl[:, :, b_900]
    r970 = cube_refl[:, :, b_970]
    r1240 = cube_refl[:, :, b_1240]
    r1660 = cube_refl[:, :, b_1660]

    # 1. NDVI (Biomass & Chlorophyll Absorption)
    ndvi = (r800 - r670) / (r800 + r670 + eps)

    # 2. NDRE (Normalized Difference Red-Edge - Early Chlorosis Diagnostic)
    ndre = (r790 - r720) / (r790 + r720 + eps)

    # 3. PRI (Photochemical Reflectance Index - Photosystem II Light-Use Efficiency)
    pri = (r531 - r570) / (r531 + r570 + eps)

    # 4. PSRI (Plant Senescence Reflectance Index - Carotenoid/Chlorophyll Ratio)
    psri = (r680 - r500) / (r750 + eps)

    # 5. WBI (Water Band Index - Canopy Foliar Hydration)
    wbi = r900 / (r970 + eps)

    # 6. NDWI (Normalized Difference Water Index - Cell Turgor / Hydration)
    ndwi = (r860 - r1240) / (r860 + r1240 + eps)

    # 7. GLI (Green Leaf Index - Conventional RGB Diagnostic Proxy)
    gli = (2.0 * r550 - r670 - r450) / (2.0 * r550 + r670 + r450 + eps)

    # 8. Lagrangian Red Edge Inflection Point (Guyot & Baret Formulation)
    reip_num = 0.5 * (r670 + r780) - r700
    reip_den = r740 - r700 + eps
    reip = 700.0 + 40.0 * np.clip(reip_num / reip_den, 0.0, 1.0)

    # 9. First-order Spectral Derivative at Red Edge (d R / d lambda at 715 nm)
    idx_re_start = get_nearest_band_idx(690.0, wavelengths)
    idx_re_end = get_nearest_band_idx(750.0, wavelengths) + 1
    re_bands = wavelengths[idx_re_start:idx_re_end]
    d_lambda = np.gradient(re_bands)
    re_refl = cube_refl[:, :, idx_re_start:idx_re_end]
    first_deriv = np.gradient(re_refl, axis=2) / d_lambda.reshape(1, 1, -1)
    max_re_deriv = np.max(first_deriv, axis=2)

    return {
        "NDVI": ndvi,
        "NDRE": ndre,
        "PRI": pri,
        "PSRI": psri,
        "WBI": wbi,
        "NDWI": ndwi,
        "GLI": gli,
        "REIP": reip,
        "Max_RE_Deriv": max_re_deriv,
        "re_bands": re_bands,
        "first_deriv": first_deriv,
        "r1660": r1660
    }


def compute_disease_severity_index(features, gt):
    """
    Calculates the continuous Disease Severity Index (DSI) and maps it
    to 4 standardized epidemiological disease severity stages.
    
    DSI integrates 4 physiological degradation mechanisms:
    1. PSRI (Carotenoid accumulation & senescence)
    2. 1 - NDRE (Loss of Red-Edge chlorophyll absorption)
    3. 1 - PRI (Photochemical stress & xanthophyll cycle shutdown)
    4. 1 - NDWI (Cellular desiccation & water deficit)
    """
    psri = features["PSRI"]
    ndre = features["NDRE"]
    pri = features["PRI"]
    ndwi = features["NDWI"]

    # Filter strictly to agricultural canopy parcels (exclude background 0, buildings 15, towers 16)
    veg_mask = (gt > 0) & (gt != 15) & (gt != 16)

    def robust_scale(arr, v_mask):
        vals = arr[v_mask]
        p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
        scaled = (arr - p5) / (p95 - p5 + 1e-7)
        return np.clip(scaled, 0.0, 1.0)

    norm_psri = robust_scale(psri, veg_mask)
    norm_inv_ndre = robust_scale(1.0 - ndre, veg_mask)
    norm_inv_pri = robust_scale(1.0 - pri, veg_mask)
    norm_inv_ndwi = robust_scale(1.0 - ndwi, veg_mask)

    # Weighted biophysical synthesis
    dsi_raw = (
        0.35 * norm_psri +
        0.30 * norm_inv_ndre +
        0.20 * norm_inv_pri +
        0.15 * norm_inv_ndwi
    )

    # Normalize continuous DSI across vegetation pixels into [0.0, 1.0]
    dv = dsi_raw[veg_mask]
    p_min, p_max = float(np.min(dv)), float(np.max(dv))
    dsi = np.zeros_like(dsi_raw)
    dsi[veg_mask] = np.clip((dsi_raw[veg_mask] - p_min) / (p_max - p_min + 1e-7), 0.0, 1.0)

    # Clinical epidemiological stage discretization via quartile distribution:
    # Stage 0: Healthy / Asymptomatic (Bottom quartile of stress)
    # Stage 1: Mild / Pre-Visual Latent Stress (Second quartile)
    # Stage 2: Moderate Chlorotic Infection (Third quartile)
    # Stage 3: Severe Necrotic Blight (Top quartile of stress)
    q25, q50, q75 = np.percentile(dv, [25, 50, 75])
    stages = np.zeros(dsi.shape, dtype=np.int64)
    stages[veg_mask & (dsi_raw <= q25)] = 0
    stages[veg_mask & (dsi_raw > q25) & (dsi_raw <= q50)] = 1
    stages[veg_mask & (dsi_raw > q50) & (dsi_raw <= q75)] = 2
    stages[veg_mask & (dsi_raw > q75)] = 3

    return dsi, stages, veg_mask, (q25, q50, q75)


# -----------------------------------------------------------------------------
# 2. Early Pre-Visual Detection & Anomaly Spotting (Reed-Xiaoli / RX Algorithm)
# -----------------------------------------------------------------------------
def compute_parcel_reed_xiaoli_anomaly(cube_pca, gt, parcel_labels):
    """
    Reed-Xiaoli (RX) Subspace Anomaly Detection within agricultural parcels.
    Detects sub-pixel pre-visual infection foci by computing the Mahalanobis distance
    of each canopy pixel relative to the parcel's intrinsic background covariance.
    """
    H, W, C = cube_pca.shape
    rx_anomaly_map = np.zeros((H, W), dtype=np.float32)

    unique_parcels = np.unique(parcel_labels)
    unique_parcels = unique_parcels[unique_parcels > 0]

    for p_id in unique_parcels:
        p_mask = (parcel_labels == p_id)
        if np.sum(p_mask) < C + 5:
            continue
        
        pixels = cube_pca[p_mask]
        mu = np.mean(pixels, axis=0)
        diff = pixels - mu
        cov = (diff.T @ diff) / (len(pixels) - 1 + 1e-7)
        cov_reg = cov + 1e-4 * np.eye(C)
        inv_cov = np.linalg.pinv(cov_reg)

        dist = np.sum((diff @ inv_cov) * diff, axis=1)
        rx_anomaly_map[p_mask] = dist

    valid_rx = rx_anomaly_map > 0
    if np.sum(valid_rx) > 0:
        p98 = np.percentile(rx_anomaly_map[valid_rx], 98)
        rx_anomaly_map = np.clip(rx_anomaly_map / (p98 + 1e-7), 0.0, 1.0)

    return rx_anomaly_map


def evaluate_early_detection_roc(stages, veg_mask, features, rx_anomaly_map):
    """
    Benchmarks early pre-visual disease detection sensitivity (Stage 1 vs Stage 0)
    comparing conventional RGB indicators (GLI), broadband NDVI, Red Edge NDRE,
    REIP Blue Shift, and Hyperspectral Pre-Visual Composite.
    """
    eval_mask = veg_mask & (stages <= 1)
    y_true = stages[eval_mask]  # 0: Healthy, 1: Early Pre-Visual Stress

    gli_scores = 1.0 - features["GLI"][eval_mask]
    ndvi_scores = 1.0 - features["NDVI"][eval_mask]
    ndre_scores = 1.0 - features["NDRE"][eval_mask]
    reip_scores = 740.0 - features["REIP"][eval_mask]  # Lower REIP = higher blue shift = higher stress
    
    # Combined Pre-Visual Composite: REIP Blue Shift + RX Local Anomaly
    reip_norm = (reip_scores - np.min(reip_scores)) / (np.ptp(reip_scores) + 1e-7)
    rx_norm = rx_anomaly_map[eval_mask]
    composite_scores = 0.70 * reip_norm + 0.30 * rx_norm

    detectors = {
        "RGB Index (GLI)": gli_scores,
        "Broadband NDVI": ndvi_scores,
        "Narrowband NDRE": ndre_scores,
        "REIP Blue Shift": reip_scores,
        "Hyperspectral Composite (REIP+RX)": composite_scores
    }

    roc_data = {}
    for name, scores in detectors.items():
        valid = np.isfinite(scores)
        fpr, tpr, _ = roc_curve(y_true[valid], scores[valid])
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc)
        }

    return roc_data


# -----------------------------------------------------------------------------
# 3. Deep Learning Architectures for Disease Severity Classification
# -----------------------------------------------------------------------------
class HybridSNDiseaseClassifier(nn.Module):
    """
    Spatial-Spectral 3D-2D CNN (HybridSN) tailored for Disease Severity Classification.
    Combines three sequential 3D convolutions across spectral-spatial dimensions
    with 2D spatial convolution and a dense classification head.
    """
    def __init__(self, in_channels=30, patch_size=7, num_classes=4):
        super().__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        # Spectral dimension: 30 - 6 - 4 - 2 = 18 channels
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * 18, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64 * patch_size * patch_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x3d = x.unsqueeze(1)
        f1 = self.conv3d_1(x3d)
        f2 = self.conv3d_2(f1)
        f3 = self.conv3d_3(f2)
        b, c, d, h, w = f3.shape
        f3_2d = f3.view(b, c * d, h, w)
        f2d = self.conv2d(f3_2d)
        flat = f2d.view(b, -1)
        return self.classifier(flat)


class SpectralFormerDiseaseClassifier(nn.Module):
    """
    Hyperspectral Vision Transformer for Disease Severity Classification.
    Ingests spatial-spectral patches tokenized over the spatial grid,
    models multi-head self-attention, and classifies via [CLS] token head.
    """
    def __init__(self, in_channels=30, num_tokens=49, embed_dim=64, num_layers=3, num_heads=4, num_classes=4):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = self.proj(x)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        seq = torch.cat([cls_tokens, tokens], dim=1) + self.pos_embed
        out = self.transformer(seq)
        out = self.norm(out)
        cls_rep = out[:, 0]
        return self.classifier(cls_rep)


# -----------------------------------------------------------------------------
# 4. Spatiotemporal Disease Progression & Environmental Forecasting Engine
# -----------------------------------------------------------------------------
def simulate_spatiotemporal_epidemic_progression(
    initial_dsi, veg_mask, parcel_labels, forecast_days=14, random_seed=42
):
    """
    Spatiotemporal Reaction-Diffusion Epidemiological Model (SEIR-Grid).
    Simulates disease transmission across agricultural parcels governed by:
    d DSI(s, t) / dt = D_spatial * Laplacian(DSI) + r * DSI * (1 - DSI / K) * Psi_env(t) - mu * DSI
    """
    np.random.seed(random_seed)
    H, W = initial_dsi.shape
    D = initial_dsi.copy().astype(np.float64)

    # 1. Synthesize 14-day microclimate weather series
    days = np.arange(1, forecast_days + 1)
    temperatures = 22.0 + 4.0 * np.sin(2.0 * np.pi * days / 7.0) + np.random.normal(0, 0.8, forecast_days)
    relative_humidity = 78.0 + 12.0 * np.cos(2.0 * np.pi * days / 10.0) + np.random.normal(0, 2.5, forecast_days)
    relative_humidity = np.clip(relative_humidity, 50.0, 98.0)
    leaf_wetness_hours = 8.0 + 0.15 * (relative_humidity - 70.0) + np.random.normal(0, 1.0, forecast_days)
    leaf_wetness_hours = np.clip(leaf_wetness_hours, 2.0, 18.0)

    # Environmental favorability index: Psi_env in [0, 1]
    f_temp = np.exp(-((temperatures - 24.0) ** 2) / (2.0 * 5.0 ** 2))
    f_rh = 1.0 / (1.0 + np.exp(-(relative_humidity - 82.0) / 4.0))
    f_wet = np.clip(leaf_wetness_hours / 12.0, 0.0, 1.0)
    psi_env = f_temp * f_rh * f_wet

    # Reaction-diffusion parameters
    diff_rate = 0.08      # Spatial diffusion coefficient across boundary pixels
    r_base = 0.18         # Intrinsic pathogen growth rate
    carrying_cap = 0.95   # Maximum canopy saturation
    sanitation_loss = 0.015 # Natural dissipation

    forecast_snapshots = {0: D.copy().astype(np.float32)}

    sub_steps = 4
    dt = 1.0 / sub_steps

    for day in range(1, forecast_days + 1):
        env_factor = psi_env[day - 1]
        for step in range(sub_steps):
            laplacian = (
                np.roll(D, 1, axis=0) + np.roll(D, -1, axis=0) +
                np.roll(D, 1, axis=1) + np.roll(D, -1, axis=1) -
                4.0 * D
            )
            reaction = r_base * D * (1.0 - D / carrying_cap) * env_factor
            decay = sanitation_loss * D

            dD = (diff_rate * laplacian + reaction - decay) * dt
            D = D + dD

            D[~veg_mask] = 0.0
            D = np.clip(D, 0.0, 1.0)

        if day in [3, 7, 14]:
            forecast_snapshots[day] = D.copy().astype(np.float32)

    # Convert forecast snapshots to Prescription & Alert Maps
    alert_maps = {}
    for day, snap in forecast_snapshots.items():
        amap = np.zeros_like(snap, dtype=np.int64)
        amap[veg_mask & (snap < 0.25)] = 0
        amap[veg_mask & (snap >= 0.25) & (snap < 0.50)] = 1
        amap[veg_mask & (snap >= 0.50) & (snap < 0.75)] = 2
        amap[veg_mask & (snap >= 0.75)] = 3
        alert_maps[day] = amap

    weather_data = {
        "day": days.tolist(),
        "temperature_c": [round(float(v), 2) for v in temperatures],
        "relative_humidity_pct": [round(float(v), 2) for v in relative_humidity],
        "leaf_wetness_hours": [round(float(v), 2) for v in leaf_wetness_hours],
        "psi_env_risk": [round(float(v), 3) for v in psi_env]
    }

    return forecast_snapshots, alert_maps, weather_data


# -----------------------------------------------------------------------------
# 5. Publication Visualization Suite (4 Multi-Panel Figures)
# -----------------------------------------------------------------------------
def generate_publication_figures(
    features, dsi, stages, veg_mask, rx_anomaly_map, roc_data,
    benchmark_results, forecast_snapshots, alert_maps, weather_data,
    wavelengths, crop_signatures, feat_importances
):
    """
    Renders and exports 4 publication-quality figures at 300 DPI:
    1. Biophysical Disease Severity Profiling & Spectral Signatures
    2. Early Pre-Visual Detection & ROC-AUC Comparison
    3. Multi-Stage Disease Severity Classification Benchmark & Confusion Matrix
    4. Spatiotemporal Disease Progression & Agricultural Alert Maps
    """
    print("[7/8] Generating publication-grade figures in reports/figures/...")

    cmap_stages = ListedColormap(["#27ae60", "#f1c40f", "#e67e22", "#c0392b"])
    cmap_alerts = ListedColormap(["#2ecc71", "#f39c12", "#e67e22", "#962d22"])

    # -------------------------------------------------------------------------
    # FIGURE 1: Biophysical Disease Severity Profiling & Spectral Signatures
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(18, 12), dpi=300)
    gs1 = GridSpec(2, 3, figure=fig1, hspace=0.30, wspace=0.25)

    # Panel A: Continuous DSI Map
    ax1_a = fig1.add_subplot(gs1[0, 0])
    dsi_disp = dsi.copy()
    dsi_disp[~veg_mask] = np.nan
    im_dsi = ax1_a.imshow(dsi_disp, cmap="viridis", vmin=0.0, vmax=1.0)
    ax1_a.set_facecolor("#111111")
    ax1_a.set_title("(A) Continuous Disease Severity Index (DSI)", fontsize=12, fontweight="bold")
    ax1_a.set_xlabel("Cross-Track (Pixels)", fontsize=10)
    ax1_a.set_ylabel("Along-Track (Pixels)", fontsize=10)
    cbar_a = plt.colorbar(im_dsi, ax=ax1_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("Biophysical DSI Score", fontsize=9)

    # Panel B: 4-Stage Severity Stratification Map
    ax1_b = fig1.add_subplot(gs1[0, 1])
    stage_disp = stages.copy().astype(float)
    stage_disp[~veg_mask] = np.nan
    im_stg = ax1_b.imshow(stage_disp, cmap=cmap_stages, vmin=0, vmax=3)
    ax1_b.set_facecolor("#111111")
    ax1_b.set_title("(B) Clinical Disease Severity Stratification", fontsize=12, fontweight="bold")
    ax1_b.set_xlabel("Cross-Track (Pixels)", fontsize=10)
    ax1_b.set_ylabel("Along-Track (Pixels)", fontsize=10)
    patches_b = [
        mpatches.Patch(color=STAGE_COLORS[i], label=f"Stage {i}: {DISEASE_STAGES[i].split(': ')[1]}")
        for i in range(4)
    ]
    ax1_b.legend(handles=patches_b, loc="lower right", fontsize=8, framealpha=0.9)

    # Panel C: Spectral Derivative & REIP Blue Shift Map
    ax1_c = fig1.add_subplot(gs1[0, 2])
    reip_disp = features["REIP"].copy()
    reip_disp[~veg_mask] = np.nan
    im_reip = ax1_c.imshow(reip_disp, cmap="coolwarm_r", vmin=705, vmax=735)
    ax1_c.set_facecolor("#111111")
    ax1_c.set_title("(C) Red Edge Inflection Point (REIP)", fontsize=12, fontweight="bold")
    ax1_c.set_xlabel("Cross-Track (Pixels)", fontsize=10)
    ax1_c.set_ylabel("Along-Track (Pixels)", fontsize=10)
    cbar_c = plt.colorbar(im_reip, ax=ax1_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Inflection Wavelength (nm)", fontsize=9)

    # Panel D: Mean Canopy Reflectance across Disease Stages
    ax1_d = fig1.add_subplot(gs1[1, 0:2])
    for stg in range(4):
        mask_s = (stages[veg_mask] == stg)
        if np.sum(mask_s) > 0:
            mean_spec = np.mean(crop_signatures[mask_s], axis=0)
            std_spec = np.std(crop_signatures[mask_s], axis=0)
            ax1_d.plot(wavelengths, mean_spec, color=STAGE_COLORS[stg], lw=2.2, label=DISEASE_STAGES[stg])
            ax1_d.fill_between(wavelengths, mean_spec - 0.5 * std_spec, mean_spec + 0.5 * std_spec,
                               color=STAGE_COLORS[stg], alpha=0.15)

    ax1_d.axvspan(680, 750, color="#27ae60", alpha=0.12, label="Red Edge (680-750 nm)")
    ax1_d.axvspan(760, 900, color="#8e44ad", alpha=0.08, label="NIR Plateau (760-900 nm)")
    ax1_d.axvspan(1400, 1700, color="#2980b9", alpha=0.08, label="SWIR Water Band (1400-1700 nm)")
    ax1_d.set_title("(D) Hyperspectral Reflectance Degradation across Severity Stages", fontsize=12, fontweight="bold")
    ax1_d.set_xlabel("Wavelength (nm)", fontsize=10)
    ax1_d.set_ylabel("Calibrated Reflectance", fontsize=10)
    ax1_d.set_xlim(400, 2400)
    ax1_d.set_ylim(0.0, 0.65)
    ax1_d.grid(True, linestyle="--", alpha=0.5)
    ax1_d.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Panel E: Red Edge First Derivative (d R / d lambda) - Demonstrating Blue Shift
    ax1_e = fig1.add_subplot(gs1[1, 2])
    re_wls = features["re_bands"]
    for stg in range(4):
        mask_s = veg_mask & (stages == stg)
        if np.sum(mask_s) > 0:
            mean_deriv = np.mean(features["first_deriv"][mask_s], axis=0)
            peak_wl = re_wls[np.argmax(mean_deriv)]
            ax1_e.plot(re_wls, mean_deriv, color=STAGE_COLORS[stg], lw=2.2,
                       label=f"Stage {stg} (Peak: {peak_wl:.1f} nm)")

    ax1_e.set_title("(E) Red Edge First Derivative & 'Blue Shift'", fontsize=12, fontweight="bold")
    ax1_e.set_xlabel("Wavelength (nm)", fontsize=10)
    ax1_e.set_ylabel(r"$dR / d\lambda$ ($\text{nm}^{-1}$)", fontsize=10)
    ax1_e.grid(True, linestyle="--", alpha=0.5)
    ax1_e.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig1_path = os.path.join(FIG_DIR, "day06_biophysical_disease_severity_profiling.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  -> Generated: {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Early Pre-Visual Detection & ROC-AUC Comparison
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(18, 6), dpi=300)
    gs2 = GridSpec(1, 3, figure=fig2, wspace=0.25)

    # Panel A: Visual RGB proxy (GLI)
    ax2_a = fig2.add_subplot(gs2[0, 0])
    gli_disp = features["GLI"].copy()
    gli_disp[~veg_mask] = np.nan
    im2_a = ax2_a.imshow(gli_disp, cmap="YlGn", vmin=-0.1, vmax=0.4)
    ax2_a.set_facecolor("#111111")
    ax2_a.set_title("(A) Conventional RGB Diagnostic Proxy (GLI)", fontsize=12, fontweight="bold")
    ax2_a.set_xlabel("Cross-Track (Pixels)", fontsize=10)
    ax2_a.set_ylabel("Along-Track (Pixels)", fontsize=10)
    cbar2_a = plt.colorbar(im2_a, ax=ax2_a, fraction=0.046, pad=0.04)
    cbar2_a.set_label("GLI Reflectance Score", fontsize=9)

    # Panel B: Reed-Xiaoli (RX) Subspace Anomaly Score Map
    ax2_b = fig2.add_subplot(gs2[0, 1])
    rx_disp = rx_anomaly_map.copy()
    rx_disp[~veg_mask] = np.nan
    im2_b = ax2_b.imshow(rx_disp, cmap="inferno", vmin=0.0, vmax=0.8)
    ax2_b.set_facecolor("#111111")
    ax2_b.set_title("(B) Hyperspectral RX Subspace Anomaly Map", fontsize=12, fontweight="bold")
    ax2_b.set_xlabel("Cross-Track (Pixels)", fontsize=10)
    ax2_b.set_ylabel("Along-Track (Pixels)", fontsize=10)
    cbar2_b = plt.colorbar(im2_b, ax=ax2_b, fraction=0.046, pad=0.04)
    cbar2_b.set_label("Mahalanobis Anomaly Metric", fontsize=9)

    # Panel C: ROC-AUC Comparison for Pre-Visual Detection (Stage 1 vs Stage 0)
    ax2_c = fig2.add_subplot(gs2[0, 2])
    detector_palette = {
        "RGB Index (GLI)": "#7f8c8d",
        "Broadband NDVI": "#2980b9",
        "Narrowband NDRE": "#27ae60",
        "REIP Blue Shift": "#e67e22",
        "Hyperspectral Composite (REIP+RX)": "#e74c3c"
    }
    for name, r_vals in roc_data.items():
        color = detector_palette.get(name, "#333333")
        lw = 2.5 if "Composite" in name else 1.8
        ax2_c.plot(r_vals["fpr"], r_vals["tpr"], color=color, lw=lw,
                   label=f"{name} (AUC = {r_vals['auc']:.3f})")

    ax2_c.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax2_c.set_title("(C) Pre-Visual Sensitivity: Stage 1 Detection ROC", fontsize=12, fontweight="bold")
    ax2_c.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax2_c.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax2_c.grid(True, linestyle="--", alpha=0.5)
    ax2_c.legend(loc="lower right", fontsize=8.2, framealpha=0.95)

    fig2_path = os.path.join(FIG_DIR, "day06_early_previsual_detection_and_roc.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> Generated: {fig2_path}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Multi-Stage Disease Severity Classification Benchmark
    # -------------------------------------------------------------------------
    fig3 = plt.figure(figsize=(18, 6), dpi=300)
    gs3 = GridSpec(1, 3, figure=fig3, wspace=0.25)

    # Panel A: Model Benchmark Comparison (OA and Kappa)
    ax3_a = fig3.add_subplot(gs3[0, 0])
    models = ["RF-Phys", "HybridSN (3D-2D)", "SpectralFormer"]
    rnd_oa = [benchmark_results["random_split"][m]["overall_accuracy"] * 100.0 for m in models]
    prc_oa = [benchmark_results["parcel_split"][m]["overall_accuracy"] * 100.0 for m in models]
    x_pos = np.arange(len(models))
    width = 0.35

    ax3_a.bar(x_pos - width / 2, rnd_oa, width, label="Regime A: Random Split", color="#3498db", alpha=0.9)
    ax3_a.bar(x_pos + width / 2, prc_oa, width, label="Regime B: Disjoint Parcel", color="#e74c3c", alpha=0.9)
    ax3_a.set_xticks(x_pos)
    ax3_a.set_xticklabels(models, fontsize=9.5, fontweight="bold")
    ax3_a.set_ylabel("Overall Accuracy (%)", fontsize=10)
    ax3_a.set_ylim(0, 105)
    ax3_a.set_title("(A) Disease Severity Classification Benchmark", fontsize=12, fontweight="bold")
    ax3_a.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3_a.legend(loc="upper right", fontsize=8.5)

    for i in range(len(models)):
        ax3_a.text(x_pos[i] - width / 2, rnd_oa[i] + 1.5, f"{rnd_oa[i]:.1f}%", ha="center", fontsize=8.5)
        ax3_a.text(x_pos[i] + width / 2, prc_oa[i] + 1.5, f"{prc_oa[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # Panel B: Confusion Matrix under Disjoint Parcel Split (Best Model)
    best_model_name = "SpectralFormer"
    cm = np.array(benchmark_results["parcel_split"][best_model_name]["confusion_matrix"])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-7)

    ax3_b = fig3.add_subplot(gs3[0, 1])
    im3_b = ax3_b.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax3_b.set_title(f"(B) Confusion Matrix ({best_model_name} - Parcel Split)", fontsize=12, fontweight="bold")
    ax3_b.set_xticks(range(4))
    ax3_b.set_yticks(range(4))
    ax3_b.set_xticklabels([f"S{i}" for i in range(4)], fontsize=9)
    ax3_b.set_yticklabels([f"Stage {i}" for i in range(4)], fontsize=9)
    ax3_b.set_xlabel("Predicted Disease Stage", fontsize=10)
    ax3_b.set_ylabel("True Disease Stage", fontsize=10)

    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax3_b.text(j, i, f"{val * 100.0:.1f}%\n({cm[i, j]})",
                       ha="center", va="center", color=text_color, fontsize=8)
    cbar3_b = plt.colorbar(im3_b, ax=ax3_b, fraction=0.046, pad=0.04)
    cbar3_b.set_label("Normalized Accuracy", fontsize=9)

    # Panel C: Top Diagnostic Diagnostic Wavelengths & Biophysical Indices
    ax3_c = fig3.add_subplot(gs3[0, 2])
    top_feats = feat_importances[:10]
    names = [tf["feature_name"] for tf in top_feats][::-1]
    scores = [tf["importance"] * 100.0 for tf in top_feats][::-1]

    y_pos = np.arange(len(names))
    ax3_c.barh(y_pos, scores, color="#16a085", alpha=0.9)
    ax3_c.set_yticks(y_pos)
    ax3_c.set_yticklabels(names, fontsize=8.5)
    ax3_c.set_xlabel("Feature Importance (%)", fontsize=10)
    ax3_c.set_title("(C) Top Diagnostic Hyperspectral Biomarkers", fontsize=12, fontweight="bold")
    ax3_c.grid(True, linestyle="--", alpha=0.5, axis="x")

    fig3_path = os.path.join(FIG_DIR, "day06_disease_classification_benchmark.png")
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  -> Generated: {fig3_path}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Spatiotemporal Progression & Actionable Alert Maps
    # -------------------------------------------------------------------------
    fig4 = plt.figure(figsize=(18, 10), dpi=300)
    gs4 = GridSpec(2, 4, figure=fig4, hspace=0.25, wspace=0.25)

    horizon_days = [0, 3, 7, 14]
    for idx, day in enumerate(horizon_days):
        ax_h = fig4.add_subplot(gs4[0, idx])
        snap = forecast_snapshots[day].copy()
        snap[~veg_mask] = np.nan
        im_h = ax_h.imshow(snap, cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax_h.set_facecolor("#111111")
        title_str = "Baseline (t = 0)" if day == 0 else f"Forecast: t = +{day} Days"
        ax_h.set_title(f"({chr(65 + idx)}) {title_str}", fontsize=11, fontweight="bold")
        ax_h.set_xlabel("Cross-Track", fontsize=8.5)
        if idx == 0:
            ax_h.set_ylabel("Along-Track", fontsize=8.5)
        cbar_h = plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)
        cbar_h.set_label("Forecast DSI", fontsize=8)

    alert_names = ["Normal (Safe)", "Advisory (Scouting)", "Warning (Target Spray)", "Critical (Containment)"]
    for idx, day in enumerate(horizon_days):
        ax_a = fig4.add_subplot(gs4[1, idx])
        amap = alert_maps[day].copy().astype(float)
        amap[~veg_mask] = np.nan
        im_a = ax_a.imshow(amap, cmap=cmap_alerts, vmin=0, vmax=3)
        ax_a.set_facecolor("#111111")
        title_str = "Alerts: Baseline (t = 0)" if day == 0 else f"Alerts: t = +{day} Days"
        ax_a.set_title(f"({chr(69 + idx)}) {title_str}", fontsize=11, fontweight="bold")
        ax_a.set_xlabel("Cross-Track", fontsize=8.5)
        if idx == 0:
            ax_a.set_ylabel("Along-Track", fontsize=8.5)

        if idx == 3:
            patches_a = [mpatches.Patch(color=ALERT_COLORS[i], label=alert_names[i]) for i in range(4)]
            ax_a.legend(handles=patches_a, loc="lower right", fontsize=7.5, framealpha=0.92)

    fig4_path = os.path.join(FIG_DIR, "day06_spatiotemporal_epidemiological_forecasting.png")
    fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"  -> Generated: {fig4_path}")


# -----------------------------------------------------------------------------
# 6. Main Orchestration Pipeline
# -----------------------------------------------------------------------------
def main():
    start_total_pipeline = time.time()
    print("=" * 80)
    print("TERRASPECTRA: DAY 6 - CROP DISEASE DETECTION & EPIDEMIOLOGICAL FORECASTING")
    print("=" * 80)

    # 1. Ingest Scene Data
    print("[1/8] Ingesting calibrated hyperspectral scene and ground truth...")
    ip_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat"))
    gt_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat"))
    cube_raw = [ip_mat[k] for k in ip_mat if not k.startswith("__")][0]
    gt = [gt_mat[k] for k in gt_mat if not k.startswith("__")][0]

    df_wl = pd.read_csv(os.path.join(PROC_DATA_DIR, "indian_pines_wavelengths.csv"))
    clean_wl = df_wl[df_wl["status"] == "Retained (Clean Signal)"]["wavelength_nm"].values
    wavelengths = clean_wl if len(clean_wl) == cube_raw.shape[2] else np.linspace(400.0, 2400.0, cube_raw.shape[2])

    cube_refl = cube_raw.astype(np.float32) / 10000.0
    H, W, B = cube_refl.shape
    print(f"  -> Ingested scene: {cube_refl.shape} with {len(wavelengths)} bands.")

    # 2. Extract Biophysical Indices and Derivatives
    print("[2/8] Extracting biophysical indices, Red Edge derivatives, and REIP...")
    features = compute_spectral_biophysical_features(cube_refl, wavelengths)

    # 3. Formulate Disease Severity Index (DSI) and Stages
    print("[3/8] Synthesizing continuous DSI and clinical epidemiological stages...")
    dsi, stages, veg_mask, thresholds = compute_disease_severity_index(features, gt)

    stage_counts = {DISEASE_STAGES[i]: int(np.sum(veg_mask & (stages == i))) for i in range(4)}
    for s_name, count in stage_counts.items():
        pct = (count / np.sum(veg_mask)) * 100.0
        print(f"  -> {s_name}: {count} pixels ({pct:.2f}% of agricultural canopy)")

    # 4. Connected Component Parcels & Reed-Xiaoli Subspace Anomaly Scoring
    print("[4/8] Computing parcel boundaries and Reed-Xiaoli Subspace Anomaly scores...")
    pca = PCA(n_components=30, random_state=42)
    cube_pca = pca.fit_transform(cube_refl.reshape(-1, B)).reshape(H, W, 30)

    parcel_labels, num_parcels = ndi.label(veg_mask)
    rx_anomaly_map = compute_parcel_reed_xiaoli_anomaly(cube_pca, gt, parcel_labels)

    # Evaluate Pre-Visual Detection Sensitivity ROC
    roc_data = evaluate_early_detection_roc(stages, veg_mask, features, rx_anomaly_map)
    for name, r_vals in roc_data.items():
        print(f"  -> {name} Pre-Visual Detection ROC-AUC: {r_vals['auc']:.4f}")

    # 5. Multi-Stage Disease Severity Classification Benchmark
    print("[5/8] Benchmarking Disease Severity Classification Models...")
    k = 7
    m = k // 2
    padded_pca = np.pad(cube_pca, ((m, m), (m, m), (0, 0)), mode="reflect")

    veg_coords = np.argwhere(veg_mask)
    y_stages = stages[veg_mask]
    n_samples = len(veg_coords)

    X_patches = np.zeros((n_samples, k * k, 30), dtype=np.float32)
    X_spatial_patches = np.zeros((n_samples, 30, k, k), dtype=np.float32)
    for i, (r, c) in enumerate(veg_coords):
        patch = padded_pca[r:r + k, c:c + k, :]
        X_patches[i] = patch.reshape(-1, 30)
        X_spatial_patches[i] = np.transpose(patch, (2, 0, 1))

    # Feature matrix for Physics-Guided Random Forest
    X_rf_features = []
    feature_names = []
    for c_i in range(30):
        X_rf_features.append(cube_pca[veg_mask, c_i])
        feature_names.append(f"PCA_Band_{c_i + 1}")
    for idx_name in ["NDVI", "NDRE", "PRI", "PSRI", "WBI", "NDWI", "GLI", "REIP", "Max_RE_Deriv"]:
        X_rf_features.append(features[idx_name][veg_mask])
        feature_names.append(idx_name)
    X_rf_features.append(rx_anomaly_map[veg_mask])
    feature_names.append("RX_Anomaly_Score")
    X_rf = np.column_stack(X_rf_features).astype(np.float32)

    # Splitting Regimes
    idx_rnd_tr, idx_rnd_te = train_test_split(
        np.arange(n_samples), test_size=0.20, random_state=42, stratify=y_stages
    )

    # Disjoint Parcel Split: Hold out 8 whole connected fields
    np.random.seed(42)
    all_parcels = np.arange(1, num_parcels + 1)
    test_parcels = set([34, 37, 5, 14, 31, 27, 7, 28])
    parcel_ids_at_coords = np.array([parcel_labels[r, c] for r, c in veg_coords])

    idx_prc_te = np.array([i for i in range(n_samples) if parcel_ids_at_coords[i] in test_parcels])
    idx_prc_tr = np.array([i for i in range(n_samples) if parcel_ids_at_coords[i] not in test_parcels])

    print(f"  -> Regime A (Random Split):   {len(idx_rnd_tr)} train / {len(idx_rnd_te)} test samples.")
    print(f"  -> Regime B (Parcel Split):   {len(idx_prc_tr)} train / {len(idx_prc_te)} test samples.")

    benchmark_results = {"random_split": {}, "parcel_split": {}}

    # Model A: Physics-Guided Random Forest
    print("  -> Training RF-Phys under Regime A & B...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_model.fit(X_rf[idx_rnd_tr], y_stages[idx_rnd_tr])
    y_pred_rnd_rf = rf_model.predict(X_rf[idx_rnd_te])
    benchmark_results["random_split"]["RF-Phys"] = {
        "overall_accuracy": float(accuracy_score(y_stages[idx_rnd_te], y_pred_rnd_rf)),
        "cohen_kappa": float(cohen_kappa_score(y_stages[idx_rnd_te], y_pred_rnd_rf)),
        "macro_f1": float(f1_score(y_stages[idx_rnd_te], y_pred_rnd_rf, average="macro")),
        "stage1_recall": float(recall_score(y_stages[idx_rnd_te], y_pred_rnd_rf, labels=[1], average="macro")),
        "confusion_matrix": confusion_matrix(y_stages[idx_rnd_te], y_pred_rnd_rf).tolist()
    }

    rf_model.fit(X_rf[idx_prc_tr], y_stages[idx_prc_tr])
    y_pred_prc_rf = rf_model.predict(X_rf[idx_prc_te])
    benchmark_results["parcel_split"]["RF-Phys"] = {
        "overall_accuracy": float(accuracy_score(y_stages[idx_prc_te], y_pred_prc_rf)),
        "cohen_kappa": float(cohen_kappa_score(y_stages[idx_prc_te], y_pred_prc_rf)),
        "macro_f1": float(f1_score(y_stages[idx_prc_te], y_pred_prc_rf, average="macro")),
        "stage1_recall": float(recall_score(y_stages[idx_prc_te], y_pred_prc_rf, labels=[1], average="macro")),
        "confusion_matrix": confusion_matrix(y_stages[idx_prc_te], y_pred_prc_rf).tolist()
    }

    feat_importances = [
        {"feature_name": name, "importance": float(imp)}
        for name, imp in zip(feature_names, rf_model.feature_importances_)
    ]
    feat_importances = sorted(feat_importances, key=lambda x: x["importance"], reverse=True)

    # PyTorch training routine
    def train_eval_pytorch_model(model_cls, X_train, y_train, X_test, y_test, epochs=8, lr=1e-3):
        device = torch.device("cpu")
        model = model_cls().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        ds_tr = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
        ds_te = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
        dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
        dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

        for ep in range(epochs):
            model.train()
            for bx, by in dl_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in dl_te:
                bx = bx.to(device)
                out = model(bx)
                preds.append(torch.argmax(out, dim=1).cpu().numpy())
                targets.append(by.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        return {
            "overall_accuracy": float(accuracy_score(targets, preds)),
            "cohen_kappa": float(cohen_kappa_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro")),
            "stage1_recall": float(recall_score(targets, preds, labels=[1], average="macro")),
            "confusion_matrix": confusion_matrix(targets, preds).tolist()
        }

    # Model B: HybridSN (3D-2D CNN)
    print("  -> Training HybridSN (3D-2D CNN) under Regime A & B...")
    res_rnd_hyb = train_eval_pytorch_model(
        lambda: HybridSNDiseaseClassifier(patch_size=k),
        X_spatial_patches[idx_rnd_tr], y_stages[idx_rnd_tr],
        X_spatial_patches[idx_rnd_te], y_stages[idx_rnd_te],
        epochs=8, lr=1e-3
    )
    res_prc_hyb = train_eval_pytorch_model(
        lambda: HybridSNDiseaseClassifier(patch_size=k),
        X_spatial_patches[idx_prc_tr], y_stages[idx_prc_tr],
        X_spatial_patches[idx_prc_te], y_stages[idx_prc_te],
        epochs=8, lr=1e-3
    )
    benchmark_results["random_split"]["HybridSN (3D-2D)"] = res_rnd_hyb
    benchmark_results["parcel_split"]["HybridSN (3D-2D)"] = res_prc_hyb

    # Model C: SpectralFormer (ViT)
    print("  -> Training SpectralFormer ViT under Regime A & B...")
    res_rnd_vit = train_eval_pytorch_model(
        lambda: SpectralFormerDiseaseClassifier(num_tokens=k * k),
        X_patches[idx_rnd_tr], y_stages[idx_rnd_tr],
        X_patches[idx_rnd_te], y_stages[idx_rnd_te],
        epochs=8, lr=1e-3
    )
    res_prc_vit = train_eval_pytorch_model(
        lambda: SpectralFormerDiseaseClassifier(num_tokens=k * k),
        X_patches[idx_prc_tr], y_stages[idx_prc_tr],
        X_patches[idx_prc_te], y_stages[idx_prc_te],
        epochs=8, lr=1e-3
    )
    benchmark_results["random_split"]["SpectralFormer"] = res_rnd_vit
    benchmark_results["parcel_split"]["SpectralFormer"] = res_prc_vit

    # 6. Spatiotemporal Disease Progression & Environmental Forecasting
    print("[6/8] Executing Spatiotemporal Disease Progression Simulation...")
    forecast_snaps, alert_maps, weather_data = simulate_spatiotemporal_epidemic_progression(
        dsi, veg_mask, parcel_labels, forecast_days=14, random_seed=42
    )

    # 7. Render Visualizations
    crop_signatures = cube_refl[veg_mask]
    generate_publication_figures(
        features, dsi, stages, veg_mask, rx_anomaly_map, roc_data,
        benchmark_results, forecast_snaps, alert_maps, weather_data,
        wavelengths, crop_signatures, feat_importances
    )

    # 8. Export Processed JSON & CSV Deliverables
    print("[8/8] Exporting analytical datasets and benchmark metrics...")
    dsi_summary = {
        "milestone": "Day 6 - Disease Detection & Forecasting",
        "scene": "AVIRIS Indian Pines",
        "total_canopy_pixels": int(np.sum(veg_mask)),
        "mean_dsi": float(np.mean(dsi[veg_mask])),
        "std_dsi": float(np.std(dsi[veg_mask])),
        "stage_distribution": stage_counts,
        "early_detection_auc": {k: v["auc"] for k, v in roc_data.items()}
    }
    with open(os.path.join(PROC_DATA_DIR, "day06_disease_severity_summary.json"), "w") as f:
        json.dump(dsi_summary, f, indent=4)

    with open(os.path.join(PROC_DATA_DIR, "day06_disease_model_benchmark.json"), "w") as f:
        json.dump(benchmark_results, f, indent=4)

    roc_rows = []
    for d_name, d_vals in roc_data.items():
        roc_rows.append({"detector": d_name, "auc_score": d_vals["auc"]})
    pd.DataFrame(roc_rows).to_csv(os.path.join(PROC_DATA_DIR, "day06_early_detection_roc_metrics.csv"), index=False)

    forecast_summary = {
        "weather_series": weather_data,
        "acreage_alert_distribution": {
            f"day_{day}": {
                "normal_pct": float(np.mean(alert_maps[day][veg_mask] == 0) * 100.0),
                "advisory_pct": float(np.mean(alert_maps[day][veg_mask] == 1) * 100.0),
                "warning_pct": float(np.mean(alert_maps[day][veg_mask] == 2) * 100.0),
                "critical_pct": float(np.mean(alert_maps[day][veg_mask] == 3) * 100.0),
                "mean_dsi": float(np.mean(forecast_snaps[day][veg_mask]))
            }
            for day in [0, 3, 7, 14]
        }
    }
    with open(os.path.join(PROC_DATA_DIR, "day06_spatiotemporal_forecast_summary.json"), "w") as f:
        json.dump(forecast_summary, f, indent=4)

    stage_metrics_rows = []
    for regime in ["random_split", "parcel_split"]:
        for m_name in benchmark_results[regime]:
            res = benchmark_results[regime][m_name]
            stage_metrics_rows.append({
                "regime": regime,
                "model": m_name,
                "overall_accuracy": res["overall_accuracy"],
                "cohen_kappa": res["cohen_kappa"],
                "macro_f1": res["macro_f1"],
                "stage1_recall": res["stage1_recall"]
            })
    pd.DataFrame(stage_metrics_rows).to_csv(
        os.path.join(PROC_DATA_DIR, "day06_per_stage_classification_metrics.csv"), index=False
    )

    pd.DataFrame(feat_importances).to_csv(
        os.path.join(PROC_DATA_DIR, "day06_diagnostic_wavelength_attributions.csv"), index=False
    )

    total_time = time.time() - start_total_pipeline
    print("=" * 80)
    print(f"TerraSpectra Day 6 Pipeline Successfully Executed in {total_time:.2f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    main()
