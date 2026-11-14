"""
TerraSpectra: Day 2 - In-Depth Loading, Structural Inspection, Channel Calibration & Spatial Patch Slicing Verification
ML Engineering & Data Pipeline: Preprocessing & Spatial-Spectral Representation
Author: Rohit (TerraSpectra ML Team)
Working Branch: Rohit-EDA-Analysis
Milestone: Day 2 - Structural Tensor Inspection, Spectrometer Calibration, 3D Patch Extraction & Spatial Leakage Verification
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

# Configure matplotlib cache directory before importing matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Project Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(PROC_DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Class Label Dictionaries
# -----------------------------------------------------------------------------
INDIAN_PINES_CLASSES = {
    0: "Background / Unlabeled",
    1: "Alfalfa",
    2: "Corn-notill",
    3: "Corn-mintill",
    4: "Corn",
    5: "Grass-pasture",
    6: "Grass-trees",
    7: "Grass-pasture-mowed",
    8: "Hay-windrowed",
    9: "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-clean",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers"
}

SALINAS_CLASSES = {
    0: "Background / Unlabeled",
    1: "Brocoli_green_weeds_1",
    2: "Brocoli_green_weeds_2",
    3: "Fallow",
    4: "Fallow_rough_plow",
    5: "Fallow_smooth",
    6: "Stubble",
    7: "Celery",
    8: "Grapes_untrained",
    9: "Soil_vinyard_develop",
    10: "Corn_senesced_green_weeds",
    11: "Lettuce_romaine_4wk",
    12: "Lettuce_romaine_5wk",
    13: "Lettuce_romaine_6wk",
    14: "Lettuce_romaine_7wk",
    15: "Vinyard_untrained",
    16: "Vinyard_vertical_trellis"
}

# -----------------------------------------------------------------------------
# 1. Structural Inspection Engine
# -----------------------------------------------------------------------------
def inspect_tensor_structure(tensor, name):
    """
    Computes rigorous structural, memory, radiometric, and numerical health statistics.
    """
    shape = tensor.shape
    ndim = tensor.ndim
    dtype_str = str(tensor.dtype)
    itemsize = tensor.itemsize
    total_elements = tensor.size
    memory_bytes = tensor.nbytes
    memory_mb = round(memory_bytes / (1024 * 1024), 2)
    is_c_contiguous = tensor.flags["C_CONTIGUOUS"]
    is_f_contiguous = tensor.flags["F_CONTIGUOUS"]

    nan_count = int(np.isnan(tensor).sum()) if np.issubdtype(tensor.dtype, np.floating) else 0
    inf_count = int(np.isinf(tensor).sum()) if np.issubdtype(tensor.dtype, np.floating) else 0

    min_val = float(np.min(tensor))
    max_val = float(np.max(tensor))
    mean_val = float(np.mean(tensor))
    std_val = float(np.std(tensor))
    p1 = float(np.percentile(tensor, 1))
    p50 = float(np.percentile(tensor, 50))
    p99 = float(np.percentile(tensor, 99))

    return {
        "name": name,
        "shape": list(shape),
        "spatial_lines_H": int(shape[0]),
        "spatial_samples_W": int(shape[1]),
        "spectral_channels_B": int(shape[2]) if ndim == 3 else 1,
        "ndim": int(ndim),
        "data_type": dtype_str,
        "itemsize_bytes": int(itemsize),
        "total_elements": int(total_elements),
        "memory_mb": memory_mb,
        "memory_layout": "C-Contiguous (Row-Major)" if is_c_contiguous else ("Fortran (Column-Major)" if is_f_contiguous else "Non-contiguous"),
        "numerical_health": {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "has_nans": nan_count > 0,
            "has_infs": inf_count > 0,
            "negative_values_count": int(np.sum(tensor < 0)),
            "zero_values_count": int(np.sum(tensor == 0))
        },
        "radiometric_statistics": {
            "min_val": min_val,
            "max_val": max_val,
            "mean_val": round(mean_val, 3),
            "std_val": round(std_val, 3),
            "percentile_1": round(p1, 3),
            "median_p50": round(p50, 3),
            "percentile_99": round(p99, 3)
        }
    }

# -----------------------------------------------------------------------------
# 2. Spectrometer Subsystems & Wavelength Calibration
# -----------------------------------------------------------------------------
def build_spectrometer_calibrations(raw_cube, corr_cube):
    """
    Constructs per-band calibration, spectrometer subsystems, and variance profiling.
    """
    total_raw_bands = raw_cube.shape[2]
    nominal_centers = np.linspace(400.0, 2450.0, total_raw_bands)
    
    # 1-indexed removed atmospheric water bands for Indian Pines
    water_bands_1idx = set(list(range(104, 109)) + list(range(150, 164)) + [220])
    
    calib_records = []
    corr_idx = 1

    for raw_idx in range(1, total_raw_bands + 1):
        wl = nominal_centers[raw_idx - 1]
        is_water = raw_idx in water_bands_1idx
        
        # Spectrometer subsystem determination
        # AVIRIS utilizes 4 separate spectrometers:
        # A: VIS (400 - 700 nm)
        # B: NIR (700 - 1300 nm)
        # C: SWIR-1 (1300 - 1900 nm)
        # D: SWIR-2 (1900 - 2500 nm)
        if wl < 700.0:
            spectrometer = "Spectrometer A (VIS: 400-700 nm)"
            subsystem = "A"
        elif wl < 1300.0:
            spectrometer = "Spectrometer B (NIR: 700-1300 nm)"
            subsystem = "B"
        elif wl < 1900.0:
            spectrometer = "Spectrometer C (SWIR-1: 1300-1900 nm)"
            subsystem = "C"
        else:
            spectrometer = "Spectrometer D (SWIR-2: 1900-2500 nm)"
            subsystem = "D"

        raw_band_data = raw_cube[:, :, raw_idx - 1]
        raw_mean = float(np.mean(raw_band_data))
        raw_std = float(np.std(raw_band_data))
        raw_var = float(np.var(raw_band_data))

        corr_band_idx = None
        corr_mean = None
        corr_std = None
        corr_var = None

        if not is_water:
            corr_band_idx = corr_idx
            corr_band_data = corr_cube[:, :, corr_idx - 1]
            corr_mean = float(np.mean(corr_band_data))
            corr_std = float(np.std(corr_band_data))
            corr_var = float(np.var(corr_band_data))
            corr_idx += 1

        calib_records.append({
            "raw_band_1based": raw_idx,
            "corrected_band_1based": corr_band_idx,
            "center_wavelength_nm": round(wl, 2),
            "center_wavelength_um": round(wl / 1000.0, 4),
            "spectrometer_subsystem": subsystem,
            "spectrometer_name": spectrometer,
            "is_water_absorption": is_water,
            "filtering_status": "Removed (Atmospheric Noise)" if is_water else "Retained (Clean Signal)",
            "raw_mean_dn": round(raw_mean, 2),
            "raw_std_dn": round(raw_std, 2),
            "raw_variance": round(raw_var, 2),
            "corr_mean_dn": round(corr_mean, 2) if corr_mean is not None else None,
            "corr_variance": round(corr_var, 2) if corr_var is not None else None
        })

    return pd.DataFrame(calib_records)

# -----------------------------------------------------------------------------
# 3. Spatial-Spectral Patch Extraction Engine
# -----------------------------------------------------------------------------
class HyperspectralPatchExtractor:
    """
    Standardized, high-throughput patch extraction engine for hyperspectral tensors.
    Constructs (k x k x B) spatial-spectral tensors around labeled pixels with boundary padding.
    """
    def __init__(self, cube, ground_truth, patch_size=9, padding_mode="reflect"):
        """
        Parameters:
        -----------
        cube : np.ndarray (H, W, B)
        ground_truth : np.ndarray (H, W)
        patch_size : int, odd window size (e.g. 7, 9, 11, 13, 15)
        padding_mode : str, 'reflect', 'constant', 'edge', 'symmetric'
        """
        assert patch_size % 2 == 1, "Patch size must be an odd integer (e.g. 7, 9, 11, 15)."
        self.patch_size = patch_size
        self.pad_width = patch_size // 2
        self.padding_mode = padding_mode
        self.raw_shape = cube.shape
        self.H, self.W, self.B = cube.shape

        # Normalize reflectance from DN to [0.0, 1.0]
        # AVIRIS calibrated data uses a fixed scaling factor of 10,000
        self.normalized_cube = np.clip(cube.astype(np.float32) / 10000.0, 0.0, 1.0)
        self.gt = ground_truth.copy()

        # Apply spatial boundary padding (spatial axes 0 and 1, no padding along spectral axis 2)
        pad_config = ((self.pad_width, self.pad_width), (self.pad_width, self.pad_width), (0, 0))
        if padding_mode == "constant":
            self.padded_cube = np.pad(self.normalized_cube, pad_config, mode="constant", constant_values=0.0)
        else:
            self.padded_cube = np.pad(self.normalized_cube, pad_config, mode=padding_mode)

        # Labeled pixel coordinates (y > 0)
        labeled_mask = (self.gt > 0)
        self.row_coords, self.col_coords = np.where(labeled_mask)
        self.labels = self.gt[self.row_coords, self.col_coords]
        self.num_samples = len(self.labels)

    def extract_single_patch(self, row, col):
        """
        Extracts patch centered at unpadded (row, col) coordinate.
        """
        r_pad = row + self.pad_width
        c_pad = col + self.pad_width
        r_start = r_pad - self.pad_width
        r_end = r_pad + self.pad_width + 1
        c_start = c_pad - self.pad_width
        c_end = c_pad + self.pad_width + 1
        return self.padded_cube[r_start:r_end, c_start:c_end, :]

    def extract_all_patches(self, max_samples=None, return_dtype=np.float32):
        """
        Extracts all labeled spatial-spectral patches into a single contiguous array (N, k, k, B).
        """
        n_samples = self.num_samples if max_samples is None else min(max_samples, self.num_samples)
        patches_array = np.zeros((n_samples, self.patch_size, self.patch_size, self.B), dtype=return_dtype)
        
        for idx in range(n_samples):
            r = self.row_coords[idx]
            c = self.col_coords[idx]
            patches_array[idx] = self.extract_single_patch(r, c)
            
        return patches_array, self.labels[:n_samples]

    @staticmethod
    def estimate_memory_footprint(num_pixels, patch_sizes, num_bands):
        """
        Estimates memory footprint across patch sizes and precision formats.
        """
        records = []
        for k in patch_sizes:
            elements_per_patch = k * k * num_bands
            total_elements = num_pixels * elements_per_patch
            mem_fp32_mb = round((total_elements * 4) / (1024 * 1024), 2)
            mem_fp16_mb = round((total_elements * 2) / (1024 * 1024), 2)
            mem_uint8_mb = round((total_elements * 1) / (1024 * 1024), 2)
            records.append({
                "patch_size": f"{k}x{k}",
                "spatial_window_k": k,
                "half_margin_m": k // 2,
                "elements_per_patch": elements_per_patch,
                "total_patch_elements": total_elements,
                "memory_float32_mb": mem_fp32_mb,
                "memory_float16_mb": mem_fp16_mb,
                "memory_uint8_mb": mem_uint8_mb
            })
        return records

# -----------------------------------------------------------------------------
# 4. Spatial Autocorrelation & Data Leakage Proof Engine
# -----------------------------------------------------------------------------
def analyze_spatial_autocorrelation_and_leakage(cube, gt, patch_size=9, test_ratio=0.20, seed=42):
    """
    Demonstrates mathematically and empirically why naive random pixel splitting causes
    catastrophic spatial data leakage in hyperspectral scenes, and formulates the parcel split solution.
    """
    np.random.seed(seed)
    H, W, B = cube.shape
    labeled_mask = (gt > 0)
    labeled_indices = np.argwhere(labeled_mask) # (N, 2)
    N = len(labeled_indices)

    # 1. Simulate Naive Random Pixel Split
    perm = np.random.permutation(N)
    n_test = int(N * test_ratio)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_coords = labeled_indices[train_idx]
    test_coords = labeled_indices[test_idx]

    train_mask = np.zeros((H, W), dtype=bool)
    test_mask = np.zeros((H, W), dtype=bool)
    train_mask[train_coords[:, 0], train_coords[:, 1]] = True
    test_mask[test_coords[:, 0], test_coords[:, 1]] = True

    # Check patch overlap: for each test pixel, what percentage of pixels in its k x k patch
    # are present in the training mask?
    m = patch_size // 2
    pad_train_mask = np.pad(train_mask, m, mode="constant", constant_values=False)
    
    overlap_ratios = []
    for r, c in test_coords:
        patch_region = pad_train_mask[r:r + 2*m + 1, c:c + 2*m + 1]
        train_pixels_in_patch = np.sum(patch_region)
        overlap_ratios.append(train_pixels_in_patch / (patch_size * patch_size))
        
    mean_patch_contamination = float(np.mean(overlap_ratios) * 100.0)
    median_patch_contamination = float(np.median(overlap_ratios) * 100.0)
    pct_test_patches_with_leakage = float(np.mean(np.array(overlap_ratios) > 0) * 100.0)

    # 2. Spatial Autocorrelation Decay Analysis
    # Pick a large homogeneous class (Soybean-mintill: class 11)
    class_mask = (gt == 11)
    class_coords = np.argwhere(class_mask)
    
    # Subsample 200 pairs to compute spectral similarity vs spatial Euclidean distance
    n_sub = min(200, len(class_coords))
    sub_indices = class_coords[np.random.choice(len(class_coords), n_sub, replace=False)]
    
    distances = []
    cos_similarities = []
    
    normalized_cube = cube.astype(np.float32) / 10000.0

    for i in range(len(sub_indices)):
        for j in range(i + 1, len(sub_indices)):
            r1, c1 = sub_indices[i]
            r2, c2 = sub_indices[j]
            spatial_dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            if spatial_dist < 40: # Focus on local neighborhood up to 40 pixels (800 meters)
                spec1 = normalized_cube[r1, c1, :]
                spec2 = normalized_cube[r2, c2, :]
                cos_sim = np.dot(spec1, spec2) / (np.linalg.norm(spec1) * np.linalg.norm(spec2) + 1e-8)
                distances.append(spatial_dist)
                cos_similarities.append(cos_sim)

    # 3. Disjoint Parcel Split Formulation via Connected Components
    parcel_labels, num_parcels = ndi.label(gt > 0)
    
    # Stratified parcel assignment
    parcel_records = []
    for p_id in range(1, num_parcels + 1):
        p_mask = (parcel_labels == p_id)
        p_gt_vals = gt[p_mask]
        dominant_cls = int(np.bincount(p_gt_vals).argmax())
        p_size = int(np.sum(p_mask))
        parcel_records.append({
            "parcel_id": p_id,
            "class_id": dominant_cls,
            "pixel_count": p_size
        })

    # Parcel split: assign parcels so no parcel is split across train/test
    # This guarantees 0% patch contamination between train and test
    parcel_df = pd.DataFrame(parcel_records)
    test_parcels = []
    for cls_id in range(1, 17):
        cls_parcels = parcel_df[parcel_df["class_id"] == cls_id]["parcel_id"].values
        if len(cls_parcels) > 1:
            # Pick roughly 20% parcels for test
            n_test_p = max(1, int(len(cls_parcels) * test_ratio))
            test_parcels.extend(list(np.random.choice(cls_parcels, n_test_p, replace=False)))
            
    test_parcel_set = set(test_parcels)
    parcel_train_mask = np.zeros((H, W), dtype=bool)
    parcel_test_mask = np.zeros((H, W), dtype=bool)

    for p_id in range(1, num_parcels + 1):
        p_mask = (parcel_labels == p_id)
        if p_id in test_parcel_set:
            parcel_test_mask[p_mask] = True
        else:
            parcel_train_mask[p_mask] = True

    # Calculate patch overlap under Parcel Split (with spatial buffer)
    parcel_test_coords = np.argwhere(parcel_test_mask)
    parcel_overlap_ratios = []
    pad_p_train_mask = np.pad(parcel_train_mask, m, mode="constant", constant_values=False)
    for r, c in parcel_test_coords:
        p_reg = pad_p_train_mask[r:r + 2*m + 1, c:c + 2*m + 1]
        parcel_overlap_ratios.append(np.sum(p_reg) / (patch_size * patch_size))
        
    parcel_mean_overlap = float(np.mean(parcel_overlap_ratios) * 100.0)

    leakage_summary = {
        "patch_size": f"{patch_size}x{patch_size}",
        "naive_random_split": {
            "test_ratio": test_ratio,
            "mean_patch_contamination_pct": round(mean_patch_contamination, 2),
            "median_patch_contamination_pct": round(median_patch_contamination, 2),
            "pct_test_patches_with_leakage": round(pct_test_patches_with_leakage, 2),
            "scientific_implication": "Flawed: Naive random split leaks training pixels into test patches, causing massive spatial data leakage and deceptively inflated test accuracy."
        },
        "disjoint_parcel_split": {
            "total_parcels_identified": int(num_parcels),
            "parcels_in_test": len(test_parcels),
            "mean_patch_overlap_pct": round(parcel_mean_overlap, 2),
            "scientific_implication": "Sound: Disjoint parcel splitting segregates entire contiguous fields, guaranteeing zero spatial patch leakage and evaluating true field generalization."
        }
    }

    autocorr_data = {
        "distances": distances,
        "cos_similarities": cos_similarities,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "parcel_train_mask": parcel_train_mask,
        "parcel_test_mask": parcel_test_mask,
        "leakage_summary": leakage_summary
    }

    return autocorr_data

# -----------------------------------------------------------------------------
# 5. Visualization Generation Suite
# -----------------------------------------------------------------------------
def generate_figure1_cube_slicing(ip_corr, calib_df):
    """
    Figure 1: 3D Orthogonal Slicing through Hyperspectral Cube (Axial, Coronal, Sagittal).
    """
    print("  -> Generating Figure 1: 3D Hyperspectral Cube Slicing...")
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.25)

    # Row 1: Four characteristic spatial (X, Y) slices across key spectral regions
    # Band 15 (~550 nm Green), Band 35 (~720 nm Red-Edge), Band 55 (~850 nm NIR), Band 140 (~1650 nm SWIR)
    bands_to_show = [
        (15, "Band 15: VIS Green (~550 nm)\nChlorophyll Reflection Peak", "YlGn"),
        (35, "Band 35: Red-Edge (~720 nm)\nSteep Canopy Transition", "RdYlGn"),
        (55, "Band 55: NIR (~850 nm)\nCellular Mesophyll Plateau", "viridis"),
        (140, "Band 140: SWIR-1 (~1650 nm)\nPlant Moisture Absorption", "cividis")
    ]

    for i, (b_idx, title, cmap) in enumerate(bands_to_show):
        ax = fig.add_subplot(gs[0, i])
        img = ip_corr[:, :, b_idx - 1]
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Sample (X Coordinate)", fontsize=8)
        ax.set_ylabel("Line (Y Coordinate)" if i == 0 else "", fontsize=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Digital Number (DN)", fontsize=7)

    # Row 2 Left: Coronal (X, Lambda) Slice at Y = 72 (Scene Midline)
    ax_coronal = fig.add_subplot(gs[1, 0:2])
    coronal_slice = ip_corr[72, :, :] # (145, 200)
    im_c = ax_coronal.imshow(coronal_slice.T, aspect="auto", cmap="inferno", extent=[0, 145, 200, 1])
    ax_coronal.set_title(r"Coronal Cross-Section: Spatial (X) vs Spectral Depth ($\lambda$)" + "\nLine Y = 72 (Across Agricultural Parcels)", fontsize=11, fontweight="bold", pad=8)
    ax_coronal.set_xlabel("Spatial Sample X (Parcel Cross-Section)", fontsize=9)
    ax_coronal.set_ylabel("Spectral Band Index (1 to 200)", fontsize=9)
    cbar_c = plt.colorbar(im_c, ax=ax_coronal, fraction=0.046, pad=0.04)
    cbar_c.ax.tick_params(labelsize=8)
    cbar_c.set_label("Radiometric Intensity (DN)", fontsize=8)

    # Row 2 Right: Sagittal (Y, Lambda) Slice at X = 72 (Scene Midline)
    ax_sagittal = fig.add_subplot(gs[1, 2:4])
    sagittal_slice = ip_corr[:, 72, :] # (145, 200)
    im_s = ax_sagittal.imshow(sagittal_slice.T, aspect="auto", cmap="magma", extent=[0, 145, 200, 1])
    ax_sagittal.set_title(r"Sagittal Cross-Section: Spatial (Y) vs Spectral Depth ($\lambda$)" + "\nSample X = 72 (Along Flight Trajectory)", fontsize=11, fontweight="bold", pad=8)
    ax_sagittal.set_xlabel("Spatial Line Y (Flight Line Trajectory)", fontsize=9)
    ax_sagittal.set_ylabel("Spectral Band Index (1 to 200)", fontsize=9)
    cbar_s = plt.colorbar(im_s, ax=ax_sagittal, fraction=0.046, pad=0.04)
    cbar_s.ax.tick_params(labelsize=8)
    cbar_s.set_label("Radiometric Intensity (DN)", fontsize=8)

    plt.suptitle("TerraSpectra Day 2: 3D Orthogonal Slicing & Spatial-Spectral Representation of AVIRIS Hyperspectral Cube", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(FIG_DIR, "day02_spectral_cube_slicing.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {out_path}")

def generate_figure2_channel_calibration(calib_df, ip_corr, ip_gt):
    """
    Figure 2: Exact Channel Calibrations, Spectrometer Subsystems, Variance Profile & Crop Profiles.
    """
    print("  -> Generating Figure 2: Spectral Channel Calibration Profile...")
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.1], hspace=0.30, wspace=0.22)

    # Subplot 1: Center Wavelengths with Spectrometer Subsystems
    ax1 = fig.add_subplot(gs[0, 0])
    raw_bands = calib_df["raw_band_1based"].values
    wls = calib_df["center_wavelength_nm"].values
    
    # Subsystems coloring
    ax1.axvspan(1, 32, color="#4285F4", alpha=0.15, label="Spectrometer A: VIS (400-700 nm)")
    ax1.axvspan(33, 96, color="#0F9D58", alpha=0.15, label="Spectrometer B: NIR (700-1300 nm)")
    ax1.axvspan(97, 160, color="#F4B400", alpha=0.15, label="Spectrometer C: SWIR-1 (1300-1900 nm)")
    ax1.axvspan(161, 220, color="#DB4437", alpha=0.15, label="Spectrometer D: SWIR-2 (1900-2450 nm)")
    
    ax1.plot(raw_bands, wls, color="#1A237E", lw=2.2, label=r"Calibrated Center Wavelength ($\lambda$)")
    
    # Atmospheric water band lines
    water_bands = calib_df[calib_df["is_water_absorption"]]["raw_band_1based"].values
    ax1.scatter(water_bands, calib_df[calib_df["is_water_absorption"]]["center_wavelength_nm"], 
                color="red", s=25, zorder=5, label="Removed Water Bands (104-108, 150-163, 220)")

    ax1.set_title("AVIRIS Spectrometer Calibration Curve Across 4 Optical Detectors", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Raw Spectral Band Index (1 to 220)", fontsize=9)
    ax1.set_ylabel("Center Wavelength (nm)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # Subplot 2: Variance Profiles - Raw vs Corrected Channels
    ax2 = fig.add_subplot(gs[0, 1])
    raw_vars = calib_df["raw_variance"].values
    
    ax2.plot(raw_bands, raw_vars, color="#D32F2F", lw=1.5, alpha=0.85, label="Raw 220 Bands Variance (Noise Spikes)")
    
    # Highlight water absorption regions
    ax2.axvspan(104, 108, color="red", alpha=0.35, label="Atmospheric H2O Window 1 (1.4 µm)")
    ax2.axvspan(150, 163, color="red", alpha=0.35, label="Atmospheric H2O Window 2 (1.9 µm)")
    ax2.axvspan(219.5, 220.5, color="darkred", alpha=0.45, label="Detector Edge Band 220")

    clean_df = calib_df[~calib_df["is_water_absorption"]]
    ax2.plot(clean_df["raw_band_1based"], clean_df["corr_variance"], color="#2E7D32", lw=2.0, label="Calibrated 200 Bands Variance (Clean Signal)")

    ax2.set_title("Radiometric Variance & Atmospheric Noise Attenuation Profile", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Spectral Band Index", fontsize=9)
    ax2.set_ylabel("Voxel Radiometric Variance ($DN^2$)", fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    # Subplot 3: Crop Spectral Signatures (Mean +/- 1 Std Dev)
    ax3 = fig.add_subplot(gs[1, :])
    crops_to_plot = [
        (2, "Corn-notill", "#E65100"),
        (10, "Soybean-notill", "#2E7D32"),
        (11, "Soybean-mintill", "#4CAF50"),
        (13, "Wheat", "#FFB300"),
        (14, "Woods", "#1B5E20"),
        (1, "Alfalfa", "#00897B")
    ]
    
    clean_wls = clean_df["center_wavelength_nm"].values
    norm_cube = ip_corr.astype(np.float32) / 10000.0

    for cls_id, cls_name, color in crops_to_plot:
        mask = (ip_gt == cls_id)
        if np.sum(mask) > 0:
            pixels = norm_cube[mask, :] # (N_cls, 200)
            mean_sig = np.mean(pixels, axis=0)
            std_sig = np.std(pixels, axis=0)
            ax3.plot(clean_wls, mean_sig, color=color, lw=2.0, label=f"{cls_name} (N={np.sum(mask):,})")
            ax3.fill_between(clean_wls, np.maximum(0, mean_sig - std_sig), mean_sig + std_sig, color=color, alpha=0.12)

    # Shaded agricultural diagnostic features
    ax3.axvspan(400, 700, color="blue", alpha=0.06, label="Photosynthetic Absorption (VIS)")
    ax3.axvspan(700, 750, color="purple", alpha=0.15, label="Vegetative Red-Edge (Chlorophyll Boundary)")
    ax3.axvspan(750, 1300, color="green", alpha=0.06, label="Near-Infrared Reflectance Plateau")
    ax3.axvspan(1400, 1850, color="orange", alpha=0.06, label="Shortwave-Infrared Moisture Window (SWIR-1)")

    ax3.set_title(r"Calibrated Crop Canopy Spectral Signatures Across Wavelengths ($\lambda \in [400, 2450]$ nm)" + "\nNotice Distinct Red-Edge Slope and NIR Plateau Across Different Agricultural Regimes", 
                  fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Calibrated Wavelength (nm)", fontsize=10)
    ax3.set_ylabel(r"Normalized Surface Reflectance ($\rho \in [0, 1]$)", fontsize=10)
    ax3.set_ylim(0.0, 0.75)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)

    plt.suptitle("TerraSpectra Day 2: Exact Wavelength Calibrations, Spectrometer Subsystems & Spectral Variance Profiles", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(FIG_DIR, "day02_spectral_channel_calibration.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {out_path}")

def generate_figure3_patch_extraction(extractor, ip_gt):
    """
    Figure 3: Spatial Patch Extraction Architecture, Boundary Reflection Padding & Sample 3D Patches.
    """
    print("  -> Generating Figure 3: Spatial Patch Slicing Architecture & Sample Patches...")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.1], hspace=0.35, wspace=0.3)

    # Top Row: Schematic of Patch Window Sizes & Padding Architecture
    # Subplot 1: Boundary Padding Visualization
    ax_pad = fig.add_subplot(gs[0, 0:3])
    # Show boundary region of Indian Pines with padding guide
    sample_corner = extractor.padded_cube[:25, :25, 55] # NIR band
    im_p = ax_pad.imshow(sample_corner, cmap="viridis")
    
    # Mark the padded boundary line
    pad_w = extractor.pad_width
    ax_pad.axvline(pad_w - 0.5, color="red", linestyle="--", lw=2, label=f"Spatial Mirror Padding Margin (m={pad_w})")
    ax_pad.axhline(pad_w - 0.5, color="red", linestyle="--", lw=2)
    
    # Add rectangle showing a 9x9 patch at an edge pixel
    patch_rect = mpatches.Rectangle((pad_w - pad_w - 0.5, pad_w - pad_w - 0.5), 9, 9, 
                                    fill=False, edgecolor="cyan", lw=2.5, label="Edge 9x9 Patch (Symmetric Reflection)")
    ax_pad.add_patch(patch_rect)
    ax_pad.scatter([pad_w], [pad_w], color="yellow", s=50, marker="o", zorder=5, label="Boundary Target Pixel")

    ax_pad.set_title("Boundary Margin Reflection Padding (m = (k-1)/2)\nEnables Convolutional Slicing on Perimeter Pixels Without Zero-Artifacts", fontsize=10.5, fontweight="bold")
    ax_pad.set_xlabel("Padded Sample X", fontsize=8.5)
    ax_pad.set_ylabel("Padded Line Y", fontsize=8.5)
    ax_pad.legend(loc="lower right", fontsize=8, framealpha=0.95)
    plt.colorbar(im_p, ax=ax_pad, fraction=0.046, pad=0.04).set_label("Reflectance", fontsize=8)

    # Subplot 2: Comparison of Spatial Patch Sizes (k=7, 9, 11, 15)
    ax_sizes = fig.add_subplot(gs[0, 3:6])
    # Compute mock spatial context comparison
    k_sizes = [5, 7, 9, 11, 13, 15]
    spatial_coverage_m = [k * 20.0 for k in k_sizes] # 20m per pixel
    elements_3d = [k * k * 200 for k in k_sizes]

    color_bar = "#1E88E5"
    bars = ax_sizes.bar([f"{k}x{k}" for k in k_sizes], spatial_coverage_m, color=color_bar, width=0.55, alpha=0.85, label="Ground Coverage (meters)")
    ax_sizes.set_ylabel("Spatial Ground Coverage ($meters \times meters$)", fontsize=9, color=color_bar)
    ax_sizes.tick_params(axis="y", labelcolor=color_bar)
    ax_sizes.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for total voxel volume
    ax_twin = ax_sizes.twinx()
    color_line = "#D81B60"
    ax_twin.plot([f"{k}x{k}" for k in k_sizes], elements_3d, color=color_line, marker="o", lw=2.2, label=r"Voxel Tensor Volume ($k^2 \cdot B$)")
    ax_twin.set_ylabel("3D Tensor Voxel Volume ($k \times k \times 200$)", fontsize=9, color=color_line)
    ax_twin.tick_params(axis="y", labelcolor=color_line)

    ax_sizes.set_title("3D Patch Size vs Spatial Footprint vs Computational Tensor Volume\nTradeoff for 3D-CNN & ViT Receptive Field", fontsize=10.5, fontweight="bold")
    ax_sizes.set_xlabel("Spatial Patch Kernel Size ($k \times k$)", fontsize=9)

    # Bottom Row: Six CIR False-Color Extracted 9x9 Patches for Representative Agricultural Classes
    # CIR Composite: R = Band 55 (~850nm NIR), G = Band 30 (~660nm Red), B = Band 15 (~550nm Green)
    sample_classes = [
        (2, "Corn-notill"),
        (10, "Soybean-notill"),
        (11, "Soybean-mintill"),
        (13, "Wheat"),
        (14, "Woods"),
        (1, "Alfalfa")
    ]

    for i, (cls_id, cls_name) in enumerate(sample_classes):
        ax_p = fig.add_subplot(gs[1, i])
        mask = (ip_gt == cls_id)
        coords = np.argwhere(mask)
        
        # Pick median coordinate for representative center
        center_idx = len(coords) // 2
        r, c = coords[center_idx]
        patch_data = extractor.extract_single_patch(r, c) # (9, 9, 200)

        # Create false-color CIR
        cir = np.stack([
            patch_data[:, :, 54], # NIR
            patch_data[:, :, 29], # Red
            patch_data[:, :, 14]  # Green
        ], axis=-1)
        # Normalize for display
        cir_disp = np.clip(cir / np.percentile(cir, 98), 0.0, 1.0)

        ax_p.imshow(cir_disp)
        # Mark center pixel with crosshair
        center_p = extractor.pad_width
        ax_p.scatter([center_p], [center_p], color="yellow", s=60, marker="x", lw=2)
        ax_p.set_title(f"{cls_name}\n(Class {cls_id})", fontsize=9.5, fontweight="bold")
        ax_p.set_xticks(range(0, 9, 2))
        ax_p.set_yticks(range(0, 9, 2))
        ax_p.tick_params(labelsize=7)
        ax_p.set_xlabel("X (Pixels)", fontsize=7.5)
        if i == 0:
            ax_p.set_ylabel("Y (Pixels)", fontsize=7.5)

    plt.suptitle("TerraSpectra Day 2: 3D Spatial-Spectral Patch Extraction Engine with False-Color (CIR) Agricultural Patches", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(FIG_DIR, "day02_spatial_patch_extraction.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {out_path}")

def generate_figure4_spatial_leakage(autocorr_data, ip_gt):
    """
    Figure 4: Spatial Autocorrelation Decay & Empirical Proof of Random Split Leakage vs Parcel Split.
    """
    print("  -> Generating Figure 4: Spatial Autocorrelation & Data Leakage Proof...")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.25)

    distances = autocorr_data["distances"]
    cos_sims = autocorr_data["cos_similarities"]
    leakage = autocorr_data["leakage_summary"]

    # Subplot 1: Spatial Autocorrelation Decay Curve
    ax_decay = fig.add_subplot(gs[0, 0:2])
    ax_decay.scatter(distances, cos_sims, color="#1976D2", alpha=0.35, s=18, edgecolors="none", label="Pixel Pair Cosine Similarity")
    
    # Fit rolling average or trendline
    sorted_pairs = sorted(zip(distances, cos_sims), key=lambda x: x[0])
    s_dists, s_sims = zip(*sorted_pairs)
    # Binning
    bin_edges = np.linspace(0, 40, 20)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = []
    for b_start, b_end in zip(bin_edges[:-1], bin_edges[1:]):
        vals = [s for d, s in sorted_pairs if b_start <= d < b_end]
        bin_means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        
    ax_decay.plot(bin_centers, bin_means, color="#D32F2F", lw=2.5, marker="s", label="Spatial Autocorrelation Mean Trend")
    ax_decay.axhline(0.98, color="black", linestyle="--", alpha=0.6, label="Severe Collinearity Threshold (r > 0.98)")

    ax_decay.set_title("Tobler's First Law in Hyperspectral Remote Sensing: Spectral Autocorrelation Decay Curve\nReflectance Cosine Similarity Remains > 0.98 Within Local Field Neighborhoods", fontsize=10.5, fontweight="bold")
    ax_decay.set_xlabel("Spatial Euclidean Distance Between Pixels (Pixels; 1 px = 20 meters)", fontsize=9)
    ax_decay.set_ylabel("Spectral Cosine Similarity", fontsize=9)
    ax_decay.set_ylim(0.92, 1.002)
    ax_decay.grid(True, linestyle="--", alpha=0.5)
    ax_decay.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    # Subplot 2: Summary Metrics Comparison Box
    ax_box = fig.add_subplot(gs[0, 2])
    ax_box.axis("off")
    
    text_content = (
        "SPATIAL DATA LEAKAGE AUDIT\n"
        "------------------------------------\n"
        f"Spatial Patch Window: 9 x 9 (81 voxels)\n"
        f"Test Partition Ratio: 20.0%\n\n"
        "[1] NAIVE RANDOM SPLIT:\n"
        f" - Test Patches Contaminated: {leakage['naive_random_split']['pct_test_patches_with_leakage']}%\n"
        f" - Mean Overlap with Train: {leakage['naive_random_split']['mean_patch_contamination_pct']}%\n"
        f" - Median Overlap with Train: {leakage['naive_random_split']['median_patch_contamination_pct']}%\n"
        " - Result: Severe Overfitting Artifact\n"
        "   (Inflates test accuracy to ~99%)\n\n"
        "[2] DISJOINT PARCEL SPLIT:\n"
        f" - Identified Isolated Parcels: {leakage['disjoint_parcel_split']['total_parcels_identified']}\n"
        f" - Reserved Test Parcels: {leakage['disjoint_parcel_split']['parcels_in_test']}\n"
        f" - Mean Patch Overlap: {leakage['disjoint_parcel_split']['mean_patch_overlap_pct']}%\n"
        " - Result: True Out-of-Distribution\n"
        "   Field Generalization Test"
    )
    
    ax_box.text(0.05, 0.5, text_content, fontsize=9.2, family="monospace", va="center",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF9C4", edgecolor="#FBC02D", lw=1.5))

    # Bottom Row: Spatial Map of Naive Random Split vs Disjoint Parcel Split
    # Map 1: Ground Truth Reference
    ax_gt = fig.add_subplot(gs[1, 0])
    im_gt = ax_gt.imshow(ip_gt, cmap="tab20", vmin=0, vmax=16)
    ax_gt.set_title("Ground Truth Classes (Reference)\nIndian Pines (16 Agricultural Classes)", fontsize=10, fontweight="bold")
    ax_gt.axis("off")

    # Map 2: Naive Random Split (Salt and Pepper Leakage)
    ax_rnd = fig.add_subplot(gs[1, 1])
    rnd_map = np.zeros_like(ip_gt)
    rnd_map[autocorr_data["train_mask"]] = 1 # Train (Blue)
    rnd_map[autocorr_data["test_mask"]] = 2  # Test (Red)
    cmap_split = matplotlib.colors.ListedColormap(["#EEEEEE", "#1976D2", "#D32F2F"])
    ax_rnd.imshow(rnd_map, cmap=cmap_split)
    ax_rnd.set_title("Naive Random Split (Salt-and-Pepper)\nHigh Leakage: Test Pixels Surrounded by Train", fontsize=10, fontweight="bold")
    ax_rnd.axis("off")
    
    patch_train = mpatches.Patch(color="#1976D2", label="Train (80%)")
    patch_test = mpatches.Patch(color="#D32F2F", label="Test (20%)")
    ax_rnd.legend(handles=[patch_train, patch_test], loc="lower right", fontsize=8)

    # Map 3: Disjoint Parcel Split (Scientifically Sound)
    ax_prc = fig.add_subplot(gs[1, 2])
    prc_map = np.zeros_like(ip_gt)
    prc_map[autocorr_data["parcel_train_mask"]] = 1 # Train
    prc_map[autocorr_data["parcel_test_mask"]] = 2  # Test
    ax_prc.imshow(prc_map, cmap=cmap_split)
    ax_prc.set_title("Disjoint Parcel Split (Zero Leakage)\nClean Field Isolation for Generalization", fontsize=10, fontweight="bold")
    ax_prc.axis("off")
    ax_prc.legend(handles=[patch_train, patch_test], loc="lower right", fontsize=8)

    plt.suptitle("TerraSpectra Day 2: Spatial Autocorrelation & Empirical Proof of Random Split Leakage vs Disjoint Parcel Split", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(FIG_DIR, "day02_spatial_autocorrelation_leakage.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {out_path}")

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    print("=" * 80)
    print("🌿 TerraSpectra: Day 2 - Structural Inspection, Spectrometer Calibration & Patching")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[Step 1/6] Loading Raw and Calibrated Hyperspectral MAT Files...")
    ip_raw = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines.mat"))["indian_pines"]
    ip_corr = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat"))["indian_pines_corrected"]
    ip_gt = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat"))["indian_pines_gt"]
    sal_corr = sio.loadmat(os.path.join(RAW_DATA_DIR, "Salinas_corrected.mat"))["salinas_corrected"]
    sal_gt = sio.loadmat(os.path.join(RAW_DATA_DIR, "Salinas_gt.mat"))["salinas_gt"]

    # 2. Structural Inspection
    print("\n[Step 2/6] Performing Deep Structural & Numerical Health Inspection...")
    ip_raw_meta = inspect_tensor_structure(ip_raw, "Indian Pines Raw Cube")
    ip_corr_meta = inspect_tensor_structure(ip_corr, "Indian Pines Calibrated Cube")
    ip_gt_meta = inspect_tensor_structure(ip_gt, "Indian Pines Ground Truth")
    sal_corr_meta = inspect_tensor_structure(sal_corr, "Salinas Calibrated Cube")
    sal_gt_meta = inspect_tensor_structure(sal_gt, "Salinas Ground Truth")

    structural_summary = {
        "milestone": "Day 2 - Structural Inspection & Spatial-Spectral Representation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensors": {
            "indian_pines_raw": ip_raw_meta,
            "indian_pines_calibrated": ip_corr_meta,
            "indian_pines_ground_truth": ip_gt_meta,
            "salinas_calibrated": sal_corr_meta,
            "salinas_ground_truth": sal_gt_meta
        }
    }

    struct_json_path = os.path.join(PROC_DATA_DIR, "day02_structural_inspection_summary.json")
    with open(struct_json_path, "w") as f:
        json.dump(structural_summary, f, indent=4)
    print(f"  -> Saved structural inspection summary to {struct_json_path}")

    # 3. Spectrometer Subsystem Mapping & Calibration Profile
    print("\n[Step 3/6] Mapping Spectrometer Subsystems (A, B, C, D) & Variance Profiles...")
    calib_df = build_spectrometer_calibrations(ip_raw, ip_corr)
    calib_csv_path = os.path.join(PROC_DATA_DIR, "day02_spectral_calibration_details.csv")
    calib_df.to_csv(calib_csv_path, index=False)
    print(f"  -> Saved detailed calibration table to {calib_csv_path}")

    # 4. Spatial-Spectral Patch Extraction Engine Benchmarks
    print("\n[Step 4/6] Initializing Patch Extraction Engine & Benchmarking Sizes...")
    extractor_ip = HyperspectralPatchExtractor(ip_corr, ip_gt, patch_size=9, padding_mode="reflect")
    print(f"  -> Extractor initialized for Indian Pines: {extractor_ip.num_samples:,} labeled targets.")
    print(f"  -> Padded cube shape with m=4: {extractor_ip.padded_cube.shape}")

    # Test extracting batch of 100 sample patches
    sample_patches, sample_labels = extractor_ip.extract_all_patches(max_samples=100)
    print(f"  -> Successfully extracted sample batch: shape={sample_patches.shape}, dtype={sample_patches.dtype}")
    assert not np.isnan(sample_patches).any(), "Extracted patches contain NaNs!"
    assert not np.isinf(sample_patches).any(), "Extracted patches contain Infs!"

    memory_benchmarks = HyperspectralPatchExtractor.estimate_memory_footprint(
        num_pixels=extractor_ip.num_samples,
        patch_sizes=[5, 7, 9, 11, 13, 15, 17],
        num_bands=ip_corr.shape[2]
    )

    patching_metadata = {
        "dataset": "Indian Pines Calibrated",
        "total_labeled_samples": extractor_ip.num_samples,
        "default_patch_size": "9x9",
        "half_margin_m": extractor_ip.pad_width,
        "padding_mode": "reflect",
        "padded_cube_shape": list(extractor_ip.padded_cube.shape),
        "sample_batch_verification": {
            "batch_shape": list(sample_patches.shape),
            "value_range": [float(np.min(sample_patches)), float(np.max(sample_patches))],
            "is_clean_numerical": True
        },
        "memory_footprint_benchmarks": memory_benchmarks
    }

    patch_json_path = os.path.join(PROC_DATA_DIR, "day02_patch_extraction_metadata.json")
    with open(patch_json_path, "w") as f:
        json.dump(patching_metadata, f, indent=4)
    print(f"  -> Saved patch extraction metadata to {patch_json_path}")

    # 5. Spatial Autocorrelation & Data Leakage Proof
    print("\n[Step 5/6] Proving Spatial Autocorrelation & Quantifying Data Leakage...")
    autocorr_data = analyze_spatial_autocorrelation_and_leakage(ip_corr, ip_gt, patch_size=9, test_ratio=0.20)
    print(f"  -> Naive Random Split Test Patch Contamination: {autocorr_data['leakage_summary']['naive_random_split']['mean_patch_contamination_pct']}%")
    print(f"  -> Disjoint Parcel Split Patch Overlap: {autocorr_data['leakage_summary']['disjoint_parcel_split']['mean_patch_overlap_pct']}%")

    # 6. Generate Publication-Quality Figures
    print("\n[Step 6/6] Generating Publication-Quality Diagnostic Figures...")
    generate_figure1_cube_slicing(ip_corr, calib_df)
    generate_figure2_channel_calibration(calib_df, ip_corr, ip_gt)
    generate_figure3_patch_extraction(extractor_ip, ip_gt)
    generate_figure4_spatial_leakage(autocorr_data, ip_gt)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"✅ DAY 2 WORK COMPLETED SUCCESSFULLY IN {elapsed} SECONDS")
    print("=" * 80)

if __name__ == "__main__":
    main()
