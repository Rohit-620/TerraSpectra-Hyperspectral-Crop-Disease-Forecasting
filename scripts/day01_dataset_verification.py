"""
TerraSpectra: Day 1 - Hyperspectral Dataset Exploration, Metadata Extraction, and Visualization
Generates metadata catalogs, wavelength centers, class distributions, and research figures.
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure output directories exist
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(PROC_DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Class Dictionaries
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Wavelength Calculation (AVIRIS Indian Pines 220 & 200 bands)
# ---------------------------------------------------------
def generate_aviris_wavelengths():
    """
    Generate calibrated center wavelengths for AVIRIS 220 bands (approx 400 nm to 2450 nm).
    Marks water absorption bands (104-108, 150-163, and 220).
    """
    # AVIRIS channels nominal wavelength calculation
    # Subdivided across 4 spectrometers: A (VIS: 400-700nm), B (NIR: 700-1300nm), C (SWIR1: 1300-1900nm), D (SWIR2: 1900-2500nm)
    start_nm = 400.0
    end_nm = 2450.0
    nominal_centers = np.linspace(start_nm, end_nm, 220)

    # 1-indexed water absorption bands in Indian Pines
    water_bands_1idx = set(list(range(104, 109)) + list(range(150, 164)) + [220])

    wavelength_records = []
    corrected_idx = 1

    for raw_idx in range(1, 221):
        wl = nominal_centers[raw_idx - 1]
        is_water = raw_idx in water_bands_1idx
        
        # Determine spectral region
        if wl < 700:
            region = "VIS"
        elif wl < 1100:
            region = "NIR"
        elif wl < 1400:
            region = "SWIR-1 (Water Absorption / Transition)"
        elif wl < 1850:
            region = "SWIR-1"
        else:
            region = "SWIR-2"

        wavelength_records.append({
            "raw_band_index_1based": raw_idx,
            "corrected_band_index_1based": None if is_water else corrected_idx,
            "wavelength_nm": round(wl, 2),
            "wavelength_um": round(wl / 1000.0, 4),
            "spectral_region": region,
            "is_water_absorption_band": is_water,
            "status": "Removed (Atmospheric Noise)" if is_water else "Retained (Clean Signal)"
        })

        if not is_water:
            corrected_idx += 1

    return pd.DataFrame(wavelength_records)

# ---------------------------------------------------------
# 3. Process Indian Pines Dataset
# ---------------------------------------------------------
def process_indian_pines():
    print("[1/4] Processing AVIRIS Indian Pines Dataset...")
    cube_corr_path = os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat")
    cube_raw_path = os.path.join(RAW_DATA_DIR, "Indian_pines.mat")
    gt_path = os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat")

    cube_corr = sio.loadmat(cube_corr_path)["indian_pines_corrected"]
    cube_raw = sio.loadmat(cube_raw_path)["indian_pines"]
    gt = sio.loadmat(gt_path)["indian_pines_gt"]

    # Class distribution
    unique, counts = np.unique(gt, return_counts=True)
    total_pixels = gt.size
    labeled_pixels = np.sum(counts[unique > 0])

    dist_records = []
    for cls_id, count in zip(unique, counts):
        cls_name = INDIAN_PINES_CLASSES.get(cls_id, f"Class {cls_id}")
        pct_total = (count / total_pixels) * 100
        pct_labeled = (count / labeled_pixels * 100) if cls_id > 0 else 0.0
        dist_records.append({
            "class_id": int(cls_id),
            "class_name": cls_name,
            "pixel_count": int(count),
            "pct_total_image": round(pct_total, 2),
            "pct_labeled_samples": round(pct_labeled, 2)
        })

    dist_df = pd.DataFrame(dist_records)
    dist_csv_path = os.path.join(PROC_DATA_DIR, "indian_pines_class_distribution.csv")
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"  -> Saved class distribution to {dist_csv_path}")

    # Metadata dictionary
    metadata = {
        "dataset_name": "AVIRIS Indian Pines Hyperspectral Scene",
        "sensor": "Airborne Visible / Infrared Imaging Spectrometer (AVIRIS)",
        "platform": "NASA ER-2 Aircraft (Airborne)",
        "acquisition_date": "June 12, 1992",
        "location": "Northwest Indiana, USA (Pine Township)",
        "spatial_coverage": "Agricultural fields and forested areas",
        "spatial_dimensions": {
            "lines_height": int(cube_corr.shape[0]),
            "samples_width": int(cube_corr.shape[1]),
            "total_pixels": int(total_pixels)
        },
        "spatial_resolution_meters": 20.0,
        "spectral_channels": {
            "raw_bands": int(cube_raw.shape[2]),
            "corrected_clean_bands": int(cube_corr.shape[2]),
            "removed_noise_bands_count": int(cube_raw.shape[2] - cube_corr.shape[2]),
            "removed_band_indices_1based": [104, 105, 106, 107, 108, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 220]
        },
        "spectral_range_nm": [400.0, 2450.0],
        "spectral_bandwidth_nm": "approx. 10 nm nominal",
        "data_format": "MATLAB .mat (v7)",
        "data_type": str(cube_corr.dtype),
        "radiometric_range": {
            "min_dn": float(np.min(cube_corr)),
            "max_dn": float(np.max(cube_corr)),
            "mean_dn": float(np.mean(cube_corr)),
            "std_dn": float(np.std(cube_corr))
        },
        "ground_truth": {
            "total_classes": 16,
            "labeled_pixels": int(labeled_pixels),
            "background_pixels": int(counts[unique == 0][0]),
            "labeled_percentage": round(float(labeled_pixels / total_pixels * 100), 2)
        }
    }

    meta_json_path = os.path.join(PROC_DATA_DIR, "indian_pines_metadata.json")
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  -> Saved metadata to {meta_json_path}")

    return cube_corr, gt, dist_df, metadata

# ---------------------------------------------------------
# 4. Process Salinas Dataset
# ---------------------------------------------------------
def process_salinas():
    print("[2/4] Processing AVIRIS Salinas Valley Dataset...")
    cube_corr_path = os.path.join(RAW_DATA_DIR, "Salinas_corrected.mat")
    gt_path = os.path.join(RAW_DATA_DIR, "Salinas_gt.mat")

    cube_corr = sio.loadmat(cube_corr_path)["salinas_corrected"]
    gt = sio.loadmat(gt_path)["salinas_gt"]

    unique, counts = np.unique(gt, return_counts=True)
    total_pixels = gt.size
    labeled_pixels = np.sum(counts[unique > 0])

    dist_records = []
    for cls_id, count in zip(unique, counts):
        cls_name = SALINAS_CLASSES.get(cls_id, f"Class {cls_id}")
        pct_total = (count / total_pixels) * 100
        pct_labeled = (count / labeled_pixels * 100) if cls_id > 0 else 0.0
        dist_records.append({
            "class_id": int(cls_id),
            "class_name": cls_name,
            "pixel_count": int(count),
            "pct_total_image": round(pct_total, 2),
            "pct_labeled_samples": round(pct_labeled, 2)
        })

    dist_df = pd.DataFrame(dist_records)
    dist_csv_path = os.path.join(PROC_DATA_DIR, "salinas_class_distribution.csv")
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"  -> Saved class distribution to {dist_csv_path}")

    metadata = {
        "dataset_name": "AVIRIS Salinas Valley Hyperspectral Scene",
        "sensor": "Airborne Visible / Infrared Imaging Spectrometer (AVIRIS)",
        "platform": "NASA ER-2 Aircraft (Airborne)",
        "acquisition_date": "October 1998",
        "location": "Salinas Valley, California, USA",
        "spatial_coverage": "Intensive agricultural valley (vegetables, vineyards, fallow)",
        "spatial_dimensions": {
            "lines_height": int(cube_corr.shape[0]),
            "samples_width": int(cube_corr.shape[1]),
            "total_pixels": int(total_pixels)
        },
        "spatial_resolution_meters": 3.7,
        "spectral_channels": {
            "corrected_clean_bands": int(cube_corr.shape[2]),
            "removed_noise_bands_count": 20,
            "removed_band_indices_1based": [108, 109, 110, 111, 112, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 224]
        },
        "spectral_range_nm": [400.0, 2500.0],
        "spectral_bandwidth_nm": "approx. 10 nm nominal",
        "data_format": "MATLAB .mat (v7)",
        "data_type": str(cube_corr.dtype),
        "radiometric_range": {
            "min_dn": float(np.min(cube_corr)),
            "max_dn": float(np.max(cube_corr)),
            "mean_dn": float(np.mean(cube_corr)),
            "std_dn": float(np.std(cube_corr))
        },
        "ground_truth": {
            "total_classes": 16,
            "labeled_pixels": int(labeled_pixels),
            "background_pixels": int(counts[unique == 0][0]),
            "labeled_percentage": round(float(labeled_pixels / total_pixels * 100), 2)
        }
    }

    meta_json_path = os.path.join(PROC_DATA_DIR, "salinas_metadata.json")
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  -> Saved metadata to {meta_json_path}")

    return cube_corr, gt, dist_df, metadata

# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------
def generate_visualizations(ip_cube, ip_gt, ip_dist, sal_cube, sal_gt, sal_dist, wl_df):
    print("[3/4] Generating Publication-Quality Research Visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Sensor Comparison Matrix (Satellite vs Airborne)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sensors = [
        "NASA EO-1 Hyperion (Satellite)",
        "ASI PRISMA (Satellite)",
        "DLR EnMAP (Satellite)",
        "AVIRIS Indian Pines (Airborne Benchmark)",
        "AVIRIS Salinas (Airborne Benchmark)"
    ]
    total_bands = [242, 239, 242, 220, 224]
    usable_bands = [198, 230, 224, 200, 204]
    spatial_res = [30.0, 30.0, 30.0, 20.0, 3.7]

    y = np.arange(len(sensors))
    width = 0.35

    rects1 = ax.barh(y - width/2, total_bands, width, label='Total Spectral Channels', color='#2b5c8f', alpha=0.9)
    rects2 = ax.barh(y + width/2, usable_bands, width, label='Usable Bands (Post Water-Vapor Filtering)', color='#2e8b57', alpha=0.9)

    ax.set_xlabel('Number of Spectral Bands', fontsize=12, fontweight='bold')
    ax.set_title('Hyperspectral Sensor Comparison: Spaceborne vs. Airborne Precision Datasets\nTerraSpectra Research Analysis (Day 1)', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(sensors, fontsize=11, fontweight='semibold')
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.set_xlim(0, 280)

    # Annotate spatial resolution
    for i, res in enumerate(spatial_res):
        ax.text(250, i, f'GSD: {res} m', va='center', ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#f4f4f4', edgecolor='#ccc'))

    # Annotate bar values
    for rect in rects1:
        w = rect.get_width()
        ax.text(w + 3, rect.get_y() + rect.get_height()/2, f'{int(w)}', va='center', fontsize=9, color='#1e3d59', fontweight='bold')
    for rect in rects2:
        w = rect.get_width()
        ax.text(w + 3, rect.get_y() + rect.get_height()/2, f'{int(w)}', va='center', fontsize=9, color='#1c5233', fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "day01_sensor_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # Figure 2: Indian Pines Class Distribution
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    ip_labeled = ip_dist[ip_dist["class_id"] > 0].sort_values(by="pixel_count", ascending=True)
    bars = ax.barh(ip_labeled["class_name"], ip_labeled["pixel_count"], color='#388e3c', edgecolor='#1b5e20', alpha=0.85)

    ax.set_xlabel('Sample Count (Pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Indian Pines Ground-Truth Class Distribution\n(16 Agricultural Vegetation & Land Cover Classes)', fontsize=14, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        pct = (w / ip_labeled["pixel_count"].sum()) * 100
        ax.text(w + 25, bar.get_y() + bar.get_height()/2, f'{int(w)} ({pct:.1f}%)', va='center', fontsize=9, fontweight='semibold')

    ax.set_xlim(0, max(ip_labeled["pixel_count"]) * 1.18)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "day01_indian_pines_classes.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # Figure 3: Salinas Valley Class Distribution
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    sal_labeled = sal_dist[sal_dist["class_id"] > 0].sort_values(by="pixel_count", ascending=True)
    bars = ax.barh(sal_labeled["class_name"], sal_labeled["pixel_count"], color='#1976d2', edgecolor='#0d47a1', alpha=0.85)

    ax.set_xlabel('Sample Count (Pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Salinas Valley Ground-Truth Class Distribution\n(Agricultural Phenology, Weed Infestation, & Senescence Classes)', fontsize=14, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        pct = (w / sal_labeled["pixel_count"].sum()) * 100
        ax.text(w + 50, bar.get_y() + bar.get_height()/2, f'{int(w)} ({pct:.1f}%)', va='center', fontsize=9, fontweight='semibold')

    ax.set_xlim(0, max(sal_labeled["pixel_count"]) * 1.18)
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, "day01_salinas_classes.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig3_path}")

    # Figure 4: Hyperspectral Cube False-Color & Ground Truth Spatial Map
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=300)

    # Normalize helper
    def norm_band(b):
        b_min, b_max = np.percentile(b, 2), np.percentile(b, 98)
        clipped = np.clip(b, b_min, b_max)
        return (clipped - b_min) / (b_max - b_min + 1e-6)

    # Indian Pines False Color (NIR: Band 40 ~800nm, Red: Band 25 ~650nm, Green: Band 15 ~550nm)
    # Band indexing in 200-band cube (approx):
    r_band = norm_band(ip_cube[:, :, 40]) # NIR
    g_band = norm_band(ip_cube[:, :, 25]) # Red
    b_band = norm_band(ip_cube[:, :, 15]) # Green
    ip_cir = np.dstack([r_band, g_band, b_band])

    axes[0].imshow(ip_cir)
    axes[0].set_title('Indian Pines False-Color (CIR)\nR: NIR (800nm), G: Red, B: Green', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Indian Pines Ground Truth Map
    cmap_ip = plt.cm.get_cmap('tab20', 17)
    im1 = axes[1].imshow(ip_gt, cmap=cmap_ip, vmin=0, vmax=16)
    axes[1].set_title('Indian Pines Ground Truth Map\n(16 Crop & Vegetation Classes)', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Salinas False Color CIR
    sal_r = norm_band(sal_cube[:, :, 40])
    sal_g = norm_band(sal_cube[:, :, 25])
    sal_b = norm_band(sal_cube[:, :, 15])
    sal_cir = np.dstack([sal_r, sal_g, sal_b])

    axes[2].imshow(sal_cir)
    axes[2].set_title('Salinas False-Color (CIR)\nHigh Spatial GSD (3.7m)', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # Salinas Ground Truth Map
    im2 = axes[3].imshow(sal_gt, cmap=cmap_ip, vmin=0, vmax=16)
    axes[3].set_title('Salinas Ground Truth Map\n(Phenology & Growth Stages)', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    plt.suptitle('TerraSpectra Day 1: Hyperspectral Image Cubes & Ground-Truth Spatial Verification', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, "day01_hyperspectral_cube_preview.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig4_path}")

def main():
    print("=" * 75)
    print(" TerraSpectra Day 1: Hyperspectral Dataset Exploration & Analysis")
    print("=" * 75)

    # 1. Wavelengths
    wl_df = generate_aviris_wavelengths()
    wl_csv_path = os.path.join(PROC_DATA_DIR, "indian_pines_wavelengths.csv")
    wl_df.to_csv(wl_csv_path, index=False)
    print(f"[Done] Generated wavelength lookup table with {len(wl_df)} bands -> {wl_csv_path}")

    # 2. Indian Pines
    ip_cube, ip_gt, ip_dist, ip_meta = process_indian_pines()

    # 3. Salinas
    sal_cube, sal_gt, sal_dist, sal_meta = process_salinas()

    # 4. Visualizations
    generate_visualizations(ip_cube, ip_gt, ip_dist, sal_cube, sal_gt, sal_dist, wl_df)

    print("=" * 75)
    print(" DAY 1 PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    main()
